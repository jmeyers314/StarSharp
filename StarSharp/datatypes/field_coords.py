from __future__ import annotations

from dataclasses import dataclass, replace

import astropy.units as u
import batoid
import galsim
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from batoid_rubin import LSSTBuilder
from galsim.fitswcs import FittedSIPWCS
from galsim.wcs import BaseWCS, CelestialWCS
from lsst.afw.cameraGeom import FOCAL_PLANE, Camera
from numpy.typing import NDArray


@dataclass(frozen=True)
class FieldCoords:
    """Field coordinates in any supported frame and space.

    Parameters
    ----------
    x, y : Quantity
        Coordinate values with units.  Angular units (e.g. ``u.deg``,
        ``u.arcsec``) indicate field-angle space; length units
        (e.g. ``u.mm``) indicate focal-plane space.
    frame : str
        Coordinate frame: ``'ocs'`` (optical, default), ``'ccs'`` (camera),
        ``'dvcs'`` (detector), or ``'edcs'`` (engineering).
    rtp : Angle or None
        Rotation angle from OCS to CCS frame.  Required for frame conversions.
    wcs : BaseWCS or None
        WCS for converting between field-angle and focal-plane space.  Required for
        space conversions.  WCS assumes field angles are in radians in OCS and
        focal-plane coordinates are in mm in CCS.
    camera : Camera or None
        Camera geometry for determining detector numbers.  Required for
        ``detnum`` property.
    """

    VALID_FRAMES = ("ocs", "ccs", "dvcs", "edcs")

    x: Quantity
    y: Quantity
    frame: str = "ocs"
    rtp: Angle | None = None
    wcs: BaseWCS | None = None
    camera: Camera | None = None

    def __post_init__(self):
        if not isinstance(self.x, Quantity) or not isinstance(self.y, Quantity):
            raise TypeError("x and y must be astropy Quantity instances with units")
        object.__setattr__(self, "x", np.atleast_1d(self.x))
        object.__setattr__(self, "y", np.atleast_1d(self.y))
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have the same shape, got {self.x.shape} and {self.y.shape}"
            )
        # Coerce frame to lower case, but preserve original name (including 'edcs')
        frame = self.frame.lower()
        object.__setattr__(self, "frame", frame)
        if frame not in self.VALID_FRAMES:
            raise ValueError(
                f"frame must be one of {self.VALID_FRAMES}, got {self.frame!r}"
            )
        if not (self.x.unit.physical_type == self.y.unit.physical_type):
            raise ValueError(
                f"x and y must have compatible units, "
                f"got {self.x.unit} and {self.y.unit}"
            )
        if self.x.unit.physical_type not in ("angle", "length"):
            raise ValueError(
                f"units must be angular or length, "
                f"got physical type {self.x.unit.physical_type!r}"
            )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(
                f"{name} must be set on the FieldCoords to use this property"
            )
        return val

    @classmethod
    def _build_wcs(
        cls,
        builder: LSSTBuilder,
        rtp: Angle,
        wavelength: Quantity,
    ) -> BaseWCS:
        rotated = builder.with_rtp(rtp).build()
        nrad = 20
        th_u, th_v = batoid.utils.hexapolar(
            np.deg2rad(2.0), nrad=nrad, naz=int(2 * np.pi * nrad)
        )
        rays = batoid.RayVector.fromFieldAngles(
            theta_x=th_u,
            theta_y=th_v,
            projection="gnomonic",
            optic=rotated,
            wavelength=wavelength.to_value(u.m),
        )
        rotated.trace(rays)
        wcs = galsim.FittedSIPWCS(
            rays.x * 1000, rays.y * 1000, th_u, th_v, order=3
        )  # Use mm <-> radians
        return wcs

    @classmethod
    def from_builder(
        cls,
        x: Quantity,
        y: Quantity,
        *,
        builder: LSSTBuilder,
        wavelength: Quantity,
        rtp: Angle | None = None,
        **kwargs,
    ) -> FieldCoords:
        if rtp is None:
            rtp = Angle("0 deg")
        wcs = cls._build_wcs(builder, rtp, wavelength)
        return cls(x, y, rtp=rtp, wcs=wcs, **kwargs)

    @property
    def space(self) -> str:
        """Inferred coordinate space: ``'angle'`` or ``'focal_plane'``."""
        if self.x.unit.physical_type == "angle":
            return "angle"
        return "focal_plane"

    def _rot(self, angle: Angle, frame: str) -> FieldCoords:
        """Return a new FieldCoords with x/y rotated by *angle*."""
        rx = self.x * np.cos(angle) + self.y * np.sin(angle)
        ry = -self.x * np.sin(angle) + self.y * np.cos(angle)
        return replace(self, x=rx, y=ry, frame=frame)

    @property
    def ocs(self) -> FieldCoords:
        """This coordinate in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self.ccs._rot(-rtp, "ocs")

    @property
    def ccs(self) -> FieldCoords:
        """This coordinate in the CCS frame (always frame='ccs')."""
        if self.frame == "ccs":
            return self
        if self.frame == "ocs":
            rtp = self._require("rtp")
            return self._rot(rtp, "ccs")
        x = self.x
        y = self.y
        if self.frame == "dvcs":
            x, y = y, x
        return replace(self, x=x, y=y, frame="ccs")

    @property
    def edcs(self) -> FieldCoords:
        """This coordinate in the EDCS frame (synonym for CCS, but preserves name)."""
        if self.frame == "edcs":
            return self
        ccs = self.ccs
        return replace(ccs, frame="edcs")

    @property
    def dvcs(self) -> FieldCoords:
        """This coordinate in the DVCS frame (transpose of EDCS/CCS)."""
        if self.frame == "dvcs":
            return self
        ccs = self.ccs
        return replace(ccs, x=ccs.y, y=ccs.x, frame="dvcs")

    @property
    def nfield(self) -> int:
        """Number of field points (last axis)."""
        return self.x.shape[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the leading batch dimensions."""
        return self.x.shape[:-1]

    @property
    def focal_plane(self) -> FieldCoords:
        """This coordinate in focal-plane space."""
        if self.space == "focal_plane":
            return self
        wcs = self._require("wcs")
        field = self.ocs
        orig_shape = field.x.shape
        fx = field.x.to_value(u.radian).ravel()
        fy = field.y.to_value(u.radian).ravel()
        args = [fx, fy]
        kwargs = {}
        if isinstance(wcs, CelestialWCS):
            kwargs["units"] = "radians"
        fpx, fpy = wcs.toImage(*args, **kwargs)
        fpx = fpx.reshape(orig_shape)
        fpy = fpy.reshape(orig_shape)
        fp_ccs = replace(self, x=fpx << u.mm, y=fpy << u.mm, frame="ccs")
        return getattr(fp_ccs, self.frame)

    @property
    def angle(self) -> FieldCoords:
        """This coordinate in field-angle space (OCS frame)."""
        if self.space == "angle":
            return self
        wcs = self._require("wcs")
        fp = self.ccs
        orig_shape = fp.x.shape
        fpx = fp.x.to_value(u.mm).ravel()
        fpy = fp.y.to_value(u.mm).ravel()
        args = [fpx, fpy]
        kwargs = {}
        if isinstance(wcs, CelestialWCS):
            kwargs["units"] = "radians"
        fx, fy = wcs.toWorld(*args, **kwargs)
        # Really need to move away from CelestialWCS, but for now can just wrap
        # towards 0
        fx[fx > np.pi] -= 2 * np.pi
        fy[fy > np.pi] -= 2 * np.pi
        fx = fx.reshape(orig_shape)
        fy = fy.reshape(orig_shape)
        field_ocs = replace(self, x=fx << u.radian, y=fy << u.radian, frame="ocs")
        return getattr(field_ocs, self.frame)

    @property
    def detnum(self) -> int | NDArray[np.integer]:
        """Detector number(s). Returns -1 for points off any detector."""
        camera = self._require("camera")
        fp = self.focal_plane.ccs
        orig_shape = fp.x.shape
        x = fp.x.to_value(u.mm).ravel()
        y = fp.y.to_value(u.mm).ravel()

        result = np.full(x.shape, -1, dtype=int)
        for det in camera:
            corners = det.getCorners(FOCAL_PLANE)
            xs = [c[1] for c in corners]  # Transpose DVCS -> EDCS = CCS
            ys = [c[0] for c in corners]
            mask = (x >= min(xs)) & (x <= max(xs)) & (y >= min(ys)) & (y <= max(ys))
            result[mask] = det.getId()

        return result.reshape(orig_shape)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx) -> FieldCoords:
        return replace(self, x=self.x[idx], y=self.y[idx])

    def __repr__(self) -> str:
        return f"FieldCoords({self.x!r}, {self.y!r}, frame={self.frame!r})"
