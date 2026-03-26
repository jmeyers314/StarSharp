from __future__ import annotations

from dataclasses import dataclass, replace

import astropy.units as u
import galsim
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from galsim.wcs import BaseWCS, CelestialWCS
from numpy.typing import NDArray

from .field_coords import FieldCoords


@dataclass(frozen=True)
class Spots:
    VALID_FRAMES = ("ocs", "ccs", "dvcs", "edcs")
    _sensitivity_fields = ("dx", "dy")
    _broadcast_fields = ("vignetted",)
    """Spot diagrams: ray intersection positions on the focal plane.

    May represent one or many field points.  When batched,
    ``dx`` and ``dy`` have shape ``(n_field, n_ray)``.

    Parameters
    ----------
    dx, dy : Quantity
        Ray positions relative to the centroid.
    vignetted : NDArray[bool]
        Vignetting mask.
    field : FieldCoords
        Field coordinate(s) at which the spot was computed.
    wavelength : Quantity or None
        Wavelength of the traced rays.
    frame : str
        Coordinate frame: ``'ocs'`` (optical, default) or ``'ccs'``
        (camera).
    rtp : Angle or None
        Rotation angle from OCS to CCS frame.  Required for frame conversions.
    px, py : Quantity or None
        Pupil (stop-plane) coordinates of each ray.  1-D arrays of shape
        ``(n_ray,)`` shared across all field points.  Units are typically
        meters.  Always interpretted as OCS frame.
    """

    dx: Quantity
    dy: Quantity
    vignetted: NDArray[np.bool_]
    field: FieldCoords
    wavelength: Quantity | None = None
    frame: str = "ccs"
    rtp: Angle | None = None
    wcs: BaseWCS | None = None
    px: Quantity | None = None
    py: Quantity | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dx", np.atleast_2d(self.dx))
        object.__setattr__(self, "dy", np.atleast_2d(self.dy))
        object.__setattr__(self, "vignetted", np.atleast_2d(self.vignetted))
        # Coerce frame to lower case, but preserve original name (including 'edcs')
        frame = self.frame.lower()
        object.__setattr__(self, "frame", frame)
        if frame not in self.VALID_FRAMES:
            raise ValueError(
                f"frame must be one of {self.VALID_FRAMES}, got {self.frame!r}"
            )
        if (
            self.rtp is not None
            and self.field.rtp is not None
            and not np.allclose(
                self.rtp.rad, self.field.rtp.rad  # type: ignore[union-attr]
            )
        ):
            raise ValueError(
                f"Spots.rtp ({self.rtp}) is inconsistent with "
                f"field.rtp ({self.field.rtp})"
            )

    def __len__(self) -> int:
        return self.dx.shape[0]

    def __getitem__(self, idx) -> Spots:
        return replace(
            self,
            dx=self.dx[idx],
            dy=self.dy[idx],
            vignetted=self.vignetted[idx],
            field=self.field[idx],
        )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Spots to use this property")
        return val

    @property
    def nfield(self) -> int:
        """Number of field points (second-to-last axis)."""
        return self.dx.shape[-2]

    @property
    def nray(self) -> int:
        """Number of rays per field point (last axis)."""
        return self.dx.shape[-1]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the leading batch dimensions."""
        return self.dx.shape[:-2]

    @property
    def space(self) -> str:
        """Inferred coordinate space: ``'angle'`` or ``'focal_plane'``."""
        if self.dx.unit.physical_type == "angle":
            return "angle"
        return "focal_plane"

    def _rot(self, angle: Angle, frame: str) -> Spots:
        """Return a new Spots with dx/dy rotated by *angle*."""
        rdx = self.dx * np.cos(angle) + self.dy * np.sin(angle)
        rdy = -self.dx * np.sin(angle) + self.dy * np.cos(angle)
        return replace(self, dx=rdx, dy=rdy, frame=frame)

    @property
    def ocs(self) -> Spots:
        """This spot in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self.ccs._rot(-rtp, "ocs")

    @property
    def ccs(self) -> Spots:
        """This spot in the CCS frame (always frame='ccs')."""
        if self.frame == "ccs":
            return self
        if self.frame == "ocs":
            rtp = self._require("rtp")
            return self._rot(rtp, "ccs")
        dx = self.dx
        dy = self.dy
        if self.frame == "dvcs":
            dx, dy = dy, dx
        return replace(self, dx=dx, dy=dy, frame="ccs")

    @property
    def edcs(self) -> Spots:
        """This spot in the EDCS frame (synonym for CCS, but preserves name)."""
        if self.frame == "edcs":
            return self
        ccs = self.ccs
        return replace(ccs, frame="edcs")

    @property
    def dvcs(self) -> Spots:
        """This spot in the DVCS frame (transpose of EDCS/CCS)."""
        if self.frame == "dvcs":
            return self
        ccs = self.ccs
        return replace(ccs, dx=ccs.dy, dy=ccs.dx, frame="dvcs")

    @property
    def focal_plane(self) -> Spots:
        """This spot in focal-plane space."""
        if self.space == "focal_plane":
            return self
        wcs = self._require("wcs")
        fx = self.field.angle.ocs.x.to_value(u.radian)  # (..., nfield)
        fy = self.field.angle.ocs.y.to_value(u.radian)
        sfx = self.ocs.dx.to_value(u.radian) + fx[..., np.newaxis]
        sfy = self.ocs.dy.to_value(u.radian) + fy[..., np.newaxis]
        orig_shape = sfx.shape
        args = [sfx.ravel(), sfy.ravel()]
        if isinstance(wcs, CelestialWCS):
            args.append("radians")
        sfpx, sfpy = wcs.toImage(*args)
        sfpx = sfpx.reshape(orig_shape)
        sfpy = sfpy.reshape(orig_shape)
        cfx = self.field.focal_plane.ccs.x.to_value(u.mm)  # (..., nfield)
        cfy = self.field.focal_plane.ccs.y.to_value(u.mm)
        sfpx -= cfx[..., np.newaxis]
        sfpy -= cfy[..., np.newaxis]

        fp_ccs = replace(
            self,
            dx=sfpx << u.mm,
            dy=sfpy << u.mm,
            field=self.field.focal_plane,
            frame="ccs",
        )
        return getattr(fp_ccs, self.frame)

    @property
    def angle(self) -> Spots:
        """This spot in field-angle space (OCS frame)."""
        if self.space == "angle":
            return self
        wcs = self._require("wcs")
        cfpx = self.field.focal_plane.ccs.x.to_value(u.mm)  # (..., nfield)
        cfpy = self.field.focal_plane.ccs.y.to_value(u.mm)
        sfpx = self.ccs.dx.to_value(u.mm) + cfpx[..., np.newaxis]
        sfpy = self.ccs.dy.to_value(u.mm) + cfpy[..., np.newaxis]
        orig_shape = sfpx.shape
        args = [sfpx.ravel(), sfpy.ravel()]
        if isinstance(wcs, CelestialWCS):
            args.append("radians")

        sfx, sfy = wcs.toWorld(*args)
        sfx = sfx.reshape(orig_shape)
        sfy = sfy.reshape(orig_shape)
        fax = self.field.angle.ocs.x.to_value(u.radian)  # (..., nfield)
        fay = self.field.angle.ocs.y.to_value(u.radian)
        sfx -= fax[..., np.newaxis]
        sfy -= fay[..., np.newaxis]

        field_ocs = replace(
            self,
            dx=sfx << u.radian,
            dy=sfy << u.radian,
            field=self.field.angle.ocs,
            frame="ocs",
        )
        return getattr(field_ocs, self.frame)

    def __repr__(self) -> str:
        return f"Spots(frame={self.frame!r})"

    def moments(self, order: int = 2):
        """Compute 2d moments of the spot diagrams (excluding vignetted spots).

        Parameters
        ----------
        order : int
            Order of the moments to compute (default: 2).

        Returns
        -------
        Moments
            Moments of the spot diagram.
        """
        from .moments import Moments

        if order < 1:
            raise ValueError("Order must be at least 1")

        # Convert to micron as a working unit and copy only if necessary
        dx = self.dx.to_value(u.micron)
        if np.shares_memory(dx, self.dx.value):
            dx = dx.copy()
        dy = self.dy.to_value(u.micron)
        if np.shares_memory(dy, self.dy.value):
            dy = dy.copy()
        dx[self.vignetted] = np.nan
        dy[self.vignetted] = np.nan

        vals = {}
        for ny in range(order + 1):
            nx = order - ny
            name = "x" * nx + "y" * ny
            vals[name] = np.nanmean(dx**nx * dy**ny, axis=-1) << (u.micron**order)

        return Moments[order](**vals, frame=self.frame, field=self.field, rtp=self.rtp)
