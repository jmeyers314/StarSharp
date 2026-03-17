from __future__ import annotations

from dataclasses import dataclass

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
    """

    dx: Quantity
    dy: Quantity
    vignetted: NDArray[np.bool_]
    field: FieldCoords
    wavelength: Quantity | None = None
    frame: str = "ccs"
    rtp: Angle | None = None
    wcs: BaseWCS | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dx", np.atleast_1d(self.dx))
        object.__setattr__(self, "dy", np.atleast_1d(self.dy))
        object.__setattr__(self, "vignetted", np.atleast_1d(self.vignetted))
        # Coerce frame to lower case, but preserve original name (including 'edcs')
        frame = self.frame.lower()
        object.__setattr__(self, "frame", frame)
        if frame not in self.VALID_FRAMES:
            raise ValueError(f"frame must be one of {self.VALID_FRAMES}, got {self.frame!r}")
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
        if self.dx.ndim == 1:
            return 1
        return self.dx.shape[0]

    def __getitem__(self, idx: int | slice) -> Spots:
        return Spots(
            dx=self.dx[idx],
            dy=self.dy[idx],
            field=self.field[idx],
            vignetted=self.vignetted[idx],
            wavelength=self.wavelength,
            frame=self.frame,
            rtp=self.rtp,
            wcs=self.wcs,
        )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Spots to use this property")
        return val

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
        return Spots(
            dx=rdx,
            dy=rdy,
            vignetted=self.vignetted,
            field=self.field,
            wavelength=self.wavelength,
            frame=frame,
            rtp=self.rtp,
            wcs=self.wcs,
        )

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
        return Spots(
            dx=dx,
            dy=dy,
            vignetted=self.vignetted,
            field=self.field,
            wavelength=self.wavelength,
            frame="ccs",
            rtp=self.rtp,
            wcs=self.wcs,
        )

    @property
    def edcs(self) -> Spots:
        """This spot in the EDCS frame (synonym for CCS, but preserves name)."""
        if self.frame == "edcs":
            return self
        ccs = self.ccs
        return Spots(
            dx=ccs.dx,
            dy=ccs.dy,
            vignetted=ccs.vignetted,
            field=ccs.field,
            wavelength=ccs.wavelength,
            frame="edcs",
            rtp=ccs.rtp,
            wcs=ccs.wcs,
        )

    @property
    def dvcs(self) -> Spots:
        """This spot in the DVCS frame (transpose of EDCS/CCS)."""
        if self.frame == "dvcs":
            return self
        ccs = self.ccs
        return Spots(
            dx=ccs.dy,
            dy=ccs.dx,
            vignetted=ccs.vignetted,
            field=ccs.field,
            wavelength=ccs.wavelength,
            frame="dvcs",
            rtp=ccs.rtp,
            wcs=ccs.wcs,
        )

    @property
    def focal_plane(self) -> Spots:
        """This spot in focal-plane space."""
        if self.space == "focal_plane":
            return self
        wcs = self._require("wcs")
        sfx = self.ocs.dx.to_value(u.radian) + self.field.angle.ocs.x.to_value(u.radian)[:, np.newaxis]
        sfy = self.ocs.dy.to_value(u.radian) + self.field.angle.ocs.y.to_value(u.radian)[:, np.newaxis]
        args = [sfx, sfy]
        if isinstance(wcs, CelestialWCS):
            args.append("radians")
        sfpx, sfpy = wcs.toImage(*args)
        sfpx -= self.field.focal_plane.ccs.x.to_value(u.mm)[:, np.newaxis]
        sfpy -= self.field.focal_plane.ccs.y.to_value(u.mm)[:, np.newaxis]

        fp_ccs = Spots(
            dx=sfpx << u.mm,
            dy=sfpy << u.mm,
            vignetted=self.vignetted,
            field=self.field.focal_plane,
            wavelength=self.wavelength,
            frame="ccs",
            rtp=self.rtp,
            wcs=self.wcs,
        )
        return getattr(fp_ccs, self.frame)

    @property
    def angle(self) -> Spots:
        """This spot in field-angle space (OCS frame)."""
        if self.space == "angle":
            return self
        wcs = self._require("wcs")
        sfpx = self.ccs.dx.to_value(u.mm) + self.field.focal_plane.ccs.x.to_value(u.mm)[:, np.newaxis]
        sfpy = self.ccs.dy.to_value(u.mm) + self.field.focal_plane.ccs.y.to_value(u.mm)[:, np.newaxis]
        args = [sfpx, sfpy]
        if isinstance(wcs, CelestialWCS):
            args.append("radians")

        sfx, sfy = wcs.toWorld(*args)
        sfx -= self.field.angle.ocs.x.to_value(u.radian)[:, np.newaxis]
        sfy -= self.field.angle.ocs.y.to_value(u.radian)[:, np.newaxis]

        field_ocs = Spots(
            dx=sfx << u.radian,
            dy=sfy << u.radian,
            vignetted=self.vignetted,
            field=self.field.angle.ocs,
            wavelength=self.wavelength,
            frame="ocs",
            rtp=self.rtp,
            wcs=self.wcs,
        )
        return getattr(field_ocs, self.frame)

    def __repr__(self) -> str:
        return f"Spots(frame={self.frame!r})"

    def compute_moments(self, order: int = 2):
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
            vals[name] = np.nanmean(
                dx ** nx * dy ** ny,
                axis=-1
            ) << (u.micron ** order)

        return Moments[order](**vals, frame=self.frame, field=self.field, rtp=self.rtp)
