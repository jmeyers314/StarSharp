from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import galsim
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from numpy.typing import NDArray

from .field_coords import FieldCoords


@dataclass(frozen=True)
class Zernikes:
    """Zernike coefficients at one or many field points.

    When batched, ``coefs`` has shape ``(..., jmax + 1)``.

    Parameters
    ----------
    coefs : Quantity
        Zernike coefficient array.  The last axis indexes Noll
        index *j* (0 … jmax).  Leading axes are batch dimensions.
    R_outer : float
        Outer radius of the annular pupil.
    R_inner : float
        Inner radius of the annular pupil.
    jmax : int or None
        Maximum Noll index.  Inferred from ``coefs`` if *None*.
    rtp : Angle or None
        Rotation angle from OCS to CCS frame.  Required for frame conversions.
    """
    _sensitivity_fields = ('coefs',)

    coefs: Quantity
    field: FieldCoords
    R_outer: Quantity = None
    R_inner: Quantity = None
    wavelength: Quantity | None = None
    jmax: int | None = None
    frame: str = "ocs"
    rtp: Angle | None = None

    def __post_init__(self):
        object.__setattr__(self, "coefs", np.atleast_2d(self.coefs))
        if self.jmax is None:
            object.__setattr__(self, "jmax", self.coefs.shape[-1] - 1)
        if (
            self.rtp is not None
            and self.field.rtp is not None
            and not np.allclose(
                self.rtp.rad, self.field.rtp.rad  # type: ignore[union-attr]
            )
        ):
            raise ValueError(
                f"Zernikes.rtp ({self.rtp}) is inconsistent with "
                f"field.rtp ({self.field.rtp})"
            )

    @property
    def nfield(self) -> int:
        """Number of field points (second-to-last axis)."""
        return self.coefs.shape[-2]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the leading batch dimensions."""
        return self.coefs.shape[:-2]

    @property
    def eps(self) -> float:
        """Obscuration ratio R_inner / R_outer."""
        return (self.R_inner / self.R_outer).value

    def __len__(self) -> int:
        return self.coefs.shape[0]

    def __getitem__(self, idx) -> Zernikes:
        return Zernikes(
            coefs=self.coefs[idx],
            field=self.field[idx],
            R_outer=self.R_outer,
            R_inner=self.R_inner,
            wavelength=self.wavelength,
            jmax=self.jmax,
            frame=self.frame,
            rtp=self.rtp,
        )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Zernikes to use this property")
        return val

    def _rot(self, angle: Angle, frame: str) -> Zernikes:
        """Return a new Zernikes with the coefficients rotated by *angle*."""
        rot = galsim.zernike.zernikeRotMatrix(self.jmax, angle.radian)
        coefs_rot = self.coefs @ rot
        return Zernikes(
            coefs=coefs_rot,
            field=self.field,
            R_outer=self.R_outer,
            R_inner=self.R_inner,
            wavelength=self.wavelength,
            jmax=self.jmax,
            frame=frame,
            rtp=self.rtp,
        )

    @property
    def ocs(self) -> Zernikes:
        """This Zernikes in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self._rot(-rtp, "ocs")

    @property
    def ccs(self) -> Zernikes:
        """This Zernikes in the CCS frame."""
        if self.frame == "ccs":
            return self
        rtp = self._require("rtp")
        return self._rot(rtp, "ccs")

    def to_galsim(
        self,
        idx: int | None = None,
        unit: u.Unit = u.um,
        radius_unit: u.Unit = u.m,
    ) -> galsim.zernike.Zernike:
        """Return a `galsim.zernike.Zernike` for one set of coefficients.

        Parameters
        ----------
        idx : int or None
            Index into the batch/field dimensions.  Required when
            ``coefs`` has more than two dimensions or more than one
            field point; ignored for a single (1, jmax+1) coefficient array.
        unit : astropy.units.Unit
            Unit for the output coefficients (default: micron).
        """
        if idx is not None:
            c = self.coefs[idx]
        else:
            c = self.coefs
        # Squeeze to 1-D if possible
        c = c.squeeze()
        if c.ndim != 1:
            raise TypeError("to_galsim() requires a single coefficient vector")
        return galsim.zernike.Zernike(
            c.to(unit).value,
            R_outer=self.R_outer.to_value(radius_unit),
            R_inner=self.R_inner.to_value(radius_unit),
        )

    def __repr__(self) -> str:
        return f"Zernikes({self.coefs!r}, jmax={self.jmax}, frame={self.frame!r})"

    def double(self, kmax, field_outer, field_inner=None) -> DoubleZernikes:
        from .double_zernikes import DoubleZernikes

        if field_inner is None:
            field_inner = 0.0 * field_outer

        R_outer = self._require("R_outer")
        R_inner = self._require("R_inner")

        field_ang = getattr(self.field.angle, self.frame)
        unit = field_outer.unit
        fx = field_ang.x.to_value(unit)
        fy = field_ang.y.to_value(unit)

        # Field Zernike basis: (kmax+1, nfield) -> transpose to (nfield, kmax+1)
        B = galsim.zernike.zernikeBasis(
            kmax, fx, fy,
            R_outer=field_outer.to_value(unit),
            R_inner=field_inner.to_value(unit),
        ).T  # (nfield, kmax+1)

        # Use pinv for batch-safe least-squares fitting
        dz_coefs = np.linalg.pinv(B) @ self.coefs.value

        return DoubleZernikes(
            coefs=dz_coefs << self.coefs.unit,
            field_outer=field_outer,
            field_inner=field_inner,
            pupil_outer=R_outer,
            pupil_inner=R_inner,
            jmax=self.jmax,
            kmax=kmax,
            wavelength=self.wavelength,
            frame=self.frame,
            rtp=self.rtp,
        )
