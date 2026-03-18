from __future__ import annotations

from dataclasses import dataclass

import galsim
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle

from .field_coords import FieldCoords
from .zernikes import Zernikes


@dataclass(frozen=True)
class DoubleZernikes:
    coefs: Quantity
    field_outer: Quantity
    field_inner: Quantity
    pupil_outer: Quantity
    pupil_inner: Quantity
    jmax: int | None = None
    kmax: int | None = None
    wavelength: Quantity | None = None
    frame: str = "ocs" # Applies to both field and pupil coordinates
    rtp: Angle | None = None

    def __post_init__(self):
        object.__setattr__(self, "coefs", np.atleast_2d(self.coefs))
        if self.jmax is None:
            object.__setattr__(self, "jmax", self.coefs.shape[-1] - 1)
        if self.kmax is None:
            object.__setattr__(self, "kmax", self.coefs.shape[-2] - 1)

    @property
    def eps(self) -> Quantity:
        return (self.pupil_inner / self.pupil_outer).value

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the leading batch dimensions."""
        return self.coefs.shape[:-2]

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the DoubleZernikes to use this property")
        return val

    def __len__(self) -> int:
        return self.coefs.shape[0]

    def __getitem__(self, idx: int | slice) -> DoubleZernikes:
        return DoubleZernikes(
            coefs=self.coefs[idx],
            field_outer=self.field_outer,
            field_inner=self.field_inner,
            pupil_outer=self.pupil_outer,
            pupil_inner=self.pupil_inner,
            jmax=self.jmax,
            kmax=self.kmax,
            wavelength=self.wavelength,
            frame=self.frame,
            rtp=self.rtp,
        )

    def _rot(self, angle: Angle, frame: str) -> DoubleZernikes:
        """Return a new DoubleZernikes with the coefficients rotated by *angle*."""
        jrot = galsim.zernike.zernikeRotMatrix(self.jmax, angle.radian)
        krot = galsim.zernike.zernikeRotMatrix(self.kmax, angle.radian)

        coefs_rot = np.einsum(
            "lk,...kj,jm->...lm", krot, self.coefs, jrot
        )

        return DoubleZernikes(
            coefs=coefs_rot,
            field_outer=self.field_outer,
            field_inner=self.field_inner,
            pupil_outer=self.pupil_outer,
            pupil_inner=self.pupil_inner,
            jmax=self.jmax,
            kmax=self.kmax,
            wavelength=self.wavelength,
            frame=frame,
            rtp=self.rtp,
        )

    @property
    def ocs(self) -> DoubleZernikes:
        """Return a copy of this DoubleZernikes with the frame set to 'ocs'."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self._rot(-rtp, frame="ocs")

    @property
    def ccs(self) -> DoubleZernikes:
        """Return a copy of this DoubleZernikes with the frame set to 'ccs'."""
        if self.frame == "ccs":
            return self
        rtp = self._require("rtp")
        return self._rot(rtp, frame="ccs")

    def to_galsim(
        self,
        idx: int | None = None,
    ) -> galsim.GSObject:
        """Return a galsim.GSObject representing the wavefront described by this DoubleZernikes."""
        if idx is not None:
            coefs = self.coefs[idx]
        else:
            coefs = self.coefs

        return galsim.zernike.DoubleZernike(
            coef=coefs,
            xy_outer=self.field_outer.value,
            xy_inner=self.field_inner.value,
            uv_outer=self.pupil_outer.value,
            uv_inner=self.pupil_inner.value,
        )

    def single(self, field: FieldCoords) -> Zernikes:
        """Return a Zernikes for a single field point."""
        from .zernikes import Zernikes

        if field.rtp != self.rtp:
            raise ValueError("Field rtp must match DoubleZernikes rtp")
        field = getattr(field, self.frame)

        # Field Zernike basis: (kmax+1, nfield) -> transpose to (nfield, kmax+1)
        B = galsim.zernike.zernikeBasis(
            self.kmax,
            field.x.value, field.y.value,
            R_outer=self.field_outer.to_value(field.x.unit),
            R_inner=self.field_inner.to_value(field.x.unit),
        ).T
        coefs = B @ self.coefs

        return Zernikes(
            coefs=coefs,
            field=field,
            R_outer=self.pupil_outer,
            R_inner=self.pupil_inner,
            wavelength=self.wavelength,
            jmax=self.jmax,
            frame=self.frame,
            rtp=self.rtp,
        )