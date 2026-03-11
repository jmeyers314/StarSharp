import astropy.units as u
import batoid
import numpy as np

from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from tqdm import tqdm
from typing import List, Optional

from datatypes import FieldCoords, Spots, Zernikes, State


FIELD_OUTER = 1.75
PUPIL_OUTER = 4.18
PUPIL_INNER = PUPIL_OUTER * 0.612
SIGMA_TO_FWHM = np.sqrt(np.log(256))


# class OpticalModel:
#     def __init__(
#         self,
#         pupil_radii: int = 10,
#         jmax: int = 28,
#         wavelength: Optional[float] = None,
#         rtp: Optional[Angle] = None,
#     ):
#         self.pupil_radii = pupil_radii
#         self.jmax = jmax
#         self.wavelength = wavelength
#         self.rtp = rtp

#     def spot(self, state: State, coord: FieldCoords) -> Spots:
#         ...

#     def zernikes(self, state: State, coord: FieldCoords) -> Zernikes:
#         ...


class RaytracedOpticalModel:
    def __init__(
        self,
        builder: LSSTBuilder,
        rtp: Angle,
        wavelength: float,
        tqdm: Optional[tqdm] = None,
    ):
        self.builder = builder
        self.rtp = rtp
        self.wavelength = wavelength
        self.tqdm = tqdm

    def spots(
        self,
        state: State,
        field: FieldCoords,
        nrad: int = 10,
    ) -> Spots:
        field = field.ocs
        if not np.allclose(field.rtp, self.rtp):
            raise ValueError(f"FieldCoords RTP ({field.rtp}) does not match model RTP ({self.rtp})")
        # Pick an outer radius that is half a grid radius smaller than the outer pupil
        # And similarly for the inner radius (but larger).
        dr = (PUPIL_OUTER - PUPIL_INNER) / (nrad + 1)
        px, py = batoid.utils.hexapolar(
            outer = PUPIL_OUTER - dr/2,
            inner = PUPIL_INNER + dr/2,
            nrad = nrad,
            naz = int(2 * np.pi * nrad / (1 - PUPIL_INNER / PUPIL_OUTER))
        )

        builder = (
            self.builder
            .with_rtp(self.rtp)
            .with_aos_dof(state.f.state)
        )

        dx = []
        dy = []
        vignetted = []

        bar = self.tqdm(desc="Raytracing", total=len(field.x)) if self.tqdm is not None else None
        for thx, thy, detnum in zip(field.x, field.y, field.detnum):
            if bar:
                bar.update(1)
            if detnum == -1:
                dx.append(np.full_like(px, np.nan))
                dy.append(np.full_like(px, np.nan))
                vignetted.append(np.ones_like(px, dtype=bool))
                continue
            telescope = builder.build_det(detnum)
            rays = batoid.RayVector.fromStop(
                np.array(px), np.array(py),
                theta_x=thx.to_value(u.rad),
                theta_y=thy.to_value(u.rad),
                optic=telescope,
                wavelength=self.wavelength.to_value(u.m),
            )
            rays = telescope.trace(rays)
            w = ~rays.vignetted
            dx.append(rays.x - np.mean(rays.x[w]))
            dy.append(rays.y - np.mean(rays.y[w]))
            vignetted.append(rays.vignetted)

        return Spots(
            dx * u.m,
            dy * u.m,
            vignetted,
            field=field,
            wavelength=self.wavelength,
            frame="ccs",
            rtp=self.rtp
        )

    def zernikes(
        self,
        state: State,
        field: FieldCoords,
        jmax: int = 28,
        rings: int = 10,
    ) -> Zernikes:
        field = field.ocs
        if not np.allclose(field.rtp, self.rtp):
            raise ValueError(f"FieldCoords RTP ({field.rtp}) does not match model RTP ({self.rtp})")
        builder = (
            self.builder
            .with_rtp(self.rtp)
            .with_aos_dof(state.f.state)
        )

        bar = self.tqdm(desc="Raytracing", total=len(field.x)) if self.tqdm is not None else None
        zk = []
        for thx, thy, detnum in zip(field.x, field.y, field.detnum):
            if bar:
                bar.update(1)
            if detnum == -1:
                zk.append(np.full(jmax+1, np.nan))
                continue
            telescope = builder.build_det(detnum)
            try:
                zkgq = batoid.zernikeGQ(
                    telescope,
                    theta_x=thx.to_value(u.rad),
                    theta_y=thy.to_value(u.rad),
                    wavelength=self.wavelength.to_value(u.m),
                    rings=rings,
                    jmax=jmax,
                    eps=PUPIL_INNER / PUPIL_OUTER,
                )
            except ValueError:
                zkgq = np.full(jmax+1, np.nan)
            zk.append(zkgq)

        return Zernikes(
            coefs=np.array(zk) * self.wavelength.to(u.um),
            field=field,
            R_outer=PUPIL_OUTER * u.m,
            R_inner=PUPIL_INNER * u.m,
            wavelength=self.wavelength,
            jmax=jmax,
            frame="ocs",
            rtp=self.rtp
        )







# class LinearOpticalModel(OpticalModel):
#     def __init__(
#         self,
#         intrinsic_ocs_spots: Spots,
#         spot_derivatives: List[Spots],
#         intrinsic_ocs_wavefront: Zernikes,
#         wavefront_derivatives: List[Zernikes],
#         ccs_spot_perturbation: Optional[Spots] = None,
#         ccs_wavefront_perturbation: Optional[Zernikes] = None,
#         wavelength: Optional[float] = None,
#         rtp: Optional[Angle] = None,
#     ):
#         super().__init__(wavelength=wavelength, rtp=rtp)
#         self.intrinsic_ocs_spots = intrinsic_ocs_spots
#         self.spot_derivatives = spot_derivatives
#         self.intrinsic_ocs_wavefront = intrinsic_ocs_wavefront
#         self.wavefront_derivatives = wavefront_derivatives
#         self.ccs_spot_perturbation = ccs_spot_perturbation
#         self.ccs_wavefront_perturbation = ccs_wavefront_perturbation

#     def spot(self, state: State, coord: FieldCoords) -> Spots:
#         ...

#     def zernikes(self, state: State, coord: FieldCoords) -> Zernikes:
#         ...
