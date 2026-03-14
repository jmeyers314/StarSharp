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

        dx = np.full((len(field.x), len(px)), np.nan)
        dy = np.full((len(field.x), len(px)), np.nan)
        vignetted = np.full((len(field.x), len(px)), True, dtype=bool)
        fpx = np.full(len(field.x), np.nan)
        fpy = np.full(len(field.x), np.nan)

        bar = self.tqdm(desc="Raytracing", total=len(field.x)) if self.tqdm is not None else None
        for ifield, (thx, thy, detnum) in enumerate(zip(field.x, field.y, field.detnum)):
            if bar:
                bar.update(1)
            if detnum == -1:
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
            fpx_ = np.mean(rays.x[w])
            fpy_ = np.mean(rays.y[w])

            dx[ifield] = rays.x - fpx_
            dy[ifield] = rays.y - fpy_
            vignetted[ifield] = rays.vignetted
            fpx[ifield] = fpx_
            fpy[ifield] = fpy_

        out_field = FieldCoords(
            x=fpx * 1e3 << u.m,
            y=fpy * 1e3 << u.m,
            frame="ccs",
            rtp=self.rtp,
            wcs=field.wcs,
            camera=field.camera
        )
        return Spots(
            dx * 1e6 << u.micron,
            dy * 1e6 << u.micron,
            vignetted,
            field=out_field,
            wavelength=self.wavelength,
            frame="ccs",
            rtp=self.rtp,
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
            coefs=zk * self.wavelength.to(u.um),
            field=field,
            R_outer=PUPIL_OUTER * u.m,
            R_inner=PUPIL_INNER * u.m,
            wavelength=self.wavelength,
            jmax=jmax,
            frame="ocs",
            rtp=self.rtp
        )
