import astropy.units as u
import batoid
import numpy as np

from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from lsst.afw.cameraGeom import Camera, FOCAL_PLANE
from tqdm import tqdm
from numpy.typing import NDArray

from ..datatypes import FieldCoords, Spots, Zernikes, State


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
        camera: Camera | None = None,
        tqdm: tqdm | None = None,
    ):
        self.builder = builder
        self.rtp = rtp
        self.wavelength = wavelength
        self.camera = camera
        self.tqdm = tqdm

    def make_hex_field(
        self,
        outer = 2.0 * u.deg,
        nrad = 20,
        naz = None,
        frame = "ocs"
    ):
        if naz is None:
            naz = int(2 * np.pi * nrad)
        thx, thy = batoid.utils.hexapolar(
            outer=outer.to_value(u.deg),
            nrad=nrad,
            naz=naz
        )
        return FieldCoords.from_builder(
            thx * u.deg, thy * u.deg,
            builder=self.builder,
            wavelength=self.wavelength,
            rtp=self.rtp,
            camera=self.camera,
            frame=frame
        )

    def make_ccd_field(
        self,
        nx: int = 1,
        frame = "ocs",
        types = ("E2V", "ITL"),
        detnums: set[int] | None = None,
    ):
        if self.camera is None:
            raise ValueError("Camera must be set on the model to use this method")
        if detnums is None:
            detnums = set(range(len(self.camera)))
        x = []
        y = []
        for det in self.camera:
            if det.getId() not in detnums:
                continue
            if det.getPhysicalType() not in types:
                continue
            corners = det.getCorners(FOCAL_PLANE)
            xmin = min(c[1] for c in corners)
            xmax = max(c[1] for c in corners)
            ymin = min(c[0] for c in corners)
            ymax = max(c[0] for c in corners)
            xx = np.linspace(xmin, xmax, nx+1)
            yy = np.linspace(ymin, ymax, nx+1)
            xx = 0.5*(xx[1:] + xx[:-1])
            yy = 0.5*(yy[1:] + yy[:-1])
            xx, yy = np.meshgrid(xx, yy)
            x.extend(xx.ravel())
            y.extend(yy.ravel())
        fc = FieldCoords.from_builder(
            np.array(x) << u.mm, np.array(y) << u.mm,
            builder=self.builder,
            wavelength=self.wavelength,
            rtp=self.rtp,
            camera=self.camera,
            frame="ccs"
        )
        return getattr(fc, frame)

    def spots(
        self,
        state: State,
        field: FieldCoords,
        nrad: int = 10,
    ) -> Spots:
        field = field.angle.ocs
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
            .with_aos_dof(state.f.value)
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
        field = field.angle.ocs
        if not np.allclose(field.rtp, self.rtp):
            raise ValueError(f"FieldCoords RTP ({field.rtp}) does not match model RTP ({self.rtp})")
        builder = (
            self.builder
            .with_rtp(self.rtp)
            .with_aos_dof(state.f.value)
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

    def _optimize_dx_func(
        self,
        params: np.ndarray,
        field: FieldCoords,
        nrad: int = 10,
        use_dof: NDArray[np.integer] | None = None,
        offset: State | None = None,
    ):
        value = np.zeros(50, dtype=np.float64)
        value[use_dof] = params
        if offset is not None:
            value += offset.f.value
        state = State(
            value=value,
            basis="f",
        )
        spots = self.spots(state, field, nrad=nrad)
        vignetted = spots.vignetted
        return np.concatenate([spots.dx[~vignetted].to_value(u.micron), spots.dy[~vignetted].to_value(u.micron)])

    def _optimize_func(
        self,
        params: np.ndarray,
        field: FieldCoords,
        nrad: int = 10,
        use_dof: NDArray[np.integer] | None = None,
        offset: State | None = None,
    ):
        value = np.zeros(50, dtype=np.float64)
        value[use_dof] = params
        if offset is not None:
            value += offset.f.value
        state = State(
            value=value,
            basis="f",
        )
        spots = self.spots(state, field, nrad=nrad)
        sizes = []
        for ispot in range(len(spots)):
            vignetted = spots.vignetted[ispot]
            dx = spots.dx[ispot][~vignetted].to_value(u.micron)
            dy = spots.dy[ispot][~vignetted].to_value(u.micron)
            if len(dx) == 0 or len(dy) == 0:
                sizes.append(np.inf)
                continue
            sizes.append(np.var(dx) + np.var(dy))
        return np.array(sizes)

    def optimize(
        self,
        guess: State,
        field: FieldCoords,
        offset: State | None = None,
        nrad: int = 10,
        mode: str = "dx",
        **kwargs,
    ) -> State:
        from scipy.optimize import least_squares
        func = self._optimize_dx_func if mode == "dx" else self._optimize_func

        x0 = guess.x.value
        result = least_squares(
            func,
            x0,
            args=(field, nrad, guess.use_dof, offset),
            **kwargs,
        )
        return State(
            value=result.x,
            basis="x",
            use_dof=guess.use_dof,
            n_dof=guess.n_dof,
        )
