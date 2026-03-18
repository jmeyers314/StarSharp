import astropy.units as u
import batoid
import numpy as np

from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from lsst.afw.cameraGeom import Camera, FOCAL_PLANE
from tqdm import tqdm
from numpy.typing import NDArray

from ..datatypes import FieldCoords, Spots, Zernikes, State, Sensitivity


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

        # Living a little dangerously here by assuming the builder's use_m1m3_modes and
        # use_m2_modes are enough to indicate "standard" dofs and thus step sizes.
        if (self.builder.use_m1m3_modes == list(range(19))+[26]
            and self.builder.use_m2_modes == list(range(17))+[25, 26, 27]
        ):
            self.steps = [
                10.0,  # M2 dz
                500.0,
                500.0,  # M2 dx, dy
                10.0,
                10.0,  # M2 rx, ry
                10.0,  # cam dz
                2000.0,
                2000.0,  # cam dx, dy
                10.0,
                10.0,  # cam rx, ry
            ]
            self.steps += [0.1] * 40  # bending modes
            if self.builder.dof_angle_units == "degree":
                self.steps[3:5] = [0.003, 0.003]
                self.steps[8:10] = [0.003, 0.003]
            self.steps = np.array(self.steps)
        else:
            self.steps = None

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

        # Flatten batch dims so we iterate over every field point individually,
        # then reshape outputs back to (*batch_shape, nfield, ...).
        batch_shape = field.x.shape[:-1]
        nfield = field.x.shape[-1]
        total = int(np.prod(field.x.shape))
        x_flat = field.x.reshape(total)
        y_flat = field.y.reshape(total)
        detnum_flat = field.detnum.reshape(total)

        nray = len(px)
        dx = np.full((total, nray), np.nan)
        dy = np.full((total, nray), np.nan)
        vignetted = np.full((total, nray), True, dtype=bool)
        fpx = np.full(total, np.nan)
        fpy = np.full(total, np.nan)

        bar = self.tqdm(desc="Raytracing", total=total) if self.tqdm is not None else None
        for ifield, (thx, thy, detnum) in enumerate(zip(x_flat, y_flat, detnum_flat)):
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
            x=fpx.reshape(batch_shape + (nfield,)) * 1e3 << u.m,
            y=fpy.reshape(batch_shape + (nfield,)) * 1e3 << u.m,
            frame="ccs",
            rtp=self.rtp,
            wcs=field.wcs,
            camera=field.camera
        )
        return Spots(
            dx.reshape(batch_shape + (nfield, nray)) * 1e6 << u.micron,
            dy.reshape(batch_shape + (nfield, nray)) * 1e6 << u.micron,
            vignetted.reshape(batch_shape + (nfield, nray)),
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

        # Flatten batch dims so we iterate over every field point individually,
        # then reshape outputs back to (*batch_shape, nfield, jmax+1).
        batch_shape = field.x.shape[:-1]
        nfield = field.x.shape[-1]
        total = int(np.prod(field.x.shape))
        x_flat = field.x.reshape(total)
        y_flat = field.y.reshape(total)
        detnum_flat = field.detnum.reshape(total)

        bar = self.tqdm(desc="Raytracing", total=total) if self.tqdm is not None else None
        zk = []
        for thx, thy, detnum in zip(x_flat, y_flat, detnum_flat):
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

        coefs = np.array(zk, dtype=float).reshape(batch_shape + (nfield, jmax + 1))
        return Zernikes(
            coefs=coefs * self.wavelength.to(u.um),
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
            x_scale=self.steps[guess.use_dof],
            args=(field, nrad, guess.use_dof, offset),
            **kwargs,
        )
        return State(
            value=result.x,
            basis="x",
            use_dof=guess.use_dof,
            n_dof=guess.n_dof,
        )

    def zernikes_sensitivity(
        self,
        field: FieldCoords,
        steps: State,
        jmax: int = 28,
        rings: int = 10,
        offset: State | None = None,
    ) -> Sensitivity:
        if offset is None:
            offset = State(
                value=np.zeros(50, dtype=np.float64),
                basis="f",
            )
        nominal = self.zernikes(
            state=offset,
            field=field,
            jmax=jmax,
            rings=rings,
        )

        perturbed = []
        for i, step in enumerate(steps.value):
            dval = np.zeros_like(steps.value)
            dval[i] = step
            dstate = State(
                value=dval,
                basis=steps.basis,
                use_dof=steps.use_dof,
                n_dof=steps.n_dof,
                Vh=steps.Vh,
            )
            perturbed.append(
                self.zernikes(
                    state=offset + dstate,
                    field=field,
                    jmax=jmax,
                    rings=rings,
                )
            )
        return Sensitivity.from_finite_differences(nominal, perturbed, steps)

    def spots_sensitivity(
        self,
        field: FieldCoords,
        steps: State,
        nrad: int = 10,
        offset: State | None = None,
    ) -> Sensitivity:
        if offset is None:
            offset = State(
                value=np.zeros(50, dtype=np.float64),
                basis="f",
            )
        nominal = self.spots(
            state=offset,
            field=field,
            nrad=nrad,
        )

        perturbed = []
        for i, step in enumerate(steps.value):
            dval = np.zeros_like(steps.value)
            dval[i] = step
            dstate = State(
                value=dval,
                basis=steps.basis,
                use_dof=steps.use_dof,
                n_dof=steps.n_dof,
                Vh=steps.Vh,
            )
            perturbed.append(
                self.spots(
                    state=offset + dstate,
                    field=field,
                    nrad=nrad,
                )
            )
        return Sensitivity.from_finite_differences(nominal, perturbed, steps)

    def double_zernikes_sensitivity(
        self,
        field: FieldCoords,
        steps: State,
        kmax: int = 28,
        field_outer=None,
        field_inner=None,
        jmax: int = 28,
        rings: int = 10,
        offset: State | None = None,
    ) -> Sensitivity:
        zk_sens = self.zernikes_sensitivity(
            field=field,
            steps=steps,
            jmax=jmax,
            rings=rings,
            offset=offset,
        )
        dz_nominal = zk_sens.nominal.double(kmax, field_outer, field_inner)
        dz_gradient = zk_sens.gradient.double(kmax, field_outer, field_inner)
        return Sensitivity(
            nominal=dz_nominal,
            gradient=dz_gradient,
            steps=zk_sens.steps,
        )
