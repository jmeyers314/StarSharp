from dataclasses import replace
from typing import Literal

import astropy.units as u
import batoid
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from batoid_rubin import LSSTBuilder
from lsst.afw.cameraGeom import FOCAL_PLANE, Camera
from numpy.typing import NDArray
from tqdm.std import tqdm as TqdmType

from ..datatypes import (
    DoubleZernikes,
    FieldCoords,
    Sensitivity,
    Spots,
    State,
    StateSchema,
    Zernikes,
)

PUPIL_OUTER = 4.18
PUPIL_INNER = PUPIL_OUTER * 0.612
EXTRA_FOCAL_DETS = [191, 195, 199, 203]
INTRA_FOCAL_DETS = [192, 196, 200, 204]


class RaytracedOpticalModel:
    """
    Ray-traced optical model for the Rubin Observatory LSST camera.

    Wraps a ``batoid_rubin.LSSTBuilder`` to provide wavefront, spot-diagram,
    and sensitivity computations at arbitrary field positions on the focal
    plane. All coordinates are in the OCS by default; conversions to other
    frames are handled by the returned :class:`~StarSharp.datatypes.FieldCoords`,
    :class:`~StarSharp.datatypes.Zernikes`, and :class:`~StarSharp.datatypes.Spots`
    objects.

    Parameters
    ----------
    builder : LSSTBuilder
        A ``batoid_rubin.LSSTBuilder`` instance that encodes the telescope
        design and allows perturbation via AOS DOFs.
    rtp : Angle
        Rotator position angle (OCS → CCS rotation).
    wavelength : Quantity
        Monochromatic wavelength for ray tracing (e.g. ``620 * u.nm``).
    state_schema : StateSchema
        Schema defining the AOS state DOFs and their mapping to the builder.
    camera : Camera or None
        LSST camera geometry object.  Required for methods that need
        detector-level information (e.g. :meth:`make_ccd_field`,
        :meth:`spots`).
    offset : State or None
        Optional fixed offset applied to the AOS DOFs before ray tracing.
    """

    def __init__(
        self,
        builder: LSSTBuilder,
        rtp: Angle,
        wavelength: Quantity,
        state_schema: StateSchema,
        camera: Camera | None = None,
        offset: State | None = None,
    ):
        self.builder = builder
        self.rtp = rtp
        self.fiducial = builder.with_rtp(rtp).build()
        self.wavelength = wavelength
        self.state_schema = state_schema
        self.camera = camera
        if offset is None:
            offset = State(
                value=np.zeros(50, dtype=np.float64),
                schema=state_schema,
                basis="f",
            )
        self.offset = offset

        # Living a little dangerously here by assuming the builder's use_m1m3_modes and
        # use_m2_modes are enough to indicate "standard" dofs and thus step sizes.
        if self.builder.use_m1m3_modes == list(range(19)) + [
            26
        ] and self.builder.use_m2_modes == list(range(17)) + [25, 26, 27]:
            self.steps = [
                10.0,  # M2 dz
                500.0,
                500.0,  # M2 dx, dy
                10.0,
                10.0,  # M2 rx, ry
                10.0,  # cam dz
                500.0,
                500.0,  # cam dx, dy
                10.0,
                10.0,  # cam rx, ry
            ]
            self.steps += [0.1] * 40  # bending modes
            if self.builder.dof_angle_units == "degree":
                self.steps[3:5] = [10./3600, 10./3600]  # M2 rx, ry
                self.steps[8:10] = [10./3600, 10./3600]  # cam rx, ry
            self.steps = np.array(self.steps)
        else:
            self.steps = None

    def make_hex_field(self, outer=2.0 * u.deg, nrad=20, naz=None):
        """Create a hexapolar grid of field coordinates.

        Parameters
        ----------
        outer : Quantity
            Outer radius of the hexapolar grid (default: 2 deg).
        nrad : int
            Number of radial rings.
        naz : int or None
            Number of azimuthal points on the outermost ring.  Defaults to
            ``int(2π * nrad)``.

        Returns
        -------
        FieldCoords
        """
        if naz is None:
            naz = int(2 * np.pi * nrad)
        thx, thy = batoid.utils.hexapolar(
            outer=outer.to_value(u.deg), nrad=nrad, naz=naz
        )
        return FieldCoords(
            thx * u.deg,
            thy * u.deg,
            frame="ocs",
            rtp=self.rtp,
            camera=self.camera
        )

    def make_ccd_field(
        self,
        nx: int = 1,
        types=("E2V", "ITL"),
        detnums: list[int] | None = None,
    ):
        """Create a field grid with one point per CCD (or per CCD sub-cell).

        Requires ``camera`` to be set on the model.

        Parameters
        ----------
        nx : int
            Number of grid points per axis within each detector (default: 1,
            i.e. the detector center).
        types : tuple[str, ...]
            Physical detector types to include (default: ``('E2V', 'ITL')``).
        detnums : list[int] or None
            Explicit list of detector IDs to include.  Defaults to all detectors.

        Returns
        -------
        FieldCoords
        """
        if isinstance(types, str): # Turn "ITL" into ("ITL",)
            types = (types,)
        if self.camera is None:
            raise ValueError("Camera must be set on the model to use this method")
        if detnums is None:
            detnums = list(range(len(self.camera)))
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
            xx = np.linspace(xmin, xmax, nx + 1)
            yy = np.linspace(ymin, ymax, nx + 1)
            xx = 0.5 * (xx[1:] + xx[:-1])
            yy = 0.5 * (yy[1:] + yy[:-1])
            xx, yy = np.meshgrid(xx, yy)
            x.extend(xx.ravel())
            y.extend(yy.ravel())
        fc = FieldCoords(
            np.array(x) << u.mm,
            np.array(y) << u.mm,
            frame="ccs",
            rtp=self.rtp,
            camera=self.camera
        )
        return fc

    def make_wfs_mean_field(
        self,
    ) -> FieldCoords:
        """Create a field with one point per wavefront sensor pair,
        located at the mean center of the pair.

        Requires ``camera`` to be set on the model and include WFS detectors.

        Returns
        -------
        FieldCoords
        """
        if self.camera is None:
            raise ValueError("Camera must be set on the model to use this method")
        x = []
        y = []
        for det1, det2 in [
            [191, 192],
            [195, 196],
            [199, 200],
            [203, 204],
        ]:
            corner_fc = self.make_ccd_field(nx=1, types="ITL_WF", detnums=[det1, det2])
            x.append(np.mean(corner_fc.ccs.x.to_value(u.mm)))
            y.append(np.mean(corner_fc.ccs.y.to_value(u.mm)))
        return FieldCoords(
            np.array(x) << u.mm,
            np.array(y) << u.mm,
            frame="ccs",
            rtp=self.rtp,
            camera=self.camera
        )

    def _iter_thx_thy_telescope(
        self,
        field: FieldCoords,
        state: State | None = None,
        zk: Zernikes | DoubleZernikes | None = None,
        include_chip_heights: bool = True,
        camera_piston: Quantity = None,
        detector_piston: Quantity = None,
    ):
        """Build and iterate over per-field telescopes for ray-tracing operations."""
        field = field.angle.ocs
        if not np.allclose(field.rtp, self.rtp):
            raise ValueError(
                f"FieldCoords RTP ({field.rtp}) does not match model RTP ({self.rtp})"
            )
        if isinstance(zk, DoubleZernikes):
            zk = zk.single(field)

        builder = self.builder.with_rtp(self.rtp)
        if camera_piston is not None:
            builder = builder.with_camera_piston(camera_piston.to_value(u.micron))
        if state is not None:
            builder = builder.with_aos_dof((state + self.offset).f.value)
        else:
            builder = builder.with_aos_dof(self.offset.f.value)

        # Flatten batch dims so we iterate over every field point individually.
        batch_shape = field.x.shape[:-1]
        nfield = field.x.shape[-1]
        total = int(np.prod(field.x.shape))
        x_flat = field.x.reshape(total)
        y_flat = field.y.reshape(total)
        detnum_flat = field.detnum.reshape(total)

        # Broadcast and flatten extra Zernike coefs to (total, jmax+1) in meters.
        # None entries signal "no extra zk for this field point".
        if zk is not None:
            target_shape = batch_shape + (nfield, zk.coefs.shape[-1])
            zk_broad = np.broadcast_to(zk.coefs.to_value(u.m), target_shape)
            zk_coefs_flat = zk_broad.reshape(total, -1)
        else:
            zk_coefs_flat = [None] * total

        def iterator():
            for ifield, (thx, thy, detnum, zk_coef) in enumerate(
                zip(x_flat, y_flat, detnum_flat, zk_coefs_flat)
            ):
                this_builder = builder
                if zk_coef is not None:
                    this_builder = this_builder.with_extra_zk(
                        zk_coef, PUPIL_INNER / PUPIL_OUTER
                    )
                if detector_piston is None:
                    if detnum in INTRA_FOCAL_DETS:
                        this_builder = this_builder.with_detector_piston(-1500)
                    elif detnum in EXTRA_FOCAL_DETS:
                        this_builder = this_builder.with_detector_piston(+1500)
                else:
                    this_builder = this_builder.with_detector_piston(
                        detector_piston.to_value(u.micron)
                    )
                if include_chip_heights:
                    if detnum == -1:
                        yield ifield, thx, thy, None
                        continue
                    telescope = this_builder.build_det(detnum)
                else:
                    telescope = this_builder.build()
                yield ifield, thx, thy, telescope

        return field, batch_shape, nfield, total, iterator()

    def spots(
        self,
        field: FieldCoords,
        state: State | None = None,
        zk: Zernikes | DoubleZernikes | None = None,
        nrad: int = 10,
        reference: Literal["chief", "mean", "ring"] = "ring",
        include_chip_heights: bool = True,
        camera_piston: Quantity = None,
        detector_piston: Quantity = None,
    ) -> Spots:
        """Trace rays and return spot diagrams.

        Parameters
        ----------
        state : State | None
            AOS alignment state (DOF perturbation from the nominal).
        field : FieldCoords
            Field coordinates at which to trace.  Any frame or space is
            accepted; coordinates are converted to OCS angles internally.
        zk : Zernikes | DoubleZernikes | None
            Extra Zernike phase perturbation to apply at the entrance pupil.
            If ``DoubleZernikes``, it is first converted to ``Zernikes``
            via ``.single(field)``.
            The coefs array is broadcast against the field shape:
            ``(1, jmax+1)`` applies the same perturbation everywhere;
            ``(nfield, jmax+1)`` gives per-field perturbations broadcast
            over batch dims; ``(*batch_shape, nfield, jmax+1)`` is fully
            specified.
        nrad : int
            Number of radial rings in the hexapolar stop sampling grid.
        reference: Literal["chief", "mean", "ring"]
            Whether to center spot diagrams on the chief ray intersection,
            the mean intersection of all unvignetted rays, or the mean of a
            ring of (possibly vignetted) rays.
        include_chip_heights : bool
            Whether to include the effect of CCD chip height variations in the
            spot diagrams.
        camera_piston : Quantity or None
            Optional extra piston term applied to the entire camera.
        detector_piston : Quantity or None
            Optional extra piston term applied to each detector independently.  If
            None, then use the canonical piston for science/intra/extra detectors.

        Returns
        -------
        Spots
            Spot diagrams in the CCS frame with focal-plane units (micron).
            Shape ``(*batch_shape, nfield, nray)``.
        """
        # Pick an outer radius that is half a grid step smaller than the outer pupil
        # And inner radius that is half a grid step larger than inner pupil.
        dr = (PUPIL_OUTER - PUPIL_INNER) / (nrad + 1)
        px, py = batoid.utils.hexapolar(
            outer=PUPIL_OUTER - dr / 2,
            inner=PUPIL_INNER + dr / 2,
            nrad=nrad,
            naz=int(2 * np.pi * nrad / (1 - PUPIL_INNER / PUPIL_OUTER)),
        )

        field, batch_shape, nfield, total, iterator = self._iter_thx_thy_telescope(
            field=field,
            state=state,
            zk=zk,
            include_chip_heights=include_chip_heights,
            camera_piston=camera_piston,
            detector_piston=detector_piston,
        )

        nray = len(px)
        dx = np.full((total, nray), np.nan)
        dy = np.full((total, nray), np.nan)
        vignetted = np.full((total, nray), True, dtype=bool)
        fpx = np.full(total, np.nan)
        fpy = np.full(total, np.nan)

        for ifield, thx, thy, telescope in iterator:
            if telescope is None:
                continue
            rays = batoid.RayVector.fromStop(
                np.array(px),
                np.array(py),
                theta_x=thx.to_value(u.rad),
                theta_y=thy.to_value(u.rad),
                optic=telescope,
                wavelength=self.wavelength.to_value(u.m),
                projection="gnomonic",
            )
            rays = telescope.trace(rays)
            w = ~rays.vignetted
            if reference == "chief":
                cr = batoid.RayVector.fromStop(
                    0.0,
                    0.0,
                    theta_x=thx.to_value(u.rad),
                    theta_y=thy.to_value(u.rad),
                    optic=telescope,
                    wavelength=self.wavelength.to_value(u.m),
                    projection="gnomonic",
                )
                cr = telescope.trace(cr)
                fpx_ = cr.x[0]
                fpy_ = cr.y[0]
            elif reference == "ring":
                # Use a ring of rays ~30% of the way out in the pupil,
                # which should be more stable than the (always vignetted) chief
                # ray but less noisy than the mean of all unvignetted rays.
                # This radius is generally unvignetted for a long time.
                ring_radius = 0.3 * (PUPIL_OUTER - PUPIL_INNER) + PUPIL_INNER
                ph = np.linspace(0, 2 * np.pi, 24, endpoint=False)
                ring = batoid.RayVector.fromStop(
                    ring_radius * np.cos(ph),
                    ring_radius * np.sin(ph),
                    theta_x=thx.to_value(u.rad),
                    theta_y=thy.to_value(u.rad),
                    optic=telescope,
                    wavelength=self.wavelength.to_value(u.m),
                    projection="gnomonic",
                )
                ring = telescope.trace(ring)
                fpx_ = np.mean(ring.x)
                fpy_ = np.mean(ring.y)
            elif reference == "mean":
                fpx_ = np.mean(rays.x[w])
                fpy_ = np.mean(rays.y[w])
            else:
                raise ValueError(f"Invalid reference {reference!r}")

            dx[ifield] = rays.x - fpx_
            dy[ifield] = rays.y - fpy_
            vignetted[ifield] = rays.vignetted
            fpx[ifield] = fpx_
            fpy[ifield] = fpy_

        out_field = FieldCoords(
            x=fpx.reshape(batch_shape + (nfield,)) * 1e3 << u.mm,
            y=fpy.reshape(batch_shape + (nfield,)) * 1e3 << u.mm,
            frame="ccs",
            rtp=self.rtp,
            camera=field.camera,
        )
        return Spots(
            dx.reshape(batch_shape + (nfield, nray)) * 1e6 << u.micron,
            dy.reshape(batch_shape + (nfield, nray)) * 1e6 << u.micron,
            vignetted.reshape(batch_shape + (nfield, nray)),
            field=out_field,
            wavelength=self.wavelength,
            frame="ccs",
            rtp=self.rtp,
            px=np.array(px) << u.m,
            py=np.array(py) << u.m,
        )

    def zernikes(
        self,
        field: FieldCoords,
        state: State | None = None,
        zk: Zernikes | DoubleZernikes | None = None,
        jmax: int = 28,
        rings: int = 10,
        reference: Literal["chief", "mean", "ring"] = "ring",
        algorithm: Literal["ta", "gq"] = "gq",
        include_chip_heights: bool = True,
        camera_piston: Quantity = None,
        detector_piston: Quantity = None,
    ) -> Zernikes:
        """Compute Zernike wavefront coefficients.

        Parameters
        ----------
        state : State | None
            AOS alignment state.
        field : FieldCoords
            Field coordinates.  Converted to OCS angles internally.
        zk : Zernikes | DoubleZernikes | None
            Extra Zernike phase perturbation to apply at the entrance pupil.
            If ``DoubleZernikes``, it is first converted to ``Zernikes``
            via ``.single(field)``.
            The coefs array is broadcast against the field shape:
            ``(1, jmax_zk+1)`` applies the same perturbation everywhere;
            ``(nfield, jmax_zk+1)`` gives per-field perturbations broadcast
            over batch dims; ``(*batch_shape, nfield, jmax_zk+1)`` is fully
            specified.
        jmax : int
            Maximum Noll index (default: 28).
        rings : int
            Number of radial rings for computing zernikes.
            (default: 10).
        reference : Literal["chief", "mean", "ring"]
            Whether to compute Zernikes relative to the chief ray intersection,
            the mean intersection of all unvignetted rays, or the mean of a
            ring of (possibly vignetted) rays.
        algorithm : Literal["ta", "gq"]
            Algorithm to use for computing zernikes.  "ta" uses the
            transverse aberration approach, "gq" uses Gaussian quadrature.
        include_chip_heights : bool
            Whether to include the effect of CCD chip height variations in the
            spot diagrams.
        camera_piston : Quantity or None
            Optional extra piston term applied to the entire camera.
        detector_piston : Quantity or None
            Optional extra piston term applied to each detector independently.  If
            None, then use the canonical piston for science/intra/extra detectors.

        Returns
        -------
        Zernikes
            Coefficients in units of microns (OCS frame).
            Shape ``(*batch_shape, nfield, jmax+1)``.
        """
        field, batch_shape, nfield, _, iterator = self._iter_thx_thy_telescope(
            field=field,
            state=state,
            zk=zk,
            include_chip_heights=include_chip_heights,
            camera_piston=camera_piston,
            detector_piston=detector_piston,
        )

        zk_out = []
        for _, thx, thy, telescope in iterator:
            if telescope is None:
                zk_out.append(np.full(jmax + 1, np.nan))
                continue
            if algorithm == "gq":
                try:
                    zk1 = batoid.zernikeGQ(
                        telescope,
                        theta_x=thx.to_value(u.rad),
                        theta_y=thy.to_value(u.rad),
                        wavelength=self.wavelength.to_value(u.m),
                        rings=rings,
                        jmax=jmax,
                        eps=PUPIL_INNER / PUPIL_OUTER,
                        reference=reference,
                        projection="gnomonic",
                    )
                except ValueError:
                    zk1 = np.full(jmax + 1, np.nan)
            elif algorithm == "ta":
                try:
                    zk1 = batoid.zernikeTA(
                        telescope,
                        theta_x=thx.to_value(u.rad),
                        theta_y=thy.to_value(u.rad),
                        wavelength=self.wavelength.to_value(u.m),
                        nrad=rings,
                        naz=int(2 * np.pi * rings / (1 - PUPIL_INNER / PUPIL_OUTER)),
                        jmax=jmax,
                        eps=PUPIL_INNER / PUPIL_OUTER,
                        focal_length=10.31,
                        reference=reference,
                        projection="gnomonic",
                    )
                except ValueError:
                    zk1 = np.full(jmax + 1, np.nan)
            zk_out.append(zk1)

        coefs = np.array(zk_out, dtype=float).reshape(batch_shape + (nfield, jmax + 1))
        return Zernikes(
            coefs=coefs * self.wavelength.to(u.um),
            field=field,
            R_outer=PUPIL_OUTER * u.m,
            R_inner=PUPIL_INNER * u.m,
            wavelength=self.wavelength,
            frame="ocs",
            rtp=self.rtp,
        )

    def double_zernikes(
        self,
        kmax: int,
        field_outer: Quantity,
        field_inner: Quantity = None,
        **kwargs,
    ):
        """Compute the double-Zernike basis functions.

        Parameters
        ----------
        kmax : int
            Maximum field Noll index for the double-Zernike fit (default: 28).
        field_outer : Quantity
            Outer field radius for the field-Zernike basis.  Required.
        field_inner : Quantity or None
            Inner field radius.  Defaults to zero.
        **kwargs
            Extra keyword arguments forwarded to :meth:`zernikes`.

        Returns
        -------
        DoubleZernikes
            Coefficients in units of microns (OCS frame).
            Shape ``(kmax+1, jmax+1)``.
        """
        zk = self.zernikes(**kwargs)
        return zk.double(kmax, field_outer, field_inner)

    def _optimize_dx_func(
        self,
        params: NDArray[np.float64],
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
        spots = self.spots(field=field, state=state, nrad=nrad)
        vignetted = spots.vignetted
        return np.concatenate(
            [
                spots.dx[~vignetted].to_value(u.micron),
                spots.dy[~vignetted].to_value(u.micron),
            ]
        )

    def _optimize_func(
        self,
        params: NDArray[np.float64],
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
        spots = self.spots(field=field, state=state, nrad=nrad)
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
        nrad: int = 10,
        mode: str = "dx",
        **kwargs,
    ) -> State:
        """Optimise the AOS DOFs to minimise spot size.

        Uses ``scipy.optimize.least_squares`` to find the DOF vector that
        minimises residual spot positions (``mode='dx'``) or spot size
        variance (``mode='var'``) across the supplied field.

        Parameters
        ----------
        guess : State
            Initial state; its ``use_dof`` defines which DOFs are free.
        field : FieldCoords
            Field coordinates to evaluate.
        nrad : int
            Pupil sampling rings passed to :meth:`spots`.
        mode : str
            ``'dx'`` (default) minimises raw spot displacements;
            ``'var'`` minimises per-field spot-size variance.
        **kwargs
            Extra keyword arguments forwarded to ``least_squares``.

        Returns
        -------
        State
            Optimised state in the ``'x'`` (active-DOF) basis.
        """
        from scipy.optimize import least_squares

        func = self._optimize_dx_func if mode == "dx" else self._optimize_func

        x0 = guess.x.value
        result = least_squares(
            func,
            x0,
            x_scale=self.steps[guess.use_dof],
            args=(field, nrad, guess.use_dof),
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
        reference: Literal["chief", "mean", "ring"] = "ring",
        algorithm: Literal["ta", "gq"] = "gq",
        include_chip_heights: bool = True,
        tqdm: type[TqdmType] | None = None,
    ) -> Sensitivity:
        """Compute the Zernike wavefront sensitivity matrix via finite differences.

        For each DOF in *steps*, perturbs the alignment state by the
        corresponding step size and records
        ``(zernikes(offset + step) - zernikes(offset)) / step``.

        Parameters
        ----------
        field : FieldCoords
            Field coordinates at which to evaluate the sensitivity.
        steps : State
            Step sizes in the desired DOF basis.  Its ``basis`` and
            ``use_dof`` define which DOFs are included.
        jmax : int
            Maximum Noll index (default: 28).
        rings : int
            Quadrature rings for :meth:`zernikes` (default: 10).
        reference : Literal["chief", "mean", "ring"]
            Whether to compute Zernikes relative to the chief ray intersection,
            the mean intersection of all unvignetted rays, or the mean of a
            ring of (possibly vignetted) rays.
        algorithm : Literal["ta", "gq"]
            Algorithm to use for computing zernikes.  "ta" uses the
            transverse aberration approach, "gq" uses Gaussian quadrature.
        include_chip_heights : bool
            Whether to include the effect of CCD chip height variations in the
            spot diagrams used for Zernike computation.
        tqdm : tqdm | None
            Optional tqdm progress bar.
        Returns
        -------
        Sensitivity[Zernikes]
            Gradient shape ``(ndof, nfield, jmax+1)``.
        """
        nominal = self.zernikes(
            field=field,
            jmax=jmax,
            rings=rings,
            algorithm=algorithm,
            reference=reference,
            include_chip_heights=include_chip_heights,
        )

        perturbed = []
        if tqdm is not None:
            iterator = tqdm(enumerate(steps.value), total=len(steps.value), desc="Computing Zernike sensitivities")
        else:
            iterator = enumerate(steps.value)
        for i, step in iterator:
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
                    field=field,
                    state=dstate,
                    jmax=jmax,
                    rings=rings,
                    reference=reference,
                    algorithm=algorithm,
                    include_chip_heights=include_chip_heights,
                )
            )
        return Sensitivity.from_finite_differences(nominal, perturbed, steps)

    def spots_sensitivity(
        self,
        field: FieldCoords,
        steps: State,
        nrad: int = 10,
        reference: Literal["chief", "mean", "ring"] = "ring",
        include_chip_heights: bool = True,
        tqdm: type[TqdmType] | None = None,
    ) -> Sensitivity:
        """Compute the spot-diagram sensitivity matrix via finite differences.

        For each DOF in *steps*, perturbs the alignment state and records
        ``(spots(offset + step) - spots(offset)) / step`` for ``dx`` and ``dy``.

        Parameters
        ----------
        field : FieldCoords
            Field coordinates.
        steps : State
            Per-DOF step sizes.
        nrad : int
            Pupil sampling rings for :meth:`spots` (default: 10).
        reference: Literal["chief", "mean", "ring"] = "ring"
            Whether to center spot diagrams on the chief ray intersection,
            the mean intersection of all unvignetted rays, or the mean of a
            ring of (possibly vignetted) rays.
        include_chip_heights : bool
            Whether to include the effect of CCD chip height variations in the
            spot diagrams used for Zernike computation.
        tqdm : tqdm | None
            Optional tqdm progress bar.

        Returns
        -------
        Sensitivity[Spots]
            Gradient shape ``(ndof, nfield, nray)`` for ``dx`` and ``dy``.
        """
        nominal = self.spots(
            field=field,
            nrad=nrad,
            reference=reference,
            include_chip_heights=include_chip_heights,
        )

        perturbed = []
        if tqdm is not None:
            iterator = tqdm(enumerate(steps.value), total=len(steps.value), desc="Computing Spot sensitivities")
        else:
            iterator = enumerate(steps.value)
        for i, step in iterator:
            dval = np.zeros_like(steps.value)
            dval[i] = step
            dstate = replace(
                steps,
                value=dval
            )
            perturbed.append(
                self.spots(
                    field=field,
                    state=dstate,
                    nrad=nrad,
                    reference=reference,
                    include_chip_heights=include_chip_heights,
                )
            )
        return Sensitivity.from_finite_differences(nominal, perturbed, steps)

    def double_zernikes_sensitivity(
        self,
        kmax: int,
        field_outer: Quantity,
        field_inner: Quantity = None,
        tqdm: type[TqdmType] | None = None,
        **kwargs,
    ) -> Sensitivity:
        """Compute the double-Zernike sensitivity matrix via finite differences.

        Internally calls :meth:`zernikes_sensitivity` and then projects each
        per-field Zernike result into the double-Zernike (field x pupil)
        basis via :meth:`~StarSharp.datatypes.Zernikes.double`.

        Parameters
        ----------
        kmax : int
            Maximum field Noll index for the double-Zernike fit (default: 28).
        field_outer : Quantity
            Outer field radius for the field-Zernike basis.  Required.
        field_inner : Quantity or None
            Inner field radius.  Defaults to zero.
        tqdm : tqdm | None
            Optional tqdm progress bar forwarded to :meth:`zernikes_sensitivity`.
        **kwargs
            Extra keyword arguments forwarded to :meth:`zernikes_sensitivity`.

        Returns
        -------
        Sensitivity[DoubleZernikes]
            Gradient shape ``(ndof, kmax+1, jmax+1)``.
        """
        zk_sens = self.zernikes_sensitivity(
            **kwargs
        )
        dz_nominal = zk_sens.nominal.double(kmax, field_outer, field_inner)
        dz_gradient = zk_sens.gradient.double(kmax, field_outer, field_inner)
        return Sensitivity(
            gradient=dz_gradient,
            nominal=dz_nominal,
            basis=zk_sens.basis,
            use_dof=zk_sens.use_dof,
            n_dof=zk_sens.n_dof,
            Vh=zk_sens.Vh,
        )
