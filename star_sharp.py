from functools import cached_property
from pathlib import Path
from typing import Callable, Optional, Sequence, Union


import asdf
import batoid
import numpy as np
from astropy.table import Table
from batoid.utils import hexapolar
from batoid_rubin import LSSTBuilder
from galsim.zernike import DoubleZernike, doubleZernikeBasis, zernikeBasis
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial import KDTree


FloatArray = Union[
    Sequence[float],
    NDArray[np.floating],
]


ScalarOrArray = Union[
    float,
    int,
    Sequence[float],
    Sequence[int],
    NDArray[np.floating],
    NDArray[np.integer],
]


IntegerArray = Union[Sequence[int], NDArray[np.integer]]

import typing
if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes


WAVELENGTHS = dict(
    u=370.9e-9,
    g=476.7e-9,
    r=619.4e-9,
    i=753.9e-9,
    z=866.8e-9,
    y=973.9e-9
)


SIGMA_TO_FWHM = np.sqrt(np.log(256))


def spot_size(
    optic: batoid.Optic,
    wavelength: float,
    nrad: int = 5,
    naz: int = 30,
    outer: float = 1.75
) -> NDArray[np.float64]:
    """
    Estimate spot sizes for a given optic and wavelength.

    Parameters
    ----------
    optic : batoid.Optic
        Telescope to analyze.
    wavelength : float
        Wavelength in meters.
    nrad : int, optional
        Number of radial points in the field.
    naz : int, optional
        Number of azimuthal points at max field radius.
    outer : float, optional
        Outer field radius in degrees.

    Returns
    -------
    sizes : NDArray[np.float64]
        Spot sizes in arcseconds.

    Notes
    -----
    Ignores vignetting.
    Pupil is tuned to Rubin.
    """
    thx, thy = batoid.utils.hexapolar(
        outer=np.deg2rad(outer),
        inner=0.0,
        nrad=nrad,
        naz=naz,
    )

    sizes = []
    for thx_, thy_ in zip(thx, thy):
        rays = batoid.RayVector.asPolar(
            optic,
            wavelength=wavelength,
            theta_x=thx_,
            theta_y=thy_,
            nrad=nrad * 3,  # 3x more pupil points than field
            naz=naz * 3,
            outer=4.15,  # Avoid clipping the actual pupil
            inner=2.6,
        )
        rays = optic.trace(rays)
        xs = rays.x
        ys = rays.y
        sizes.extend([np.std(xs), np.std(ys)])

    # convert to arcsec FWHM
    sizes = np.array(sizes) * 0.2 * np.sqrt(np.log(256)) / 10e-6
    assert isinstance(sizes, np.ndarray)

    return sizes


class FocusObjective:
    """
    Focus objective function for optimizing telescope spot size.

    Parameters
    ----------
    optic : batoid.Optic
        Telescope to analyze.
    wavelength : float
        Wavelength in meters.
    use_dof : IntegerArray
        Which degrees of freedom to optimize.
    **kwargs
        Additional keyword arguments to pass to spot_size.
    """
    # Set up an objective function for least_squares to optimize.
    def __init__(
        self,
        optic: batoid.Optic,
        wavelength: float,
        use_dof: IntegerArray,
        **kwargs
    ) -> None:
        self.optic = optic
        self.wavelength = wavelength

        # which dof to use.  dz,dy,dy,rx,ry = [5,6,7,8,9]
        # only consider camera for now.
        self.use_dof = use_dof

        self.kwargs = kwargs

    def get_optic(
        self,
        x: FloatArray,
    ) -> batoid.Optic:
        """
        Get a new optic with the specified degrees of freedom applied.

        Parameters
        ----------
        x : FloatArray
            Values for the degrees of freedom to apply.
            Units are meters for translation and radians for rotation.

        Returns
        -------
        batoid.Optic
            The modified optic.
        """
        optic = self.optic
        m2_dr = np.zeros(3, dtype=float)
        m2_rot = np.eye(3, dtype=float)
        cam_dr = np.zeros(3, dtype=float)
        cam_rot = np.eye(3, dtype=float)
        for idof, dof in enumerate(self.use_dof):
            if dof == 0:
                m2_dr[2] = x[idof]
            elif dof == 1:
                m2_dr[0] = x[idof]
            elif dof == 2:
                m2_dr[1] = x[idof]
            elif dof == 3:
                m2_rot @= batoid.RotX(x[idof])
            elif dof == 4:
                m2_rot @= batoid.RotY(x[idof])
            elif dof == 5:
                cam_dr[2] = x[idof]
            elif dof == 6:
                cam_dr[0] = x[idof]
            elif dof == 7:
                cam_dr[1] = x[idof]
            elif dof == 8:
                cam_rot @= batoid.RotX(x[idof])
            elif dof == 9:
                cam_rot @= batoid.RotY(x[idof])

        optic = optic.withGloballyShiftedOptic(
            "LSSTCamera",
            cam_dr,
        ).withGloballyShiftedOptic(
            "M2",
            m2_dr,
        ).withLocallyRotatedOptic(
            "LSSTCamera",
            cam_rot,
        ).withLocallyRotatedOptic(
            "M2",
            m2_rot,
        )
        return optic

    def __call__(
        self,
        x: FloatArray,
    ) -> NDArray[np.float64]:
        """
        Evaluate the objective function at the given state.

        Parameters
        ----------
        x : FloatArray
            Values for the degrees of freedom to apply.
            Units are meters for translation and radians for rotation.

        Returns
        -------
        sizes : NDArray[np.float64]
            Spot sizes in arcseconds.
        """
        optic = self.get_optic(x)
        return spot_size(optic, self.wavelength, **self.kwargs)


def get_dzs(
    optic: batoid.Optic,
    wavelength: float,
    kmax: int = 36,
    jmax: int = 28,
    nrad: int = 7,
    naz: int = 45,
    outer: float = 1.75,
    verbose=False,
) -> NDArray[np.float64]:
    """
    Compute double Zernike coefficients for given telescope.

    Parameters
    ----------
    optic : batoid.Optic
        Telescope to analyze.
    wavelength : float
        Wavelength in meters.
    kmax : int, optional
        Maximum field Noll index.
    jmax : int, optional
        Maximum pupil Noll index.
    nrad : int, optional
        Number of radial points in the field.
    naz : int, optional
        Number of azimuthal points at max field radius.
    outer : float
        The outer field radius in degrees.
    verbose : bool
        Print verbose output?

    Returns
    -------
    NDArray[np.float64]
        The double Zernike coefficients.

    Notes
    -----
    Ignores vignetting.
    Pupil is tuned to Rubin.
    """
    thx, thy = batoid.utils.hexapolar(
        outer=outer,
        inner=0.0,
        nrad=nrad,
        naz=naz,
    )
    basis = zernikeBasis(kmax, thx, thy, R_outer=outer)

    zk = np.zeros((len(thx), jmax+1))
    for i, (thx_, thy_) in enumerate(zip(thx, thy)):
        zk[i] = (
            batoid.zernikeGQ(
                optic,
                np.deg2rad(thx_),
                np.deg2rad(thy_),
                wavelength,
                jmax=jmax,
                eps=0.612,
                rings=15,
            )
            * wavelength
            * 1e6
        )  # convert to microns
    dzs, *_ = np.linalg.lstsq(basis.T, zk, rcond=None)
    dzs[:, :4] = 0.0  # Zero out PTT
    dzs[0, :] = 0.0  # k=0 is unused

    if verbose:
        # What are the dominant remaining terms?
        asort = np.argsort(np.square(dzs).ravel())[::-1]
        ks, js = np.unravel_index(asort[:20], dzs.shape)
        cumsum = 0.0
        for k, j in zip(ks, js):
            val = dzs[k, j]
            cumsum += val**2
            print("{:3d} {:3d} {:8.4f} {:8.4f}".format(k, j, val, np.sqrt(cumsum)))
        print("sum sqr dz {:8.4f}".format(np.sqrt(np.sum(dzs**2))))
    return dzs


def grid_measurements(
    u: FloatArray, # [n,]
    v: FloatArray, # [n,]
    vals: dict[str, FloatArray],  # process each key...
    ugrid: FloatArray, # [k,]
    vgrid: FloatArray, # [k,]
) -> dict[str, NDArray[np.float64]]:
    if isinstance(vals, Table):
        vals = {k:vals[k] for k in vals.colnames}
    ref = np.array(list(zip(ugrid, vgrid)))
    test = np.array(list(zip(u, v)))
    tree = KDTree(ref)
    distances, dest_indices = tree.query(test)

    val_sums = {}
    for k in vals.keys():
        val_sums[k] = {i:0.0 for i in range(len(ugrid))}
    counts = {i:0 for i in range(len(ugrid))}

    for src_idx, dest_idx in enumerate(dest_indices):
        for k in vals.keys():
            val_sums[k][dest_idx] += vals[k][src_idx]
        counts[dest_idx] += 1

    out = {
        k: np.array(
            [
                float(val_sums[k][i] / counts[i])
                if counts[i] != 0
                else float("nan")
                for i in range(len(ugrid))
            ],
            dtype=np.float64
        )
        for k in vals
    }

    return out


class StarSharp:
    """ Fit optical state either from wavefront measurements
    (Zernikes) or 2nd moment measurements.

    Paramters
    ---------
    band : str
        Rubin ugrizy band
    use_dof : IntegerArray, optional
        Degrees of freedom to use in the fit.
    nkeep : int, optional
        Number of modes to keep.
    transverse_pupil_radii : int, optional
        Number of radii when creating sample points in the pupil.
    transverse_field_radii : int, optional
        Number of radii when creating sample points in the field.
    wf_kmax : int, optional
        Maximum field index for sensitivity matrix and intrinsic Zernikes.
    wf_jmax : int, optional
        Maximum pupil index for sensitivity matrix and intrinsic Zernikes.
    ortho_transverse : bool, optional
        Use transverse sensitivity to orthogonalize.
    tqdm : callable, optional
        Optional progressbar callable.
    """
    def __init__(
        self,
        band: str,
        use_dof: Optional[IntegerArray] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        nkeep: Optional[int] = 6,
        transverse_pupil_radii: int = 10,
        transverse_field_radii: int = 14,
        wf_kmax: int = 15,
        wf_jmax: int = 28,
        ortho_transverse: bool = False,
        tqdm: Optional[Callable] = None,
    ) -> None:
        # Use builder instead of fiducial so don't need to pass in
        # the fea_dir and bend_dir.
        self.band = band
        self.fiducial = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
        self.builder = LSSTBuilder(self.fiducial)
        self.wavelength = WAVELENGTHS[band]
        self.use_dof = np.array(use_dof, dtype=int)
        self.nkeep = nkeep
        self.transverse_pupil_radii = transverse_pupil_radii
        self.transverse_field_radii = transverse_field_radii
        self.wf_kmax = wf_kmax
        self.wf_jmax = wf_jmax
        self.ortho_transverse = ortho_transverse
        self.tqdm = tqdm

        self.n_dof = len(self.builder.dof)
        assert self.n_dof == 50  # Future: make this dynamic?

        # For transverse model, we pre-select field and pupil evaluation points
        self.pupil_x, self.pupil_y = hexapolar(
            outer=4.15,
            inner=2.6,
            nrad=transverse_pupil_radii,
            naz=int(2 * np.pi * transverse_pupil_radii),
        )
        self.field_u, self.field_v = hexapolar(
            outer=1.75,
            nrad=transverse_field_radii,
            naz=int(2 * np.pi * transverse_field_radii),
        )
        self.n_pupil = len(self.pupil_x)
        self.n_field = len(self.field_u)

        # To be populated on demand
        self._dx: Optional[NDArray[np.float64]] = None
        self._dy: Optional[NDArray[np.float64]] = None
        self._x0: Optional[NDArray[np.float64]] = None
        self._y0: Optional[NDArray[np.float64]] = None
        # Full sensitivity matrix
        self._A: Optional[NDArray[np.float64]] = None
        # Intrinsic double Zernike coefficients
        self._zk0: Optional[NDArray[np.float64]] = None

    @property
    def dx(self) -> NDArray[np.float64]:
        """x-displacement sensitivity of ray indexed by
        [degree-of-freedom, field, pupil] in meters per micron-or-arcsec.
        """
        if self._dx is None:
            self._build_transverse_sensitivity()
            assert self._dx is not None
        return self._dx

    @property
    def dy(self) -> NDArray[np.float64]:
        """y-displacement sensitivity of ray indexed by
        [degree-of-freedom, field, pupil] in meters per micron-or-arcsec.
        """
        if self._dy is None:
            self._build_transverse_sensitivity()
            assert self._dy is not None
        return self._dy

    @property
    def x0(self) -> NDArray[np.float64]:
        """Initial x-displacement of ray indexed by field and pupil in meters.
        """
        if self._x0 is None:
            self._build_transverse_sensitivity()
            assert self._x0 is not None
        return self._x0

    @property
    def y0(self) -> NDArray[np.float64]:
        """Initial y-displacement of ray indexed by field and pupil in meters.
        """
        if self._y0 is None:
            self._build_transverse_sensitivity()
            assert self._y0 is not None
        return self._y0

    @property
    def A(self) -> NDArray[np.float64]:
        """Wavefront sensitivity matrix as double Zernike series.
        Indexed as [degree-of-freedom, field, pupil] in meters per micron / arcsec.
        """
        if self._A is None:
            self._build_wf_sensitivity()
            assert self._A is not None
        return self._A

    @property
    def zk0(self) -> NDArray[np.float64]:
        """Intrinsic double Zernike coefficients.
        Indexed as [field, pupil] in meters.
        """
        if self._zk0 is None:
            self._build_wf_sensitivity()
            assert self._zk0 is not None
        return self._zk0

    @property
    def A_dz(self) -> DoubleZernike:
        """Wavefront sensitivity matrix as a list of DoubleZernike objects -
        one for each degree-of-freedom.
        """
        return [
            DoubleZernike(
                self.A[i],
                uv_outer=1.75,
                xy_outer=4.18,
                xy_inner=4.18 * 0.612,
            )
            for i in range(self.n_dof)
        ]

    @property
    def zk0_dz(self) -> DoubleZernike:
        """Intrinsic aberrations as DoubleZernike object.
        """
        return DoubleZernike(
            self.zk0,
            uv_outer=1.75,
            xy_outer=4.18,
            xy_inner=4.18 * 0.612,
        )

    @property
    def _steps(self) -> NDArray[np.float64]:
        """Initial step sizes to use when computing sensitivities.
        """
        steps = [
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
        steps += [0.1] * 40  # bending modes
        return np.array(steps)

    @property
    def dof_signs(self) -> NDArray[np.float64]:
        signs = np.ones(self.n_dof, dtype=float)
        signs[[0, 1, 3, 5, 6, 8]] = -1  # flip z and y coords
        signs[list(range(30, 50))] = -1  # flip M2 response
        return signs

    def _build_transverse_sensitivity(self) -> None:
        """Build the transverse aberration sensitivities.
        """
        # Check if we already have the sensitivities on disk:
        fn = "transverse_sensitivity_"
        fn += self.band
        fn += f"_{self.transverse_field_radii}"
        fn += f"_{self.transverse_pupil_radii}"
        fn += ".asdf"
        path = Path(fn)
        if path.is_file():
            with asdf.open(path, lazy_load=False) as af:
                self._dx = af["dx"]
                self._dy = af["dy"]
                self._x0 = af["x0"]
                self._y0 = af["y0"]
            return

        self._dx = np.empty((self.n_dof, self.n_field, self.n_pupil))
        self._dy = np.empty((self.n_dof, self.n_field, self.n_pupil))
        self._x0 = np.empty((self.n_field, self.n_pupil))
        self._y0 = np.empty((self.n_field, self.n_pupil))
        bar = None
        if self.tqdm is not None:
            bar = self.tqdm(total=self.n_dof, desc="Building transverse sensitivity")
        for idof, (step, sign) in enumerate(zip(self._steps, self.dof_signs)):
            dof = np.zeros(self.n_dof)
            dof[idof] = step * sign
            perturbed = self.builder.with_aos_dof(dof).build()
            for ith, (u, v) in enumerate(zip(self.field_u, self.field_v)):
                rays = batoid.RayVector.fromStop(
                    np.array(self.pupil_x),
                    np.array(self.pupil_y),
                    theta_x=np.deg2rad(u),
                    theta_y=np.deg2rad(v),
                    optic=self.fiducial,
                    wavelength=self.wavelength,
                )
                frays = self.fiducial.trace(rays.copy())
                prays = perturbed.trace(rays)

                # Subtract the motion of the chief ray.
                cr = batoid.RayVector.fromStop(
                    np.array([0]),
                    np.array([0]),
                    theta_x=np.deg2rad(u),
                    theta_y=np.deg2rad(v),
                    optic=self.fiducial,
                    wavelength=self.wavelength,
                )
                fcr = self.fiducial.trace(cr.copy())
                pcr = perturbed.trace(cr)
                crdx = pcr.x - fcr.x
                crdy = pcr.y - fcr.y

                self._dx[idof][ith] = (prays.x - frays.x - crdx) / step
                self._dy[idof][ith] = (prays.y - frays.y - crdy) / step
                self._dx[idof][ith][frays.vignetted] = np.nan
                self._dy[idof][ith][frays.vignetted] = np.nan
                if idof == 0:
                    self._x0[ith] = np.array(frays.x)
                    self._y0[ith] = np.array(frays.y)
                    self._x0[ith][frays.vignetted] = np.nan
                    self._y0[ith][frays.vignetted] = np.nan
                    self._x0[ith] -= np.nanmean(self._x0[ith])
                    self._y0[ith] -= np.nanmean(self._y0[ith])
            if bar is not None:
                bar.update(1)
        # Save the transverse sensitivity to disk
        with asdf.AsdfFile(
            {
                "dx": self._dx,
                "dy": self._dy,
                "x0": self._x0,
                "y0": self._y0,
            }
        ) as af:
            af.write_to(path)

    def _build_wf_sensitivity(self) -> None:
        """Build the wavefront sensivities.
        """
        # Check if we already have the sensitivities on disk:
        fn = "wf_sensitivity_"
        fn += self.band
        fn += f"_{self.wf_kmax}"
        fn += f"_{self.wf_jmax}"
        fn += ".asdf"
        path = Path(fn)
        if path.is_file():
            with asdf.open(path, lazy_load=False) as af:
                self._A = af["A"]
                self._zk0 = af["zk0"]
            return
        self._zk0 = get_dzs(
            self.fiducial,
            self.wavelength,
            kmax=self.wf_kmax,
            jmax=self.wf_jmax,
        )
        self._A = np.empty((self.n_dof, self.wf_kmax + 1, self.wf_jmax + 1))
        bar = None
        if self.tqdm is not None:
            bar = self.tqdm(total=self.n_dof, desc="Building wavefront sensitivity")
        for idof, step in enumerate(self._steps):
            dof = np.zeros(self.n_dof, dtype=float)
            dof[idof] = step
            perturbed = self.builder.with_aos_dof(dof).build()
            perturbed_dzs = get_dzs(
                perturbed,
                self.wavelength,
                kmax=self.wf_kmax,
                jmax=self.wf_jmax,
            )
            self._A[idof] = (perturbed_dzs - self._zk0) / step
            if bar is not None:
                bar.update(1)
        # Save the wavefront sensitivity to disk
        with asdf.AsdfFile(
            {
                "A": self._A,
                "zk0": self._zk0,
            }
        ) as af:
            af.write_to(path)

    @cached_property
    def _ranges(self) -> NDArray[np.float64]:
        """Operational ranges of each degree-of-freedom.
        """
        ranges = np.array(
            [
                6700.0,  # M2 dz
                5900.0,  # M2 dx
                5900.0,  # M2 dy
                0.12 * 3600,  # M2 rx
                0.12 * 3600,  # M2 ry
                7600.0,  # cam dz
                8700.0,  # cam dx
                8700.0,  # cam dy
                0.1 * 3600,  # cam rx
                0.1 * 3600,  # cam ry
                0.454,  # M1M3 B1
                0.452,  # M1M3 B2
                0.087,  # M1M3 B3
                0.066,  # M1M3 B4
                0.066,  # M1M3 B5
                0.023,  # M1M3 B6
                0.021,  # M1M3 B7
                0.022,  # M1M3 B8
                0.019,  # M1M3 B9
                0.013,  # M1M3 B10
                0.013,  # M1M3 B11
                0.009,  # M1M3 B12
                0.009,  # M1M3 B13
                0.009,  # M1M3 B14
                0.004,  # M1M3 B15
                0.004,  # M1M3 B16
                0.006,  # M1M3 B17
                0.006,  # M1M3 B18
                0.005,  # M1M3 B19
                0.002,  # M1M3 B20
                4.287,  # M2 B1
                4.306,  # M2 B2
                0.609,  # M2 B3
                0.557,  # M2 B4
                0.331,  # M2 B5
                0.136,  # M2 B6
                0.138,  # M2 B7
                0.160,  # M2 B8
                0.159,  # M2 B9
                0.076,  # M2 B10
                0.075,  # M2 B11
                0.064,  # M2 B12
                0.065,  # M2 B13
                0.039,  # M2 B14
                0.033,  # M2 B15
                0.030,  # M2 B16
                0.032,  # M2 B17
                0.011,  # M2 B18
                0.008,  # M2 B19
                0.007,  # M2 B20
            ]
        )
        # Adjust M2 mode ranges to account for lbf to N conversion
        ranges[30:] /= 4.448222
        return ranges

    @cached_property
    def _moments_power(self) -> NDArray[np.float64]:
        """Relative ability of degree of freedom to affect second moments.
        """
        # Variance over the pupil, Mean over the field, one entry per dof.
        return np.mean(
            np.sqrt(np.nanvar(self.dx, axis=2) + np.nanvar(self.dy, axis=2)), axis=1
        )

    @cached_property
    def _transverse_double_zernikes(self) -> NDArray[np.float64]:
        fn = "transverse_double_zernikes_"
        fn += self.band
        fn += f"_{self.transverse_field_radii}"
        fn += f"_{self.transverse_pupil_radii}"
        fn += f"_{self.wf_kmax}_{self.wf_jmax}"
        fn += ".asdf"
        path = Path(fn)
        if path.is_file():
            with asdf.open(path, lazy_load=False) as af:
                tdz = af["tdz"]
                return tdz

        kmax = self.wf_kmax
        jmax = self.wf_jmax
        u = np.repeat(self.field_u, self.n_pupil)
        v = np.repeat(self.field_v, self.n_pupil)
        x = np.tile(self.pupil_x, self.n_field)
        y = np.tile(self.pupil_y, self.n_field)
        dx = self.dx.reshape(self.n_dof, -1)
        dy = self.dy.reshape(self.n_dof, -1)

        # Filter out the NaNs
        good = np.isfinite(dx[0]) & np.isfinite(dy[0])
        dx = dx[:, good]
        dy = dy[:, good]
        u = u[good]
        v = v[good]
        x = x[good]
        y = y[good]

        dz_basis = doubleZernikeBasis(
            kmax,
            jmax,
            u,
            v,
            x,
            y,
            uv_outer=1.75,
            xy_outer=4.18,
            xy_inner=2.55,
        )
        dz_basis = dz_basis.reshape(-1, dz_basis.shape[-1])  # ravel the jk indices
        tdz = np.empty((self.n_dof, 2, kmax + 1, jmax + 1))
        xycoefs, *_ = np.linalg.lstsq(dz_basis.T, np.concatenate([dx, dy]).T, rcond=None)
        tdz[:, 0] = xycoefs[:, :self.n_dof].T.reshape((self.n_dof, kmax + 1, jmax + 1))
        tdz[:, 1] = xycoefs[:, self.n_dof:].T.reshape((self.n_dof, kmax + 1, jmax + 1))
        # k=0 and j=0 are unused
        # We also don't care about constant pupil terms
        tdz[:, :, 0, :] = 0.0
        tdz[:, :, :, :2] = 0.0
        with asdf.AsdfFile(
            {
                "tdz":tdz,
            }
        ) as af:
            af.write_to(path)
        return tdz

    @cached_property
    def _svd(self):
        if self.ortho_transverse:
            factor2 = self._transverse_double_zernikes
        else:
            factor2 = self.A
        sh = (-1,) + tuple([1]*(factor2.ndim-1))
        factor1 = (self._ranges * self._moments_power).reshape(sh)
        A = factor1 * factor2
        A = A[self.use_dof]
        A = A.reshape(len(self.use_dof), -1)
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        return U, S, Vh

    def orthogonal_to_nominal(
        self,
        state,
    ):
        if self.nkeep is None:
            return state
        U, S, Vh = self._svd
        assert len(state) == self.nkeep
        return U[:, :self.nkeep] @ state

    def moments_model(
        self,
        state: FloatArray,
        dIxx: Optional[float] = 0.0,
        dIxy: Optional[float] = 0.0,
        dIyy: Optional[float] = 0.0,
        include_intrinsic: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute the second moments of the field based on the state.

        Parameters
        ----------
        state : FloatArray
            The state of the degrees of freedom.
            Units are meters for translation/bending modes and arcsec for tilts.
        dIxx : float, optional
            Additional variance in Ixx to add.
        dIxy : float, optional
            Additional covariance in Ixy to add.
        dIyy : float, optional
            Additional variance in Iyy to add.
        include_intrinsic : bool, optional
            Whether to include the intrinsic aberrations in the model.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            A dictionary containing the second moments and related quantities.
        """
        state = np.asarray(state)
        assert len(state) == len(self.use_dof)
        this_dx = np.sum(state[:, None, None] * self.dx[self.use_dof], axis=0)
        this_dy = np.sum(state[:, None, None] * self.dy[self.use_dof], axis=0)
        if include_intrinsic:
            this_dx += self.x0
            this_dy += self.y0

        Ixx = np.nanvar(this_dx, axis=1)
        Iyy = np.nanvar(this_dy, axis=1)
        Ixy = np.nanmean(this_dx * this_dy, axis=1)
        Ixy -= np.nanmean(this_dx, axis=1) * np.nanmean(this_dy, axis=1)

        Ixx *= (1e6 * 0.02) ** 2  # meters -> arcsec
        Iyy *= (1e6 * 0.02) ** 2
        Ixy *= (1e6 * 0.02) ** 2

        # Add seeing
        Ixx += dIxx
        Iyy += dIyy
        Ixy += dIxy

        # Note to self.  "Bonus" stats below are negligible compute time
        T = Ixx + Iyy
        w1 = Ixx - Iyy
        w2 = 2 * Ixy
        w = np.hypot(w1, w2)
        e1 = w1 / T
        e2 = w2 / T
        e = np.hypot(e1, e2)
        beta = 0.5 * np.arctan2(e2, e1)
        ex = e * np.cos(beta)
        ey = e * np.sin(beta)
        wx = w * np.cos(beta)
        wy = w * np.sin(beta)
        FWHM = np.sqrt(T / 2 * np.log(256))
        return dict(
            u=self.field_u,
            v=self.field_v,
            Ixx=Ixx,
            Ixy=Ixy,
            Iyy=Iyy,
            T=T,
            w1=w1,
            w2=w2,
            w=w,
            e1=e1,
            e2=e2,
            e=e,
            beta=beta,
            ex=ex,
            ey=ey,
            wx=wx,
            wy=wy,
            FWHM=FWHM,
        )

    def wf_model(
        self,
        u: FloatArray,
        v: FloatArray,
        state: FloatArray,
        include_intrinsic: bool = True,
    ) -> NDArray[np.float64]:
        zk0 = self.zk0_dz.xycoef(u, v)
        A = np.empty((len(self.use_dof), len(u), self.wf_jmax + 1))
        for i, idof in enumerate(self.use_dof):
            A[i] = self.A_dz[idof].xycoef(u, v)

        wf = np.einsum("i, ijk -> jk", state, A)
        if include_intrinsic:
            wf += zk0

        return wf

    def _moments_residual(
        self,
        x: NDArray[np.float64],  # state + dIxx/xy/yy
        gridded_moment_measurements: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Compute the residuals for the moments model.

        Parameters
        ----------
        x : NDArray[np.float64]
            The input vector containing the (optionally rotated) state and
            additional moments.
        gridded_moment_measurements : dict[str, NDArray[np.float64]]
            The gridded moment measurements to compare against.

        Returns
        -------
        NDArray[np.float64]
            The residuals between the model and the measurements.
        """
        state = x[:-3]
        dIxx, dIxy, dIyy = x[-3:]
        Ixx = gridded_moment_measurements["Ixx"]
        Ixy = gridded_moment_measurements["Ixy"]
        Iyy = gridded_moment_measurements["Iyy"]

        if self.nkeep is not None:
            assert len(state) == self.nkeep
            state = self.orthogonal_to_nominal(state)
        else:
            assert len(state) == len(self.use_dof)
        test_mom = self.moments_model(
            state, dIxx=dIxx, dIxy=dIxy, dIyy=dIyy
        )

        good = ~np.isnan(Ixx)

        return np.concatenate(
            [
                (test_mom["Ixx"] - Ixx)[good],
                (test_mom["Iyy"] - Iyy)[good],
                (test_mom["Ixy"] - Ixy)[good],
            ]
        )

    def _wf_residual(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        state: NDArray[np.float64],
        use_zk: NDArray[np.int64],
        zk: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.nkeep is not None:
            state = self.orthogonal_to_nominal(state)
        test_wf = self.wf_model(
            u=u, v=v, state=state, include_intrinsic=False
        )
        return (test_wf[:, use_zk] - zk).ravel()

    def _combined_residual(
        self,
        x: NDArray[np.float64],  # state + dIxx/xy/yy
        gridded_moment_measurements: dict[str, NDArray[np.float64]],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        zk: NDArray[np.float64],
        use_zk: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        # Equally weighted for now
        return np.concatenate([
            self._moments_residual(x, gridded_moment_measurements),
            self._wf_residual(u, v, state=x[:-3], use_zk=use_zk, zk=zk)
        ])

    def plot_modes(
        self,
        ax0: "Axes",
        ax1: "Axes",
        **kwargs,
    ):
        U, S, Vh = self._svd
        ax0.imshow(U, origin="lower", aspect="auto", **kwargs)
        # Show breaks between hexapods and mirrors
        m2_hex_idx = np.where(self.use_dof < 5)[0]
        cam_hex_idx = np.where((self.use_dof >= 5) & (self.use_dof < 10))[0]
        m1m3_bend_idx = np.where((self.use_dof >= 10) & (self.use_dof < 30))[0]
        m2_bend_idx = np.where(self.use_dof >= 30)[0]
        if len(m2_hex_idx) > 0:
            if any(len(block) > 0 for block in [cam_hex_idx, m1m3_bend_idx, m2_bend_idx]):
                ax0.axhline(m2_hex_idx[-1] + 0.5, color="k", alpha=0.2)
        if len(cam_hex_idx) > 0:
            if any(len(block) > 0 for block in [m1m3_bend_idx, m2_bend_idx]):
                ax0.axhline(cam_hex_idx[-1] + 0.5, color="k", alpha=0.2)
        if len(m1m3_bend_idx) > 0:
            if len(m2_bend_idx) > 0:
                ax0.axhline(m1m3_bend_idx[-1] + 0.5, color="k", alpha=0.2)

        nkeep = self.nkeep if self.nkeep is not None else len(S)
        ax0.axvspan(nkeep - 0.5, len(S)-0.5, color="k", alpha=0.2)

        ax1.plot(np.arange(1, len(S)+1), S)
        ax1.set_xlim(0.5, len(S) + 0.5)
        ax1.set_xticks([i for i in range(5, len(self.use_dof)+1, 5)])
        ax1.set_xticklabels([f"{i}" for i in range(5, len(self.use_dof)+1, 5)])
        ax1.set_yscale("log")
        ax1.set_yticks([])
        ax1.axvspan(nkeep + 0.5, len(S)+0.5, color="k", alpha=0.2)

    def plot_sens_dz(
        self,
        ax: "Axes",
        **kwargs
    ):
        # Declare that interesting DZs to plot are
        dzs = [(k, j) for j in range(4, 8+1) for k in range(1, 3+1)]
        dzs.extend([(1, j) for j in range(9, 15+1)])
        rot, S, Vh = self._svd
        A = self.A[self.use_dof]
        # sensitivity of each vmode
        Achar = np.einsum("ab,ajk->bjk", rot, A)
        # Normalize Achar
        for i in range(Achar.shape[0]):
            Achar[i] /= np.linalg.norm(Achar[i])
        show = np.zeros((len(dzs), len(self.use_dof)))
        for i, (k, j) in enumerate(dzs):
            Acol = Achar[:, k, j]
            show[i, :] = Acol
        ax.imshow(show, origin="lower", aspect="auto", **kwargs)
        ax.set_yticks(list(range(len(dzs))))
        ax.set_yticklabels([f"({i},{j})" for i, j in dzs])
        ax.set_xticks([i-1 for i in range(5, len(self.use_dof)+1, 5)])
        ax.set_xticklabels([f"{i}" for i in range(5, len(self.use_dof)+1, 5)])
        nkeep = self.nkeep if self.nkeep is not None else len(self.use_dof)
        ax.axvspan(nkeep - 0.5, len(self.use_dof)-0.5, color="k", alpha=0.2)

    def fit_moments(
        self,
        gridded_moment_measurements: dict[str, NDArray[np.float64]],
        **kwargs,
    ) -> dict[str, ScalarOrArray]:
        nguess = len(self.use_dof) if self.nkeep is None else self.nkeep
        guess = [0]*nguess + [0.5**2 / np.log(256)] * 2 + [0]
        result = least_squares(
            self._moments_residual,
            guess,
            args=(gridded_moment_measurements,),
            **kwargs
        )
        assert len(result.x) == nguess + 3
        dof = self.orthogonal_to_nominal(result.x[:-3])
        Ixx, Ixy, Iyy = result.x[-3:]
        result = dict(
            state=dof,
            Ixx=float(Ixx),
            Ixy=float(Ixy),
            Iyy=float(Iyy),
        )
        return result

    def fit_wf(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        zk: NDArray[np.float64],
        use_zk: Optional[IntegerArray] = None,
    ) -> dict[str, ScalarOrArray]:
        if use_zk is None:
            use_zk = list(range(4, self.wf_jmax + 1))

        zk0 = self.zk0_dz.xycoef(u, v)[:, use_zk]
        A = np.empty((len(self.use_dof), len(u), len(use_zk)))
        for i, idof in enumerate(self.use_dof):
            A[i] = self.A_dz[idof].xycoef(u, v)[:, use_zk]

        dzk = zk - zk0

        A = A.reshape(A.shape[0], -1)
        dzk = dzk.reshape(-1)

        if self.nkeep is not None:
            U, S, Vh = self._svd
            A = U[:, :self.nkeep].T @ A

        state, *_ = np.linalg.lstsq(A.T, dzk, rcond=None)
        state = self.orthogonal_to_nominal(state)
        result = dict(
            state=state,
            Ixx=float("nan"),
            Ixy=float("nan"),
            Iyy=float("nan"),
        )
        return result

    def fit_both(
        self,
        gridded_moment_measurements: dict[str, NDArray[np.float64]],
        u: FloatArray,
        v: FloatArray,
        wf_measurements: FloatArray,
        use_zk: Optional[IntegerArray] = None,
        **kwargs,
    ) -> dict[str, ScalarOrArray]:
        nguess = len(self.use_dof) if self.nkeep is None else self.nkeep
        guess = [0]*nguess + [(1.0 / SIGMA_TO_FWHM)**2] * 2 + [0]
        result = least_squares(
            self._combined_residual,
            guess,
            args=(gridded_moment_measurements, u, v, wf_measurements, use_zk),
            **kwargs
        )
        assert len(result.x) == nguess + 3
        dof = self.orthogonal_to_nominal(result.x[:-3])
        Ixx, Ixy, Iyy = result.x[-3:]
        result = dict(
            state=dof,
            Ixx=float(Ixx),
            Ixy=float(Ixy),
            Iyy=float(Iyy),
        )
        return result
