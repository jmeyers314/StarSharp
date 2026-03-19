"""Tests for the LinearOpticalModel class."""

from __future__ import annotations

from unittest.mock import MagicMock

import astropy.units as u
import galsim
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import (
    DoubleZernikes,
    FieldCoords,
    Sensitivity,
    State,
    Zernikes,
)
from StarSharp.models.linear import LinearOpticalModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RTP = Angle(0.0, unit=u.deg)  # zero rotation simplifies checking
JMAX = 6
KMAX = 4
NDOF = 3
FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m

# ---------------------------------------------------------------------------
# Helpers: mock camera
# ---------------------------------------------------------------------------


def _make_mock_camera(xlo=-50.0, xhi=50.0, ylo=-50.0, yhi=50.0, det_id=1):
    """Return a mock Camera with one rectangular detector in CCS mm.

    getCorners(FOCAL_PLANE) returns DVCS corners, which are the
    *transpose* of CCS: corner[0] → CCS y, corner[1] → CCS x.
    """
    det = MagicMock()
    det.getId.return_value = det_id
    # Corners in DVCS: (dvcs_x, dvcs_y) = (ccs_y, ccs_x)
    det.getCorners.return_value = [
        (ylo, xlo),
        (yhi, xlo),
        (yhi, xhi),
        (ylo, xhi),
    ]
    camera = MagicMock()
    camera.__iter__ = MagicMock(return_value=iter([det]))
    return camera


def _make_wcs(rtp: Angle | None = None) -> galsim.BaseWCS:
    """Simple affine WCS: field angle (rad) ↔ focal plane (mm)."""
    if rtp is None:
        rtp = RTP
    c = np.cos(rtp.rad)
    s = np.sin(rtp.rad)
    # 20 arcsec / mm
    scale = np.deg2rad(20 / 3600)
    rot = np.array([[c, -s], [s, c]]) * scale
    return galsim.AffineTransform(
        rot[0, 0],
        rot[0, 1],
        rot[1, 0],
        rot[1, 1],
        galsim.PositionD(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Helpers: synthetic data
# ---------------------------------------------------------------------------


def _make_grid_field(nx=5, ny=5, camera=None, rtp=None, wcs=None):
    """Regular grid of focal-plane CCS positions in [-40, 40] mm."""
    xs = np.linspace(-40, 40, nx)
    ys = np.linspace(-40, 40, ny)
    xx, yy = np.meshgrid(xs, ys)
    return FieldCoords(
        x=xx.ravel() * u.mm,
        y=yy.ravel() * u.mm,
        frame="ccs",
        rtp=rtp or RTP,
        wcs=wcs,
        camera=camera,
    )


def _make_nominal_zernikes(field, jmax=JMAX):
    """Create a nominal Zernikes on *field* with linearly varying coefs.

    coefs[i, j] = 0.1 * x_mm[i] + 0.05 * y_mm[i] + j
    so that interpolation can be verified exactly.
    """
    x_mm = field.x.to_value(u.mm)
    y_mm = field.y.to_value(u.mm)
    nfield = len(x_mm)
    coefs = np.zeros((nfield, jmax + 1))
    for j in range(jmax + 1):
        coefs[:, j] = 0.1 * x_mm + 0.05 * y_mm + j
    return Zernikes(
        coefs=coefs * u.um,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        wavelength=622.0 * u.nm,
        frame="ccs",
        rtp=field.rtp,
    )


def _make_dz_sensitivity(ndof=NDOF, jmax=JMAX, kmax=KMAX, rtp=None):
    """Build a Sensitivity[DoubleZernikes] with known gradient."""
    rtp = rtp or RTP
    rng = np.random.default_rng(77)

    nominal_coefs = rng.standard_normal((kmax + 1, jmax + 1)) * u.um
    nominal = DoubleZernikes(
        coefs=nominal_coefs,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        pupil_outer=PUPIL_OUTER,
        pupil_inner=PUPIL_INNER,
        frame="ocs",
        rtp=rtp,
    )

    steps = State(
        value=np.ones(ndof),
        basis="f",
        use_dof=np.arange(ndof),
        n_dof=ndof,
    )

    perturbed = []
    for i in range(ndof):
        delta = rng.standard_normal((kmax + 1, jmax + 1))
        perturbed.append(
            DoubleZernikes(
                coefs=(nominal_coefs.value + delta) * u.um,
                field_outer=FIELD_OUTER,
                field_inner=FIELD_INNER,
                pupil_outer=PUPIL_OUTER,
                pupil_inner=PUPIL_INNER,
                frame="ocs",
                rtp=rtp,
            )
        )

    return Sensitivity.from_finite_differences(nominal, perturbed, steps)


# ---------------------------------------------------------------------------
# Tests: interpolator construction + interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    """Verify per-detector RegularGridInterpolator round-trips."""

    def test_interpolate_on_grid(self):
        """Querying at the exact grid points reproduces the nominal coefs."""
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field = _make_grid_field(camera=camera, wcs=wcs)
        nominal = _make_nominal_zernikes(field)
        sens = _make_dz_sensitivity()

        model = LinearOpticalModel(nominal, sens, RTP, camera)

        # Query at the same grid points — reinitialize camera mock
        # because __iter__ is exhausted after __init__.
        camera2 = _make_mock_camera()
        field2 = _make_grid_field(camera=camera2, wcs=wcs)
        result = model._interpolate_zernikes(field2)

        np.testing.assert_allclose(result, nominal.coefs.to_value(u.um), atol=1e-12)

    def test_interpolate_between_grid(self):
        """Querying between grid points gives linearly interpolated values."""
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field = _make_grid_field(nx=5, ny=5, camera=camera, wcs=wcs)
        nominal = _make_nominal_zernikes(field)
        sens = _make_dz_sensitivity()

        model = LinearOpticalModel(nominal, sens, RTP, camera)

        # Query at the midpoint of four grid cells.
        camera2 = _make_mock_camera()
        mid_x = np.array([0.0, 10.0]) * u.mm
        mid_y = np.array([0.0, 10.0]) * u.mm
        query = FieldCoords(
            x=mid_x,
            y=mid_y,
            frame="ccs",
            rtp=RTP,
            wcs=wcs,
            camera=camera2,
        )
        result = model._interpolate_zernikes(query)

        # Since the coefs are exactly linear in (x, y), any interpolation
        # should reproduce the function exactly.
        expected = np.zeros((2, JMAX + 1))
        for j in range(JMAX + 1):
            expected[:, j] = (
                0.1 * mid_x.to_value(u.mm) + 0.05 * mid_y.to_value(u.mm) + j
            )
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: zernikes (full predict)
# ---------------------------------------------------------------------------


class TestZernikes:
    """Test zernikes() with zero-rtp and known sensitivity."""

    def test_zero_state_returns_nominal(self):
        """With zero perturbation, zernikes() returns the rotated nominal."""
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field = _make_grid_field(camera=camera, wcs=wcs)
        nominal = _make_nominal_zernikes(field)
        sens = _make_dz_sensitivity()

        model = LinearOpticalModel(nominal, sens, RTP, camera)

        state = State(
            value=np.zeros(NDOF),
            basis="f",
            use_dof=np.arange(NDOF),
            n_dof=NDOF,
        )

        camera2 = _make_mock_camera()
        field2 = _make_grid_field(camera=camera2, wcs=wcs)
        zk = model.zernikes(state, field2)

        # With rtp=0 the rot matrix is identity, so OCS == CCS
        np.testing.assert_allclose(
            zk.coefs.to_value(u.um),
            nominal.coefs.to_value(u.um),
            atol=1e-10,
        )
        assert zk.frame == "ocs"

    def test_nonzero_state_adds_delta(self):
        """A non-zero state adds the sensitivity prediction to the nominal."""
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field = _make_grid_field(nx=3, ny=3, camera=camera, wcs=wcs)
        nominal = _make_nominal_zernikes(field)
        sens = _make_dz_sensitivity()

        model = LinearOpticalModel(nominal, sens, RTP, camera)

        weights = np.array([1.0, -0.5, 0.3])
        state = State(
            value=weights,
            basis="f",
            use_dof=np.arange(NDOF),
            n_dof=NDOF,
        )

        camera2 = _make_mock_camera()
        field2 = _make_grid_field(nx=3, ny=3, camera=camera2, wcs=wcs)
        zk = model.zernikes(state, field2)

        # Manually compute the expected delta from the sensitivity
        field_ocs = field2.angle.ocs
        grad_zk = sens.gradient.single(field_ocs)
        delta = np.einsum("i...,i->...", grad_zk.coefs.to_value(u.um), weights)

        # With rtp=0, nominal_ocs == nominal_ccs
        expected = nominal.coefs.to_value(u.um) + delta
        np.testing.assert_allclose(zk.coefs.to_value(u.um), expected, atol=1e-10)

    def test_prepare_field_attaches_metadata(self):
        """_prepare_field fills in rtp, camera, and wcs from self."""
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field_grid = _make_grid_field(camera=camera, wcs=wcs)
        nominal = _make_nominal_zernikes(field_grid)
        sens = _make_dz_sensitivity()

        model = LinearOpticalModel(nominal, sens, RTP, camera)

        bare = FieldCoords(
            x=np.array([0.0]) * u.mm, y=np.array([0.0]) * u.mm, frame="ccs"
        )
        prepared = model._prepare_field(bare)
        assert prepared.rtp is not None
        assert prepared.camera is not None
        assert prepared.wcs is not None


# ---------------------------------------------------------------------------
# Tests: optimize
# ---------------------------------------------------------------------------


class TestOptimize:
    """Test the linear least-squares solver."""

    def test_recovers_injected_state(self):
        """optimize recovers a state that was used to perturb a zero nominal.

        With a zero nominal the perturbed wavefront is purely ``G @ w``,
        and ``optimize`` should find ``x = -w`` to drive it back to zero.
        """
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field = _make_grid_field(nx=5, ny=5, camera=camera, wcs=wcs)

        # Use a zero nominal so the only signal is the injected perturbation
        zero_nominal = Zernikes(
            coefs=np.zeros((25, JMAX + 1)) * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            wavelength=622.0 * u.nm,
            frame="ccs",
            rtp=RTP,
        )
        sens = _make_dz_sensitivity()
        model = LinearOpticalModel(zero_nominal, sens, RTP, camera)

        # Inject a known perturbation
        true_weights = np.array([0.5, -1.0, 0.8])
        true_state = State(
            value=true_weights,
            basis="f",
            use_dof=np.arange(NDOF),
            n_dof=NDOF,
        )

        # Compute perturbed Zernikes (= 0 + G @ true_weights)
        camera2 = _make_mock_camera()
        field2 = _make_grid_field(nx=5, ny=5, camera=camera2, wcs=wcs)
        zk_perturbed = model.zernikes(true_state, field2)

        # Build a new model whose "nominal" is the perturbed wavefront
        camera3 = _make_mock_camera()
        nominal_perturbed = Zernikes(
            coefs=zk_perturbed.ccs.coefs,
            field=_make_grid_field(nx=5, ny=5, camera=camera3, wcs=wcs),
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            wavelength=622.0 * u.nm,
            frame="ccs",
            rtp=RTP,
        )

        camera4 = _make_mock_camera()
        model2 = LinearOpticalModel(nominal_perturbed, sens, RTP, camera4)

        guess = State(
            value=np.zeros(NDOF),
            basis="f",
            use_dof=np.arange(NDOF),
            n_dof=NDOF,
        )

        # optimize minimises ||nominal + G @ x||²  →  x ≈ -true_weights
        camera5 = _make_mock_camera()
        field5 = _make_grid_field(nx=5, ny=5, camera=camera5, wcs=wcs)
        recovered = model2.optimize(guess, field5, jmin=0)

        np.testing.assert_allclose(recovered.value, -true_weights, atol=1e-6)

    def test_optimize_with_offset(self):
        """optimize with offset adds the offset to the result."""
        camera = _make_mock_camera()
        wcs = _make_wcs()
        field = _make_grid_field(nx=4, ny=4, camera=camera, wcs=wcs)
        nominal = _make_nominal_zernikes(field)
        sens = _make_dz_sensitivity()
        model = LinearOpticalModel(nominal, sens, RTP, camera)

        guess = State(
            value=np.zeros(NDOF),
            basis="f",
            use_dof=np.arange(NDOF),
            n_dof=NDOF,
        )

        offset = State(
            value=np.array([10.0, 20.0, 30.0]),
            basis="f",
            use_dof=np.arange(NDOF),
            n_dof=NDOF,
        )

        camera2 = _make_mock_camera()
        field2 = _make_grid_field(nx=4, ny=4, camera=camera2, wcs=wcs)
        result_no_off = model.optimize(guess, field2, jmin=0)

        camera3 = _make_mock_camera()
        field3 = _make_grid_field(nx=4, ny=4, camera=camera3, wcs=wcs)
        result_off = model.optimize(guess, field3, offset=offset, jmin=0)

        np.testing.assert_allclose(
            result_off.f.value,
            result_no_off.f.value + offset.f.value,
            atol=1e-12,
        )
