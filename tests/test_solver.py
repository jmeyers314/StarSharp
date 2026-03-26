"""Tests for ZernikeSolver."""

from __future__ import annotations

from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest

from StarSharp.datatypes import (
    DoubleZernikes,
    FieldCoords,
    Sensitivity,
    State,
    Zernikes,
)
from StarSharp.solver import ZernikeSolver

from .utils import _make_field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NDOF = 10
NKEEP = 4
NFIELD = 8
JMAX = 22
KMAX = 10
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m
FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg


def _make_zk_sensitivity(rng, ndof=NDOF, nfield=NFIELD, jmax=JMAX):
    """Build a ``Sensitivity[Zernikes]`` with a random design matrix."""
    grad_coefs = rng.standard_normal((ndof, nfield, jmax + 1)) * u.um
    field = _make_field(nfield)
    gradient = Zernikes(
        coefs=grad_coefs,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
    )
    nom_coefs = rng.standard_normal((nfield, jmax + 1)) * u.um
    nominal = Zernikes(
        coefs=nom_coefs,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
    )
    return Sensitivity(
        gradient=gradient,
        nominal=nominal,
        basis="x",
        use_dof=np.arange(ndof),
        n_dof=ndof,
    )


def _make_zk_sensitivity_v(rng, ndof=NDOF, nkeep=NKEEP, nfield=NFIELD, jmax=JMAX):
    """Build a ``Sensitivity[Zernikes]`` in the SVD (v) basis."""
    sens_x = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
    A = rng.standard_normal((ndof, ndof))
    Q, _ = np.linalg.qr(A)
    Vh = Q[:nkeep]  # (nkeep, ndof) — orthonormal rows
    return replace(sens_x, Vh=Vh).v, Vh


def _make_dz_sensitivity(rng, ndof=NDOF, kmax=KMAX, jmax=JMAX):
    """Build a ``Sensitivity[DoubleZernikes]`` with a random design matrix."""
    grad_coefs = rng.standard_normal((ndof, kmax + 1, jmax + 1)) * u.um
    gradient = DoubleZernikes(
        coefs=grad_coefs,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        pupil_outer=PUPIL_OUTER,
        pupil_inner=PUPIL_INNER,
        frame="ocs",
    )
    nom_coefs = rng.standard_normal((kmax + 1, jmax + 1)) * u.um
    nominal = DoubleZernikes(
        coefs=nom_coefs,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        pupil_outer=PUPIL_OUTER,
        pupil_inner=PUPIL_INNER,
        frame="ocs",
    )
    return Sensitivity(
        gradient=gradient,
        nominal=nominal,
        basis="x",
        use_dof=np.arange(ndof),
        n_dof=ndof,
    )


def _make_wfs_field(nfield=NFIELD):
    """Approximate WFS-like field positions near 1.5° radius."""
    rng = np.random.default_rng(77)
    r = rng.uniform(1.3, 1.75, nfield) * u.deg
    theta = rng.uniform(0, 2 * np.pi, nfield)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return FieldCoords(x=x, y=y, frame="ocs")


# ---------------------------------------------------------------------------
# Tests: ZernikeSolver construction
class TestZernikeSolverConstruction:
    def test_constructs_from_zk_sensitivity(self):
        rng = np.random.default_rng(10)
        sens = _make_zk_sensitivity(rng)
        solver = ZernikeSolver(sens)
        assert solver.sensitivity is sens

    def test_constructs_from_dz_sensitivity(self):
        rng = np.random.default_rng(11)
        sens = _make_dz_sensitivity(rng)
        solver = ZernikeSolver(sens)
        assert solver.sensitivity is sens

    def test_constructs_from_v_basis_sensitivity(self):
        rng = np.random.default_rng(12)
        sens_v, _ = _make_zk_sensitivity_v(rng)
        solver = ZernikeSolver(sens_v)
        assert solver.sensitivity.basis == "v"


# ---------------------------------------------------------------------------
# Tests: ZernikeSolver.solve  —  Sensitivity[Zernikes]
# ---------------------------------------------------------------------------


class TestSolveZernikes:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(20)
        ndof = 8
        nfield = 6
        jmax = 20
        sens = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
        solver = ZernikeSolver(sens)
        return solver, sens, ndof

    def test_returns_state(self, setup):
        solver, sens, ndof = setup
        rng = np.random.default_rng(21)
        obs_coefs = rng.standard_normal(sens.nominal.coefs.shape) * u.um
        obs = Zernikes(
            coefs=obs_coefs,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        assert isinstance(result, State)

    def test_result_basis_matches_sensitivity(self, setup):
        solver, sens, ndof = setup
        obs = Zernikes(
            coefs=np.zeros(sens.nominal.coefs.shape) * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        assert result.basis == sens.basis

    def test_deviation_mode_roundtrip(self, setup):
        """Given obs = grad @ true_state, solve should recover true_state."""
        solver, sens, ndof = setup
        rng = np.random.default_rng(22)
        true_x = rng.standard_normal(ndof)

        grad_val = sens.gradient.coefs.to_value(u.um)  # (ndof, nfield, jmax+1)
        obs_val = np.einsum("dnj,d->nj", grad_val, true_x)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_total_mode_roundtrip(self, setup):
        """Given obs = nominal + grad @ true_state, solve should recover true_state."""
        solver, sens, ndof = setup
        rng = np.random.default_rng(23)
        true_x = rng.standard_normal(ndof)

        grad_val = sens.gradient.coefs.to_value(u.um)
        nom_val = sens.nominal.coefs.to_value(u.um)
        obs_val = nom_val + np.einsum("dnj,d->nj", grad_val, true_x)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="total")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_total_and_deviation_differ_when_nominal_nonzero(self, setup):
        """If the nominal is non-zero, total and deviation modes give different answers."""
        solver, sens, ndof = setup
        rng = np.random.default_rng(24)
        obs_coefs = rng.standard_normal(sens.nominal.coefs.shape) * u.um
        obs = Zernikes(
            coefs=obs_coefs,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        r_dev = solver.solve(obs, mode="deviation")
        r_tot = solver.solve(obs, mode="total")
        assert not np.allclose(r_dev.x.value, r_tot.x.value)

    def test_invalid_mode_raises(self, setup):
        solver, sens, ndof = setup
        obs = Zernikes(
            coefs=sens.nominal.coefs,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        with pytest.raises(ValueError, match="mode must be"):
            solver.solve(obs, mode="wrong")

    def test_zero_deviation_returns_zero_state(self, setup):
        """Solving zero observations should return zero state."""
        solver, sens, ndof = setup
        obs = Zernikes(
            coefs=np.zeros(sens.nominal.coefs.shape) * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: ZernikeSolver.solve  —  Sensitivity[DoubleZernikes]
# ---------------------------------------------------------------------------


class TestSolveDoubleZernikes:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(30)
        ndof = 6
        kmax = 8
        jmax = 18
        field = _make_wfs_field(nfield=8)
        sens = _make_dz_sensitivity(rng, ndof=ndof, kmax=kmax, jmax=jmax)
        solver = ZernikeSolver(sens)
        return solver, sens, field, ndof, jmax

    def test_returns_state(self, setup):
        solver, sens, field, ndof, jmax = setup
        nfield_obs = len(field.x)
        obs = Zernikes(
            coefs=np.zeros((nfield_obs, jmax + 1)) * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        assert isinstance(result, State)

    def test_deviation_mode_roundtrip(self, setup):
        """DZ sensitivity: solve should recover the true state."""
        solver, sens, field, ndof, jmax = setup
        rng = np.random.default_rng(31)
        true_x = rng.standard_normal(ndof)

        # Project DZ gradient to the observed field — same operation solve() does
        projected_gradient = sens.gradient.single(field)
        grad_val = projected_gradient.ocs.coefs.to_value(u.um)  # (ndof, nfield, jmax+1)
        obs_val = np.einsum("dnj,d->nj", grad_val, true_x)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_total_mode_roundtrip(self, setup):
        solver, sens, field, ndof, jmax = setup
        rng = np.random.default_rng(32)
        true_x = rng.standard_normal(ndof)

        projected_gradient = sens.gradient.single(field)
        projected_nominal = sens.nominal.single(field)
        grad_val = projected_gradient.ocs.coefs.to_value(u.um)
        nom_val = projected_nominal.ocs.coefs.to_value(u.um)
        obs_val = nom_val + np.einsum("dnj,d->nj", grad_val, true_x)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="total")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: v-basis sensitivity  (SVD-truncated recovery)
# ---------------------------------------------------------------------------


class TestVBasisSolver:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(40)
        ndof = 8
        nkeep = 4
        nfield = 6
        jmax = 20
        sens_x = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
        A = rng.standard_normal((ndof, ndof))
        Q, _ = np.linalg.qr(A)
        Vh = Q[:nkeep]  # (nkeep, ndof) orthonormal rows
        sens_v = replace(sens_x, Vh=Vh).v
        solver = ZernikeSolver(sens_v)
        return solver, sens_x, sens_v, Vh, ndof, nkeep

    def test_result_basis_is_v(self, setup):
        solver, sens_x, sens_v, Vh, ndof, nkeep = setup
        obs = Zernikes(
            coefs=np.zeros(sens_x.nominal.coefs.shape) * u.um,
            field=sens_x.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        assert result.basis == "v"

    def test_result_shape_is_nkeep(self, setup):
        solver, sens_x, sens_v, Vh, ndof, nkeep = setup
        obs = Zernikes(
            coefs=np.zeros(sens_x.nominal.coefs.shape) * u.um,
            field=sens_x.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        assert result.value.shape == (nkeep,)

    def test_deviation_mode_roundtrip_v(self, setup):
        """A state expressed in the v-basis subspace is exactly recovered."""
        solver, sens_x, sens_v, Vh, ndof, nkeep = setup
        rng = np.random.default_rng(41)
        true_v = rng.standard_normal(nkeep)

        grad_val = sens_v.gradient.coefs.to_value(u.um)  # (nkeep, nfield, jmax+1)
        obs_val = np.einsum("knj,k->nj", grad_val, true_v)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=sens_x.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.value, true_v, atol=1e-8)

    def test_x_conversion_recovers_x_projection(self, setup):
        """result.x should give the x-basis projection of the recovered v state."""
        solver, sens_x, sens_v, Vh, ndof, nkeep = setup
        rng = np.random.default_rng(42)
        true_v = rng.standard_normal(nkeep)
        true_x_proj = true_v @ Vh  # shape (ndof,)

        grad_val = sens_v.gradient.coefs.to_value(u.um)
        obs_val = np.einsum("knj,k->nj", grad_val, true_v)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=sens_x.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x_proj, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: gradient-only Sensitivity (nominal auto-zeroed)
# ---------------------------------------------------------------------------


class TestGradientOnlyInput:
    def test_total_and_deviation_equal_when_nominal_is_zero(self):
        rng = np.random.default_rng(50)
        ndof, nfield, jmax = 5, 4, 15
        grad_coefs = rng.standard_normal((ndof, nfield, jmax + 1)) * u.um
        field = _make_field(nfield)
        gradient = Zernikes(
            coefs=grad_coefs,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        # nominal is auto-zeroed by __post_init__
        sens = Sensitivity(
            gradient=gradient,
            basis="x",
            use_dof=np.arange(ndof),
            n_dof=ndof,
        )
        solver = ZernikeSolver(sens)

        true_x = rng.standard_normal(ndof)
        grad_val = gradient.coefs.to_value(u.um)
        obs_val = np.einsum("dnj,d->nj", grad_val, true_x)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        r_dev = solver.solve(obs, mode="deviation")
        r_tot = solver.solve(obs, mode="total")
        # Both should recover true_x since nominal is zero
        np.testing.assert_allclose(r_dev.x.value, true_x, atol=1e-8)
        np.testing.assert_allclose(r_tot.x.value, true_x, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: jmin — ignoring low-order Zernike modes
# ---------------------------------------------------------------------------


class TestJmin:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(70)
        ndof = 6
        nfield = 5
        jmax = 20
        sens = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
        return sens, ndof, nfield, jmax

    def _obs(self, sens, true_x, jmin=0):
        """Build synthetic obs = grad @ true_x (no nominal), full jmax width."""
        grad_val = sens.gradient.coefs.to_value(u.um)  # (ndof, nfield, jmax+1)
        obs_val = np.einsum("dnj,d->nj", grad_val, true_x)
        # Zero out the modes below jmin just like a real WFS would
        obs_val[..., :jmin] = 0.0
        return Zernikes(
            coefs=obs_val * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )

    def test_jmin_zero_is_default(self, setup):
        """jmin=0 is equivalent to constructing without jmin."""
        sens, ndof, nfield, jmax = setup
        rng = np.random.default_rng(71)
        true_x = rng.standard_normal(ndof)
        obs = self._obs(sens, true_x)
        r_default = ZernikeSolver(sens).solve(obs, mode="deviation")
        r_zero    = ZernikeSolver(sens, jmin=0).solve(obs, mode="deviation")
        np.testing.assert_array_equal(r_default.value, r_zero.value)

    def test_jmin_roundtrip(self, setup):
        """With jmin=4, roundtrip recovers true_x (sensitivity and obs agree)."""
        sens, ndof, nfield, jmax = setup
        rng = np.random.default_rng(72)
        true_x = rng.standard_normal(ndof)
        jmin = 4
        # Build obs from grad columns jmin: only — matches what the solver will use
        grad_val = sens.gradient.coefs.to_value(u.um)[:, :, jmin:]
        obs_val_trimmed = np.einsum("dnj,d->nj", grad_val, true_x)
        # Pad j<jmin with zeros so the Zernikes object has the full jmax+1 width
        nfield_ = sens.nominal.coefs.shape[0]
        obs_full = np.zeros((nfield_, jmax + 1))
        obs_full[:, jmin:] = obs_val_trimmed
        obs = Zernikes(
            coefs=obs_full * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = ZernikeSolver(sens, jmin=jmin).solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_jmin_ignores_low_order_noise(self, setup):
        """Adding large noise to j<jmin should not affect the solution when jmin>0."""
        sens, ndof, nfield, jmax = setup
        rng = np.random.default_rng(73)
        true_x = rng.standard_normal(ndof)
        jmin = 4

        grad_val = sens.gradient.coefs.to_value(u.um)[:, :, jmin:]
        obs_val_trimmed = np.einsum("dnj,d->nj", grad_val, true_x)
        nfield_ = sens.nominal.coefs.shape[0]
        obs_full = np.zeros((nfield_, jmax + 1))
        obs_full[:, jmin:] = obs_val_trimmed
        # Inject large noise into the ignored modes
        obs_full[:, :jmin] = rng.standard_normal((nfield_, jmin)) * 1e6

        obs = Zernikes(
            coefs=obs_full * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = ZernikeSolver(sens, jmin=jmin).solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_jmin_too_large_raises(self, setup):
        """jmin >= nj should raise ValueError."""
        sens, ndof, nfield, jmax = setup
        rng = np.random.default_rng(74)
        obs = self._obs(sens, rng.standard_normal(ndof))
        with pytest.raises(ValueError, match="jmin"):
            ZernikeSolver(sens, jmin=jmax + 1).solve(obs, mode="deviation")

    def test_jmin_stored_on_solver(self, setup):
        sens, *_ = setup
        assert ZernikeSolver(sens, jmin=4).jmin == 4
        assert ZernikeSolver(sens).jmin == 0

    def test_jmin_total_mode(self, setup):
        """jmin also slices the nominal in total mode."""
        sens, ndof, nfield, jmax = setup
        rng = np.random.default_rng(75)
        true_x = rng.standard_normal(ndof)
        jmin = 4

        grad_val = sens.gradient.coefs.to_value(u.um)[:, :, jmin:]
        nom_val  = sens.nominal.coefs.to_value(u.um)[:, jmin:]
        obs_val_trimmed = nom_val + np.einsum("dnj,d->nj", grad_val, true_x)
        nfield_ = sens.nominal.coefs.shape[0]
        obs_full = np.zeros((nfield_, jmax + 1))
        obs_full[:, jmin:] = obs_val_trimmed

        obs = Zernikes(
            coefs=obs_full * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = ZernikeSolver(sens, jmin=jmin).solve(obs, mode="total")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: jmax mismatch between sensitivity and observations
# ---------------------------------------------------------------------------


class TestJmaxMismatch:
    """Tests for allow_narrow_zk / allow_narrow_sens behaviour."""

    @pytest.fixture
    def sens_and_field(self):
        """Sensitivity with jmax=14 (15 columns) over 5 field points."""
        rng = np.random.default_rng(60)
        ndof = 4
        nfield = 5
        jmax_sens = 14
        sens = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax_sens)
        solver = ZernikeSolver(sens)
        field = sens.nominal.field
        return solver, sens, field, ndof, jmax_sens

    def _obs(self, field, nfield, jmax):
        """Build a zero Zernikes observation at *field* with *jmax*."""
        return Zernikes(
            coefs=np.zeros((nfield, jmax + 1)) * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )

    # --- narrow_zk: obs has fewer Zernikes than sensitivity ---

    def test_narrow_zk_allowed_by_default(self, sens_and_field):
        """obs jmax < sens jmax → allowed by default (allow_narrow_zk=True)."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        nfield = len(field.x)
        obs = self._obs(field, nfield, jmax_sens - 4)  # fewer than sensitivity
        result = solver.solve(obs, mode="deviation")
        assert isinstance(result, State)

    def test_narrow_zk_truncates_sensitivity_not_result(self, sens_and_field):
        """When obs has fewer Zernikes, the sensitivity is truncated to match;
        the returned state has the same DOF count regardless."""
        solver, sens, sens_field, ndof, jmax_sens = sens_and_field
        nfield = len(sens_field.x)
        obs_full = self._obs(sens_field, nfield, jmax_sens)
        obs_narrow = self._obs(sens_field, nfield, jmax_sens - 4)
        result_full = solver.solve(obs_full, mode="deviation")
        result_narrow = solver.solve(obs_narrow, mode="deviation")
        # Both return a state with the right number of DOFs
        assert result_full.value.shape == result_narrow.value.shape

    def test_narrow_zk_raises_when_disallowed(self, sens_and_field):
        """obs jmax < sens jmax with allow_narrow_zk=False → ValueError."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        nfield = len(field.x)
        obs = self._obs(field, nfield, jmax_sens - 4)
        strict_solver = ZernikeSolver(sens, allow_narrow_zk=False)
        with pytest.raises(ValueError, match="allow_narrow_zk"):
            strict_solver.solve(obs, mode="deviation")

    def test_narrow_zk_roundtrip(self, sens_and_field):
        """Roundtrip still works when obs covers fewer Zernikes than sensitivity."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        jmax_obs = jmax_sens - 4
        rng = np.random.default_rng(61)
        true_x = rng.standard_normal(ndof)
        # Build obs from the truncated gradient
        grad_val = sens.gradient.coefs.to_value(u.um)[:, :, : jmax_obs + 1]
        obs_val = np.einsum("dnj,d->nj", grad_val, true_x)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    # --- narrow_sens: obs has more Zernikes than sensitivity ---

    def test_narrow_sens_raises_by_default(self, sens_and_field):
        """obs jmax > sens jmax → raises by default (allow_narrow_sens=False)."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        nfield = len(field.x)
        obs = self._obs(field, nfield, jmax_sens + 4)  # more than sensitivity
        with pytest.raises(ValueError, match="allow_narrow_sens"):
            solver.solve(obs, mode="deviation")

    def test_narrow_sens_allowed_explicitly(self, sens_and_field):
        """obs jmax > sens jmax with allow_narrow_sens=True → truncates obs."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        nfield = len(field.x)
        obs = self._obs(field, nfield, jmax_sens + 4)
        lax_solver = ZernikeSolver(sens, allow_narrow_sens=True)
        result = lax_solver.solve(obs, mode="deviation")
        assert isinstance(result, State)

    def test_narrow_sens_roundtrip(self, sens_and_field):
        """Roundtrip with allow_narrow_sens=True (extra observed modes ignored)."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        rng = np.random.default_rng(62)
        true_x = rng.standard_normal(ndof)
        # Build obs with extra zero-padded columns beyond jmax_sens
        grad_val = sens.gradient.coefs.to_value(u.um)  # (ndof, nfield, jmax_sens+1)
        obs_val_base = np.einsum("dnj,d->nj", grad_val, true_x)
        # Pad with zeros to simulate extra higher-order terms
        nfield = obs_val_base.shape[0]
        extra = np.zeros((nfield, 4))
        obs_val = np.concatenate([obs_val_base, extra], axis=-1)
        obs = Zernikes(
            coefs=obs_val * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        lax_solver = ZernikeSolver(sens, allow_narrow_sens=True)
        result = lax_solver.solve(obs, mode="deviation")
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    # --- exact match: no flags needed ---

    def test_exact_jmax_match_works_with_both_flags_false(self, sens_and_field):
        """When jmax matches exactly, both flags can be False without error."""
        solver, sens, field, ndof, jmax_sens = sens_and_field
        nfield = len(field.x)
        obs = self._obs(field, nfield, jmax_sens)
        strict_solver = ZernikeSolver(sens, allow_narrow_zk=False, allow_narrow_sens=False)
        result = strict_solver.solve(obs, mode="deviation")
        assert isinstance(result, State)

