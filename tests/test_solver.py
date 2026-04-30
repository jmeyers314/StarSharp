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
    StateSchema,
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
JMAX = 30
KMAX = 10
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m
FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg


def _make_schema(ndof: int) -> StateSchema:
    """Minimal StateSchema with dummy DOF names/units for solver tests."""
    return StateSchema(
        dof_names=tuple(f"dof{i}" for i in range(ndof)),
        dof_units=(u.mm,) * ndof,
    )


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
        schema=_make_schema(ndof),
    )


def _make_zk_sensitivity_v(rng, ndof=NDOF, nkeep=NKEEP, nfield=NFIELD, jmax=JMAX):
    """Build a ``Sensitivity[Zernikes]`` in the SVD (v) basis."""
    sens_x = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
    A = rng.standard_normal((ndof, ndof))
    Q, _ = np.linalg.qr(A)
    Vh = Q[:nkeep]  # (nkeep, ndof) — orthonormal rows
    return replace(sens_x, schema=replace(sens_x.schema, Vh=Vh)).v, Vh


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
        schema=_make_schema(ndof),
    )


def _make_wfs_field(nfield=NFIELD):
    """Approximate WFS-like field positions near 1.5° radius."""
    rng = np.random.default_rng(77)
    r = rng.uniform(1.3, 1.75, nfield) * u.deg
    theta = rng.uniform(0, 2 * np.pi, nfield)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return FieldCoords(x=x, y=y, frame="ocs")


def _deviation_obs(sens_or_grad_val, use_zk, true_x, field, jmax):
    """Build a Zernikes observation as grad[use_zk] @ true_x."""
    grad_val = sens_or_grad_val[..., use_zk]  # (ndof, nfield, nzk)
    obs_val = np.zeros((len(field.x), jmax + 1))
    obs_val[..., use_zk] = np.einsum("dnj,d->nj", grad_val, true_x)
    return Zernikes(
        coefs=obs_val * u.um,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
    )


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
        jmax = 30
        sens = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
        solver = ZernikeSolver(sens)
        return solver, sens, ndof

    def test_returns_state(self, setup):
        solver, sens, ndof = setup
        obs = Zernikes(
            coefs=np.zeros(sens.nominal.coefs.shape) * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs)
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
        result = solver.solve(obs)
        assert result.basis == sens.basis

    def test_roundtrip(self, setup):
        """Given obs = grad @ true_state, solve recovers true_state."""
        solver, sens, ndof = setup
        rng = np.random.default_rng(22)
        true_x = rng.standard_normal(ndof)
        obs = _deviation_obs(
            sens.gradient.coefs.to_value(u.um), solver.use_zk, true_x,
            sens.nominal.field, sens.nominal.jmax,
        )
        result = solver.solve(obs)
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_zero_obs_returns_zero_state(self, setup):
        solver, sens, ndof = setup
        obs = Zernikes(
            coefs=np.zeros(sens.nominal.coefs.shape) * u.um,
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs)
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
        jmax = 30
        field = _make_wfs_field(nfield=8)
        sens = _make_dz_sensitivity(rng, ndof=ndof, kmax=kmax, jmax=jmax)
        solver = ZernikeSolver(sens)
        return solver, sens, field, ndof, jmax

    def test_returns_state(self, setup):
        solver, sens, field, ndof, jmax = setup
        obs = Zernikes(
            coefs=np.zeros((len(field.x), jmax + 1)) * u.um,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        result = solver.solve(obs)
        assert isinstance(result, State)

    def test_roundtrip(self, setup):
        """DZ sensitivity: solve recovers the true state."""
        solver, sens, field, ndof, jmax = setup
        rng = np.random.default_rng(31)
        true_x = rng.standard_normal(ndof)

        projected_gradient = sens.gradient.single(field)
        obs = _deviation_obs(
            projected_gradient.ocs.coefs.to_value(u.um), solver.use_zk, true_x,
            field, jmax,
        )
        result = solver.solve(obs)
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
        jmax = 30
        sens_x = _make_zk_sensitivity(rng, ndof=ndof, nfield=nfield, jmax=jmax)
        A = rng.standard_normal((ndof, ndof))
        Q, _ = np.linalg.qr(A)
        Vh = Q[:nkeep]  # (nkeep, ndof) orthonormal rows
        sens_v = replace(sens_x, schema=replace(sens_x.schema, Vh=Vh)).v
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
        result = solver.solve(obs)
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
        result = solver.solve(obs)
        assert result.value.shape == (nkeep,)

    def test_roundtrip_v(self, setup):
        """A state in the v-basis subspace is exactly recovered."""
        solver, sens_x, sens_v, Vh, ndof, nkeep = setup
        rng = np.random.default_rng(41)
        true_v = rng.standard_normal(nkeep)
        obs = _deviation_obs(
            sens_v.gradient.coefs.to_value(u.um), solver.use_zk, true_v,
            sens_x.nominal.field, sens_x.nominal.jmax,
        )
        result = solver.solve(obs)
        np.testing.assert_allclose(result.value, true_v, atol=1e-8)

    def test_x_conversion_recovers_x_projection(self, setup):
        """result.x gives the x-basis projection of the recovered v state."""
        solver, sens_x, sens_v, Vh, ndof, nkeep = setup
        rng = np.random.default_rng(42)
        true_v = rng.standard_normal(nkeep)
        true_x_proj = true_v @ Vh
        obs = _deviation_obs(
            sens_v.gradient.coefs.to_value(u.um), solver.use_zk, true_v,
            sens_x.nominal.field, sens_x.nominal.jmax,
        )
        result = solver.solve(obs)
        np.testing.assert_allclose(result.x.value, true_x_proj, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: gradient-only Sensitivity (nominal auto-zeroed)
# ---------------------------------------------------------------------------


class TestGradientOnlyInput:
    def test_roundtrip_with_zero_nominal(self):
        rng = np.random.default_rng(50)
        ndof, nfield, jmax = 5, 4, 30
        grad_coefs = rng.standard_normal((ndof, nfield, jmax + 1)) * u.um
        field = _make_field(nfield)
        gradient = Zernikes(
            coefs=grad_coefs,
            field=field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        sens = Sensitivity(
            gradient=gradient,
            basis="x",
            schema=_make_schema(ndof),
        )
        solver = ZernikeSolver(sens)
        true_x = rng.standard_normal(ndof)
        obs = _deviation_obs(grad_coefs.to_value(u.um), solver.use_zk, true_x, field, jmax)
        result = solver.solve(obs)
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: use_zk — Noll index selection
# ---------------------------------------------------------------------------


class TestUseZk:
    @pytest.fixture
    def sens(self):
        rng = np.random.default_rng(60)
        return _make_zk_sensitivity(rng, ndof=6, nfield=5, jmax=JMAX)

    def test_default_use_zk(self, sens):
        solver = ZernikeSolver(sens)
        np.testing.assert_array_equal(solver.use_zk, np.arange(4, 29))

    def test_use_zk_as_list(self, sens):
        solver = ZernikeSolver(sens, use_zk=[4, 5, 6, 7, 8])
        np.testing.assert_array_equal(solver.use_zk, [4, 5, 6, 7, 8])

    def test_use_zk_as_str(self, sens):
        solver_list = ZernikeSolver(sens, use_zk=[4, 5, 6, 7, 8])
        solver_str  = ZernikeSolver(sens, use_zk="4-8")
        np.testing.assert_array_equal(solver_list.use_zk, solver_str.use_zk)

    def test_missing_index_in_obs_raises(self, sens):
        solver = ZernikeSolver(sens)  # use_zk includes j=28
        obs = Zernikes(
            coefs=np.zeros((5, 20)) * u.um,  # only 20 terms (0..19)
            field=sens.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        with pytest.raises(ValueError, match="observations"):
            solver.solve(obs)

    def test_missing_index_in_sens_raises(self):
        rng = np.random.default_rng(61)
        sens_narrow = _make_zk_sensitivity(rng, ndof=4, nfield=5, jmax=18)
        solver = ZernikeSolver(sens_narrow)  # use_zk="4-28", sens only has 0..18
        obs = Zernikes(
            coefs=np.zeros((5, JMAX + 1)) * u.um,
            field=sens_narrow.nominal.field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        with pytest.raises(ValueError, match="sensitivity"):
            solver.solve(obs)

    def test_roundtrip_with_custom_use_zk(self, sens):
        use_zk = [4, 7, 11, 22]
        solver = ZernikeSolver(sens, use_zk=use_zk)
        ndof = sens.gradient.coefs.shape[0]
        rng = np.random.default_rng(62)
        true_x = rng.standard_normal(ndof)
        obs = _deviation_obs(
            sens.gradient.coefs.to_value(u.um), solver.use_zk, true_x,
            sens.nominal.field, sens.nominal.jmax,
        )
        result = solver.solve(obs)
        np.testing.assert_allclose(result.x.value, true_x, atol=1e-8)

    def test_field_mismatch_raises(self, sens):
        solver = ZernikeSolver(sens)
        wrong_field = _make_field(sens.gradient.coefs.shape[1] + 1)  # different nfield
        obs = Zernikes(
            coefs=np.zeros((wrong_field.nfield, JMAX + 1)) * u.um,
            field=wrong_field,
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            frame="ocs",
        )
        with pytest.raises(ValueError, match="field"):
            solver.solve(obs)
