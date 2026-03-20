"""Tests for RaytracedOpticalModel methods (no ray-tracer required)."""

from __future__ import annotations

from unittest.mock import MagicMock

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
from StarSharp.models.raytraced import RaytracedOpticalModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m

NDOF = 5
NFIELD = 9   # >= KMAX+1 = 7 for a well-determined DZ fit
JMAX = 10
KMAX = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_field() -> FieldCoords:
    """Regular grid of field-angle positions in OCS degrees."""
    x = np.linspace(-1.0, 1.0, 3) * u.deg
    y = np.linspace(-1.0, 1.0, 3) * u.deg
    xx, yy = np.meshgrid(x, y)
    return FieldCoords(x=xx.ravel(), y=yy.ravel(), frame="ocs")


def _make_steps(ndof: int = NDOF) -> State:
    return State(
        value=np.ones(ndof),
        basis="f",
        use_dof=np.arange(ndof),
        n_dof=ndof,
    )


def _make_zk_sensitivity(rng=None) -> Sensitivity:
    """Synthetic Sensitivity[Zernikes] with well-determined field sampling."""
    if rng is None:
        rng = np.random.default_rng(42)

    field = _make_field()
    nominal_coefs = rng.standard_normal((NFIELD, JMAX + 1)) * u.um
    gradient_coefs = rng.standard_normal((NDOF, NFIELD, JMAX + 1)) * u.um

    nominal = Zernikes(
        coefs=nominal_coefs,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
    )
    gradient = Zernikes(
        coefs=gradient_coefs,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
    )

    steps = _make_steps()
    return Sensitivity(
        gradient=gradient,
        nominal=nominal,
        basis=steps.basis,
        use_dof=steps.use_dof,
        n_dof=steps.n_dof,
        Vh=steps.Vh,
    )


def _call_method(mock_model, field, steps, kmax=KMAX, jmax=JMAX, rings=10,
                 offset=None):
    """Call double_zernikes_sensitivity as an unbound method on mock_model."""
    return RaytracedOpticalModel.double_zernikes_sensitivity(
        mock_model,
        field=field,
        steps=steps,
        kmax=kmax,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        jmax=jmax,
        rings=rings,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDoubleZernikesSensitivity:
    """Unit tests for RaytracedOpticalModel.double_zernikes_sensitivity.

    The underlying ray-tracer is replaced by a mock whose
    ``zernikes_sensitivity`` returns a pre-built Sensitivity[Zernikes].
    """

    @pytest.fixture
    def setup(self):
        zk_sens = _make_zk_sensitivity()
        mock_model = MagicMock()
        mock_model.zernikes_sensitivity.return_value = zk_sens
        field = _make_field()
        steps = _make_steps()
        return mock_model, zk_sens, field, steps

    # --- return types ---

    def test_returns_sensitivity(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert isinstance(result, Sensitivity)

    def test_nominal_is_double_zernikes(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert isinstance(result.nominal, DoubleZernikes)

    def test_gradient_is_double_zernikes(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert isinstance(result.gradient, DoubleZernikes)

    # --- shapes ---

    def test_gradient_coefs_shape(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.gradient.coefs.shape == (NDOF, KMAX + 1, JMAX + 1)

    def test_nominal_coefs_shape(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.nominal.coefs.shape == (KMAX + 1, JMAX + 1)

    # --- metadata passthrough from steps ---

    def test_basis_passthrough(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.basis == steps.basis

    def test_use_dof_passthrough(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        np.testing.assert_array_equal(result.use_dof, steps.use_dof)

    def test_n_dof_passthrough(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.n_dof == steps.n_dof

    def test_vh_passthrough(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.Vh is steps.Vh  # both None

    # --- physical parameters ---

    def test_field_outer_on_nominal(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.nominal.field_outer == FIELD_OUTER

    def test_field_inner_on_nominal(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.nominal.field_inner == FIELD_INNER

    def test_pupil_params_on_nominal(self, setup):
        mock_model, _, field, steps = setup
        result = _call_method(mock_model, field, steps)
        assert result.nominal.pupil_outer == PUPIL_OUTER
        assert result.nominal.pupil_inner == PUPIL_INNER

    # --- projection correctness ---

    def test_nominal_matches_direct_double(self, setup):
        """result.nominal should equal zk_sens.nominal.double() directly."""
        mock_model, zk_sens, field, steps = setup
        result = _call_method(mock_model, field, steps)
        expected = zk_sens.nominal.double(KMAX, FIELD_OUTER, FIELD_INNER)
        np.testing.assert_allclose(
            result.nominal.coefs.to_value(u.um),
            expected.coefs.to_value(u.um),
            atol=1e-12,
        )

    def test_gradient_matches_direct_double(self, setup):
        """result.gradient should equal zk_sens.gradient.double() directly."""
        mock_model, zk_sens, field, steps = setup
        result = _call_method(mock_model, field, steps)
        expected = zk_sens.gradient.double(KMAX, FIELD_OUTER, FIELD_INNER)
        np.testing.assert_allclose(
            result.gradient.coefs.to_value(u.um),
            expected.coefs.to_value(u.um),
            atol=1e-12,
        )

    # --- mock call verification ---

    def test_zernikes_sensitivity_called_once(self, setup):
        mock_model, _, field, steps = setup
        _call_method(mock_model, field, steps)
        mock_model.zernikes_sensitivity.assert_called_once()

    def test_zernikes_sensitivity_called_with_field_and_steps(self, setup):
        mock_model, _, field, steps = setup
        _call_method(mock_model, field, steps)
        call_kwargs = mock_model.zernikes_sensitivity.call_args.kwargs
        assert call_kwargs["field"] is field
        assert call_kwargs["steps"] is steps

    def test_zernikes_sensitivity_receives_jmax(self, setup):
        mock_model, _, field, steps = setup
        _call_method(mock_model, field, steps, jmax=JMAX)
        call_kwargs = mock_model.zernikes_sensitivity.call_args.kwargs
        assert call_kwargs["jmax"] == JMAX

    def test_zernikes_sensitivity_receives_rings(self, setup):
        mock_model, _, field, steps = setup
        _call_method(mock_model, field, steps, rings=5)
        call_kwargs = mock_model.zernikes_sensitivity.call_args.kwargs
        assert call_kwargs["rings"] == 5

    def test_zernikes_sensitivity_receives_offset(self, setup):
        mock_model, _, field, steps = setup
        offset = State(value=np.zeros(NDOF), basis="f")
        _call_method(mock_model, field, steps, offset=offset)
        call_kwargs = mock_model.zernikes_sensitivity.call_args.kwargs
        assert call_kwargs["offset"] is offset

    def test_offset_defaults_to_none(self, setup):
        mock_model, _, field, steps = setup
        _call_method(mock_model, field, steps)
        call_kwargs = mock_model.zernikes_sensitivity.call_args.kwargs
        assert call_kwargs["offset"] is None
