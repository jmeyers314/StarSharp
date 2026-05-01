"""Tests for LinearModel."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from StarSharp.datatypes import (
    FieldCoords,
    Sensitivity,
    StateSchema,
    Zernikes,
)
from StarSharp.datatypes.state import StateFactory
from StarSharp.models.linear import LinearModel

from .utils import _make_field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NDOF = 10
NFIELD = 5
JMAX = 30
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m


def _make_schema(ndof: int, use_dof=None) -> StateSchema:
    return StateSchema(
        dof_names=tuple(f"dof{i}" for i in range(ndof)),
        dof_units=(u.mm,) * ndof,
        step=np.ones(ndof) * 0.1,
        use_dof=use_dof,
    )


def _make_zk_sensitivity(rng, ndof=NDOF, nfield=NFIELD, jmax=JMAX, use_dof=None):
    schema   = _make_schema(ndof, use_dof=use_dof)
    n_active = schema.n_active
    field    = _make_field(nfield)
    gradient = Zernikes(
        coefs=rng.standard_normal((n_active, nfield, jmax + 1)) * u.um,
        field=field, R_outer=PUPIL_OUTER, R_inner=PUPIL_INNER, frame="ocs",
    )
    nominal = Zernikes(
        coefs=rng.standard_normal((nfield, jmax + 1)) * u.um,
        field=field, R_outer=PUPIL_OUTER, R_inner=PUPIL_INNER, frame="ocs",
    )
    return Sensitivity(gradient=gradient, nominal=nominal, basis="x", schema=schema)


class _StubRaytraced:
    """Lightweight stand-in for RaytracedOpticalModel.

    Returns a pre-built sensitivity from both sensitivity methods so tests
    can exercise LinearModel without any ray tracing.  Tracks calls so tests
    can assert on argument values and call counts.
    """

    def __init__(self, sensitivity: Sensitivity):
        self._sensitivity = sensitivity
        self.spots_calls: list[dict] = []
        self.zernikes_calls: list[dict] = []

    def spots_sensitivity(self, field, basis="x", use_dof=None, **kwargs):
        self.spots_calls.append(dict(field=field, basis=basis, use_dof=use_dof, **kwargs))
        return self._sensitivity

    def zernikes_sensitivity(self, field, basis="x", use_dof=None, **kwargs):
        self.zernikes_calls.append(dict(field=field, basis=basis, use_dof=use_dof, **kwargs))
        return self._sensitivity


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLinearModelConstruction:
    def test_stores_raytraced_and_field(self):
        rng  = np.random.default_rng(0)
        sens = _make_zk_sensitivity(rng)
        rt   = _StubRaytraced(sens)
        field = _make_field(NFIELD)
        model = LinearModel(rt, field)
        assert model.raytraced is rt
        assert model.field is field

    def test_use_dof_none_stored_as_none(self):
        rng  = np.random.default_rng(1)
        rt   = _StubRaytraced(_make_zk_sensitivity(rng))
        model = LinearModel(rt, _make_field(NFIELD))
        assert model.use_dof is None

    def test_use_dof_list(self):
        rng  = np.random.default_rng(2)
        rt   = _StubRaytraced(_make_zk_sensitivity(rng))
        model = LinearModel(rt, _make_field(NFIELD), use_dof=[0, 1, 2])
        np.testing.assert_array_equal(model.use_dof, [0, 1, 2])

    def test_use_dof_str(self):
        rng  = np.random.default_rng(3)
        rt   = _StubRaytraced(_make_zk_sensitivity(rng))
        model = LinearModel(rt, _make_field(NFIELD), use_dof="0-4,8")
        np.testing.assert_array_equal(model.use_dof, [0, 1, 2, 3, 4, 8])

    def test_use_dof_str_and_list_equivalent(self):
        rng  = np.random.default_rng(4)
        sens = _make_zk_sensitivity(rng)
        m_str  = LinearModel(_StubRaytraced(sens), _make_field(NFIELD), use_dof="0-4")
        m_list = LinearModel(_StubRaytraced(sens), _make_field(NFIELD), use_dof=[0, 1, 2, 3, 4])
        np.testing.assert_array_equal(m_str.use_dof, m_list.use_dof)

    def test_kwargs_stored(self):
        rng  = np.random.default_rng(5)
        rt   = _StubRaytraced(_make_zk_sensitivity(rng))
        model = LinearModel(rt, _make_field(NFIELD),
                            spots_kwargs={"nrad": 5},
                            zernikes_kwargs={"jmax": 22})
        assert model.spots_kwargs == {"nrad": 5}
        assert model.zernikes_kwargs == {"jmax": 22}

    def test_sensitivity_not_computed_at_construction(self):
        rng  = np.random.default_rng(6)
        rt   = _StubRaytraced(_make_zk_sensitivity(rng))
        LinearModel(rt, _make_field(NFIELD))
        assert len(rt.spots_calls) == 0
        assert len(rt.zernikes_calls) == 0


# ---------------------------------------------------------------------------
# Sensitivity caching and arguments
# ---------------------------------------------------------------------------


class TestSensitivityCaching:
    @pytest.fixture
    def setup(self):
        rng   = np.random.default_rng(10)
        sens  = _make_zk_sensitivity(rng, use_dof=np.arange(NDOF))
        field = _make_field(NFIELD)
        rt    = _StubRaytraced(sens)
        model = LinearModel(rt, field, use_dof=list(range(NDOF)))
        return model, rt, sens, field

    def test_spots_sensitivity_computed_on_first_access(self, setup):
        model, rt, sens, _ = setup
        result = model.spots_sensitivity
        assert result is sens
        assert len(rt.spots_calls) == 1

    def test_spots_sensitivity_cached(self, setup):
        model, rt, _, _ = setup
        _ = model.spots_sensitivity
        _ = model.spots_sensitivity
        assert len(rt.spots_calls) == 1

    def test_zernikes_sensitivity_computed_on_first_access(self, setup):
        model, rt, sens, _ = setup
        result = model.zernikes_sensitivity
        assert result is sens
        assert len(rt.zernikes_calls) == 1

    def test_zernikes_sensitivity_cached(self, setup):
        model, rt, _, _ = setup
        _ = model.zernikes_sensitivity
        _ = model.zernikes_sensitivity
        assert len(rt.zernikes_calls) == 1

    def test_spots_and_zernikes_independent(self, setup):
        model, rt, _, _ = setup
        _ = model.spots_sensitivity
        _ = model.zernikes_sensitivity
        assert len(rt.spots_calls) == 1
        assert len(rt.zernikes_calls) == 1

    def test_basis_is_always_x(self, setup):
        model, rt, _, _ = setup
        _ = model.spots_sensitivity
        _ = model.zernikes_sensitivity
        assert rt.spots_calls[0]["basis"] == "x"
        assert rt.zernikes_calls[0]["basis"] == "x"

    def test_field_forwarded(self, setup):
        model, rt, _, field = setup
        _ = model.spots_sensitivity
        assert rt.spots_calls[0]["field"] is field

    def test_use_dof_forwarded(self, setup):
        model, rt, _, _ = setup
        _ = model.zernikes_sensitivity
        np.testing.assert_array_equal(rt.zernikes_calls[0]["use_dof"], list(range(NDOF)))

    def test_use_dof_none_forwarded(self):
        rng   = np.random.default_rng(11)
        sens  = _make_zk_sensitivity(rng)
        rt    = _StubRaytraced(sens)
        model = LinearModel(rt, _make_field(NFIELD))  # use_dof=None
        _ = model.spots_sensitivity
        assert rt.spots_calls[0]["use_dof"] is None

    def test_extra_kwargs_forwarded(self):
        rng   = np.random.default_rng(12)
        sens  = _make_zk_sensitivity(rng)
        rt    = _StubRaytraced(sens)
        model = LinearModel(rt, _make_field(NFIELD),
                            spots_kwargs={"nrad": 7},
                            zernikes_kwargs={"jmax": 22})
        _ = model.spots_sensitivity
        _ = model.zernikes_sensitivity
        assert rt.spots_calls[0]["nrad"] == 7
        assert rt.zernikes_calls[0]["jmax"] == 22


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    @pytest.fixture
    def setup(self):
        rng     = np.random.default_rng(20)
        use_dof = np.arange(NDOF)
        sens    = _make_zk_sensitivity(rng, use_dof=use_dof)
        rt      = _StubRaytraced(sens)
        field   = _make_field(NFIELD)
        model   = LinearModel(rt, field, use_dof=list(use_dof))
        sf      = StateFactory(sens.schema)
        return model, sens, sf

    def test_zernikes_returns_zernikes(self, setup):
        model, sens, sf = setup
        state = sf.x(np.zeros(sens.schema.n_active))
        assert isinstance(model.zernikes(state), Zernikes)

    def test_zero_state_returns_nominal(self, setup):
        model, sens, sf = setup
        state  = sf.x(np.zeros(sens.schema.n_active))
        result = model.zernikes(state)
        np.testing.assert_allclose(
            result.coefs.to_value(u.um),
            sens.nominal.coefs.to_value(u.um),
        )

    def test_prediction_roundtrip(self, setup):
        """nominal + gradient @ true_x should be reproduced exactly."""
        model, sens, sf = setup
        rng    = np.random.default_rng(21)
        true_x = rng.standard_normal(sens.schema.n_active)
        state  = sf.x(true_x)

        grad = sens.gradient.coefs.to_value(u.um)   # (n_active, nfield, jmax+1)
        nom  = sens.nominal.coefs.to_value(u.um)    # (nfield, jmax+1)
        expected = nom + np.einsum("dnj,d->nj", grad, true_x)

        result = model.zernikes(state)
        np.testing.assert_allclose(result.coefs.to_value(u.um), expected, atol=1e-12)

    def test_spots_delegates_to_sensitivity_predict(self, setup):
        """spots() returns sensitivity.predict(state) — verified with a second
        independent call to predict to confirm the values match."""
        model, sens, sf = setup
        true_x = np.random.default_rng(22).standard_normal(sens.schema.n_active)
        state  = sf.x(true_x)

        from_model = model.spots(state)
        direct     = sens.predict(state)

        np.testing.assert_allclose(
            from_model.coefs.to_value(u.um),
            direct.coefs.to_value(u.um),
        )


# ---------------------------------------------------------------------------
# Integration: LinearModel vs full ray tracing
# ---------------------------------------------------------------------------


class TestLinearModelVsRaytraced:
    """Compare LinearModel predictions against full ray-traced outputs.

    Requires batoid and batoid_rubin.  The model is constructed once for the
    whole class (``scope="class"``) to avoid rebuilding it for every test.
    """

    @pytest.fixture(scope="class")
    def rt_model(self):
        from StarSharp.models.fiducial import default_raytraced_model
        return default_raytraced_model(pointing_model=None, rtp_lookup=None)

    @pytest.fixture(scope="class")
    def field(self, rt_model):
        x = np.array([0.0,  0.5, -0.5,  0.9, -0.9]) * u.deg
        y = np.array([0.0,  0.5,  0.5, -0.9,  0.9]) * u.deg
        return FieldCoords(x=x, y=y, frame="ocs", rtp=rt_model.rtp)

    def test_zernikes_single_dof_perturbation(self, rt_model, field):
        """Linear zernike prediction matches raytraced to <0.01 nm for a
        5 µm M2 dz perturbation (half the default step)."""
        linear = LinearModel(rt_model, field, use_dof=[0])
        sf     = StateFactory(linear.zernikes_sensitivity.schema)
        state  = sf.x(np.array([5.0]))  # 5 µm M2 dz

        np.testing.assert_allclose(
            linear.zernikes(state).coefs.to_value(u.nm),
            rt_model.zernikes(field, state).coefs.to_value(u.nm),
            atol=0.01,
        )

    def test_zernikes_rigid_body_perturbations(self, rt_model, field):
        """Linear zernike prediction matches raytraced to <0.05 nm for
        small simultaneous perturbations across all 10 rigid-body DOFs."""
        use_dof = list(range(10))
        linear  = LinearModel(rt_model, field, use_dof=use_dof)
        schema  = linear.zernikes_sensitivity.schema
        sf      = StateFactory(schema)

        rng    = np.random.default_rng(42)
        # 10 % of each DOF's step size — well within the linear regime
        steps  = schema.step[use_dof]
        x_vals = rng.standard_normal(len(use_dof)) * steps * 0.1
        state  = sf.x(x_vals)

        np.testing.assert_allclose(
            linear.zernikes(state).coefs.to_value(u.nm),
            rt_model.zernikes(field, state).coefs.to_value(u.nm),
            atol=0.05,
        )

    def test_spots_single_dof_perturbation(self, rt_model, field):
        """Linear spot prediction matches raytraced centroids to <0.001 nm
        for a 5 µm M2 dz perturbation."""
        linear = LinearModel(rt_model, field, use_dof=[0])
        sf     = StateFactory(linear.spots_sensitivity.schema)
        state  = sf.x(np.array([5.0]))

        linear_spots = linear.spots(state)
        rt_spots     = rt_model.spots(field, state)

        np.testing.assert_allclose(
            linear_spots.dx.to_value(u.nm),
            rt_spots.dx.to_value(u.nm),
            atol=0.001,
        )
        np.testing.assert_allclose(
            linear_spots.dy.to_value(u.nm),
            rt_spots.dy.to_value(u.nm),
            atol=0.001,
        )
