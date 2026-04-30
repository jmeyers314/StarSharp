"""Tests for Sensitivity and StateSchema.with_svd."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from dataclasses import replace

from StarSharp.datatypes import Zernikes, Spots
from StarSharp.datatypes.sensitivity import Sensitivity
from StarSharp.datatypes.state import State, StateSchema

from .utils import RTP, _make_field

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m

N_DOF = 10
N_ACTIVE = 4
ACTIVE_IDX = [0, 1, 2, 3]
N_FIELD = 5
JMAX = 10
N_RAY = 16

# ---------------------------------------------------------------------------
# Shared schema / steps helpers
# ---------------------------------------------------------------------------

_SCHEMA = StateSchema(
    dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
    dof_units=(u.mm,) * N_DOF,
    use_dof=np.array(ACTIVE_IDX, dtype=int),
)


def _make_steps(schema=None):
    if schema is None:
        schema = _SCHEMA
    return State(
        value=np.array([1.0, 2.0, 0.5, 3.0]),
        basis="x",
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Zernike helpers
# ---------------------------------------------------------------------------


def _make_zk_nominal(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    field = _make_field(N_FIELD, rtp=RTP)
    coefs = rng.standard_normal((N_FIELD, JMAX + 1)) * u.um
    return Zernikes(
        coefs=coefs,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
        rtp=RTP,
    )


def _make_zk_perturbed(nominal, step, seed):
    rng = np.random.default_rng(seed)
    delta = rng.standard_normal(nominal.coefs.shape)
    return Zernikes(
        coefs=nominal.coefs + delta * step * u.um,
        field=nominal.field,
        R_outer=nominal.R_outer,
        R_inner=nominal.R_inner,
        frame=nominal.frame,
        rtp=nominal.rtp,
    )


def _make_zk_sensitivity(schema=None):
    if schema is None:
        schema = _SCHEMA
    rng = np.random.default_rng(7)
    nominal = _make_zk_nominal(rng)
    steps = _make_steps(schema)
    perturbed = [
        _make_zk_perturbed(nominal, s, seed=10 + i)
        for i, s in enumerate(steps.value)
    ]
    sens = Sensitivity.from_finite_differences(nominal, perturbed, steps)
    return sens, nominal, steps


# ---------------------------------------------------------------------------
# Spots helpers
# ---------------------------------------------------------------------------


def _make_spots_nominal(rng=None):
    if rng is None:
        rng = np.random.default_rng(55)
    field = _make_field(N_FIELD, rtp=RTP)
    dx = rng.standard_normal((N_FIELD, N_RAY)) * u.um
    dy = rng.standard_normal((N_FIELD, N_RAY)) * u.um
    vig = np.zeros((N_FIELD, N_RAY), dtype=bool)
    return Spots(
        dx=dx,
        dy=dy,
        vignetted=vig,
        field=field,
        wavelength=620.0 * u.nm,
        frame="ccs",
        rtp=RTP,
    )


def _make_spots_perturbed(nominal, step, seed):
    rng = np.random.default_rng(seed)
    ddx = rng.standard_normal(nominal.dx.shape)
    ddy = rng.standard_normal(nominal.dy.shape)
    return Spots(
        dx=nominal.dx + ddx * step * u.um,
        dy=nominal.dy + ddy * step * u.um,
        vignetted=nominal.vignetted,
        field=nominal.field,
        wavelength=nominal.wavelength,
        frame=nominal.frame,
        rtp=nominal.rtp,
    )


def _make_spots_sensitivity(schema=None):
    if schema is None:
        schema = _SCHEMA
    rng = np.random.default_rng(88)
    nominal = _make_spots_nominal(rng)
    steps = _make_steps(schema)
    perturbed = [
        _make_spots_perturbed(nominal, s, seed=100 + i)
        for i, s in enumerate(steps.value)
    ]
    return Sensitivity.from_finite_differences(nominal, perturbed, steps)


# ---------------------------------------------------------------------------
# TestSensitivityConstruction
# ---------------------------------------------------------------------------


class TestSensitivityConstruction:
    def test_gradient_shape_zk(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.gradient.coefs.shape == (N_ACTIVE, N_FIELD, JMAX + 1)

    def test_nominal_preserved(self):
        sens, nominal, _ = _make_zk_sensitivity()
        np.testing.assert_allclose(
            sens.nominal.coefs.to_value(u.um),
            nominal.coefs.to_value(u.um),
        )

    def test_gradient_values_zk(self):
        sens, nominal, steps = _make_zk_sensitivity()
        rng = np.random.default_rng(7)
        _ = _make_zk_nominal(rng)  # advance rng to same state used in helper
        for i, step in enumerate(steps.value):
            perturbed = _make_zk_perturbed(nominal, step, seed=10 + i)
            expected = (perturbed.coefs - nominal.coefs) / step
            np.testing.assert_allclose(
                sens.gradient.coefs[i].to_value(u.um),
                expected.to_value(u.um),
            )

    def test_basis_is_x(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.basis == "x"

    def test_schema_propagated(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.schema is _SCHEMA

    def test_n_dof_n_active_passthrough(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.n_dof == N_DOF
        assert sens.n_active == N_ACTIVE

    def test_schema_required(self):
        """Sensitivity.__post_init__ rejects a non-StateSchema schema."""
        nominal = _make_zk_nominal()
        with pytest.raises(TypeError, match="StateSchema"):
            Sensitivity(gradient=nominal, nominal=nominal, basis="x", schema=None)

    def test_invalid_basis_raises(self):
        nominal = _make_zk_nominal()
        with pytest.raises(ValueError, match="basis must be"):
            Sensitivity(gradient=nominal, nominal=nominal, basis="z", schema=_SCHEMA)

    def test_repr(self):
        sens, *_ = _make_zk_sensitivity()
        r = repr(sens)
        assert "Zernikes" in r
        assert "basis=" in r


# ---------------------------------------------------------------------------
# TestSensitivityBasisNarrowing
# ---------------------------------------------------------------------------


class TestSensitivityBasisNarrowing:
    def test_x_identity(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.x is sens  # already x-basis

    def test_f_to_x_narrows_gradient(self):
        """Build an f-basis sensitivity and verify .x selects active DOFs."""
        # Manually wrap sensitivity built from all N_DOF DOFs, then narrow.
        sens_x, nominal, steps = _make_zk_sensitivity()
        # Pretend it's f-basis by rebuilding with full schema (no use_dof narrowing).
        full_schema = StateSchema(
            dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
            use_dof=np.array(ACTIVE_IDX, dtype=int),
        )
        full_coefs = np.zeros((N_DOF, N_FIELD, JMAX + 1))
        for i, ai in enumerate(ACTIVE_IDX):
            full_coefs[ai] = sens_x.gradient.coefs[i].to_value(u.um)
        full_gradient = replace(sens_x.gradient, coefs=full_coefs * u.um)
        sens_f = Sensitivity(
            gradient=full_gradient,
            nominal=nominal,
            basis="f",
            schema=full_schema,
        )
        sens_x2 = sens_f.x
        assert sens_x2.basis == "x"
        assert sens_x2.gradient.coefs.shape == (N_ACTIVE, N_FIELD, JMAX + 1)
        # The active rows must match the original.
        for local_i, global_i in enumerate(ACTIVE_IDX):
            np.testing.assert_allclose(
                sens_x2.gradient.coefs[local_i].to_value(u.um),
                full_coefs[global_i],
            )

    def test_f_identity(self):
        """Sensitivity already in f-basis returns self."""
        sens_x, nominal, steps = _make_zk_sensitivity()
        full_schema = StateSchema(
            dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
        )
        full_coefs = np.zeros((N_DOF, N_FIELD, JMAX + 1))
        full_gradient = replace(sens_x.gradient, coefs=full_coefs * u.um)
        sens_f = Sensitivity(
            gradient=full_gradient,
            nominal=nominal,
            basis="f",
            schema=full_schema,
        )
        assert sens_f.f is sens_f

    def test_f_raises_from_x(self):
        sens, *_ = _make_zk_sensitivity()
        with pytest.raises(ValueError, match="one-way"):
            _ = sens.f

    def test_v_from_x(self):
        n_active = N_ACTIVE
        Vh = _make_Vh_ortho(n_active)
        schema = StateSchema(
            dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
            use_dof=np.array(ACTIVE_IDX, dtype=int),
            Vh=Vh,
        )
        sens, *_ = _make_zk_sensitivity(schema=schema)
        sv = sens.v
        assert sv.basis == "v"
        assert sv.gradient.coefs.shape[0] == n_active

    def test_v_without_Vh_raises(self):
        sens, *_ = _make_zk_sensitivity()  # schema has no Vh
        with pytest.raises(ValueError, match="Vh must be set"):
            _ = sens.v

    def test_v_to_x_raises(self):
        Vh = _make_Vh_ortho(N_ACTIVE)
        schema = StateSchema(
            dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
            use_dof=np.array(ACTIVE_IDX, dtype=int),
            Vh=Vh,
        )
        sens, *_ = _make_zk_sensitivity(schema=schema)
        sv = sens.v
        with pytest.raises(ValueError, match="one-way"):
            _ = sv.x


# ---------------------------------------------------------------------------
# TestSensitivityPredict
# ---------------------------------------------------------------------------


class TestSensitivityPredict:
    def test_predict_zero_state_returns_nominal(self):
        sens, nominal, steps = _make_zk_sensitivity()
        zero_state = State(value=np.zeros(N_ACTIVE), basis="x", schema=_SCHEMA)
        result = sens.predict(zero_state)
        np.testing.assert_allclose(
            result.coefs.to_value(u.um),
            nominal.coefs.to_value(u.um),
        )

    def test_predict_linearity(self):
        """predict(2*s) == 2 * predict(s) - nominal (linear)."""
        sens, nominal, steps = _make_zk_sensitivity()
        xvals = np.array([0.01, 0.01, 0.01, 0.01])
        s1 = State(value=xvals, basis="x", schema=_SCHEMA)
        s2 = State(value=2 * xvals, basis="x", schema=_SCHEMA)
        r1 = sens.predict(s1)
        r2 = sens.predict(s2)
        delta1 = (r1.coefs - nominal.coefs).to_value(u.um)
        delta2 = (r2.coefs - nominal.coefs).to_value(u.um)
        np.testing.assert_allclose(delta2, 2.0 * delta1, rtol=1e-12)

    def test_predict_basis_mismatch_raises(self):
        sens, *_ = _make_zk_sensitivity()
        f_schema = StateSchema(
            dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
        )
        f_state = State(value=np.zeros(N_DOF), basis="f", schema=f_schema)
        with pytest.raises(ValueError, match="basis must match"):
            sens.predict(f_state)

    def test_predict_incompatible_schema_raises(self):
        sens, *_ = _make_zk_sensitivity()
        bad_schema = StateSchema(
            dof_names=tuple(f"other{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
            use_dof=np.array(ACTIVE_IDX, dtype=int),
        )
        bad_state = State(value=np.zeros(N_ACTIVE), basis="x", schema=bad_schema)
        with pytest.raises(ValueError, match="not compatible"):
            sens.predict(bad_state)


# ---------------------------------------------------------------------------
# TestStateSchemaWithSvd
# ---------------------------------------------------------------------------


def _make_Vh_ortho(n_active, n_keep=None, seed=99):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_active * 4, n_active))
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    if n_keep is None:
        n_keep = n_active
    return Vh[:n_keep]


class TestStateSchemaWithSvd:
    def _sens_x(self, schema=None):
        if schema is None:
            schema = _SCHEMA
        sens, *_ = _make_zk_sensitivity(schema=schema)
        return sens  # already x-basis (from_finite_differences uses steps.basis)

    def test_Vh_shape_default(self):
        sens = self._sens_x()
        schema2 = _SCHEMA.with_svd(sens)
        assert schema2.Vh.shape == (N_ACTIVE, N_ACTIVE)

    def test_S_shape_default(self):
        sens = self._sens_x()
        schema2 = _SCHEMA.with_svd(sens)
        assert schema2.S.shape == (N_ACTIVE,)

    def test_U_shape_default(self):
        sens = self._sens_x()
        schema2 = _SCHEMA.with_svd(sens)
        n_obs = schema2.U.shape[0]
        assert schema2.U.shape == (n_obs, N_ACTIVE)

    def test_S_is_decreasing(self):
        sens = self._sens_x()
        schema2 = _SCHEMA.with_svd(sens)
        assert np.all(np.diff(schema2.S) <= 0)

    def test_n_keep_truncation(self):
        sens = self._sens_x()
        schema2 = _SCHEMA.with_svd(sens, n_keep=2)
        assert schema2.Vh.shape == (2, N_ACTIVE)
        assert schema2.S.shape == (2,)
        assert schema2.U.shape[1] == 2

    def test_norm_n_active_length(self):
        sens = self._sens_x()
        norm = np.array([1.0, 2.0, 0.5, 3.0])
        schema2 = _SCHEMA.with_svd(sens, norm=norm)
        assert schema2.Vh is not None
        assert schema2.Vh.shape[1] == N_ACTIVE

    def test_norm_n_dof_length_sliced(self):
        """norm of length n_dof is sliced by use_dof."""
        sens = self._sens_x()
        norm_full = np.ones(N_DOF)
        norm_full[ACTIVE_IDX] = [1.0, 2.0, 0.5, 3.0]
        schema_full = _SCHEMA.with_svd(sens, norm=norm_full)

        norm_active = np.array([1.0, 2.0, 0.5, 3.0])
        schema_active = _SCHEMA.with_svd(sens, norm=norm_active)

        np.testing.assert_allclose(schema_full.Vh, schema_active.Vh, rtol=1e-10)

    def test_norm_wrong_length_raises(self):
        sens = self._sens_x()
        with pytest.raises(ValueError, match="norm must have length"):
            _SCHEMA.with_svd(sens, norm=np.ones(7))

    def test_n_keep_too_large_raises(self):
        sens = self._sens_x()
        with pytest.raises(ValueError, match="exceeds the number"):
            _SCHEMA.with_svd(sens, n_keep=N_ACTIVE + 10)

    def test_schema_mismatch_use_dof_raises(self):
        """with_svd rejects sensitivities whose use_dof differs from this schema."""
        # Build an f-basis sensitivity.
        full_schema = StateSchema(
            dof_names=tuple(f"dof{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
        )
        nominal = _make_zk_nominal()
        full_gradient = replace(
            nominal,
            coefs=np.zeros((N_DOF, N_FIELD, JMAX + 1)) * u.um,
        )
        sens_f = Sensitivity(
            gradient=full_gradient,
            nominal=nominal,
            basis="f",
            schema=full_schema,
        )
        with pytest.raises(ValueError, match="use_dof"):
            _SCHEMA.with_svd(sens_f)

    def test_not_a_sensitivity_raises(self):
        with pytest.raises(TypeError, match="Sensitivity"):
            _SCHEMA.with_svd("not_a_sensitivity")

    def test_schema_mismatch_dof_names_raises(self):
        bad_schema = StateSchema(
            dof_names=tuple(f"other{i}" for i in range(N_DOF)),
            dof_units=(u.mm,) * N_DOF,
            use_dof=np.array(ACTIVE_IDX, dtype=int),
        )
        sens, *_ = _make_zk_sensitivity(schema=bad_schema)
        with pytest.raises(ValueError, match="dof_names"):
            _SCHEMA.with_svd(sens)

    def test_original_schema_unchanged(self):
        """with_svd returns a new schema; the original is unmodified."""
        sens = self._sens_x()
        schema2 = _SCHEMA.with_svd(sens)
        assert _SCHEMA.Vh is None
        assert schema2.Vh is not None

    def test_Vh_norm_scaling(self):
        """Vh_scaled @ diag(1/norm) should recover Vh_raw up to sign/ordering."""
        sens = self._sens_x()
        norm = np.array([2.0, 3.0, 1.5, 0.5])
        schema2 = _SCHEMA.with_svd(sens, norm=norm)
        # Vh_scaled = Vh_raw[:n_keep] @ diag(norm)
        # So Vh_scaled @ diag(1/norm) == Vh_raw[:n_keep]
        Vh_unscaled = schema2.Vh @ np.diag(1.0 / norm)
        # Rows of Vh_unscaled should be unit vectors (since Vh_raw is semi-orthogonal).
        row_norms = np.linalg.norm(Vh_unscaled, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(N_ACTIVE), atol=1e-10)

    def test_with_svd_spots(self):
        """with_svd works with Spots sensitivity (tests _sensitivity_to_x_matrix)."""
        sens = _make_spots_sensitivity()
        schema2 = _SCHEMA.with_svd(sens)
        assert schema2.Vh.shape[1] == N_ACTIVE
        assert schema2.S.shape[0] == N_ACTIVE


# ---------------------------------------------------------------------------
# ASDF round-trip tests
# ---------------------------------------------------------------------------

from .utils import roundtrip_asdf, roundtrip_asdf_ctx  # noqa: E402
from .conftest import requires_starsharp_asdf  # noqa: E402


@pytest.mark.skipif(
    pytest.importorskip("asdf", reason="asdf not installed") is None,
    reason="asdf not installed",
)
class TestSensitivityAsdf:
    asdf = pytest.importorskip("asdf")

    def testroundtrip_asdf_ctx_zernikes_basis_x(self):
        sens, nominal, steps = _make_zk_sensitivity()
        sens_x = sens.x
        rt = roundtrip_asdf_ctx(sens_x)
        assert rt.basis == "x"
        assert isinstance(rt.gradient, Zernikes)
        assert rt.gradient.coefs.shape == sens_x.gradient.coefs.shape
        np.testing.assert_allclose(
            rt.gradient.coefs.to_value(u.um),
            sens_x.gradient.coefs.to_value(u.um),
        )

    def testroundtrip_asdf_ctx_zernikes_basis_f(self):
        nominal = _make_zk_nominal()
        steps_f = State(
            value=np.ones(N_DOF),
            basis="f",
            schema=_SCHEMA,
        )
        perturbed = [_make_zk_perturbed(nominal, 1.0, seed=10 + i) for i in range(N_DOF)]
        sens_f = Sensitivity.from_finite_differences(nominal, perturbed, steps_f)
        rt = roundtrip_asdf_ctx(sens_f)
        assert rt.basis == "f"
        assert len(rt.gradient) == N_DOF

    def testroundtrip_asdf_ctx_schema_preserved(self):
        sens, _, _ = _make_zk_sensitivity()
        rt = roundtrip_asdf_ctx(sens)
        assert rt.schema.dof_names == _SCHEMA.dof_names
        np.testing.assert_array_equal(rt.schema.use_dof, _SCHEMA.use_dof)

    def testroundtrip_asdf_ctx_nominal_preserved(self):
        sens, nominal, _ = _make_zk_sensitivity()
        rt = roundtrip_asdf_ctx(sens)
        assert isinstance(rt.nominal, Zernikes)
        np.testing.assert_allclose(
            rt.nominal.coefs.to_value(u.um),
            nominal.coefs.to_value(u.um),
        )

    def testroundtrip_asdf_ctx_spots(self):
        sens = _make_spots_sensitivity()
        rt = roundtrip_asdf_ctx(sens)
        assert rt.basis == sens.basis
        assert isinstance(rt.gradient, Spots)
        assert rt.gradient.dx.shape == sens.gradient.dx.shape
        np.testing.assert_allclose(
            rt.gradient.dx.to_value(u.um),
            sens.gradient.dx.to_value(u.um),
        )

    def testroundtrip_asdf_ctx_spots_nominal(self):
        sens = _make_spots_sensitivity()
        rt = roundtrip_asdf_ctx(sens)
        assert isinstance(rt.nominal, Spots)
        np.testing.assert_allclose(
            rt.nominal.dx.to_value(u.um),
            sens.nominal.dx.to_value(u.um),
        )

    def testroundtrip_asdf_ctx_with_svd_basis_v(self):
        sens, _, _ = _make_zk_sensitivity()
        schema_svd = _SCHEMA.with_svd(sens)
        sens_v = Sensitivity(
            gradient=sens.x.gradient,
            schema=schema_svd,
            nominal=sens.nominal,
            basis="x",
        ).v
        rt = roundtrip_asdf_ctx(sens_v)
        assert rt.basis == "v"
        assert rt.schema.Vh is not None
        np.testing.assert_allclose(rt.schema.Vh, schema_svd.Vh)

    def testroundtrip_asdf_ctx_predict_consistent(self):
        """Predict with round-tripped sensitivity matches original."""
        sens, _, steps = _make_zk_sensitivity()
        rt = roundtrip_asdf_ctx(sens.x)
        state = State(value=np.array([0.1, 0.2, 0.3, 0.4]), basis="x", schema=_SCHEMA)
        pred_orig = sens.x.predict(state)
        pred_rt = rt.predict(state)
        np.testing.assert_allclose(
            pred_rt.coefs.to_value(u.um),
            pred_orig.coefs.to_value(u.um),
            atol=1e-12,
        )


@requires_starsharp_asdf
class TestSensitivityAsdfEntryPoint:
    def testroundtrip_asdf_ctx_zernikes(self):
        sens, _, _ = _make_zk_sensitivity()
        rt = roundtrip_asdf(sens.x)
        assert rt.basis == "x"
        assert isinstance(rt.gradient, Zernikes)

    def testroundtrip_asdf_ctx_spots(self):
        sens = _make_spots_sensitivity()
        rt = roundtrip_asdf(sens)
        assert isinstance(rt.gradient, Spots)
