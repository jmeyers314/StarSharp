"""Tests for StateSchema, StateFactory, and State."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from StarSharp.datatypes.state import State, StateFactory, StateSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NAMES_5 = ("a", "b", "c", "d", "e")
UNITS_5 = (u.mm, u.mm, u.mm, u.mm, u.mm)

NAMES_10 = tuple(f"dof{i}" for i in range(10))
UNITS_10 = (u.mm,) * 10


def _make_schema(n_dof=5, use_dof=None, Vh=None, S=None, U=None):
    """Make a simple StateSchema with *n_dof* uniform-mm DOFs."""
    names = tuple(f"dof{i}" for i in range(n_dof))
    units = (u.mm,) * n_dof
    return StateSchema(
        dof_names=names,
        dof_units=units,
        use_dof=use_dof,
        Vh=Vh,
        S=S,
        U=U,
    )


def _make_Vh(n_active, n_keep=None, seed=42):
    """Return a (n_keep, n_active) semi-orthogonal matrix via SVD."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_active * 4, n_active))
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    if n_keep is None:
        n_keep = n_active
    return Vh[:n_keep]


# ---------------------------------------------------------------------------
# TestStateSchemaConstruction
# ---------------------------------------------------------------------------


class TestStateSchemaConstruction:
    def test_minimal_all_active(self):
        schema = _make_schema(5)
        assert schema.n_dof == 5
        assert schema.n_active == 5
        np.testing.assert_array_equal(schema.use_dof, np.arange(5))

    def test_with_use_dof(self):
        schema = _make_schema(10, use_dof=[1, 3, 7])
        assert schema.n_dof == 10
        assert schema.n_active == 3
        np.testing.assert_array_equal(schema.use_dof, [1, 3, 7])

    def test_with_Vh(self):
        Vh = _make_Vh(4, 3)
        schema = _make_schema(10, use_dof=[0, 1, 2, 3], Vh=Vh)
        assert schema.Vh.shape == (3, 4)
        assert schema.n_keep == 3

    def test_with_Vh_S_U(self):
        n_active = 4
        n_keep = 3
        n_obs = 20
        Vh = _make_Vh(n_active, n_keep)
        S = np.array([5.0, 3.0, 1.0])
        U = np.random.default_rng(0).standard_normal((n_obs, n_keep))
        schema = _make_schema(10, use_dof=list(range(n_active)), Vh=Vh, S=S, U=U)
        assert schema.S.shape == (n_keep,)
        assert schema.U.shape == (n_obs, n_keep)

    def test_coerces_names_to_str(self):
        schema = StateSchema(dof_names=(0, 1, 2), dof_units=(u.mm, u.mm, u.mm))
        assert schema.dof_names == ("0", "1", "2")

    def test_coerces_units(self):
        schema = StateSchema(dof_names=("a", "b"), dof_units=("mm", "arcsec"))
        assert schema.dof_units[0] == u.mm
        assert schema.dof_units[1] == u.arcsec

    def test_default_step_is_none(self):
        schema = _make_schema(5)
        assert schema.step is None

    def test_step_coerced_to_float_array(self):
        schema = StateSchema(
            dof_names=("a", "b", "c"),
            dof_units=(u.mm, u.mm, u.mm),
            step=[1, 2, 3],
        )
        assert isinstance(schema.step, np.ndarray)
        assert schema.step.dtype == float
        np.testing.assert_array_equal(schema.step, np.array([1.0, 2.0, 3.0]))

    def test_frozen(self):
        schema = _make_schema(3)
        with pytest.raises((AttributeError, TypeError)):
            schema.dof_names = ("x",)

    def test_empty_names_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            StateSchema(dof_names=(), dof_units=())

    def test_mismatched_units_raises(self):
        with pytest.raises(ValueError, match="same length"):
            StateSchema(dof_names=("a", "b"), dof_units=(u.mm,))

    def test_bad_use_dof_2d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            _make_schema(5, use_dof=[[0, 1], [2, 3]])

    def test_bad_step_shape_raises(self):
        with pytest.raises(ValueError, match="step must have shape"):
            StateSchema(
                dof_names=("a", "b", "c"),
                dof_units=(u.mm, u.mm, u.mm),
                step=[1.0, 2.0],
            )

    def test_use_dof_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            _make_schema(5, use_dof=[0, 1, 10])

    def test_bad_Vh_shape_raises(self):
        Vh = np.ones((3, 5))  # n_active=4, so wrong
        with pytest.raises(ValueError, match="Vh must have shape"):
            _make_schema(10, use_dof=[0, 1, 2, 3], Vh=Vh)

    def test_bad_S_length_raises(self):
        Vh = _make_Vh(4, 3)
        S = np.array([1.0, 2.0])  # wrong length for n_keep=3
        with pytest.raises(ValueError, match="S must have length"):
            _make_schema(10, use_dof=list(range(4)), Vh=Vh, S=S)

    def test_bad_S_ndim_raises(self):
        Vh = _make_Vh(4, 3)
        S = np.ones((3, 1))
        with pytest.raises(ValueError, match="1D array"):
            _make_schema(10, use_dof=list(range(4)), Vh=Vh, S=S)

    def test_bad_U_ndim_raises(self):
        Vh = _make_Vh(4, 3)
        S = np.array([3.0, 2.0, 1.0])
        U = np.ones(10)  # must be 2D
        with pytest.raises(ValueError, match="2D array"):
            _make_schema(10, use_dof=list(range(4)), Vh=Vh, S=S, U=U)

    def test_bad_U_cols_raises(self):
        Vh = _make_Vh(4, 3)  # n_keep=3
        S = np.array([3.0, 2.0, 1.0])
        U = np.ones((10, 2))  # should be (10, 3)
        with pytest.raises(ValueError, match="n_keep=3"):
            _make_schema(10, use_dof=list(range(4)), Vh=Vh, S=S, U=U)


# ---------------------------------------------------------------------------
# TestStateSchemaProperties
# ---------------------------------------------------------------------------


class TestStateSchemaProperties:
    def test_n_dof(self):
        assert _make_schema(7).n_dof == 7

    def test_n_active_default(self):
        assert _make_schema(7).n_active == 7

    def test_n_active_subset(self):
        assert _make_schema(10, use_dof=[1, 3, 5]).n_active == 3

    def test_n_keep_raises_when_no_Vh(self):
        schema = _make_schema(5)
        with pytest.raises(ValueError, match="n_keep is not defined"):
            _ = schema.n_keep

    def test_n_keep_with_Vh(self):
        Vh = _make_Vh(5, 3)
        schema = _make_schema(10, use_dof=list(range(5)), Vh=Vh)
        assert schema.n_keep == 3


# ---------------------------------------------------------------------------
# TestStateFactory
# ---------------------------------------------------------------------------


class TestStateFactory:
    def _make_factory(self, n_dof=10, use_dof=None, with_Vh=False):
        use = use_dof if use_dof is not None else list(range(4))
        n_active = len(use)
        Vh = _make_Vh(n_active) if with_Vh else None
        schema = _make_schema(n_dof, use_dof=use, Vh=Vh)
        return StateFactory(schema=schema)

    def test_passthrough_n_dof(self):
        sf = self._make_factory(n_dof=10)
        assert sf.n_dof == 10

    def test_passthrough_n_active(self):
        sf = self._make_factory(use_dof=[0, 2, 4])
        assert sf.n_active == 3

    def test_passthrough_dof_names(self):
        sf = self._make_factory(n_dof=5)
        assert sf.dof_names == tuple(f"dof{i}" for i in range(5))

    def test_passthrough_Vh(self):
        sf = self._make_factory(with_Vh=True)
        assert sf.Vh is not None

    def test_f_creates_f_state(self):
        sf = self._make_factory(n_dof=5, use_dof=list(range(5)))
        s = sf.f(np.zeros(5))
        assert s.basis == "f"

    def test_x_creates_x_state(self):
        sf = self._make_factory()
        s = sf.x(np.zeros(len(sf.use_dof)))
        assert s.basis == "x"

    def test_v_creates_v_state(self):
        sf = self._make_factory(with_Vh=True)
        n_keep = sf.n_keep
        s = sf.v(np.zeros(n_keep))
        assert s.basis == "v"

    def test_v_without_Vh_raises(self):
        sf = self._make_factory(with_Vh=False)
        with pytest.raises(ValueError, match="Vh must be set"):
            sf.v(np.zeros(len(sf.use_dof)))

    def test_zero_f(self):
        sf = self._make_factory(n_dof=5, use_dof=list(range(5)))
        s = sf.zero("f")
        assert s.basis == "f"
        np.testing.assert_array_equal(s.value, np.zeros(5))

    def test_zero_x(self):
        sf = self._make_factory()
        s = sf.zero("x")
        assert s.basis == "x"
        np.testing.assert_array_equal(s.value, np.zeros(len(sf.use_dof)))

    def test_zero_v(self):
        sf = self._make_factory(with_Vh=True)
        s = sf.zero("v")
        assert s.basis == "v"
        np.testing.assert_array_equal(s.value, np.zeros(sf.n_keep))

    def test_by_name_x_single(self):
        schema = _make_schema(5)
        sf = StateFactory(schema=schema)
        s = sf.by_name(dof0=3.0)
        assert s.basis == "x"
        assert s.value[0] == 3.0
        np.testing.assert_array_equal(s.value[1:], 0.0)

    def test_by_name_empty_returns_zero(self):
        sf = self._make_factory()
        s = sf.by_name()
        np.testing.assert_array_equal(s.value, np.zeros(len(sf.use_dof)))

    def test_by_name_vmode(self):
        sf = self._make_factory(with_Vh=True)
        s = sf.by_name(vmode1=7.0)
        assert s.basis == "v"
        assert s.value[0] == 7.0
        np.testing.assert_array_equal(s.value[1:], 0.0)

    def test_by_name_mixed_raises(self):
        sf = self._make_factory()
        with pytest.raises(ValueError, match="Cannot mix"):
            sf.by_name(dof0=1.0, vmode1=2.0)

    def test_by_name_invalid_key_raises(self):
        sf = self._make_factory()
        with pytest.raises(ValueError, match="Invalid kwargs"):
            sf.by_name(unknown_dof=1.0)


# ---------------------------------------------------------------------------
# TestStateConstruction
# ---------------------------------------------------------------------------


class TestStateConstruction:
    def test_f_basis(self):
        schema = _make_schema(5)
        s = State(value=np.zeros(5), basis="f", schema=schema)
        assert s.basis == "f"

    def test_x_basis(self):
        schema = _make_schema(5, use_dof=[0, 2, 4])
        s = State(value=np.zeros(3), basis="x", schema=schema)
        assert s.basis == "x"

    def test_v_basis(self):
        Vh = _make_Vh(4, 3)
        schema = _make_schema(10, use_dof=list(range(4)), Vh=Vh)
        s = State(value=np.zeros(3), basis="v", schema=schema)
        assert s.basis == "v"

    def test_invalid_basis_raises(self):
        schema = _make_schema(3)
        with pytest.raises(ValueError, match="basis must be"):
            State(value=np.zeros(3), basis="z", schema=schema)

    def test_wrong_schema_type_raises(self):
        with pytest.raises(TypeError, match="StateSchema"):
            State(value=np.zeros(3), basis="x", schema="not_a_schema")

    def test_f_wrong_length_raises(self):
        schema = _make_schema(5)
        with pytest.raises(ValueError, match="f-basis value must have length 5"):
            State(value=np.zeros(4), basis="f", schema=schema)

    def test_x_wrong_length_raises(self):
        schema = _make_schema(10, use_dof=[0, 1, 2])
        with pytest.raises(ValueError, match="x-basis value must have length 3"):
            State(value=np.zeros(4), basis="x", schema=schema)

    def test_v_no_Vh_raises(self):
        schema = _make_schema(5)  # no Vh
        with pytest.raises(ValueError, match="Vh must be set"):
            State(value=np.zeros(5), basis="v", schema=schema)

    def test_v_wrong_length_raises(self):
        Vh = _make_Vh(4, 3)
        schema = _make_schema(10, use_dof=list(range(4)), Vh=Vh)
        with pytest.raises(ValueError, match="v-basis value must have length 3"):
            State(value=np.zeros(2), basis="v", schema=schema)

    def test_value_coerced_to_ndarray(self):
        schema = _make_schema(3)
        s = State(value=[1.0, 2.0, 3.0], basis="x", schema=schema)
        assert isinstance(s.value, np.ndarray)

    def test_frozen(self):
        schema = _make_schema(3)
        s = State(value=np.zeros(3), basis="x", schema=schema)
        with pytest.raises((AttributeError, TypeError)):
            s.value = np.ones(3)


# ---------------------------------------------------------------------------
# TestStateProperties
# ---------------------------------------------------------------------------


class TestStateProperties:
    def _x_state(self):
        schema = _make_schema(10, use_dof=[1, 3, 5, 7])
        return State(value=np.array([1.0, 2.0, 3.0, 4.0]), basis="x", schema=schema)

    def test_n_dof(self):
        s = self._x_state()
        assert s.n_dof == 10

    def test_use_dof(self):
        s = self._x_state()
        np.testing.assert_array_equal(s.use_dof, [1, 3, 5, 7])

    def test_Vh_none(self):
        s = self._x_state()
        assert s.Vh is None

    def test_dof_names(self):
        s = self._x_state()
        assert s.dof_names == tuple(f"dof{i}" for i in range(10))

    def test_x_names(self):
        s = self._x_state()
        assert s.x_names == ("dof1", "dof3", "dof5", "dof7")

    def test_x_units(self):
        s = self._x_state()
        assert all(u_ == u.mm for u_ in s.x_units)


# ---------------------------------------------------------------------------
# TestStateBasisConversions
# ---------------------------------------------------------------------------


class TestStateBasisConversions:
    def test_x_identity(self):
        schema = _make_schema(5)
        s = State(value=np.ones(5), basis="x", schema=schema)
        assert s.x is s

    def test_f_identity(self):
        schema = _make_schema(5)
        s = State(value=np.zeros(5), basis="f", schema=schema)
        assert s.f is s

    def test_v_identity(self):
        Vh = _make_Vh(4)
        schema = _make_schema(10, use_dof=list(range(4)), Vh=Vh)
        s = State(value=np.ones(4), basis="v", schema=schema)
        assert s.v is s

    def test_x_to_f(self):
        use_dof = np.array([1, 3, 7])
        schema = _make_schema(10, use_dof=use_dof)
        xvals = np.array([10.0, 20.0, 30.0])
        s = State(value=xvals, basis="x", schema=schema)
        fs = s.f
        assert fs.basis == "f"
        assert len(fs.value) == 10
        np.testing.assert_allclose(fs.value[use_dof], xvals)
        np.testing.assert_allclose(
            fs.value[np.setdiff1d(np.arange(10), use_dof)], 0.0
        )

    def test_f_to_x(self):
        use_dof = np.array([0, 4])
        schema = _make_schema(5, use_dof=use_dof)
        fvals = np.array([5.0, 0.0, 0.0, 0.0, 9.0])
        s = State(value=fvals, basis="f", schema=schema)
        np.testing.assert_allclose(s.x.value, [5.0, 9.0])

    def test_x_to_f_to_x_roundtrip(self):
        use_dof = np.array([1, 3, 7])
        schema = _make_schema(10, use_dof=use_dof)
        xvals = np.array([10.0, 20.0, 30.0])
        s = State(value=xvals, basis="x", schema=schema)
        np.testing.assert_allclose(s.f.x.value, xvals)

    def test_x_to_v_to_x_full_rank(self):
        """Full-rank SVD: x→v→x is lossless."""
        n_active = 5
        Vh = _make_Vh(n_active, n_active)
        schema = _make_schema(10, use_dof=list(range(n_active)), Vh=Vh)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = State(value=xvals, basis="x", schema=schema)
        np.testing.assert_allclose(s.v.x.value, xvals, atol=1e-12)

    def test_x_to_v_truncated_is_lossy(self):
        """Truncated SVD: x→v→x loses information."""
        n_active = 5
        Vh = _make_Vh(n_active, 3)
        schema = _make_schema(10, use_dof=list(range(n_active)), Vh=Vh)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = State(value=xvals, basis="x", schema=schema)
        recovered = s.v.x.value
        assert not np.allclose(recovered, xvals)

    def test_v_to_x_to_v_stable(self):
        """Re-projecting the recovered x-state gives the same v-state."""
        n_active = 5
        Vh = _make_Vh(n_active, 3)
        schema = _make_schema(10, use_dof=list(range(n_active)), Vh=Vh)
        s = State(value=np.array([0.5, 1.5, -0.5]), basis="v", schema=schema)
        np.testing.assert_allclose(s.x.v.value, s.value, atol=1e-12)

    def test_v_to_f_to_v_roundtrip(self):
        """Converting v -> f -> v preserves v-basis coefficients."""
        use_dof = np.array([1, 3, 5, 7])
        Vh = _make_Vh(4, 3)
        schema = _make_schema(10, use_dof=use_dof, Vh=Vh)
        s = State(value=np.array([0.25, -1.0, 2.5]), basis="v", schema=schema)
        np.testing.assert_allclose(s.f.v.value, s.value, atol=1e-12)

    def test_v_to_f(self):
        use_dof = np.array([1, 3, 5, 7])
        Vh = _make_Vh(4, 3)
        schema = _make_schema(10, use_dof=use_dof, Vh=Vh)
        s = State(value=np.ones(3), basis="v", schema=schema)
        fs = s.f
        assert fs.basis == "f"
        assert len(fs.value) == 10
        inactive = np.setdiff1d(np.arange(10), use_dof)
        np.testing.assert_allclose(fs.value[inactive], 0.0)

    def test_v_no_Vh_raises(self):
        schema = _make_schema(5)
        s = State(value=np.ones(5), basis="x", schema=schema)
        with pytest.raises(ValueError, match="Vh must be set"):
            _ = s.v


# ---------------------------------------------------------------------------
# TestStateArithmetic
# ---------------------------------------------------------------------------


class TestStateArithmetic:
    def _schema_pair(self, use_dof_a, use_dof_b, n_dof=10):
        schema_a = _make_schema(n_dof, use_dof=use_dof_a)
        schema_b = _make_schema(n_dof, use_dof=use_dof_b)
        return schema_a, schema_b

    def test_add_x_x(self):
        schema = _make_schema(5)
        s1 = State(value=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), basis="x", schema=schema)
        s2 = State(value=np.array([0.5, 0.5, 0.5, 0.5, 0.5]), basis="x", schema=schema)
        s3 = s1 + s2
        np.testing.assert_allclose(s3.x.value, [1.5, 2.5, 3.5, 4.5, 5.5])

    def test_add_preserves_immutability(self):
        schema = _make_schema(3)
        s1 = State(value=np.array([1.0, 2.0, 3.0]), basis="x", schema=schema)
        s2 = State(value=np.array([1.0, 1.0, 1.0]), basis="x", schema=schema)
        _ = s1 + s2
        np.testing.assert_array_equal(s1.value, [1.0, 2.0, 3.0])

    def test_add_x_f(self):
        schema = _make_schema(5)
        s1 = State(value=np.array([1.0, 2.0, 3.0, 4.0, 5.0]), basis="x", schema=schema)
        ssum = s1 + s1.f
        np.testing.assert_allclose(ssum.x.value, 2.0 * s1.value, atol=1e-12)

    def test_add_same_use_dof_but_mismatched_Vh_drops_Vh(self):
        use_dof = [0, 1, 2, 3]
        schema_a = _make_schema(10, use_dof=use_dof, Vh=_make_Vh(4, seed=101))
        schema_b = _make_schema(10, use_dof=use_dof, Vh=_make_Vh(4, seed=202))
        sa = State(value=np.array([1.0, 0.0, 0.0, 0.0]), basis="x", schema=schema_a)
        sb = State(value=np.array([0.0, 1.0, 0.0, 0.0]), basis="x", schema=schema_b)

        s = sa + sb

        np.testing.assert_array_equal(s.schema.use_dof, use_dof)
        assert s.schema.Vh is None

    def test_add_different_use_dof_falls_back_to_f_schema(self):
        schema_a, schema_b = self._schema_pair([0, 1, 2], [3, 4, 5])
        sa = State(value=np.array([1.0, 0.0, 0.0]), basis="x", schema=schema_a)
        sb = State(value=np.array([0.0, 1.0, 0.0]), basis="x", schema=schema_b)
        s = sa + sb
        assert s.basis == "f"
        assert len(s.value) == 10
        assert np.isclose(s.value[0], 1.0)
        assert np.isclose(s.value[4], 1.0)

    def test_add_incompatible_names_raises(self):
        schema_a = StateSchema(dof_names=("a", "b"), dof_units=(u.mm, u.mm))
        schema_b = StateSchema(dof_names=("x", "y"), dof_units=(u.mm, u.mm))
        sa = State(value=np.array([1.0, 2.0]), basis="x", schema=schema_a)
        sb = State(value=np.array([1.0, 2.0]), basis="x", schema=schema_b)
        with pytest.raises(ValueError, match="incompatible"):
            _ = sa + sb

    def test_mul_scalar(self):
        schema = _make_schema(3)
        s = State(value=np.array([1.0, 2.0, 3.0]), basis="x", schema=schema)
        s2 = s * 3.0
        np.testing.assert_allclose(s2.value, [3.0, 6.0, 9.0])
        assert s2.basis == "x"

    def test_rmul_scalar(self):
        schema = _make_schema(3)
        s = State(value=np.array([1.0, 2.0, 3.0]), basis="x", schema=schema)
        s2 = 2.0 * s
        np.testing.assert_allclose(s2.value, [2.0, 4.0, 6.0])

    def test_mul_non_scalar_returns_not_implemented(self):
        schema = _make_schema(3)
        s = State(value=np.ones(3), basis="x", schema=schema)
        result = s.__mul__([1, 2])
        assert result is NotImplemented
