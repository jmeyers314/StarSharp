"""Tests for PointingModel."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from StarSharp.datatypes import PointingModel, State, StateSchema

from .utils import roundtrip_asdf, roundtrip_asdf_ctx
from .conftest import requires_starsharp_asdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NAMES_4 = ("dz", "dx", "rx", "ry")
UNITS_4 = (u.mm, u.mm, u.deg, u.deg)
N_DOF = 4


def _make_schema(names=NAMES_4, units=UNITS_4):
    return StateSchema(dof_names=names, dof_units=units)


def _make_pm(names=NAMES_4, units=UNITS_4, seed=0, angle_unit=u.arcsec):
    rng = np.random.default_rng(seed)
    schema = _make_schema(names, units)
    matrix = rng.standard_normal((2, len(names))) * angle_unit
    return PointingModel(schema=schema, matrix=matrix)


def _make_state(schema=None):
    if schema is None:
        schema = _make_schema()
    return State(value=np.ones(N_DOF), basis="f", schema=schema)


# ---------------------------------------------------------------------------
# TestPointingModelConstruction
# ---------------------------------------------------------------------------


class TestPointingModelConstruction:
    def test_basic(self):
        pm = _make_pm()
        assert pm.matrix.shape == (2, N_DOF)
        assert pm.schema.n_dof == N_DOF

    def test_schema_stripped_to_names_units(self):
        """__post_init__ discards Vh/use_dof — only names/units are kept."""
        full_schema = StateSchema(
            dof_names=NAMES_4,
            dof_units=UNITS_4,
            use_dof=[0, 1],
        )
        matrix = np.zeros((2, N_DOF)) * u.arcsec
        pm = PointingModel(schema=full_schema, matrix=matrix)
        assert pm.schema.Vh is None
        assert list(pm.schema.use_dof) == list(range(N_DOF))

    def test_wrong_schema_type_raises(self):
        with pytest.raises(TypeError, match="StateSchema"):
            PointingModel(schema="bad", matrix=np.zeros((2, 4)) * u.arcsec)

    def test_non_quantity_matrix_raises(self):
        with pytest.raises(TypeError, match="Quantity"):
            PointingModel(schema=_make_schema(), matrix=np.zeros((2, N_DOF)))

    def test_non_angular_matrix_raises(self):
        with pytest.raises(ValueError, match="angular units"):
            PointingModel(schema=_make_schema(), matrix=np.zeros((2, N_DOF)) * u.mm)

    def test_wrong_matrix_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            PointingModel(schema=_make_schema(), matrix=np.zeros((2, 3)) * u.arcsec)

    def test_frozen(self):
        pm = _make_pm()
        with pytest.raises(AttributeError):
            pm.matrix = np.zeros((2, N_DOF)) * u.arcsec

    def test_various_angle_units_accepted(self):
        for unit in (u.deg, u.arcmin, u.arcsec, u.rad):
            pm = _make_pm(angle_unit=unit)
            assert pm.matrix.unit == unit


# ---------------------------------------------------------------------------
# TestPointingModelAligned
# ---------------------------------------------------------------------------


class TestPointingModelAligned:
    def test_same_schema_is_noop(self):
        pm = _make_pm()
        aligned = pm.aligned(pm.schema)
        np.testing.assert_allclose(
            aligned.matrix.to_value(u.arcsec), pm.matrix.to_value(u.arcsec)
        )

    def test_reorder_dofs(self):
        """Swapping DOF order produces consistent matrix columns."""
        pm = _make_pm()
        schema2 = StateSchema(
            dof_names=("rx", "ry", "dz", "dx"),
            dof_units=(u.deg, u.deg, u.mm, u.mm),
        )
        aligned = pm.aligned(schema2)
        assert aligned.schema.dof_names == ("rx", "ry", "dz", "dx")
        # The response for 'dz' should be unchanged
        dz_idx_orig = NAMES_4.index("dz")
        dz_idx_new = ("rx", "ry", "dz", "dx").index("dz")
        np.testing.assert_allclose(
            aligned.matrix[:, dz_idx_new].to_value(u.arcsec),
            pm.matrix[:, dz_idx_orig].to_value(u.arcsec),
        )

    def test_missing_dof_strict_raises(self):
        pm = _make_pm()
        schema2 = StateSchema(
            dof_names=("dz", "dx", "rx", "ry", "extra"),
            dof_units=(u.mm, u.mm, u.deg, u.deg, u.mm),
        )
        with pytest.raises(ValueError, match="extra"):
            pm.aligned(schema2, strict=True)

    def test_missing_dof_non_strict_zeros(self):
        pm = _make_pm()
        schema2 = StateSchema(
            dof_names=("dz", "dx", "rx", "ry", "extra"),
            dof_units=(u.mm, u.mm, u.deg, u.deg, u.mm),
        )
        aligned = pm.aligned(schema2, strict=False)
        assert aligned.matrix.shape == (2, 5)
        np.testing.assert_allclose(aligned.matrix[:, 4].to_value(u.arcsec), 0.0)

    def test_unit_rescaling(self):
        """DOF in mm vs um: response column should scale by 1000."""
        pm = _make_pm()
        schema_um = StateSchema(
            dof_names=NAMES_4,
            dof_units=(u.um, u.um, u.deg, u.deg),
        )
        aligned = pm.aligned(schema_um)
        # 1 um = 0.001 mm, so response should be 0.001× what it was in mm
        np.testing.assert_allclose(
            aligned.matrix[:, 0].to_value(u.arcsec),
            pm.matrix[:, 0].to_value(u.arcsec) * 0.001,
        )

    def test_incompatible_units_raises(self):
        pm = _make_pm()
        schema_bad = StateSchema(
            dof_names=NAMES_4,
            dof_units=(u.kg, u.mm, u.deg, u.deg),
        )
        with pytest.raises(ValueError, match="Incompatible DOF units"):
            pm.aligned(schema_bad)

    def test_wrong_schema_type_raises(self):
        pm = _make_pm()
        with pytest.raises(TypeError, match="StateSchema"):
            pm.aligned("bad")


# ---------------------------------------------------------------------------
# TestPointingModelDelta
# ---------------------------------------------------------------------------


class TestPointingModelDelta:
    def test_zero_state_gives_zero_delta(self):
        pm = _make_pm()
        state = State(value=np.zeros(N_DOF), basis="f", schema=pm.schema)
        dx, dy = pm.delta(state)
        assert dx.to_value(u.rad) == pytest.approx(0.0)
        assert dy.to_value(u.rad) == pytest.approx(0.0)

    def test_delta_matches_manual_matmul(self):
        pm = _make_pm(seed=7)
        state = _make_state(pm.schema)
        dx, dy = pm.delta(state)
        expected = pm.matrix.to_value(u.rad) @ state.f.value
        np.testing.assert_allclose(dx.to_value(u.rad), expected[0], atol=1e-15)
        np.testing.assert_allclose(dy.to_value(u.rad), expected[1], atol=1e-15)

    def test_delta_with_reordered_state(self):
        pm = _make_pm()
        schema2 = StateSchema(
            dof_names=("rx", "ry", "dz", "dx"),
            dof_units=(u.deg, u.deg, u.mm, u.mm),
        )
        state2 = State(value=np.array([0.0, 0.0, 1.0, 0.0]), basis="f", schema=schema2)
        dx2, dy2 = pm.delta(state2)
        # Same as applying only the 'dz' DOF on the original schema
        state1 = State(value=np.array([1.0, 0.0, 0.0, 0.0]), basis="f", schema=pm.schema)
        dx1, dy1 = pm.delta(state1)
        np.testing.assert_allclose(dx2.to_value(u.rad), dx1.to_value(u.rad), atol=1e-15)
        np.testing.assert_allclose(dy2.to_value(u.rad), dy1.to_value(u.rad), atol=1e-15)

    def test_linearity(self):
        pm = _make_pm(seed=3)
        s1 = State(value=np.array([1.0, 0.0, 0.0, 0.0]), basis="f", schema=pm.schema)
        s2 = State(value=np.array([0.0, 1.0, 0.0, 0.0]), basis="f", schema=pm.schema)
        s12 = State(value=np.array([1.0, 1.0, 0.0, 0.0]), basis="f", schema=pm.schema)
        dx1, dy1 = pm.delta(s1)
        dx2, dy2 = pm.delta(s2)
        dx12, dy12 = pm.delta(s12)
        np.testing.assert_allclose(
            dx12.to_value(u.rad), (dx1 + dx2).to_value(u.rad), atol=1e-15
        )
        np.testing.assert_allclose(
            dy12.to_value(u.rad), (dy1 + dy2).to_value(u.rad), atol=1e-15
        )


# ---------------------------------------------------------------------------
# TestPointingModelTable
# ---------------------------------------------------------------------------


class TestPointingModelTable:
    def test_to_table_columns(self):
        pm = _make_pm()
        t = pm.to_table()
        assert set(t.colnames) >= {"dof_name", "dof_unit", "dtheta_x", "dtheta_y"}
        assert len(t) == N_DOF

    def test_to_table_dof_names(self):
        pm = _make_pm()
        t = pm.to_table()
        assert tuple(t["dof_name"]) == NAMES_4

    def test_to_table_values(self):
        pm = _make_pm()
        t = pm.to_table()
        np.testing.assert_allclose(
            t["dtheta_x"].to_value(u.arcsec), pm.matrix[0].to_value(u.arcsec)
        )
        np.testing.assert_allclose(
            t["dtheta_y"].to_value(u.arcsec), pm.matrix[1].to_value(u.arcsec)
        )

    def test_roundtrip_table(self):
        pm = _make_pm()
        rt = PointingModel.from_table(pm.to_table())
        assert rt.schema.dof_names == pm.schema.dof_names
        assert rt.schema.dof_units == pm.schema.dof_units
        np.testing.assert_allclose(
            rt.matrix.to_value(u.arcsec), pm.matrix.to_value(u.arcsec)
        )

    def test_from_table_missing_column_raises(self):
        t = QTable({"dof_name": ["a"], "dof_unit": ["mm"], "dtheta_x": [1.0] * u.arcsec})
        with pytest.raises(ValueError, match="dtheta_y"):
            PointingModel.from_table(t)

    def test_roundtrip_table_preserves_units(self):
        pm = _make_pm(angle_unit=u.deg)
        rt = PointingModel.from_table(pm.to_table())
        np.testing.assert_allclose(
            rt.matrix.to_value(u.deg), pm.matrix.to_value(u.deg)
        )


# ---------------------------------------------------------------------------
# ASDF round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("asdf", reason="asdf not installed") is None,
    reason="asdf not installed",
)
class TestPointingModelAsdf:
    asdf = pytest.importorskip("asdf")

    def test_roundtrip_basic(self):
        pm = _make_pm()
        rt = roundtrip_asdf_ctx(pm)
        assert isinstance(rt, PointingModel)
        assert rt.schema.dof_names == pm.schema.dof_names
        assert rt.schema.dof_units == pm.schema.dof_units
        np.testing.assert_allclose(
            rt.matrix.to_value(u.arcsec), pm.matrix.to_value(u.arcsec)
        )

    def test_roundtrip_preserves_angle_units(self):
        pm = _make_pm(angle_unit=u.deg)
        rt = roundtrip_asdf_ctx(pm)
        np.testing.assert_allclose(
            rt.matrix.to_value(u.deg), pm.matrix.to_value(u.deg)
        )

    def test_roundtrip_schema_embedded(self):
        pm = _make_pm()
        rt = roundtrip_asdf_ctx(pm)
        assert isinstance(rt.schema, StateSchema)
        assert rt.schema.n_dof == N_DOF

    def test_roundtrip_delta_consistent(self):
        """delta() with round-tripped model matches original."""
        pm = _make_pm(seed=5)
        rt = roundtrip_asdf_ctx(pm)
        state = _make_state(pm.schema)
        dx_orig, dy_orig = pm.delta(state)
        dx_rt, dy_rt = rt.delta(state)
        np.testing.assert_allclose(dx_rt.to_value(u.rad), dx_orig.to_value(u.rad), atol=1e-15)
        np.testing.assert_allclose(dy_rt.to_value(u.rad), dy_orig.to_value(u.rad), atol=1e-15)

    def test_roundtrip_mixed_dof_units(self):
        schema = StateSchema(
            dof_names=("Cam_dz", "Cam_dx", "Cam_rx", "Cam_ry"),
            dof_units=(u.um, u.um, u.arcsec, u.arcsec),
        )
        matrix = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]) * u.arcsec
        pm = PointingModel(schema=schema, matrix=matrix)
        rt = roundtrip_asdf_ctx(pm)
        assert rt.schema.dof_units == (u.um, u.um, u.arcsec, u.arcsec)
        np.testing.assert_allclose(
            rt.matrix.to_value(u.arcsec), pm.matrix.to_value(u.arcsec)
        )


@requires_starsharp_asdf
class TestPointingModelAsdfEntryPoint:
    def test_roundtrip(self):
        pm = _make_pm()
        rt = roundtrip_asdf(pm)
        assert isinstance(rt, PointingModel)
        np.testing.assert_allclose(
            rt.matrix.to_value(u.arcsec), pm.matrix.to_value(u.arcsec)
        )
