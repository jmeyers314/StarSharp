"""Tests for StarSharp/models/fiducial.py — default_schema, str_to_arr, etc."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from StarSharp.datatypes import StateSchema
from StarSharp.models.fiducial import (
    _DEFAULT,
    _DefaultType,
    default_schema,
)
from StarSharp.utils import str_to_arr

from .conftest import requires_default_sensitivity


# ---------------------------------------------------------------------------
# TestStrToArr
# ---------------------------------------------------------------------------


class TestStrToArr:
    def test_list_input(self):
        result = str_to_arr([0, 1, 2])
        np.testing.assert_array_equal(result, [0, 1, 2])
        assert result.dtype == int

    def test_array_input(self):
        arr = np.array([3, 5, 7])
        result = str_to_arr(arr)
        np.testing.assert_array_equal(result, arr)

    def test_simple_indices(self):
        np.testing.assert_array_equal(str_to_arr("0,1,2"), [0, 1, 2])

    def test_range(self):
        np.testing.assert_array_equal(str_to_arr("0-4"), [0, 1, 2, 3, 4])

    def test_mixed(self):
        np.testing.assert_array_equal(str_to_arr("0-2,5,7-9"), [0, 1, 2, 5, 7, 8, 9])

    def test_single(self):
        np.testing.assert_array_equal(str_to_arr("42"), [42])

    def test_whitespace_around_parts(self):
        np.testing.assert_array_equal(str_to_arr("0, 2, 4"), [0, 2, 4])


# ---------------------------------------------------------------------------
# TestDefaultType
# ---------------------------------------------------------------------------


class TestDefaultType:
    def test_singleton(self):
        assert _DefaultType() is _DefaultType()

    def test_default_is_instance(self):
        assert isinstance(_DEFAULT, _DefaultType)

    def test_is_check(self):
        assert _DEFAULT is _DEFAULT
        assert _DEFAULT is not None
        assert _DEFAULT is not object()


# ---------------------------------------------------------------------------
# TestDefaultSchemaBasic
# ---------------------------------------------------------------------------


class TestDefaultSchemaBasic:
    def test_returns_state_schema(self):
        assert isinstance(default_schema(), StateSchema)

    def test_fifty_dofs(self):
        schema = default_schema()
        assert schema.n_dof == 50

    def test_all_dofs_active_by_default(self):
        schema = default_schema()
        assert schema.n_active == 50
        np.testing.assert_array_equal(schema.use_dof, np.arange(50))

    def test_no_vh_by_default(self):
        assert default_schema().Vh is None

    def test_use_dof_list(self):
        schema = default_schema(use_dof=[0, 1, 2, 3, 4])
        assert schema.n_active == 5
        np.testing.assert_array_equal(schema.use_dof, [0, 1, 2, 3, 4])

    def test_use_dof_string(self):
        schema_list = default_schema(use_dof=[0, 1, 2, 3, 4])
        schema_str = default_schema(use_dof="0-4")
        np.testing.assert_array_equal(schema_list.use_dof, schema_str.use_dof)

    def test_use_dof_complex_string(self):
        schema = default_schema(use_dof="0-9,10-14,30-36")
        assert schema.n_active == 10 + 5 + 7

    def test_dof_names_well_known(self):
        schema = default_schema()
        assert schema.dof_names[0] == "M2_dz"
        assert schema.dof_names[5] == "Cam_dz"
        assert schema.dof_names[10] == "M1M3_b1"
        assert schema.dof_names[30] == "M2_b1"

    def test_rotation_dofs_in_deg(self):
        schema = default_schema()
        for i in (3, 4, 8, 9):  # M2_rx, M2_ry, Cam_rx, Cam_ry
            assert schema.dof_units[i] == u.deg

    def test_translation_dofs_in_um(self):
        schema = default_schema()
        for i in (0, 1, 2, 5, 6, 7):
            assert schema.dof_units[i] == u.um

    def test_steps_set(self):
        schema = default_schema()
        assert schema.step is not None
        assert len(schema.step) == 50


# ---------------------------------------------------------------------------
# TestDefaultSchemaSvd  (requires bundled sensitivity)
# ---------------------------------------------------------------------------


@requires_default_sensitivity
class TestDefaultSchemaSvd:
    def test_vh_set_when_n_keep_given(self):
        schema = default_schema(n_keep=10)
        assert schema.Vh is not None
        assert schema.n_keep == 10

    def test_vh_shape_full_dofs(self):
        n_keep = 8
        schema = default_schema(n_keep=n_keep)
        assert schema.Vh.shape == (n_keep, 50)  # (n_keep, n_active)

    def test_vh_shape_with_use_dof(self):
        n_keep = 5
        schema = default_schema(use_dof="0-9", n_keep=n_keep)
        assert schema.Vh.shape == (n_keep, 10)

    def test_n_keep_le_n_active(self):
        schema = default_schema(use_dof=[0, 1, 2], n_keep=2)
        assert schema.n_keep == 2
        assert schema.Vh.shape == (2, 3)

    def test_custom_exponents_accepted(self):
        # Just check it runs and returns a schema; numerical correctness
        # is tested in test_state.py / test_sensitivity.py.
        schema = default_schema(n_keep=5, fwhm_exponent=1.0, range_exponent=0.0)
        assert schema.n_keep == 5


# ---------------------------------------------------------------------------
# TestDefaultRaytracedModelArgs
# ---------------------------------------------------------------------------


class TestDefaultRaytracedModelArgs:
    """Lightweight checks on the new sentinel args — no ray tracing needed."""

    def test_pointing_model_none_accepted(self):
        from StarSharp.models.fiducial import default_raytraced_model
        model = default_raytraced_model(pointing_model=None, rtp_lookup=None)
        assert model.pointing_model is None

    def test_rtp_lookup_none_accepted(self):
        from StarSharp.models.fiducial import default_raytraced_model
        model = default_raytraced_model(pointing_model=None, rtp_lookup=None)
        assert model.rtp_lookup is None

    def test_explicit_pointing_model_used(self):
        from StarSharp.models.fiducial import default_raytraced_model, default_pointing_model
        pm = default_pointing_model()
        model = default_raytraced_model(pointing_model=pm, rtp_lookup=None)
        # __init__ calls pm.aligned() which returns a new object, so check value not identity
        assert model.pointing_model is not None
        np.testing.assert_allclose(
            model.pointing_model.matrix.to_value(u.arcsec),
            pm.matrix.to_value(u.arcsec),
        )

    def test_state_schema_and_use_dof_raises(self):
        from StarSharp.models.fiducial import default_raytraced_model
        schema = default_schema()
        with pytest.raises(ValueError, match="use_dof"):
            default_raytraced_model(
                state_schema=schema,
                use_dof=[0, 1],
                pointing_model=None,
                rtp_lookup=None,
            )

    def test_default_sentinel_is_default(self):
        from StarSharp.models.fiducial import default_raytraced_model
        import inspect
        sig = inspect.signature(default_raytraced_model)
        assert sig.parameters["pointing_model"].default is _DEFAULT
        assert sig.parameters["rtp_lookup"].default is _DEFAULT

    def test_standard_args(self):
        from StarSharp.models.fiducial import default_raytraced_model
        model = default_raytraced_model(
            use_dof="0-9,10-16,30-34",
            n_keep=12,
        )
        assert model.state_schema is not None
        assert np.array_equal(model.state_schema.use_dof, str_to_arr("0-9,10-16,30-34"))
        assert model.state_schema.Vh.shape == (12, 10 + 7 + 5)
        assert model.pointing_model is not None
        assert model.rtp_lookup is not None
        assert np.array_equal(model.sf.zero(basis="v").value, np.zeros(12))