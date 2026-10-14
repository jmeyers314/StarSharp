"""Tests for StarSharp/models/fiducial.py — default_schema, str_to_arr, etc."""

from __future__ import annotations

import inspect

import astropy.units as u
import numpy as np
import pytest
import yaml

from StarSharp.datatypes import StateSchema
from StarSharp.models.fiducial import (
    _DEFAULT,
    _DefaultType,
    default_normalizations,
    default_schema,
)
from StarSharp.utils import str_to_arr

from .conftest import (
    requires_default_sensitivity,
    requires_ts_config_mttcs,
    ts_config_mttcs_weights,
)


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


# ---------------------------------------------------------------------------
# TestNormalizationWeights
# ---------------------------------------------------------------------------


class TestNormalizationWeights:
    # Reduced set of 20 DOF being controlled: the 10 hexapod rigid-body DOF
    # plus the first 5 bending modes of each mirror.
    ACTIVE_DOF = list(range(15)) + list(range(30, 35))

    @requires_ts_config_mttcs
    def test_ts_ofc_source_reproduces_mtaos_file(self):
        """``source='ts_ofc'`` must reproduce mtaos's weights to float precision.

        This is a regression lock on the snapshot in
        ``fwhm_ts_ofc_quadrature.yaml`` plus ``range.yaml``: together with the
        default exponents they must reconstruct the operational mtaos file
        exactly.

        NOTE the mtaos filename is a typo.  It is called
        ``range0.5_fwhm-0.15.yaml`` but genuinely uses **beta = -0.5**:

        * the file's own header says ``alpha is 0.5 and beta is -0.5``;
        * the commit that added it (ts_config_mttcs ``92575b1``) is titled
          "Add range0.5_fwhm-**0.5** weights to MTAOS";
        * solving per-DOF for beta gives exactly -0.5000;
        * and ``alpha - beta == 1`` is required dimensionally.

        If this test ever fails at ~1e-1 rather than ~1e-14, suspect that
        someone regenerated the mtaos file with different exponents or a
        different field sampling.
        """
        expected = np.asarray(
            yaml.safe_load(ts_config_mttcs_weights().read_text()), dtype=float
        )
        actual = default_normalizations(source="ts_ofc")
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    @requires_ts_config_mttcs
    def test_normalizations_match_ts_ofc_v13(self):
        """StarSharp's raytraced weights must agree with mtaos's to ~20%.

        StarSharp measures the FWHM response by raytracing and taking second
        moments of the ray-displacement difference; mtaos evaluates a
        linearized double-Zernike sensitivity and converts it with
        ``convertZernikesToPsfWidth``, linearizing a second time.  The two are
        the same physics by different routes, so they agree closely but not
        exactly -- this test pins "closely".

        Restricted to lower order DOFs.  The high-order inactive bending modes
        disagree much more (M1M3_b19 by ~7x), because ts_ofc's stored
        sensitivity gives physically implausible values there: its M1M3_b19
        FWHM response is 0.218 while both neighbours are 1.98 and 1.41, where
        StarSharp's raytrace returns a smooth 1.57.  That is a limitation of
        the stored matrix at high mode order, not a StarSharp error, so those
        DOF are deliberately excluded.
        """
        expected = np.asarray(
            yaml.safe_load(ts_config_mttcs_weights().read_text()), dtype=float
        )[self.ACTIVE_DOF]
        actual = default_normalizations()[self.ACTIVE_DOF]

        ratio = actual / expected
        assert np.all(ratio > 0.85), f"weights too small: min ratio {ratio.min()}"
        assert np.all(ratio < 1.20), f"weights too large: max ratio {ratio.max()}"

        # Shape agreement across DOF matters more than the overall scale, which
        # the SVD largely absorbs.
        corr = np.corrcoef(np.log(actual), np.log(expected))[0, 1]
        assert corr > 0.99, f"log-log correlation only {corr}"

    def test_source_starsharp_and_ts_ofc_differ(self):
        """The two sources must not be silently aliased to the same file."""
        starsharp = default_normalizations(source="starsharp")
        ts_ofc = default_normalizations(source="ts_ofc")
        assert not np.allclose(starsharp, ts_ofc)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="source must be one of"):
            default_normalizations(source="nonsense")

    def test_ts_ofc_dir_env_var_is_ignored(self, monkeypatch, tmp_path):
        """A stale $TS_OFC_DIR must not break or redirect the defaults.

        ``default_normalizations`` used to prefer
        ``$TS_OFC_DIR/policy/normalization_weights/`` whenever that directory
        existed, then fail because ts_ofc ships combined weights rather than
        the separate fwhm/range files this code wants.
        """
        monkeypatch.delenv("TS_OFC_DIR", raising=False)
        expected = default_normalizations()

        # An empty-but-present policy dir is exactly what used to trip the
        # directory-existence check and then raise on the missing fwhm.yaml.
        (tmp_path / "policy" / "normalization_weights").mkdir(parents=True)
        monkeypatch.setenv("TS_OFC_DIR", str(tmp_path))

        np.testing.assert_array_equal(default_normalizations(), expected)
        assert default_schema(n_keep=12, use_dof="0-14,30-34").Vh.shape == (12, 20)
