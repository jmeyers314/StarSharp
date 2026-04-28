"""Tests for RaytracedOpticalModel using the real LSST stack.

Uses ``default_raytraced_model()`` so all tests exercise the actual batoid
ray tracer.  The model fixture is module-scoped so the builder and camera are
loaded once per session.

Sensitivity tests use a single active DOF and a tiny field to keep runtimes
short while still exercising the real finite-difference logic.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import (
    DoubleZernikes,
    FieldCoords,
    Sensitivity,
    Spots,
    StateSchema,
    StateFactory,
    Zernikes,
)
from StarSharp.models.fiducial import default_raytraced_model
from StarSharp.models.raytraced import RaytracedOpticalModel


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> RaytracedOpticalModel:
    """Real LSST ROM, r-band, zero rotator angle."""
    return default_raytraced_model(band="r", rtp=Angle(0.0, unit=u.deg))


@pytest.fixture(scope="module")
def on_axis_field(model) -> FieldCoords:
    """Single on-axis OCS field point."""
    return FieldCoords(
        x=0.0 * u.deg,
        y=0.0 * u.deg,
        frame="ocs",
        rtp=model.rtp,
        camera=model.camera,
    )


@pytest.fixture(scope="module")
def small_field(model) -> FieldCoords:
    """Three OCS field points — fast for ray-trace calls."""
    return FieldCoords(
        x=np.array([0.0, 0.5, -0.5]) * u.deg,
        y=np.array([0.0, 0.3,  0.3]) * u.deg,
        frame="ocs",
        rtp=model.rtp,
        camera=model.camera,
    )


# ---------------------------------------------------------------------------
# Field-coordinate helpers
# ---------------------------------------------------------------------------


class TestMakeHexField:
    def test_returns_field_coords(self, model):
        fc = model.make_hex_field()
        assert isinstance(fc, FieldCoords)

    def test_frame_is_ocs(self, model):
        fc = model.make_hex_field()
        assert fc.frame == "ocs"

    def test_rtp_matches_model(self, model):
        fc = model.make_hex_field()
        assert np.isclose(fc.rtp.rad, model.rtp.rad)

    def test_camera_matches_model(self, model):
        fc = model.make_hex_field()
        assert fc.camera is model.camera

    def test_nrad_controls_npoints(self, model):
        fc5 = model.make_hex_field(nrad=5)
        fc10 = model.make_hex_field(nrad=10)
        assert len(fc10.x) > len(fc5.x)

    def test_outer_controls_extent(self, model):
        fc_small = model.make_hex_field(outer=1.0 * u.deg, nrad=5)
        fc_large = model.make_hex_field(outer=2.0 * u.deg, nrad=5)
        assert fc_large.x.max() > fc_small.x.max()


class TestMakeCcdField:
    def test_returns_field_coords(self, model):
        fc = model.make_ccd_field()
        assert isinstance(fc, FieldCoords)

    def test_frame_is_ccs(self, model):
        fc = model.make_ccd_field()
        assert fc.frame == "ccs"

    def test_camera_set(self, model):
        fc = model.make_ccd_field()
        assert fc.camera is model.camera

    def test_nx1_gives_one_point_per_detector(self, model):
        fc = model.make_ccd_field(nx=1)
        n_detectors = sum(
            1 for det in model.camera
            if det.getPhysicalType() in ("E2V", "ITL")
        )
        assert len(fc.x) == n_detectors

    def test_detnums_filters_detectors(self, model):
        fc_all = model.make_ccd_field(nx=1)
        fc_sub = model.make_ccd_field(nx=1, detnums=[0, 1, 2])
        assert len(fc_sub.x) < len(fc_all.x)


class TestMakeWfsMeanField:
    def test_returns_field_coords(self, model):
        fc = model.make_wfs_mean_field()
        assert isinstance(fc, FieldCoords)

    def test_four_wfs_pairs(self, model):
        fc = model.make_wfs_mean_field()
        assert len(fc.x) == 4

    def test_frame_is_ccs(self, model):
        fc = model.make_wfs_mean_field()
        assert fc.frame == "ccs"


# ---------------------------------------------------------------------------
# spots()
# ---------------------------------------------------------------------------


class TestSpots:
    @pytest.fixture(scope="class")
    def spots(self, model, small_field):
        return model.spots(field=small_field, nrad=3, reference="ring")

    def test_returns_spots(self, spots):
        assert isinstance(spots, Spots)

    def test_frame_is_ccs(self, spots):
        assert spots.frame == "ccs"

    def test_rtp_set(self, spots, model):
        assert np.isclose(spots.rtp.rad, model.rtp.rad)

    def test_camera_is_model_camera(self, spots, model):
        assert spots.camera is model.camera

    def test_wavelength_set(self, spots, model):
        assert spots.wavelength == model.wavelength

    def test_nfield_matches_input(self, spots, small_field):
        assert spots.nfield == len(small_field.x)

    def test_dx_dy_units_micron(self, spots):
        assert spots.dx.unit == u.micron
        assert spots.dy.unit == u.micron

    def test_field_units_mm(self, spots):
        assert spots.field.x.unit == u.mm

    def test_no_all_vignetted(self, spots):
        # On-sky science field: at least some rays should get through
        assert not np.all(spots.vignetted)

    def test_state_none_matches_zero_state(self, model, small_field):
        sf = StateFactory(model.state_schema)
        spots_none = model.spots(field=small_field, nrad=3, reference="ring",
                                 state=None)
        spots_zero = model.spots(field=small_field, nrad=3, reference="ring",
                                 state=sf.zero("f"))
        np.testing.assert_allclose(
            spots_none.dx.to_value(u.micron),
            spots_zero.dx.to_value(u.micron),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# zernikes()
# ---------------------------------------------------------------------------


class TestZernikes:
    @pytest.fixture(scope="class")
    def zk(self, model, small_field):
        return model.zernikes(field=small_field, jmax=10, rings=3)

    def test_returns_zernikes(self, zk):
        assert isinstance(zk, Zernikes)

    def test_frame_is_ocs(self, zk):
        assert zk.frame == "ocs"

    def test_rtp_matches_model(self, zk, model):
        assert np.isclose(zk.rtp.rad, model.rtp.rad)

    def test_r_outer_set(self, zk):
        assert zk.R_outer is not None

    def test_r_inner_set(self, zk):
        assert zk.R_inner is not None

    def test_jmax(self, zk):
        assert zk.jmax == 10

    def test_nfield_matches_input(self, zk, small_field):
        assert zk.nfield == len(small_field.x)

    def test_coef_units_micron(self, zk):
        assert zk.coefs.unit.is_equivalent(u.micron)

    def test_state_none_matches_zero_state(self, model, small_field):
        sf = StateFactory(model.state_schema)
        zk_none = model.zernikes(field=small_field, jmax=10, rings=3, state=None)
        zk_zero = model.zernikes(field=small_field, jmax=10, rings=3,
                                 state=sf.zero("f"))
        np.testing.assert_allclose(
            zk_none.coefs.to_value(u.um),
            zk_zero.coefs.to_value(u.um),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# zernikes_sensitivity()
# ---------------------------------------------------------------------------


class TestZernikesSensitivity:
    """Use basis='x' with a single DOF (M2_dz, index 0) and a tiny field
    so each test requires only 2 ray-trace evaluations."""

    def test_step_none_raises(self, model, small_field):
        schema_no_step = StateSchema(
            dof_names=model.state_schema.dof_names,
            dof_units=model.state_schema.dof_units,
        )
        model_no_step = RaytracedOpticalModel(
            builder=model.builder,
            rtp=model.rtp,
            wavelength=model.wavelength,
            state_schema=schema_no_step,
        )
        with pytest.raises(ValueError, match="step"):
            model_no_step.zernikes_sensitivity(
                field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
            )

    def test_basis_v_raises(self, model, small_field):
        with pytest.raises(ValueError, match="v"):
            model.zernikes_sensitivity(
                field=small_field, jmax=4, rings=3, basis="v"
            )

    def test_use_dof_without_x_raises(self, model, small_field):
        with pytest.raises(ValueError, match="use_dof"):
            model.zernikes_sensitivity(
                field=small_field, jmax=4, rings=3, basis="f", use_dof=[0]
            )

    def test_returns_sensitivity(self, model, small_field):
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
        )
        assert isinstance(sens, Sensitivity)

    def test_nominal_is_zernikes(self, model, small_field):
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
        )
        assert isinstance(sens.nominal, Zernikes)

    def test_gradient_is_zernikes(self, model, small_field):
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
        )
        assert isinstance(sens.gradient, Zernikes)

    def test_gradient_shape_x_basis_single_dof(self, model, small_field):
        nfield = len(small_field.x)
        jmax = 4
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=jmax, rings=3, basis="x", use_dof=[0]
        )
        assert sens.gradient.coefs.shape == (1, nfield, jmax + 1)

    def test_gradient_shape_f_basis_all_dof(self, model, on_axis_field):
        # f-basis: gradient first dim == n_dof; use a single field point to
        # limit trace count to n_dof + 1 = 51
        jmax = 4
        sens = model.zernikes_sensitivity(
            field=on_axis_field, jmax=jmax, rings=3, basis="f"
        )
        n_dof = model.state_schema.n_dof
        assert sens.gradient.coefs.shape == (n_dof, 1, jmax + 1)

    def test_basis_matches_schema(self, model, small_field):
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
        )
        assert sens.basis == "x"

    def test_use_dof_reflected_in_schema(self, model, small_field):
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
        )
        np.testing.assert_array_equal(sens.schema.use_dof, [0])

    def test_use_dof_override_vs_schema_default(self, model, small_field):
        """x-basis sensitivity with use_dof=[0, 1] should differ from [0] alone."""
        sens1 = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0]
        )
        sens2 = model.zernikes_sensitivity(
            field=small_field, jmax=4, rings=3, basis="x", use_dof=[0, 1]
        )
        assert sens2.gradient.coefs.shape[0] == 2
        # First DOF column should match between the two
        np.testing.assert_allclose(
            sens1.gradient.coefs.to_value(u.um),
            sens2.gradient.coefs[:1].to_value(u.um),
            atol=1e-10,
        )

    def test_m2_dz_defocus_is_nonzero(self, model, small_field):
        """M2_dz perturbation should produce a measurable defocus (Z4)."""
        sens = model.zernikes_sensitivity(
            field=small_field, jmax=10, rings=3, basis="x", use_dof=[0]
        )
        # Z4 = defocus; expect a nonzero gradient
        assert not np.allclose(sens.gradient.coefs[:, :, 4].to_value(u.um), 0.0)


# ---------------------------------------------------------------------------
# spots_sensitivity()
# ---------------------------------------------------------------------------


class TestSpotsSensitivity:
    def test_step_none_raises(self, model, small_field):
        schema_no_step = StateSchema(
            dof_names=model.state_schema.dof_names,
            dof_units=model.state_schema.dof_units,
        )
        model_no_step = RaytracedOpticalModel(
            builder=model.builder,
            rtp=model.rtp,
            wavelength=model.wavelength,
            state_schema=schema_no_step,
        )
        with pytest.raises(ValueError, match="step"):
            model_no_step.spots_sensitivity(
                field=small_field, nrad=3, basis="x", use_dof=[0]
            )

    def test_basis_v_raises(self, model, small_field):
        with pytest.raises(ValueError, match="v"):
            model.spots_sensitivity(field=small_field, nrad=3, basis="v")

    def test_returns_sensitivity(self, model, small_field):
        sens = model.spots_sensitivity(
            field=small_field, nrad=3, basis="x", use_dof=[0]
        )
        assert isinstance(sens, Sensitivity)

    def test_nominal_is_spots(self, model, small_field):
        sens = model.spots_sensitivity(
            field=small_field, nrad=3, basis="x", use_dof=[0]
        )
        assert isinstance(sens.nominal, Spots)

    def test_gradient_is_spots(self, model, small_field):
        sens = model.spots_sensitivity(
            field=small_field, nrad=3, basis="x", use_dof=[0]
        )
        assert isinstance(sens.gradient, Spots)

    def test_gradient_shape_x_basis_single_dof(self, model, small_field):
        sens = model.spots_sensitivity(
            field=small_field, nrad=3, basis="x", use_dof=[0]
        )
        # gradient shape: (ndof, nfield, nray)
        assert sens.gradient.dx.shape[0] == 1
        assert sens.gradient.dx.shape[1] == len(small_field.x)

    def test_basis_reflected_in_schema(self, model, small_field):
        sens = model.spots_sensitivity(
            field=small_field, nrad=3, basis="x", use_dof=[0]
        )
        assert sens.basis == "x"
        np.testing.assert_array_equal(sens.schema.use_dof, [0])


# ---------------------------------------------------------------------------
# double_zernikes_sensitivity()
# ---------------------------------------------------------------------------


class TestDoubleZernikesSensitivity:
    @pytest.fixture(scope="class")
    def dz_sens(self, model, small_field):
        return model.double_zernikes_sensitivity(
            field=small_field,
            kmax=4,
            field_outer=1.75 * u.deg,
            jmax=4,
            rings=3,
            basis="x",
            use_dof=[0],
        )

    def test_returns_sensitivity(self, dz_sens):
        assert isinstance(dz_sens, Sensitivity)

    def test_nominal_is_double_zernikes(self, dz_sens):
        assert isinstance(dz_sens.nominal, DoubleZernikes)

    def test_gradient_is_double_zernikes(self, dz_sens):
        assert isinstance(dz_sens.gradient, DoubleZernikes)

    def test_gradient_coefs_shape(self, dz_sens):
        kmax = 4
        jmax = 4
        assert dz_sens.gradient.coefs.shape == (1, kmax + 1, jmax + 1)

    def test_nominal_coefs_shape(self, dz_sens):
        kmax = 4
        jmax = 4
        assert dz_sens.nominal.coefs.shape == (kmax + 1, jmax + 1)

    def test_basis_x(self, dz_sens):
        assert dz_sens.basis == "x"

    def test_use_dof_reflected(self, dz_sens):
        np.testing.assert_array_equal(dz_sens.schema.use_dof, [0])

    def test_consistent_with_direct_double(self, model, small_field):
        """double_zernikes_sensitivity gradient should equal
        zernikes_sensitivity.gradient.double() directly."""
        kmax = 4
        field_outer = 1.75 * u.deg
        jmax = 4

        zk_sens = model.zernikes_sensitivity(
            field=small_field, jmax=jmax, rings=3, basis="x", use_dof=[0]
        )
        dz_sens = model.double_zernikes_sensitivity(
            field=small_field, kmax=kmax, field_outer=field_outer,
            jmax=jmax, rings=3, basis="x", use_dof=[0],
        )

        expected = zk_sens.gradient.double(kmax, field_outer)
        np.testing.assert_allclose(
            dz_sens.gradient.coefs.to_value(u.um),
            expected.coefs.to_value(u.um),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# optimize()
# ---------------------------------------------------------------------------

# Cam_dz is index 5 in the 50-DOF schema.
_CAM_DZ_IDX = 5
_CAM_DZ_OFFSET_UM = 50.0  # applied as a known defocus


def _spot_rms(spots: Spots) -> float:
    """RMS spot radius across all unvignetted rays and field points (micron)."""
    w = ~spots.vignetted
    dx = spots.dx.to_value(u.micron)[w]
    dy = spots.dy.to_value(u.micron)[w]
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


class TestOptimize:
    """Verify ROM.optimize() recovers a known Cam_dz defocus in both modes.

    A model with a 50 um Cam_dz offset is built once per class.  An x-basis
    guess with only Cam_dz free is passed to optimize(); we check that the
    result cancels the offset and shrinks the spot RMS.
    """

    @pytest.fixture(scope="class")
    def defocused_model(self, model):
        """ROM with a 50 um Cam_dz offset baked into the offset State."""
        sf = StateFactory(model.state_schema)
        dof_val = np.zeros(model.state_schema.n_dof)
        dof_val[_CAM_DZ_IDX] = _CAM_DZ_OFFSET_UM
        return RaytracedOpticalModel(
            builder=model.builder,
            rtp=model.rtp,
            wavelength=model.wavelength,
            state_schema=model.state_schema,
            camera=model.camera,
            offset=sf.f(dof_val),
        )

    @pytest.fixture(scope="class")
    def cam_dz_schema(self, model):
        """Schema with only Cam_dz active."""
        from dataclasses import replace
        return replace(model.state_schema, use_dof=np.array([_CAM_DZ_IDX]), Vh=None)

    @pytest.fixture(scope="class")
    def guess(self, cam_dz_schema):
        """Zero x-basis guess with Cam_dz as the single free DOF."""
        return StateFactory(cam_dz_schema).zero("x")

    @pytest.fixture(scope="class")
    def result_dx(self, defocused_model, guess, small_field):
        return defocused_model.optimize(
            guess=guess, field=small_field, nrad=5, mode="dx"
        )

    @pytest.fixture(scope="class")
    def result_var(self, defocused_model, guess, small_field):
        return defocused_model.optimize(
            guess=guess, field=small_field, nrad=5, mode="var"
        )

    # --- return type ---

    def test_returns_state_dx(self, result_dx):
        from StarSharp.datatypes import State
        assert isinstance(result_dx, State)

    def test_returns_state_var(self, result_var):
        from StarSharp.datatypes import State
        assert isinstance(result_var, State)

    def test_result_basis_is_x(self, result_dx, result_var):
        assert result_dx.basis == "x"
        assert result_var.basis == "x"

    # --- recovery of offset ---

    def test_recovers_cam_dz_dx_mode(self, result_dx):
        """Optimized Cam_dz should be close to -offset (≈ -50 um)."""
        cam_dz = result_dx.x.value[0]
        assert abs(cam_dz - (-_CAM_DZ_OFFSET_UM)) < 5.0

    def test_recovers_cam_dz_var_mode(self, result_var):
        cam_dz = result_var.x.value[0]
        assert abs(cam_dz - (-_CAM_DZ_OFFSET_UM)) < 5.0

    # --- spot improvement ---

    def test_reduces_spot_rms_dx_mode(self, defocused_model, guess, result_dx,
                                      small_field):
        rms_before = _spot_rms(defocused_model.spots(
            field=small_field, state=guess, nrad=5
        ))
        rms_after = _spot_rms(defocused_model.spots(
            field=small_field, state=result_dx, nrad=5
        ))
        assert rms_after < rms_before

    def test_reduces_spot_rms_var_mode(self, defocused_model, guess, result_var,
                                       small_field):
        rms_before = _spot_rms(defocused_model.spots(
            field=small_field, state=guess, nrad=5
        ))
        rms_after = _spot_rms(defocused_model.spots(
            field=small_field, state=result_var, nrad=5
        ))
        assert rms_after < rms_before
