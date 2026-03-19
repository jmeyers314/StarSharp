"""Tests for the Sensitivity class."""

from __future__ import annotations
from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import DoubleZernikes, Sensitivity, Spots, State, Zernikes

from .utils import RTP, _make_field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m

NDOF = 50
NUSEDOF = 4
NFIELD = 5
JMAX = 10
KMAX = 6
NRAY = 20


def _make_zernikes_nominal(rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    coefs = rng.standard_normal((NFIELD, JMAX + 1)) * u.um
    field = _make_field(NFIELD, rtp=RTP)
    return Zernikes(
        coefs=coefs,
        field=field,
        R_outer=PUPIL_OUTER,
        R_inner=PUPIL_INNER,
        frame="ocs",
        rtp=RTP,
    )


def _make_zernikes_perturbed(nominal, rng, step):
    """Simulate a perturbation by adding a random delta scaled by step."""
    delta = rng.standard_normal(nominal.coefs.shape)
    return Zernikes(
        coefs=nominal.coefs + delta * step * u.um,
        field=nominal.field,
        R_outer=nominal.R_outer,
        R_inner=nominal.R_inner,
        frame=nominal.frame,
        rtp=nominal.rtp,
    )


def _make_steps():
    return State(
        value=np.array([1.0, 2.0, 0.5, 3.0]),
        basis="x",
        use_dof=np.array([0, 1, 2, 3]),
        n_dof=50,
    )


def _make_zk_sensitivity():
    rng = np.random.default_rng(99)
    nominal = _make_zernikes_nominal(rng)
    steps = _make_steps()
    perturbed = [
        _make_zernikes_perturbed(nominal, np.random.default_rng(i), s)
        for i, s in enumerate(steps.value)
    ]
    return Sensitivity.from_finite_differences(nominal, perturbed, steps), perturbed, steps


def _make_spots_nominal(rng=None):
    if rng is None:
        rng = np.random.default_rng(55)
    dx = rng.standard_normal((NFIELD, NRAY)) * u.um
    dy = rng.standard_normal((NFIELD, NRAY)) * u.um
    vig = np.zeros((NFIELD, NRAY), dtype=bool)
    field = _make_field(NFIELD, rtp=RTP)
    return Spots(
        dx=dx,
        dy=dy,
        vignetted=vig,
        field=field,
        wavelength=620.0 * u.nm,
        frame="ccs",
        rtp=RTP,
    )


def _make_spots_perturbed(nominal, rng, step):
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


def _make_spots_sensitivity():
    rng = np.random.default_rng(88)
    nominal = _make_spots_nominal(rng)
    steps = _make_steps()
    perturbed = [
        _make_spots_perturbed(nominal, np.random.default_rng(100 + i), s)
        for i, s in enumerate(steps.value)
    ]
    return Sensitivity.from_finite_differences(nominal, perturbed, steps), perturbed, steps


def _make_dz_nominal(rng=None):
    if rng is None:
        rng = np.random.default_rng(77)
    coefs = rng.standard_normal((KMAX + 1, JMAX + 1)) * u.um
    return DoubleZernikes(
        coefs=coefs,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        pupil_outer=PUPIL_OUTER,
        pupil_inner=PUPIL_INNER,
        frame="ocs",
        rtp=RTP,
    )


def _make_dz_perturbed(nominal, rng, step):
    delta = rng.standard_normal(nominal.coefs.shape)
    return DoubleZernikes(
        coefs=nominal.coefs + delta * step * u.um,
        field_outer=nominal.field_outer,
        field_inner=nominal.field_inner,
        pupil_outer=nominal.pupil_outer,
        pupil_inner=nominal.pupil_inner,
        frame=nominal.frame,
        rtp=nominal.rtp,
    )


def _make_dz_sensitivity():
    rng = np.random.default_rng(66)
    nominal = _make_dz_nominal(rng)
    steps = _make_steps()
    perturbed = [
        _make_dz_perturbed(nominal, np.random.default_rng(200 + i), s)
        for i, s in enumerate(steps.value)
    ]
    return Sensitivity.from_finite_differences(nominal, perturbed, steps), perturbed, steps


# ---------------------------------------------------------------------------
# Tests: __class_getitem__
# ---------------------------------------------------------------------------


class TestClassGetitem:
    def test_type_hint_sugar(self):
        assert Sensitivity[Zernikes] is Sensitivity
        assert Sensitivity[Spots] is Sensitivity
        assert Sensitivity[DoubleZernikes] is Sensitivity


# ---------------------------------------------------------------------------
# Tests: from_finite_differences  —  Zernikes
# ---------------------------------------------------------------------------


class TestFromFiniteDifferencesZernikes:
    def test_gradient_shape(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.gradient.coefs.shape == (NUSEDOF, NFIELD, JMAX + 1)

    def test_nominal_preserved(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.nominal.coefs.shape == (NFIELD, JMAX + 1)

    def test_gradient_values(self):
        sens, perturbed, steps = _make_zk_sensitivity()
        for i in range(NUSEDOF):
            expected = (perturbed[i].coefs - sens.nominal.coefs) / steps.value[i]
            np.testing.assert_allclose(
                sens.gradient.coefs[i].to_value(u.um),
                expected.to_value(u.um),
            )


# ---------------------------------------------------------------------------
# Tests: from_finite_differences  —  Spots
# ---------------------------------------------------------------------------


class TestFromFiniteDifferencesSpots:
    def test_gradient_shape(self):
        sens, *_ = _make_spots_sensitivity()
        assert sens.gradient.dx.shape == (NUSEDOF, NFIELD, NRAY)
        assert sens.gradient.dy.shape == (NUSEDOF, NFIELD, NRAY)
        assert sens.gradient.vignetted.shape == (NUSEDOF, NFIELD, NRAY)

    def test_gradient_dx_values(self):
        sens, perturbed, steps = _make_spots_sensitivity()
        for i in range(NUSEDOF):
            expected = (perturbed[i].dx - sens.nominal.dx) / steps.value[i]
            np.testing.assert_allclose(
                sens.gradient.dx[i].to_value(u.um),
                expected.to_value(u.um),
            )

    def test_vignetted_broadcast(self):
        sens, *_ = _make_spots_sensitivity()
        for i in range(NUSEDOF):
            np.testing.assert_array_equal(
                sens.gradient.vignetted[i], sens.nominal.vignetted
            )


# ---------------------------------------------------------------------------
# Tests: from_finite_differences  —  DoubleZernikes
# ---------------------------------------------------------------------------


class TestFromFiniteDifferencesDZ:
    def test_gradient_shape(self):
        sens, *_ = _make_dz_sensitivity()
        assert sens.gradient.coefs.shape == (NUSEDOF, KMAX + 1, JMAX + 1)

    def test_gradient_values(self):
        sens, perturbed, steps = _make_dz_sensitivity()
        for i in range(NUSEDOF):
            expected = (perturbed[i].coefs - sens.nominal.coefs) / steps.value[i]
            np.testing.assert_allclose(
                sens.gradient.coefs[i].to_value(u.um),
                expected.to_value(u.um),
            )


# ---------------------------------------------------------------------------
# Tests: properties and indexing
# ---------------------------------------------------------------------------


class TestProperties:
    def test_n_dof(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.n_dof == NDOF

    def test_getitem_zernikes(self):
        sens, *_ = _make_zk_sensitivity()
        z0 = sens[0]
        assert isinstance(z0, Zernikes)
        assert z0.coefs.shape == (NFIELD, JMAX + 1)
        np.testing.assert_allclose(
            z0.coefs.to_value(u.um),
            sens.gradient.coefs[0].to_value(u.um),
        )

    def test_getitem_spots(self):
        sens, *_ = _make_spots_sensitivity()
        s0 = sens[0]
        assert isinstance(s0, Spots)
        assert s0.dx.shape == (NFIELD, NRAY)

    def test_getitem_dz(self):
        sens, *_ = _make_dz_sensitivity()
        dz0 = sens[0]
        assert isinstance(dz0, DoubleZernikes)
        assert dz0.coefs.shape == (KMAX + 1, JMAX + 1)

    def test_slice(self):
        sens, *_ = _make_zk_sensitivity()
        sliced = sens[:2]
        assert isinstance(sliced, Zernikes)
        assert sliced.coefs.shape == (2, NFIELD, JMAX + 1)


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_zernikes(self):
        sens, *_ = _make_zk_sensitivity()
        delta = State(
            value=np.array([0.1, -0.2, 0.3, 0.0]),
            basis="x",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert isinstance(result, Zernikes)
        assert result.coefs.shape == (NFIELD, JMAX + 1)

        # Manual check: nominal + gradient @ weights
        weights = delta.x.value
        expected = sens.nominal.coefs.value + np.einsum(
            "i...,i->...", sens.gradient.coefs.value, weights
        )
        np.testing.assert_allclose(result.coefs.to_value(u.um), expected, rtol=1e-12)

    def test_predict_spots(self):
        sens, *_ = _make_spots_sensitivity()
        delta = State(
            value=np.array([0.5, -0.5, 1.0, -1.0]),
            basis="x",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert isinstance(result, Spots)

        weights = delta.x.value
        expected_dx = sens.nominal.dx.value + np.einsum(
            "i...,i->...", sens.gradient.dx.value, weights
        )
        np.testing.assert_allclose(result.dx.to_value(u.um), expected_dx, rtol=1e-12)

    def test_predict_double_zernikes(self):
        sens, *_ = _make_dz_sensitivity()
        delta = State(
            value=np.array([1.0, 2.0, 3.0, 4.0]),
            basis="x",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert isinstance(result, DoubleZernikes)

        weights = delta.x.value
        expected = sens.nominal.coefs.value + np.einsum(
            "i...,i->...", sens.gradient.coefs.value, weights
        )
        np.testing.assert_allclose(result.coefs.to_value(u.um), expected, rtol=1e-12)

    def test_predict_zero_is_nominal(self):
        sens, *_ = _make_zk_sensitivity()
        zero = State(
            value=np.zeros(NUSEDOF),
            basis="x",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=NDOF,
        )
        result = sens.predict(zero)
        np.testing.assert_allclose(
            result.coefs.to_value(u.um),
            sens.nominal.coefs.to_value(u.um),
            rtol=1e-12,
        )

    def test_predict_preserves_metadata(self):
        sens, *_ = _make_zk_sensitivity()
        delta = State(
            value=np.ones(NDOF),
            basis="f",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert result.jmax == sens.nominal.jmax
        assert result.R_outer == sens.nominal.R_outer
        assert result.R_inner == sens.nominal.R_inner
        assert result.frame == sens.nominal.frame


# ---------------------------------------------------------------------------
# Tests: gradient-only construction (nominal=None)
# ---------------------------------------------------------------------------


class TestGradientOnlyConstruction:
    def test_zernikes_nominal_is_zeros(self):
        sens, *_ = _make_zk_sensitivity()
        grad_only = Sensitivity(gradient=sens.gradient)
        np.testing.assert_array_equal(
            grad_only.nominal.coefs.to_value(u.um),
            np.zeros((NFIELD, JMAX + 1)),
        )

    def test_zernikes_nominal_shape_matches_gradient_slice(self):
        sens, *_ = _make_zk_sensitivity()
        grad_only = Sensitivity(gradient=sens.gradient)
        assert grad_only.nominal.coefs.shape == sens.gradient.coefs.shape[1:]

    def test_zernikes_nominal_preserves_metadata(self):
        """Non-sensitivity fields (frame, rtp, field, etc.) come from gradient[0]."""
        sens, *_ = _make_zk_sensitivity()
        grad_only = Sensitivity(gradient=sens.gradient)
        assert grad_only.nominal.frame == sens.gradient[0].frame
        assert grad_only.nominal.R_outer == sens.gradient[0].R_outer

    def test_dz_nominal_is_zeros(self):
        sens, *_ = _make_dz_sensitivity()
        grad_only = Sensitivity(gradient=sens.gradient)
        np.testing.assert_array_equal(
            grad_only.nominal.coefs.to_value(u.um),
            np.zeros((KMAX + 1, JMAX + 1)),
        )

    def test_spots_nominal_is_zeros(self):
        sens, *_ = _make_spots_sensitivity()
        grad_only = Sensitivity(gradient=sens.gradient)
        np.testing.assert_array_equal(
            grad_only.nominal.dx.to_value(u.um),
            np.zeros((NFIELD, NRAY)),
        )
        np.testing.assert_array_equal(
            grad_only.nominal.dy.to_value(u.um),
            np.zeros((NFIELD, NRAY)),
        )

    def test_spots_nominal_vignetted_copied_from_gradient(self):
        """vignetted is a broadcast field — should be copied not zeroed."""
        rng = np.random.default_rng(42)
        nominal = _make_spots_nominal(rng)
        # Mark some rays as vignetted
        vig = nominal.vignetted.copy()
        vig[0, 0] = True
        vig[2, 5] = True
        nominal = replace(nominal, vignetted=vig)
        steps = _make_steps()
        perturbed = [
            _make_spots_perturbed(nominal, np.random.default_rng(100 + i), s)
            for i, s in enumerate(steps.value)
        ]
        sens = Sensitivity.from_finite_differences(nominal, perturbed, steps)

        grad_only = Sensitivity(gradient=sens.gradient)
        # gradient[0].vignetted is the original mask, broadcast
        np.testing.assert_array_equal(grad_only.nominal.vignetted, vig)


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_zernikes(self):
        sens, *_ = _make_zk_sensitivity()
        r = repr(sens)
        assert "Sensitivity[Zernikes]" in r

    def test_repr_spots(self):
        sens, *_ = _make_spots_sensitivity()
        r = repr(sens)
        assert "Sensitivity[Spots]" in r

    def test_repr_dz(self):
        sens, *_ = _make_dz_sensitivity()
        r = repr(sens)
        assert "Sensitivity[DoubleZernikes]" in r


# ---------------------------------------------------------------------------
# Tests: frame conversions
# ---------------------------------------------------------------------------


class TestFrameConversions:
    def test_zernikes_ocs_from_ccs(self):
        """Sensitivity[Zernikes] CCS → OCS round-trips the frame attribute."""
        sens_ocs, _, steps = _make_zk_sensitivity()
        # Manually create a CCS version via the underlying type's conversion
        sens_ccs = replace(
            sens_ocs,
            gradient=sens_ocs.gradient.ccs,
            nominal=sens_ocs.nominal.ccs,
        )
        assert sens_ccs.gradient.frame == "ccs"
        assert sens_ccs.nominal.frame == "ccs"

        back = sens_ccs.ocs
        assert back.gradient.frame == "ocs"
        assert back.nominal.frame == "ocs"

    def test_zernikes_ccs_from_ocs(self):
        sens_ocs, *_ = _make_zk_sensitivity()
        sens_ccs = sens_ocs.ccs
        assert sens_ccs.gradient.frame == "ccs"
        assert sens_ccs.nominal.frame == "ccs"

    def test_zernikes_ocs_is_identity_when_already_ocs(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.gradient.frame == "ocs"
        same = sens.ocs
        assert same.gradient is sens.gradient
        assert same.nominal is sens.nominal

    def test_zernikes_ccs_ocs_roundtrip_coefs(self):
        """OCS → CCS → OCS should recover the original coefficients."""
        sens, *_ = _make_zk_sensitivity()
        roundtripped = sens.ccs.ocs
        np.testing.assert_allclose(
            roundtripped.gradient.coefs.to_value(u.um),
            sens.gradient.coefs.to_value(u.um),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            roundtripped.nominal.coefs.to_value(u.um),
            sens.nominal.coefs.to_value(u.um),
            atol=1e-12,
        )

    def test_dz_ocs_ccs_roundtrip(self):
        sens, *_ = _make_dz_sensitivity()
        assert sens.gradient.frame == "ocs"
        roundtripped = sens.ccs.ocs
        np.testing.assert_allclose(
            roundtripped.gradient.coefs.to_value(u.um),
            sens.gradient.coefs.to_value(u.um),
            atol=1e-12,
        )

    def test_spots_ccs_ocs_roundtrip(self):
        sens, *_ = _make_spots_sensitivity()
        # Spots from_finite_differences yields CCS frame
        assert sens.gradient.frame == "ccs"
        roundtripped = sens.ocs.ccs
        np.testing.assert_allclose(
            roundtripped.gradient.dx.to_value(u.um),
            sens.gradient.dx.to_value(u.um),
            atol=1e-12,
        )

    def test_spots_dvcs(self):
        sens, *_ = _make_spots_sensitivity()
        dvcs = sens.dvcs
        assert dvcs.gradient.frame == "dvcs"
        assert dvcs.nominal.frame == "dvcs"
        # In DVCS, dx and dy are swapped relative to CCS
        np.testing.assert_allclose(
            dvcs.gradient.dx.to_value(u.um),
            sens.gradient.dy.to_value(u.um),
            atol=1e-12,
        )

    def test_spots_edcs(self):
        sens, *_ = _make_spots_sensitivity()
        edcs = sens.edcs
        assert edcs.gradient.frame == "edcs"
        assert edcs.nominal.frame == "edcs"


# ---------------------------------------------------------------------------
# Helpers for basis-conversion tests
# ---------------------------------------------------------------------------

NKEEP = 3


def _make_Vh_rect():
    """Rectangular orthogonal Vh of shape (NKEEP, NUSEDOF)."""
    rng = np.random.default_rng(123)
    A = rng.standard_normal((NUSEDOF, NUSEDOF))
    Q, _ = np.linalg.qr(A)
    return Q[:NKEEP]  # (NKEEP, NUSEDOF)


def _make_Vh_square():
    """Square orthogonal Vh of shape (NUSEDOF, NUSEDOF)."""
    rng = np.random.default_rng(123)
    A = rng.standard_normal((NUSEDOF, NUSEDOF))
    Q, _ = np.linalg.qr(A)
    return Q  # (NUSEDOF, NUSEDOF)


# ---------------------------------------------------------------------------
# Tests: basis conversions
# ---------------------------------------------------------------------------


class TestBasisConversions:
    # --- x unchanged ---

    def test_x_identity(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.x is sens

    # --- x → f ---

    def test_x_to_f_gradient_shape_zk(self):
        sens, *_ = _make_zk_sensitivity()
        sf = sens.f
        assert sf.basis == "f"
        assert sf.gradient.coefs.shape == (NDOF, NFIELD, JMAX + 1)

    def test_x_to_f_values_at_use_dof(self):
        """Rows at use_dof in f-basis match the original x-basis rows."""
        sens, *_ = _make_zk_sensitivity()
        sf = sens.f
        use_dof = sens.use_dof
        np.testing.assert_allclose(
            sf.gradient.coefs[use_dof].to_value(u.um),
            sens.gradient.coefs.to_value(u.um),
        )

    def test_x_to_f_zeros_outside_use_dof(self):
        """All rows outside use_dof are zero in f-basis."""
        sens, *_ = _make_zk_sensitivity()
        sf = sens.f
        mask = np.ones(NDOF, dtype=bool)
        mask[sens.use_dof] = False
        np.testing.assert_array_equal(
            sf.gradient.coefs[mask].to_value(u.um),
            np.zeros((NDOF - NUSEDOF, NFIELD, JMAX + 1)),
        )

    def test_x_to_f_x_roundtrip_zk(self):
        """x → f → x recovers the original gradient exactly."""
        sens, *_ = _make_zk_sensitivity()
        np.testing.assert_allclose(
            sens.f.x.gradient.coefs.to_value(u.um),
            sens.gradient.coefs.to_value(u.um),
        )

    def test_x_to_f_basis_attr(self):
        sens, *_ = _make_zk_sensitivity()
        assert sens.f.basis == "f"
        assert sens.f.x.basis == "x"

    def test_x_to_f_nominal_unchanged(self):
        """Nominal is not modified by basis conversion."""
        sens, *_ = _make_zk_sensitivity()
        np.testing.assert_allclose(
            sens.f.nominal.coefs.to_value(u.um),
            sens.nominal.coefs.to_value(u.um),
        )

    # --- x → f for Spots (vignetted re-broadcast) ---

    def test_x_to_f_spots_gradient_shape(self):
        sens, *_ = _make_spots_sensitivity()
        sf = sens.f
        assert sf.gradient.dx.shape == (NDOF, NFIELD, NRAY)
        assert sf.gradient.dy.shape == (NDOF, NFIELD, NRAY)

    def test_x_to_f_spots_vignetted_shape(self):
        sens, *_ = _make_spots_sensitivity()
        assert sens.f.gradient.vignetted.shape == (NDOF, NFIELD, NRAY)

    def test_x_to_f_spots_vignetted_values(self):
        """All rows of vignetted in f-basis equal nominal.vignetted."""
        sens, *_ = _make_spots_sensitivity()
        sf = sens.f
        for i in range(NDOF):
            np.testing.assert_array_equal(
                sf.gradient.vignetted[i], sens.nominal.vignetted
            )

    def test_x_to_f_x_roundtrip_spots(self):
        sens, *_ = _make_spots_sensitivity()
        np.testing.assert_allclose(
            sens.f.x.gradient.dx.to_value(u.um),
            sens.gradient.dx.to_value(u.um),
        )
        np.testing.assert_allclose(
            sens.f.x.gradient.dy.to_value(u.um),
            sens.gradient.dy.to_value(u.um),
        )

    # --- x → v ---

    def test_x_to_v_gradient_shape_zk(self):
        sens, *_ = _make_zk_sensitivity()
        Vh = _make_Vh_rect()
        sv = replace(sens, Vh=Vh).v
        assert sv.basis == "v"
        assert sv.gradient.coefs.shape == (NKEEP, NFIELD, JMAX + 1)

    def test_x_to_v_values_zk(self):
        """Gradient in v-basis equals Vh @ x-gradient along DOF axis."""
        sens, *_ = _make_zk_sensitivity()
        Vh = _make_Vh_rect()
        sv = replace(sens, Vh=Vh).v
        expected = np.einsum("ij,j...->i...", Vh, sens.gradient.coefs.to_value(u.um))
        np.testing.assert_allclose(
            sv.gradient.coefs.to_value(u.um), expected, rtol=1e-12
        )

    def test_x_v_x_roundtrip_square_Vh_zk(self):
        """With a square orthogonal Vh, x → v → x is exact."""
        sens, *_ = _make_zk_sensitivity()
        Vh = _make_Vh_square()
        roundtripped = replace(sens, Vh=Vh).v.x
        np.testing.assert_allclose(
            roundtripped.gradient.coefs.to_value(u.um),
            sens.gradient.coefs.to_value(u.um),
            rtol=1e-12,
        )

    def test_v_to_x_gradient_shape(self):
        """Starting from v-basis, .x gives shape (NUSEDOF, *obs)."""
        sens, *_ = _make_zk_sensitivity()
        Vh = _make_Vh_rect()
        sv = replace(sens, Vh=Vh).v  # shape (NKEEP, NFIELD, JMAX+1)
        sx = sv.x
        assert sx.basis == "x"
        assert sx.gradient.coefs.shape == (NUSEDOF, NFIELD, JMAX + 1)

    def test_x_to_v_spots_vignetted_shape(self):
        """Spots vignetted is correctly re-broadcast when converting to v."""
        sens, *_ = _make_spots_sensitivity()
        Vh = _make_Vh_rect()
        sv = replace(sens, Vh=Vh).v
        assert sv.gradient.vignetted.shape == (NKEEP, NFIELD, NRAY)
        for i in range(NKEEP):
            np.testing.assert_array_equal(
                sv.gradient.vignetted[i], sens.nominal.vignetted
            )

    # --- f → x (start from f-basis sensitivity) ---

    def test_f_to_x_gradient_shape_dz(self):
        """DoubleZernikes: f-basis → x recovers (NUSEDOF, K+1, J+1) shape."""
        sens, *_ = _make_dz_sensitivity()
        sf = sens.f  # (NDOF, KMAX+1, JMAX+1)
        sx = sf.x
        assert sx.basis == "x"
        assert sx.gradient.coefs.shape == (NUSEDOF, KMAX + 1, JMAX + 1)

    def test_f_to_x_values_dz(self):
        sens, *_ = _make_dz_sensitivity()
        sx = sens.f.x
        np.testing.assert_allclose(
            sx.gradient.coefs.to_value(u.um),
            sens.gradient.coefs.to_value(u.um),
        )
