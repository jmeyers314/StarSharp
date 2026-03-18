"""Tests for the Sensitivity class."""

from __future__ import annotations

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

NDOF = 4
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
        basis="f",
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
    return Sensitivity.from_finite_differences(nominal, perturbed, steps), perturbed


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
    return Sensitivity.from_finite_differences(nominal, perturbed, steps), perturbed


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
    return Sensitivity.from_finite_differences(nominal, perturbed, steps), perturbed


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
        sens, _ = _make_zk_sensitivity()
        assert sens.gradient.coefs.shape == (NDOF, NFIELD, JMAX + 1)

    def test_nominal_preserved(self):
        sens, _ = _make_zk_sensitivity()
        assert sens.nominal.coefs.shape == (NFIELD, JMAX + 1)

    def test_gradient_values(self):
        sens, perturbed = _make_zk_sensitivity()
        steps = sens.steps
        for i in range(NDOF):
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
        sens, _ = _make_spots_sensitivity()
        assert sens.gradient.dx.shape == (NDOF, NFIELD, NRAY)
        assert sens.gradient.dy.shape == (NDOF, NFIELD, NRAY)
        assert sens.gradient.vignetted.shape == (NDOF, NFIELD, NRAY)

    def test_gradient_dx_values(self):
        sens, perturbed = _make_spots_sensitivity()
        steps = sens.steps
        for i in range(NDOF):
            expected = (perturbed[i].dx - sens.nominal.dx) / steps.value[i]
            np.testing.assert_allclose(
                sens.gradient.dx[i].to_value(u.um),
                expected.to_value(u.um),
            )

    def test_vignetted_broadcast(self):
        sens, _ = _make_spots_sensitivity()
        for i in range(NDOF):
            np.testing.assert_array_equal(
                sens.gradient.vignetted[i], sens.nominal.vignetted
            )


# ---------------------------------------------------------------------------
# Tests: from_finite_differences  —  DoubleZernikes
# ---------------------------------------------------------------------------


class TestFromFiniteDifferencesDZ:
    def test_gradient_shape(self):
        sens, _ = _make_dz_sensitivity()
        assert sens.gradient.coefs.shape == (NDOF, KMAX + 1, JMAX + 1)

    def test_gradient_values(self):
        sens, perturbed = _make_dz_sensitivity()
        steps = sens.steps
        for i in range(NDOF):
            expected = (perturbed[i].coefs - sens.nominal.coefs) / steps.value[i]
            np.testing.assert_allclose(
                sens.gradient.coefs[i].to_value(u.um),
                expected.to_value(u.um),
            )


# ---------------------------------------------------------------------------
# Tests: properties and indexing
# ---------------------------------------------------------------------------


class TestProperties:
    def test_ndof(self):
        sens, _ = _make_zk_sensitivity()
        assert sens.ndof == NDOF

    def test_len(self):
        sens, _ = _make_zk_sensitivity()
        assert len(sens) == NDOF

    def test_getitem_zernikes(self):
        sens, _ = _make_zk_sensitivity()
        z0 = sens[0]
        assert isinstance(z0, Zernikes)
        assert z0.coefs.shape == (NFIELD, JMAX + 1)
        np.testing.assert_allclose(
            z0.coefs.to_value(u.um),
            sens.gradient.coefs[0].to_value(u.um),
        )

    def test_getitem_spots(self):
        sens, _ = _make_spots_sensitivity()
        s0 = sens[0]
        assert isinstance(s0, Spots)
        assert s0.dx.shape == (NFIELD, NRAY)

    def test_getitem_dz(self):
        sens, _ = _make_dz_sensitivity()
        dz0 = sens[0]
        assert isinstance(dz0, DoubleZernikes)
        assert dz0.coefs.shape == (KMAX + 1, JMAX + 1)

    def test_slice(self):
        sens, _ = _make_zk_sensitivity()
        sliced = sens[:2]
        assert isinstance(sliced, Zernikes)
        assert sliced.coefs.shape == (2, NFIELD, JMAX + 1)


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_zernikes(self):
        sens, _ = _make_zk_sensitivity()
        delta = State(
            value=np.array([0.1, -0.2, 0.3, 0.0]),
            basis="f",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert isinstance(result, Zernikes)
        assert result.coefs.shape == (NFIELD, JMAX + 1)

        # Manual check: nominal + gradient @ weights
        weights = delta.f.value
        expected = sens.nominal.coefs.value + np.einsum(
            "i...,i->...", sens.gradient.coefs.value, weights
        )
        np.testing.assert_allclose(result.coefs.to_value(u.um), expected, rtol=1e-12)

    def test_predict_spots(self):
        sens, _ = _make_spots_sensitivity()
        delta = State(
            value=np.array([0.5, -0.5, 1.0, -1.0]),
            basis="f",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert isinstance(result, Spots)

        weights = delta.f.value
        expected_dx = sens.nominal.dx.value + np.einsum(
            "i...,i->...", sens.gradient.dx.value, weights
        )
        np.testing.assert_allclose(result.dx.to_value(u.um), expected_dx, rtol=1e-12)

    def test_predict_double_zernikes(self):
        sens, _ = _make_dz_sensitivity()
        delta = State(
            value=np.array([1.0, 2.0, 3.0, 4.0]),
            basis="f",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(delta)
        assert isinstance(result, DoubleZernikes)

        weights = delta.f.value
        expected = sens.nominal.coefs.value + np.einsum(
            "i...,i->...", sens.gradient.coefs.value, weights
        )
        np.testing.assert_allclose(result.coefs.to_value(u.um), expected, rtol=1e-12)

    def test_predict_zero_is_nominal(self):
        sens, _ = _make_zk_sensitivity()
        zero = State(
            value=np.zeros(NDOF),
            basis="f",
            use_dof=np.array([0, 1, 2, 3]),
            n_dof=50,
        )
        result = sens.predict(zero)
        np.testing.assert_allclose(
            result.coefs.to_value(u.um),
            sens.nominal.coefs.to_value(u.um),
            rtol=1e-12,
        )

    def test_predict_wrong_length_raises(self):
        sens, _ = _make_zk_sensitivity()
        bad = State(value=np.zeros(3), basis="f")
        with pytest.raises(ValueError, match="Expected 4 weights, got 3"):
            sens.predict(bad)

    def test_predict_preserves_metadata(self):
        sens, _ = _make_zk_sensitivity()
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
# Tests: repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_zernikes(self):
        sens, _ = _make_zk_sensitivity()
        r = repr(sens)
        assert "Sensitivity[Zernikes]" in r
        assert "ndof=4" in r
        assert "basis='f'" in r

    def test_repr_spots(self):
        sens, _ = _make_spots_sensitivity()
        r = repr(sens)
        assert "Sensitivity[Spots]" in r

    def test_repr_dz(self):
        sens, _ = _make_dz_sensitivity()
        r = repr(sens)
        assert "Sensitivity[DoubleZernikes]" in r
