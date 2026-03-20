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


# ---------------------------------------------------------------------------
# Batch-broadcast tests for spots() and zernikes()
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from astropy.coordinates import Angle

import StarSharp.models.raytraced as raytraced_module


class _FakeTelescope:
    """Trivial telescope stub whose trace() is the identity."""

    def trace(self, rays):
        return rays


class _FakeBuilder:
    """Chainable builder stub that always returns _FakeTelescope."""

    def with_rtp(self, _rtp):
        return self

    def with_aos_dof(self, _f):
        return self

    def build_det(self, detnum):
        return _FakeTelescope()


def _make_model_stub():
    """Create a RaytracedOpticalModel without calling __init__."""
    model = RaytracedOpticalModel.__new__(RaytracedOpticalModel)
    model.builder = _FakeBuilder()
    model.rtp = Angle(0.0, unit=u.deg)
    model.wavelength = 620.0 * u.nm
    model.camera = None
    model.tqdm = None
    model.steps = None
    return model


def _make_batched_field(batch_shape, nfield, rtp=None):
    """FieldCoords with shape (*batch_shape, nfield) in OCS degrees."""
    if rtp is None:
        rtp = Angle(0.0, unit=u.deg)
    shape = batch_shape + (nfield,)
    n = int(np.prod(shape))
    x = (np.arange(n, dtype=float) * 0.01).reshape(shape) * u.deg
    y = (0.5 + np.arange(n, dtype=float) * 0.02).reshape(shape) * u.deg
    return FieldCoords(x=x, y=y, frame="ocs", rtp=rtp)


def _patch_detnum_zero(monkeypatch):
    """Make FieldCoords.detnum always return 0 (valid detector)."""
    monkeypatch.setattr(
        FieldCoords,
        "detnum",
        property(lambda self: np.zeros(self.x.shape, dtype=int)),
    )


class TestZernikesBatchBroadcast:
    """Verify zernikes() output shapes and values with non-trivial batch dims."""

    @staticmethod
    def _patch_zernikeGQ(monkeypatch):
        """Replace batoid.zernikeGQ with a deterministic function."""

        def fake_zernikeGQ(telescope, theta_x, theta_y, wavelength,
                           rings, jmax, eps):
            base = 10.0 * theta_x + 100.0 * theta_y
            return base + np.arange(jmax + 1, dtype=float)

        monkeypatch.setattr(raytraced_module.batoid, "zernikeGQ", fake_zernikeGQ)

    def test_1d_batch(self, monkeypatch):
        """batch_shape = (2,), nfield = 3  ->  coefs.shape == (2, 3, jmax+1)."""
        model = _make_model_stub()
        field = _make_batched_field((2,), 3, rtp=model.rtp)
        state = State(value=np.zeros(50), basis="f")
        jmax = 4
        _patch_detnum_zero(monkeypatch)
        self._patch_zernikeGQ(monkeypatch)

        zk = RaytracedOpticalModel.zernikes(
            model, field=field, state=state, jmax=jmax, rings=3,
        )

        assert zk.batch_shape == (2,)
        assert zk.nfield == 3
        assert zk.coefs.shape == (2, 3, jmax + 1)

    def test_2d_batch(self, monkeypatch):
        """batch_shape = (2, 4), nfield = 3."""
        model = _make_model_stub()
        field = _make_batched_field((2, 4), 3, rtp=model.rtp)
        state = State(value=np.zeros(50), basis="f")
        jmax = 4
        _patch_detnum_zero(monkeypatch)
        self._patch_zernikeGQ(monkeypatch)

        zk = RaytracedOpticalModel.zernikes(
            model, field=field, state=state, jmax=jmax, rings=3,
        )

        assert zk.batch_shape == (2, 4)
        assert zk.nfield == 3
        assert zk.coefs.shape == (2, 4, 3, jmax + 1)

    def test_values_match_flat(self, monkeypatch):
        """Batched results match looping over each batch element individually."""
        model = _make_model_stub()
        batch_shape = (2,)
        nfield = 3
        field = _make_batched_field(batch_shape, nfield, rtp=model.rtp)
        state = State(value=np.zeros(50), basis="f")
        jmax = 4
        _patch_detnum_zero(monkeypatch)
        self._patch_zernikeGQ(monkeypatch)

        zk_batched = RaytracedOpticalModel.zernikes(
            model, field=field, state=state, jmax=jmax, rings=3,
        )

        for ib in range(batch_shape[0]):
            flat_field = FieldCoords(
                x=field.x[ib], y=field.y[ib],
                frame="ocs", rtp=model.rtp,
            )
            zk_flat = RaytracedOpticalModel.zernikes(
                model, field=flat_field, state=state, jmax=jmax, rings=3,
            )
            np.testing.assert_allclose(
                zk_batched.coefs[ib].to_value(u.um),
                zk_flat.coefs.to_value(u.um),
                atol=1e-12,
            )

    def test_state_none_accepted(self, monkeypatch):
        """zernikes() should not crash when state is None."""
        model = _make_model_stub()
        field = _make_batched_field((2,), 3, rtp=model.rtp)
        jmax = 4
        _patch_detnum_zero(monkeypatch)
        self._patch_zernikeGQ(monkeypatch)

        zk = RaytracedOpticalModel.zernikes(
            model, field=field, state=None, jmax=jmax, rings=3,
        )
        assert zk.coefs.shape == (2, 3, jmax + 1)

    def test_detnum_minus1_gives_nan(self, monkeypatch):
        """Points with detnum == -1 should produce NaN coefficients."""
        model = _make_model_stub()
        field = _make_batched_field((2,), 3, rtp=model.rtp)
        state = State(value=np.zeros(50), basis="f")
        jmax = 4
        self._patch_zernikeGQ(monkeypatch)

        def detnum_with_hole(self_fc):
            out = np.zeros(self_fc.x.shape, dtype=int)
            out[1, 1] = -1
            return out

        monkeypatch.setattr(
            FieldCoords, "detnum",
            property(detnum_with_hole),
        )

        zk = RaytracedOpticalModel.zernikes(
            model, field=field, state=state, jmax=jmax, rings=3,
        )

        assert not np.any(np.isnan(zk.coefs[0].value))   # batch 0 all valid
        assert np.all(np.isnan(zk.coefs[1, 1].value))     # batch 1, field 1 NaN
        assert not np.any(np.isnan(zk.coefs[1, 0].value))
        assert not np.any(np.isnan(zk.coefs[1, 2].value))


class TestSpotsBatchBroadcast:
    """Verify spots() output shapes and values with non-trivial batch dims."""

    @staticmethod
    def _patch_hexapolar(monkeypatch, px, py):
        monkeypatch.setattr(
            raytraced_module.batoid.utils, "hexapolar",
            lambda **kwargs: (px, py),
        )

    @staticmethod
    def _patch_fromStop(monkeypatch):
        """Replace RayVector.fromStop with a trivial deterministic function."""

        def fake_fromStop(px_in, py_in, theta_x, theta_y, optic, wavelength):
            px_arr = np.atleast_1d(np.asarray(px_in, dtype=float))
            py_arr = np.atleast_1d(np.asarray(py_in, dtype=float))
            # Chief ray (single point at origin)
            if px_arr.size == 1 and py_arr.size == 1 and px_arr[0] == 0.0:
                return SimpleNamespace(
                    x=np.array([theta_x]),
                    y=np.array([theta_y]),
                    vignetted=np.array([False]),
                )
            # Grid rays: place them at pupil coord + field angle
            return SimpleNamespace(
                x=px_arr + theta_x,
                y=py_arr + theta_y,
                vignetted=np.zeros(len(px_arr), dtype=bool),
            )

        monkeypatch.setattr(
            raytraced_module.batoid.RayVector, "fromStop",
            staticmethod(fake_fromStop),
        )

    def test_1d_batch_shapes(self, monkeypatch):
        model = _make_model_stub()
        field = _make_batched_field((2,), 3, rtp=model.rtp)
        _patch_detnum_zero(monkeypatch)
        px = np.array([-1.0, 0.0, 1.0])
        py = np.array([0.0, 0.5, -0.5])
        self._patch_hexapolar(monkeypatch, px, py)
        self._patch_fromStop(monkeypatch)

        sp = RaytracedOpticalModel.spots(
            model, field=field, state=None, nrad=2, reference="chief",
        )

        assert sp.batch_shape == (2,)
        assert sp.nfield == 3
        assert sp.nray == 3
        assert sp.dx.shape == (2, 3, 3)
        assert sp.dy.shape == (2, 3, 3)
        assert sp.vignetted.shape == (2, 3, 3)
        assert sp.field.x.shape == (2, 3)

    def test_2d_batch_shapes(self, monkeypatch):
        model = _make_model_stub()
        field = _make_batched_field((2, 4), 3, rtp=model.rtp)
        _patch_detnum_zero(monkeypatch)
        px = np.array([-1.0, 0.0, 1.0])
        py = np.array([0.0, 0.5, -0.5])
        self._patch_hexapolar(monkeypatch, px, py)
        self._patch_fromStop(monkeypatch)

        sp = RaytracedOpticalModel.spots(
            model, field=field, state=None, nrad=2, reference="chief",
        )

        assert sp.batch_shape == (2, 4)
        assert sp.nfield == 3
        assert sp.dx.shape == (2, 4, 3, 3)
        assert sp.field.x.shape == (2, 4, 3)

    def test_values_match_flat(self, monkeypatch):
        """Batched results match looping over each batch element individually."""
        model = _make_model_stub()
        field = _make_batched_field((2,), 3, rtp=model.rtp)
        _patch_detnum_zero(monkeypatch)
        px = np.array([-1.0, 0.0, 1.0])
        py = np.array([0.0, 0.5, -0.5])
        self._patch_hexapolar(monkeypatch, px, py)
        self._patch_fromStop(monkeypatch)

        sp_batched = RaytracedOpticalModel.spots(
            model, field=field, state=None, nrad=2, reference="chief",
        )

        for ib in range(2):
            flat_field = FieldCoords(
                x=field.x[ib], y=field.y[ib],
                frame="ocs", rtp=model.rtp,
            )
            sp_flat = RaytracedOpticalModel.spots(
                model, field=flat_field, state=None, nrad=2, reference="chief",
            )
            np.testing.assert_allclose(
                sp_batched.dx[ib].to_value(u.micron),
                sp_flat.dx.to_value(u.micron),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                sp_batched.dy[ib].to_value(u.micron),
                sp_flat.dy.to_value(u.micron),
                atol=1e-12,
            )
            np.testing.assert_array_equal(
                sp_batched.vignetted[ib], sp_flat.vignetted,
            )

    def test_detnum_minus1_gives_nan_and_vignetted(self, monkeypatch):
        """Off-detector points should have NaN dx/dy and vignetted=True."""
        model = _make_model_stub()
        field = _make_batched_field((2,), 3, rtp=model.rtp)
        px = np.array([-1.0, 0.0, 1.0])
        py = np.array([0.0, 0.5, -0.5])
        self._patch_hexapolar(monkeypatch, px, py)
        self._patch_fromStop(monkeypatch)

        def detnum_with_hole(self_fc):
            out = np.zeros(self_fc.x.shape, dtype=int)
            out[1, 1] = -1
            return out

        monkeypatch.setattr(
            FieldCoords, "detnum",
            property(detnum_with_hole),
        )

        sp = RaytracedOpticalModel.spots(
            model, field=field, state=None, nrad=2, reference="chief",
        )

        # batch 0: all valid -> no NaN
        assert not np.any(np.isnan(sp.dx[0].value))
        # batch 1, field 1: off-detector -> NaN + vignetted
        assert np.all(np.isnan(sp.dx[1, 1].value))
        assert np.all(sp.vignetted[1, 1])
        # batch 1, fields 0/2: still valid
        assert not np.any(np.isnan(sp.dx[1, 0].value))
        assert not np.any(np.isnan(sp.dx[1, 2].value))
