"""Tests for State and StateFactory."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from StarSharp.datatypes import Sensitivity, State, StateFactory

from .utils import _make_field


def _make_Vh(n_active, nkeep=None, rng=None):
    """Build an orthogonal Vh matrix of shape (nkeep, n_active)."""
    if rng is None:
        rng = np.random.default_rng(99)
    A = rng.standard_normal((n_active * 3, n_active))
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    if nkeep is None:
        nkeep = n_active
    return Vh[:nkeep]


class TestStateConstruction:
    def test_from_x(self):
        use_dof = np.array([0, 2, 5])
        s = State(value=np.array([1.0, 2.0, 3.0]), basis="x", use_dof=use_dof, n_dof=10)
        assert s.basis == "x"
        assert len(s.value) == 3

    def test_from_f(self):
        fvals = np.zeros(10)
        fvals[2] = 5.0
        s = State(value=fvals, basis="f")
        assert s.basis == "f"
        assert s.n_dof == 10  # inferred

    def test_from_v(self):
        Vh = _make_Vh(5, 3)
        s = State(value=np.ones(3), basis="v", Vh=Vh, use_dof=np.arange(5), n_dof=10)
        assert s.basis == "v"
        assert s.nkeep == 3

    def test_invalid_basis(self):
        with pytest.raises(ValueError, match="basis must be"):
            State(value=np.ones(3), basis="z")

    def test_frozen(self):
        s = State(value=np.ones(3), basis="x", use_dof=np.arange(3), n_dof=5)
        with pytest.raises(AttributeError):
            s.value = np.zeros(3)

    def test_state_coerced_to_ndarray(self):
        s = State(value=[1.0, 2.0], basis="x", use_dof=np.arange(2), n_dof=5)
        assert isinstance(s.value, np.ndarray)


class TestStateConversions:
    def test_x_identity(self):
        s = State(
            value=np.array([1.0, 2.0, 3.0]),
            basis="x",
            use_dof=np.array([0, 2, 5]),
            n_dof=10,
        )
        assert s.x is s

    def test_f_identity(self):
        s = State(value=np.zeros(10), basis="f")
        assert s.f is s

    def test_v_identity(self):
        Vh = _make_Vh(5, 3)
        s = State(value=np.ones(3), basis="v", Vh=Vh, use_dof=np.arange(5), n_dof=10)
        assert s.v is s

    def test_x_to_f_roundtrip(self):
        use_dof = np.array([1, 3, 7])
        xvals = np.array([10.0, 20.0, 30.0])
        s = State(value=xvals, basis="x", use_dof=use_dof, n_dof=10)
        fs = s.f
        assert fs.basis == "f"
        assert len(fs.value) == 10
        np.testing.assert_allclose(fs.value[use_dof], xvals)
        inactive = np.setdiff1d(np.arange(10), use_dof)
        np.testing.assert_allclose(fs.value[inactive], 0.0)
        np.testing.assert_allclose(fs.x.value, xvals)

    def test_f_to_x(self):
        use_dof = np.array([0, 4])
        fvals = np.array([5.0, 0.0, 0.0, 0.0, 9.0])
        s = State(value=fvals, basis="f", use_dof=use_dof)
        np.testing.assert_allclose(s.x.value, [5.0, 9.0])

    def test_x_to_v_roundtrip_full_rank(self):
        """When nkeep == len(use_dof), x->v->x is lossless."""
        n_active = 5
        Vh = _make_Vh(n_active, 5)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = State(
            value=xvals,
            basis="x",
            use_dof=np.arange(n_active),
            n_dof=10,
            Vh=Vh,
        )
        np.testing.assert_allclose(s.v.x.value, xvals, atol=1e-12)

    def test_x_to_v_lossy_truncated(self):
        """When nkeep < len(use_dof), x->v->x is lossy."""
        n_active = 5
        nkeep = 3
        Vh = _make_Vh(n_active, nkeep)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = State(
            value=xvals,
            basis="x",
            use_dof=np.arange(n_active),
            n_dof=10,
            Vh=Vh,
        )
        recovered = s.v.x.value
        assert not np.allclose(recovered, xvals)
        np.testing.assert_allclose(s.v.x.v.value, s.v.value, atol=1e-12)

    def test_v_to_f(self):
        n_active = 4
        use_dof = np.array([1, 3, 5, 7])
        Vh = _make_Vh(n_active, 3)
        vvals = np.ones(3)
        s = State(value=vvals, basis="v", use_dof=use_dof, n_dof=10, Vh=Vh)
        fs = s.f
        assert fs.basis == "f"
        assert len(fs.value) == 10
        inactive = np.setdiff1d(np.arange(10), use_dof)
        np.testing.assert_allclose(fs.value[inactive], 0.0)

    def test_f_to_v(self):
        n_active = 4
        use_dof = np.array([1, 3, 5, 7])
        Vh = _make_Vh(n_active)
        fvals = np.zeros(10)
        fvals[use_dof] = [1.0, 2.0, 3.0, 4.0]
        s = State(value=fvals, basis="f", use_dof=use_dof, Vh=Vh)
        vs = s.v
        assert vs.basis == "v"
        assert len(vs.value) == 4
        np.testing.assert_allclose(vs.x.value, [1.0, 2.0, 3.0, 4.0], atol=1e-12)


class TestStateAdd:
    def test_add_x_states(self):
        rng = np.random.default_rng(423)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(5), nkeep=5)

        s1 = sf.from_x(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        s2 = sf.from_x(np.array([0.5, -1.0, 2.0, 0.0, 1.5]))

        s3 = s1 + s2
        np.testing.assert_allclose(s3.x.value, [1.5, 1.0, 5.0, 4.0, 6.5])
        np.testing.assert_allclose(s1.x.value, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(s2.x.value, [0.5, -1.0, 2.0, 0.0, 1.5])

    def test_add_x_and_f(self):
        rng = np.random.default_rng(422)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(5), nkeep=5)

        sx = sf.from_x(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        ssum = sx + sx.f
        np.testing.assert_allclose(ssum.x.value, 2.0 * sx.x.value, atol=1e-12)
        np.testing.assert_allclose(ssum.use_dof, sx.use_dof, atol=1e-12)
        np.testing.assert_allclose(ssum.n_dof, sx.n_dof, atol=1e-12)
        np.testing.assert_allclose(ssum.Vh, sx.Vh, atol=1e-12)

    def test_add_x_and_v(self):
        rng = np.random.default_rng(421)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(5), nkeep=5)

        sx = sf.from_x(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        ssum = sx + sx.v
        np.testing.assert_allclose(ssum.x.value, 2.0 * sx.x.value, atol=1e-12)
        np.testing.assert_allclose(ssum.use_dof, sx.use_dof, atol=1e-12)
        np.testing.assert_allclose(ssum.n_dof, sx.n_dof, atol=1e-12)
        np.testing.assert_allclose(ssum.Vh, sx.Vh, atol=1e-12)

    def test_add_raises_for_mismatched_n_dof(self):
        s1 = State(value=np.ones(3), basis="x", use_dof=np.arange(3), n_dof=5)
        s2 = State(value=np.ones(3), basis="x", use_dof=np.arange(3), n_dof=6)
        with pytest.raises(ValueError, match="different n_dof"):
            _ = s1 + s2

    def test_add_falls_back_to_f_basis(self):
        rng = np.random.default_rng(424)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(7), nkeep=5)
        sf2 = StateFactory(A=A, use_dof=np.arange(8), nkeep=5)

        s1 = sf.from_x(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        s2 = sf2.from_x(np.array([0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 0.0, 0.0]))
        ssum = s1 + s2
        assert ssum.basis == "f"
        np.testing.assert_allclose(ssum.value, s1.f.value + s2.f.value, atol=1e-12)


class TestStateRequires:
    def test_f_to_x_requires_use_dof(self):
        s = State(value=np.zeros(10), basis="f")
        with pytest.raises(ValueError, match="use_dof"):
            s.x

    def test_x_to_f_requires_n_dof(self):
        s = State(value=np.ones(3), basis="x", use_dof=np.arange(3))
        with pytest.raises(ValueError, match="n_dof"):
            s.f

    def test_x_to_v_requires_Vh(self):
        s = State(value=np.ones(3), basis="x", use_dof=np.arange(3), n_dof=5)
        with pytest.raises(ValueError, match="Vh"):
            s.v

    def test_v_requires_Vh_at_construction(self):
        with pytest.raises(ValueError, match="Vh"):
            State(value=np.ones(3), basis="v", use_dof=np.arange(5), n_dof=10)


class TestStateFactory:
    def test_svd_computed(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        use_dof = np.array([0, 2, 4, 6, 8])
        sf = StateFactory(A=A, use_dof=use_dof, nkeep=3)
        assert sf.n_dof == 10
        assert sf.Vh.shape == (3, 5)
        assert sf.full_Vh.shape == (5, 5)
        assert len(sf.S) == 5
        assert sf.nkeep == 3

    def test_nkeep_defaults_to_full(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        use_dof = np.arange(5)
        sf = StateFactory(A=A, use_dof=use_dof)
        assert sf.nkeep == 5

    def test_from_x(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(5), nkeep=3)
        s = sf.from_x(np.ones(5))
        assert s.basis == "x"
        assert s.n_dof == 10
        assert s.nkeep == 3

    def test_from_f(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(5))
        s = sf.from_f(np.zeros(10))
        assert s.basis == "f"

    def test_from_v(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        sf = StateFactory(A=A, use_dof=np.arange(5), nkeep=3)
        s = sf.from_v(np.ones(3))
        assert s.basis == "v"

    def test_factory_roundtrip(self):
        """Factory-created states carry full context for all conversions."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        use_dof = np.array([0, 2, 4, 6, 8])
        sf = StateFactory(A=A, use_dof=use_dof, nkeep=5)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = sf.from_x(xvals)
        np.testing.assert_allclose(s.f.x.v.x.value, xvals, atol=1e-12)

    def test_normalization_argument(self):
        """Test that normalization rescales A as expected."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 5))
        norm = np.array([1, 2, 3, 4, 5], dtype=float)
        sf_norm = StateFactory(A=A, use_dof=np.arange(5), norm=norm)
        sf_no_norm = StateFactory(A=A, use_dof=np.arange(5))
        # The SVD of A @ diag(norm) should differ from A
        assert not np.allclose(sf_norm.S, sf_no_norm.S)
        # If norm is all ones, should match no-norm
        sf_ones = StateFactory(A=A, use_dof=np.arange(5), norm=np.ones(5))
        np.testing.assert_allclose(sf_ones.S, sf_no_norm.S)

    def test_from_sensitivity_zernikes(self):
        """StateFactory accepts a Sensitivity[Zernikes] and extracts the design matrix."""
        from StarSharp.datatypes import Zernikes

        rng = np.random.default_rng(77)
        n_dof = 5
        n_field = 3
        jmax = 10
        # Gradient Zernikes: shape (n_dof, n_field, jmax+1)
        grad_coefs = rng.standard_normal((n_dof, n_field, jmax + 1)) * u.um
        field = _make_field(n_field)
        gradient = Zernikes(coefs=grad_coefs, field=field)
        nominal = Zernikes(
            coefs=rng.standard_normal((n_field, jmax + 1)) * u.um,
            field=field,
        )
        steps = State(value=np.ones(n_dof), basis="x", use_dof=np.arange(n_dof), n_dof=n_dof)
        sens = Sensitivity(nominal=nominal, gradient=gradient, steps=steps)

        use_dof = np.arange(n_dof)
        sf = StateFactory(sens, use_dof=use_dof, nkeep=3)

        # Design matrix should be (n_field * (jmax+1), n_dof)
        nobs = n_field * (jmax + 1)
        assert sf.A.shape == (nobs, n_dof)
        assert sf.Vh.shape == (3, n_dof)

        # Result should match building from the raw array directly
        A_raw = grad_coefs.value.reshape(n_dof, -1).T
        sf_raw = StateFactory(A=A_raw, use_dof=use_dof, nkeep=3)
        np.testing.assert_allclose(sf.S, sf_raw.S)

    def test_from_sensitivity_spots(self):
        """StateFactory[Sensitivity[Spots]] concatenates dx/dy and excludes vignetted rays."""
        from StarSharp.datatypes import Spots

        rng = np.random.default_rng(88)
        n_dof = 4
        n_field = 2
        n_ray = 10
        field = _make_field(n_field)

        # Nominal spots: 2 of the 10 rays are vignetted
        vig = np.zeros((n_field, n_ray), dtype=bool)
        vig[0, 0] = True
        vig[1, 3] = True
        nom = Spots(
            dx=rng.standard_normal((n_field, n_ray)) * u.um,
            dy=rng.standard_normal((n_field, n_ray)) * u.um,
            vignetted=vig,
            field=field,
        )

        # Gradient spots: (n_dof, n_field, n_ray)
        grad = Spots(
            dx=rng.standard_normal((n_dof, n_field, n_ray)) * u.um,
            dy=rng.standard_normal((n_dof, n_field, n_ray)) * u.um,
            vignetted=np.broadcast_to(vig, (n_dof, n_field, n_ray)).copy(),
            field=field,
        )
        steps = State(value=np.ones(n_dof), basis="x", use_dof=np.arange(n_dof), n_dof=n_dof)
        sens = Sensitivity(nominal=nom, gradient=grad, steps=steps)

        sf = StateFactory(sens, use_dof=np.arange(n_dof))

        # 2 rays vignetted out of n_field * n_ray = 20; two fields dx+dy → 2*(20-2) = 36 obs
        n_valid = n_field * n_ray - 2
        assert sf.A.shape == (2 * n_valid, n_dof)

        # Verify values match manual construction
        valid = ~vig.ravel()
        dx_rows = grad.dx.value.reshape(n_dof, -1)[:, valid]
        dy_rows = grad.dy.value.reshape(n_dof, -1)[:, valid]
        A_raw = np.concatenate([dx_rows, dy_rows], axis=1).T
        sf_raw = StateFactory(A=A_raw, use_dof=np.arange(n_dof))
        np.testing.assert_allclose(sf.A, sf_raw.A)
