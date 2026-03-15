"""Unit tests for FieldCoords, Spots, and Zernikes."""

from __future__ import annotations

import itertools

import astropy.units as u
import galsim
import numpy as np
import pytest
from astropy.coordinates import Angle

from datatypes import FieldCoords, Moments, Moments2, Moments3, Moments4, Spots, State, StateFactory, Zernikes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wcs(
    rtp: Angle | None = None,
) -> galsim.CelestialWCS:
    if rtp is None:
        rtp = Angle(0, unit=u.deg)
    c = np.cos(rtp)
    s = np.sin(rtp)
    scale = np.deg2rad(20/3600)  # 20 arcsec/mm
    rot = np.array([[c, -s], [s, c]]) * scale
    return galsim.AffineTransform(
        rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], galsim.PositionD(0.0, 0.0)
    )

def _make_field(
    n: int = 3,
    frame: str = "ocs",
    rtp: Angle | None = None,
    unit: u.Unit = u.deg,
    wcs: galsim.BaseWCS | None = None,
) -> FieldCoords:
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.5, 1.5, n) * unit
    y = rng.uniform(-1.5, 1.5, n) * unit
    return FieldCoords(x=x, y=y, frame=frame, rtp=rtp, wcs=wcs)


RTP = Angle(0.25, unit=u.rad)


# ===================================================================
# FieldCoords
# ===================================================================


class TestFieldCoordsConstruction:
    def test_scalar_promoted_to_1d(self):
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg)
        assert fc.x.ndim == 1
        assert fc.y.ndim == 1

    def test_1d_stays_1d(self):
        fc = _make_field(5)
        assert fc.x.ndim == 1
        assert len(fc) == 1  # 1-D -> single field point batch

    def test_2d_stays_2d(self):
        x = np.ones((3, 4)) * u.deg
        y = np.zeros((3, 4)) * u.deg
        fc = FieldCoords(x=x, y=y)
        assert fc.x.ndim == 2
        assert len(fc) == 3

    def test_non_quantity_raises(self):
        with pytest.raises(TypeError):
            FieldCoords(x=1.0, y=2.0)

    def test_mismatched_units_raises(self):
        with pytest.raises(ValueError, match="compatible units"):
            FieldCoords(x=1.0 * u.deg, y=2.0 * u.mm)

    def test_invalid_unit_type_raises(self):
        with pytest.raises(ValueError, match="angular or length"):
            FieldCoords(x=1.0 * u.kg, y=2.0 * u.kg)

    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame must be one of"):
            FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="xyz")


class TestFieldCoordsSpace:
    def test_angle_space(self):
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg)
        assert fc.space == "angle"

    def test_focal_plane_space(self):
        fc = FieldCoords(x=1.0 * u.mm, y=2.0 * u.mm)
        assert fc.space == "focal_plane"

    def test_roundtrip_angle_focal_plane(self):
        wcs = _make_wcs(rtp=RTP)
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, wcs=wcs, rtp=RTP)
        fp = fc.focal_plane.angle
        np.testing.assert_allclose(
            fp.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            fp.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_roundtrip_focal_plane_angle(self):
        wcs = _make_wcs(rtp=RTP)
        fc = FieldCoords(x=1.0 * u.mm, y=2.0 * u.mm, wcs=wcs, rtp=RTP)
        ang = fc.angle.focal_plane
        np.testing.assert_allclose(
            ang.x.to_value(u.mm), fc.x.to_value(u.mm), atol=1e-12
        )
        np.testing.assert_allclose(
            ang.y.to_value(u.mm), fc.y.to_value(u.mm), atol=1e-12
        )


class TestFieldCoordsFrameRotation:
    def test_ocs_noop_when_already_ocs(self):
        fc = _make_field(frame="ocs", rtp=RTP)
        assert fc.ocs is fc

    def test_ccs_noop_when_already_ccs(self):
        fc = _make_field(frame="ccs", rtp=RTP)
        assert fc.ccs is fc

    def test_ocs_then_ccs_roundtrip(self):
        fc = _make_field(frame="ocs", rtp=RTP)
        rt = fc.ccs.ocs
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )
        assert rt.frame == "ocs"

    def test_ccs_then_ocs_roundtrip(self):
        fc = _make_field(frame="ccs", rtp=RTP)
        rt = fc.ocs.ccs
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )
        assert rt.frame == "ccs"

    def test_zero_rtp_is_identity(self):
        fc = _make_field(frame="ocs", rtp=Angle(0, unit=u.rad))
        ccs = fc.ccs
        np.testing.assert_allclose(
            ccs.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            ccs.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_90deg_rotation(self):
        rtp = Angle(np.pi / 2, unit=u.rad)
        fc = FieldCoords(
            x=np.array([1.0]) * u.deg,
            y=np.array([0.0]) * u.deg,
            frame="ocs",
            rtp=rtp,
        )
        ccs = fc.ccs
        np.testing.assert_allclose(ccs.x.to_value(u.deg), [0.0], atol=1e-12)
        np.testing.assert_allclose(ccs.y.to_value(u.deg), [-1.0], atol=1e-12)

    def test_rotation_without_rtp_raises(self):
        fc = _make_field(frame="ocs", rtp=None)
        with pytest.raises(ValueError, match="rtp"):
            fc.ccs

    def test_rtp_propagated(self):
        fc = _make_field(frame="ocs", rtp=RTP)
        assert fc.ccs.rtp is RTP


class TestFieldCoordsSlicing:
    def test_getitem_int(self):
        fc = FieldCoords(
            x=np.array([1.0, 2.0, 3.0]) * u.deg,
            y=np.array([4.0, 5.0, 6.0]) * u.deg,
            rtp=RTP,
        )
        s = fc[1]
        assert s.x.to_value(u.deg) == pytest.approx(2.0)
        assert s.y.to_value(u.deg) == pytest.approx(5.0)
        assert s.rtp is RTP

    def test_getitem_slice(self):
        fc = FieldCoords(
            x=np.arange(5.0) * u.deg,
            y=np.arange(5.0, 10.0) * u.deg,
        )
        s = fc[1:3]
        assert len(s.x) == 2

    def test_frozen(self):
        fc = _make_field()
        with pytest.raises(AttributeError):
            fc.x = 999 * u.deg


# ===================================================================
# Spots
# ===================================================================


def _make_spots(
    n_field: int = 3,
    n_ray: int = 50,
    frame: str = "ccs",
    rtp: Angle | None = None,
    unit: u.Unit = u.um,
    wcs: galsim.BaseWCS | None = None,
) -> Spots:
    rng = np.random.default_rng(12)
    dx = rng.normal(size=(n_field, n_ray)) * unit
    dy = rng.normal(size=(n_field, n_ray)) * unit
    vig = np.zeros((n_field, n_ray), dtype=bool)
    field = _make_field(n_field, frame="ocs", rtp=rtp, wcs=wcs)
    return Spots(
        dx=dx,
        dy=dy,
        vignetted=vig,
        field=field,
        wavelength=622.0 * u.nm,
        frame=frame,
        rtp=rtp,
        wcs=wcs,
    )


class TestSpotsConstruction:
    def test_scalar_promoted(self):
        fc = FieldCoords(x=1.0 * u.deg, y=0.5 * u.deg)
        sp = Spots(
            dx=0.1 * u.um,
            dy=0.2 * u.um,
            vignetted=np.array(False),
            field=fc,
        )
        assert sp.dx.ndim == 1
        assert sp.dy.ndim == 1
        assert sp.vignetted.ndim == 1

    def test_2d_shape_preserved(self):
        sp = _make_spots(4, 100, rtp=RTP)
        assert sp.dx.shape == (4, 100)
        assert len(sp) == 4

    def test_1d_len_is_one(self):
        fc = FieldCoords(x=1.0 * u.deg, y=0.5 * u.deg)
        sp = Spots(
            dx=np.ones(10) * u.um,
            dy=np.ones(10) * u.um,
            vignetted=np.zeros(10, dtype=bool),
            field=fc,
        )
        assert len(sp) == 1


class TestSpotsRtpConsistency:
    def test_consistent_rtp_ok(self):
        # Should not raise
        _make_spots(rtp=RTP)

    def test_inconsistent_rtp_raises(self):
        field = _make_field(3, rtp=RTP)
        with pytest.raises(ValueError, match="inconsistent"):
            Spots(
                dx=np.ones((3, 10)) * u.um,
                dy=np.ones((3, 10)) * u.um,
                vignetted=np.zeros((3, 10), dtype=bool),
                field=field,
                rtp=Angle(999.0, unit=u.deg),
            )


class TestSpotsFrameRotation:
    def test_ocs_noop(self):
        sp = _make_spots(frame="ocs", rtp=RTP)
        assert sp.ocs is sp

    def test_ccs_noop(self):
        sp = _make_spots(frame="ccs", rtp=RTP)
        assert sp.ccs is sp

    def test_roundtrip_ccs_ocs(self):
        sp = _make_spots(frame="ccs", rtp=RTP)
        rt = sp.ocs.ccs
        np.testing.assert_allclose(
            rt.dx.to_value(u.um), sp.dx.to_value(u.um), atol=1e-10
        )
        np.testing.assert_allclose(
            rt.dy.to_value(u.um), sp.dy.to_value(u.um), atol=1e-10
        )
        assert rt.frame == "ccs"

    def test_roundtrip_ocs_ccs(self):
        sp = _make_spots(frame="ocs", rtp=RTP)
        rt = sp.ccs.ocs
        np.testing.assert_allclose(
            rt.dx.to_value(u.um), sp.dx.to_value(u.um), atol=1e-10
        )

    def test_without_rtp_raises(self):
        sp = _make_spots(frame="ccs", rtp=None)
        with pytest.raises(ValueError, match="rtp"):
            sp.ocs

    def test_rotation_preserves_norm(self):
        sp = _make_spots(frame="ccs", rtp=RTP)
        orig_norm = np.sqrt(sp.dx**2 + sp.dy**2)
        rot = sp.ocs
        rot_norm = np.sqrt(rot.dx**2 + rot.dy**2)
        np.testing.assert_allclose(
            rot_norm.to_value(u.um), orig_norm.to_value(u.um), atol=1e-12
        )


class TestSpotsSpace:
    def test_space_angle(self):
        sp = _make_spots(frame="ocs", rtp=RTP, unit=u.arcsec)
        assert sp.space == "angle"

    def test_space_focal_plane(self):
        sp = _make_spots(frame="ccs", rtp=RTP, unit=u.mm)
        assert sp.space == "focal_plane"

    def test_roundtrip_focal_plane_angle(self):
        wcs = _make_wcs(rtp=RTP)
        sp = _make_spots(frame="ccs", rtp=RTP, unit=u.mm, wcs=wcs)
        rt = sp.angle.focal_plane
        np.testing.assert_allclose(
            rt.dx.to_value(u.mm), sp.dx.to_value(u.mm), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.dy.to_value(u.mm), sp.dy.to_value(u.mm), atol=1e-12
        )

    def test_roundtrip_angle_focal_plane(self):
        wcs = _make_wcs(rtp=RTP)
        sp = _make_spots(frame="ocs", rtp=RTP, unit=u.arcsec, wcs=wcs)
        rt = sp.focal_plane.angle
        np.testing.assert_allclose(
            rt.dx.to_value(u.arcsec), sp.dx.to_value(u.arcsec), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.dy.to_value(u.arcsec), sp.dy.to_value(u.arcsec), atol=1e-12
        )


class TestSpotsSlicing:
    def test_getitem_int(self):
        sp = _make_spots(5, 20, rtp=RTP)
        s = sp[2]
        assert s.dx.ndim == 1
        assert len(s) == 1
        assert s.wavelength == sp.wavelength

    def test_getitem_slice(self):
        sp = _make_spots(5, 20, rtp=RTP)
        s = sp[1:4]
        assert len(s) == 3
        assert s.frame == sp.frame

    def test_frozen(self):
        sp = _make_spots(rtp=RTP)
        with pytest.raises(AttributeError):
            sp.dx = 0 * u.um


# ===================================================================
# Zernikes
# ===================================================================


def _make_zernikes(
    n_field: int = 1,
    jmax: int = 22,
    frame: str = "ocs",
    rtp: Angle | None = None,
    R_outer: float = 4.18 << u.m,
    R_inner: float = 2.5 << u.m,
) -> Zernikes:
    rng = np.random.default_rng(99)
    if n_field == 1:
        coefs = rng.normal(size=jmax + 1) * u.um
    else:
        coefs = rng.normal(size=(n_field, jmax + 1)) * u.um
    field = _make_field(n_field, rtp=rtp)
    return Zernikes(
        coefs=coefs,
        field=field,
        R_outer=R_outer,
        R_inner=R_inner,
        frame=frame,
        rtp=rtp,
    )


class TestZernikesConstruction:
    def test_jmax_inferred(self):
        zk = _make_zernikes(jmax=15)
        assert zk.jmax == 15
        assert zk.coefs.shape[-1] == 16

    def test_jmax_explicit(self):
        coefs = np.zeros(23) * u.um
        field = _make_field(1)
        zk = Zernikes(coefs=coefs, field=field, jmax=22)
        assert zk.jmax == 22

    def test_scalar_coefs_promoted(self):
        # edge case: a single coefficient
        field = _make_field(1)
        zk = Zernikes(coefs=1.0 * u.um, field=field)
        assert zk.coefs.ndim == 1

    def test_eps(self):
        zk = _make_zernikes(R_outer=4.18 << u.m, R_inner=2.5 << u.m)
        assert zk.eps == pytest.approx(2.5 / 4.18)


class TestZernikesRtpConsistency:
    def test_consistent_ok(self):
        _make_zernikes(rtp=RTP)  # should not raise

    def test_inconsistent_raises(self):
        field = _make_field(1, rtp=RTP)
        with pytest.raises(ValueError, match="inconsistent"):
            Zernikes(
                coefs=np.zeros(10) * u.um,
                field=field,
                rtp=Angle(999.0, unit=u.deg),
            )


class TestZernikesFrameRotation:
    def test_ocs_noop(self):
        zk = _make_zernikes(frame="ocs", rtp=RTP)
        assert zk.ocs is zk

    def test_ccs_noop(self):
        zk = _make_zernikes(frame="ccs", rtp=RTP)
        assert zk.ccs is zk

    def test_roundtrip_ocs_ccs(self):
        zk = _make_zernikes(frame="ocs", rtp=RTP)
        rt = zk.ccs.ocs
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), zk.coefs.to_value(u.um), atol=1e-10
        )
        assert rt.frame == "ocs"

    def test_roundtrip_ccs_ocs(self):
        zk = _make_zernikes(frame="ccs", rtp=RTP)
        rt = zk.ocs.ccs
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), zk.coefs.to_value(u.um), atol=1e-10
        )

    def test_zero_rtp_identity(self):
        zk = _make_zernikes(frame="ocs", rtp=Angle(0, unit=u.rad))
        ccs = zk.ccs
        np.testing.assert_allclose(
            ccs.coefs.to_value(u.um), zk.coefs.to_value(u.um), atol=1e-12
        )

    def test_without_rtp_raises(self):
        zk = _make_zernikes(frame="ocs", rtp=None)
        with pytest.raises(ValueError, match="rtp"):
            zk.ccs

    def test_2d_roundtrip(self):
        zk = _make_zernikes(n_field=5, frame="ocs", rtp=RTP)
        rt = zk.ccs.ocs
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), zk.coefs.to_value(u.um), atol=1e-10
        )


class TestZernikesSlicing:
    def test_getitem_int(self):
        zk = _make_zernikes(n_field=5, rtp=RTP)
        s = zk[2]
        assert s.coefs.ndim == 1
        assert len(s) == 1
        assert s.rtp is RTP

    def test_getitem_slice(self):
        zk = _make_zernikes(n_field=5, rtp=RTP)
        s = zk[1:4]
        assert len(s) == 3
        assert s.R_outer == zk.R_outer

    def test_frozen(self):
        zk = _make_zernikes()
        with pytest.raises(AttributeError):
            zk.coefs = np.zeros(10) * u.um


class TestZernikesToGalsim:
    def test_1d(self):
        zk = _make_zernikes(jmax=10, R_outer=4.18 << u.m, R_inner=2.5 << u.m)
        gz = zk.to_galsim()
        assert isinstance(gz, galsim.zernike.Zernike)
        assert gz.R_outer == 4.18
        assert gz.R_inner == 2.5
        np.testing.assert_allclose(gz.coef, zk.coefs.to_value(u.um))

    def test_2d_requires_idx(self):
        zk = _make_zernikes(n_field=3, jmax=10)
        with pytest.raises((IndexError, TypeError)):
            zk.to_galsim()  # idx=None on 2-D should fail

    def test_2d_with_idx(self):
        zk = _make_zernikes(n_field=3, jmax=10)
        gz = zk.to_galsim(idx=1)
        np.testing.assert_allclose(gz.coef, zk.coefs[1].to_value(u.um))

    def test_unit_conversion(self):
        zk = _make_zernikes(jmax=10)
        gz_nm = zk.to_galsim(unit=u.nm)
        gz_um = zk.to_galsim(unit=u.um)
        np.testing.assert_allclose(gz_nm.coef, gz_um.coef * 1000, atol=1e-10)


# ===================================================================
# State
# ===================================================================


def _make_Vh(n_active, rng=None):
    """Build an orthogonal Vh matrix of shape (n_active, n_active)."""
    if rng is None:
        rng = np.random.default_rng(99)
    A = rng.standard_normal((n_active * 3, n_active))
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    return Vh


class TestStateConstruction:
    def test_from_x(self):
        use_dof = np.array([0, 2, 5])
        s = State(state=np.array([1.0, 2.0, 3.0]), basis="x", use_dof=use_dof, n_dof=10)
        assert s.basis == "x"
        assert len(s.state) == 3

    def test_from_f(self):
        fvals = np.zeros(10)
        fvals[2] = 5.0
        s = State(state=fvals, basis="f")
        assert s.basis == "f"
        assert s.n_dof == 10  # inferred

    def test_from_v(self):
        Vh = _make_Vh(5)
        s = State(
            state=np.ones(3), basis="v", Vh=Vh, nkeep=3, use_dof=np.arange(5), n_dof=10
        )
        assert s.basis == "v"
        assert s.nkeep == 3

    def test_nkeep_inferred_from_Vh(self):
        Vh = _make_Vh(5)
        s = State(state=np.ones(5), basis="x", use_dof=np.arange(5), n_dof=10, Vh=Vh)
        assert s.nkeep == 5  # inferred from Vh.shape[0]

    def test_invalid_basis(self):
        with pytest.raises(ValueError, match="basis must be"):
            State(state=np.ones(3), basis="z")

    def test_frozen(self):
        s = State(state=np.ones(3), basis="x", use_dof=np.arange(3), n_dof=5)
        with pytest.raises(AttributeError):
            s.state = np.zeros(3)

    def test_state_coerced_to_ndarray(self):
        s = State(state=[1.0, 2.0], basis="x", use_dof=np.arange(2), n_dof=5)
        assert isinstance(s.state, np.ndarray)


class TestStateConversions:
    def test_x_identity(self):
        s = State(
            state=np.array([1.0, 2.0, 3.0]),
            basis="x",
            use_dof=np.array([0, 2, 5]),
            n_dof=10,
        )
        assert s.x is s

    def test_f_identity(self):
        s = State(state=np.zeros(10), basis="f")
        assert s.f is s

    def test_v_identity(self):
        Vh = _make_Vh(5)
        s = State(
            state=np.ones(3), basis="v", Vh=Vh, nkeep=3, use_dof=np.arange(5), n_dof=10
        )
        assert s.v is s

    def test_x_to_f_roundtrip(self):
        use_dof = np.array([1, 3, 7])
        xvals = np.array([10.0, 20.0, 30.0])
        s = State(state=xvals, basis="x", use_dof=use_dof, n_dof=10)
        fs = s.f
        assert fs.basis == "f"
        assert len(fs.state) == 10
        np.testing.assert_allclose(fs.state[use_dof], xvals)
        # Inactive DOFs are zero
        inactive = np.setdiff1d(np.arange(10), use_dof)
        np.testing.assert_allclose(fs.state[inactive], 0.0)
        # Back to x
        np.testing.assert_allclose(fs.x.state, xvals)

    def test_f_to_x(self):
        use_dof = np.array([0, 4])
        fvals = np.array([5.0, 0.0, 0.0, 0.0, 9.0])
        s = State(state=fvals, basis="f", use_dof=use_dof)
        np.testing.assert_allclose(s.x.state, [5.0, 9.0])

    def test_x_to_v_roundtrip_full_rank(self):
        """When nkeep == len(use_dof), x->v->x is lossless."""
        n_active = 5
        Vh = _make_Vh(n_active)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = State(
            state=xvals,
            basis="x",
            use_dof=np.arange(n_active),
            n_dof=10,
            Vh=Vh,
            nkeep=n_active,
        )
        np.testing.assert_allclose(s.v.x.state, xvals, atol=1e-12)

    def test_x_to_v_lossy_truncated(self):
        """When nkeep < len(use_dof), x->v->x is lossy."""
        n_active = 5
        nkeep = 3
        Vh = _make_Vh(n_active)
        xvals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = State(
            state=xvals,
            basis="x",
            use_dof=np.arange(n_active),
            n_dof=10,
            Vh=Vh,
            nkeep=nkeep,
        )
        recovered = s.v.x.state
        # Not exactly equal (lossy)
        assert not np.allclose(recovered, xvals)
        # But v->x->v is lossless
        np.testing.assert_allclose(s.v.x.v.state, s.v.state, atol=1e-12)

    def test_v_to_f(self):
        n_active = 4
        use_dof = np.array([1, 3, 5, 7])
        Vh = _make_Vh(n_active)
        vvals = np.ones(3)
        s = State(state=vvals, basis="v", use_dof=use_dof, n_dof=10, Vh=Vh, nkeep=3)
        fs = s.f
        assert fs.basis == "f"
        assert len(fs.state) == 10
        # Inactive DOFs should be zero
        inactive = np.setdiff1d(np.arange(10), use_dof)
        np.testing.assert_allclose(fs.state[inactive], 0.0)

    def test_f_to_v(self):
        n_active = 4
        use_dof = np.array([1, 3, 5, 7])
        Vh = _make_Vh(n_active)
        fvals = np.zeros(10)
        fvals[use_dof] = [1.0, 2.0, 3.0, 4.0]
        s = State(state=fvals, basis="f", use_dof=use_dof, Vh=Vh, nkeep=4)
        vs = s.v
        assert vs.basis == "v"
        assert len(vs.state) == 4
        # Back through x should recover the original x-values
        np.testing.assert_allclose(vs.x.state, [1.0, 2.0, 3.0, 4.0], atol=1e-12)


class TestStateRequires:
    def test_f_to_x_requires_use_dof(self):
        s = State(state=np.zeros(10), basis="f")
        with pytest.raises(ValueError, match="use_dof"):
            s.x

    def test_x_to_f_requires_n_dof(self):
        s = State(state=np.ones(3), basis="x", use_dof=np.arange(3))
        with pytest.raises(ValueError, match="n_dof"):
            s.f

    def test_x_to_v_requires_Vh(self):
        s = State(state=np.ones(3), basis="x", use_dof=np.arange(3), n_dof=5)
        with pytest.raises(ValueError, match="Vh"):
            s.v

    def test_v_requires_Vh_at_construction(self):
        with pytest.raises(ValueError, match="Vh"):
            State(state=np.ones(3), basis="v", use_dof=np.arange(5), n_dof=10)


class TestStateFactory:
    def test_svd_computed(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 10))
        use_dof = np.array([0, 2, 4, 6, 8])
        sf = StateFactory(A=A, use_dof=use_dof, nkeep=3)
        assert sf.n_dof == 10
        assert sf.Vh.shape == (5, 5)
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
        # Full chain: x -> f -> x -> v -> x
        np.testing.assert_allclose(s.f.x.v.x.state, xvals, atol=1e-12)


# ===================================================================
# Moments
# ===================================================================


class TestMomentsConstruction:
    def test_moments2_fields(self):
        m = Moments[2](xx=1.0 * u.mm**2, xy=0.5 * u.mm**2, yy=2.0 * u.mm**2)
        assert m.xx == 1.0 * u.mm**2
        assert m.xy == 0.5 * u.mm**2
        assert m.yy == 2.0 * u.mm**2

    def test_moments3_fields(self):
        m = Moments[3](
            xxx=1.0 * u.mm**3,
            xxy=2.0 * u.mm**3,
            xyy=3.0 * u.mm**3,
            yyy=4.0 * u.mm**3,
        )
        assert m.xxx == 1.0 * u.mm**3
        assert m.xxy == 2.0 * u.mm**3
        assert m.xyy == 3.0 * u.mm**3
        assert m.yyy == 4.0 * u.mm**3

    def test_moments4_fields(self):
        m = Moments[4](
            xxxx=1.0 * u.mm**4,
            xxxy=0.0 * u.mm**4,
            xxyy=0.5 * u.mm**4,
            xyyy=0.0 * u.mm**4,
            yyyy=1.0 * u.mm**4,
        )
        assert m.xxxx == 1.0 * u.mm**4
        assert m.xxxy == 0.0 * u.mm**4
        assert m.xxyy == 0.5 * u.mm**4
        assert m.xyyy == 0.0 * u.mm**4
        assert m.yyyy == 1.0 * u.mm**4

    def test_concrete_class_moments2(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        assert isinstance(m, Moments)
        assert isinstance(m, Moments2)
        assert isinstance(m, Moments[2])

    def test_concrete_class_moments3(self):
        m = Moments3(
            xxx=0.0 * u.mm**3,
            xxy=0.0 * u.mm**3,
            xyy=0.0 * u.mm**3,
            yyy=0.0 * u.mm**3,
        )
        assert isinstance(m, Moments)
        assert isinstance(m, Moments3)
        assert isinstance(m, Moments[3])

    def test_concrete_class_moments4(self):
        m = Moments4(
            xxxx=1.0 * u.mm**4,
            xxxy=0.0 * u.mm**4,
            xxyy=0.5 * u.mm**4,
            xyyy=0.0 * u.mm**4,
            yyyy=1.0 * u.mm**4,
        )
        assert isinstance(m, Moments)
        assert isinstance(m, Moments4)
        assert isinstance(m, Moments[4])

    def test_generic_and_concrete_same_type(self):
        """Moments[2](...) and Moments2(...) produce instances of the same class."""
        m_generic = Moments[2](xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        m_concrete = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        assert type(m_generic) is type(m_concrete)

    def test_caching(self):
        """Moments[n] returns the same class object on repeated calls."""
        assert Moments[2] is Moments[2]
        assert Moments[3] is Moments[3]

    def test_frozen(self):
        """Moment instances are immutable."""
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        with pytest.raises(Exception):
            m.xx = 99.0 * u.mm**2


class TestMomentsMetadata:
    def test_default_frame_is_ocs(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        assert m.frame == "ocs"

    def test_explicit_frame_ccs(self):
        m = Moments2(
            xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2,
            frame="ccs",
        )
        assert m.frame == "ccs"

    def test_default_field_is_none(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        assert m.field is None

    def test_field_attached(self):
        fc = FieldCoords(x=0.5 * u.deg, y=0.5 * u.deg, frame="ocs")
        m = Moments2(
            xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2,
            field=fc,
        )
        assert m.field is fc

    def test_rtp_attached(self):
        rtp = Angle(0.25, unit=u.rad)
        m = Moments2(
            xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2,
            rtp=rtp,
        )
        assert m.rtp is rtp


class TestMomentsIsinstance:
    def test_moments2_isinstance_moments(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        assert isinstance(m, Moments)

    def test_moments3_isinstance_moments(self):
        m = Moments3(
            xxx=0.0 * u.mm**3, xxy=0.0 * u.mm**3,
            xyy=0.0 * u.mm**3, yyy=0.0 * u.mm**3,
        )
        assert isinstance(m, Moments)

    def test_moments4_isinstance_moments(self):
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.0 * u.mm**4,
            xxyy=0.5 * u.mm**4, xyyy=0.0 * u.mm**4, yyyy=1.0 * u.mm**4,
        )
        assert isinstance(m, Moments)


class TestMomentsRotation:
    """Tests for the generic .ocs / .ccs frame rotation on Moments."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _m2(frame='ocs', rtp=None):
        return Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame=frame, rtp=rtp,
        )

    @staticmethod
    def _m3(frame='ocs', rtp=None):
        return Moments3(
            xxx=1.0 * u.mm**3, xxy=2.0 * u.mm**3,
            xyy=3.0 * u.mm**3, yyy=4.0 * u.mm**3,
            frame=frame, rtp=rtp,
        )

    @staticmethod
    def _m4(frame='ocs', rtp=None):
        return Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
            frame=frame, rtp=rtp,
        )

    # ------------------------------------------------------------------
    # No-op when already in requested frame
    # ------------------------------------------------------------------
    def test_ocs_noop(self):
        m = self._m2(frame='ocs', rtp=RTP)
        assert m.ocs is m

    def test_ccs_noop(self):
        m = self._m2(frame='ccs', rtp=RTP)
        assert m.ccs is m

    # ------------------------------------------------------------------
    # Raises without rtp
    # ------------------------------------------------------------------
    def test_ccs_without_rtp_raises(self):
        m = self._m2(frame='ocs', rtp=None)
        with pytest.raises(ValueError, match='rtp'):
            m.ccs

    def test_ocs_without_rtp_raises(self):
        m = self._m2(frame='ccs', rtp=None)
        with pytest.raises(ValueError, match='rtp'):
            m.ocs

    # ------------------------------------------------------------------
    # Zero rotation is identity
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('make', ['_m2', '_m3', '_m4'])
    def test_zero_rotation_identity(self, make):
        m = getattr(self, make)(frame='ocs', rtp=Angle(0, unit=u.rad))
        ccs = m.ccs
        for name in ccs._moment_order * ['x']:  # just to get field list
            pass
        moment_names = [
            ''.join(p)
            for p in itertools.combinations_with_replacement('xy', m._moment_order)
        ]
        for name in moment_names:
            np.testing.assert_allclose(
                getattr(ccs, name).value,
                getattr(m, name).value,
                atol=1e-12,
                err_msg=f"{name} changed under zero rotation",
            )

    # ------------------------------------------------------------------
    # Roundtrip ocs -> ccs -> ocs
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('make', ['_m2', '_m3', '_m4'])
    def test_roundtrip_ocs_ccs_ocs(self, make):
        m = getattr(self, make)(frame='ocs', rtp=RTP)
        rt = m.ccs.ocs
        assert rt.frame == 'ocs'
        moment_names = [
            ''.join(p)
            for p in itertools.combinations_with_replacement('xy', m._moment_order)
        ]
        for name in moment_names:
            np.testing.assert_allclose(
                getattr(rt, name).value,
                getattr(m, name).value,
                atol=1e-12,
                err_msg=f"{name} not recovered after ocs->ccs->ocs",
            )

    @pytest.mark.parametrize('make', ['_m2', '_m3', '_m4'])
    def test_roundtrip_ccs_ocs_ccs(self, make):
        m = getattr(self, make)(frame='ccs', rtp=RTP)
        rt = m.ocs.ccs
        assert rt.frame == 'ccs'
        moment_names = [
            ''.join(p)
            for p in itertools.combinations_with_replacement('xy', m._moment_order)
        ]
        for name in moment_names:
            np.testing.assert_allclose(
                getattr(rt, name).value,
                getattr(m, name).value,
                atol=1e-12,
                err_msg=f"{name} not recovered after ccs->ocs->ccs",
            )

    # ------------------------------------------------------------------
    # Spin-0 invariant: trace of 2nd moments is preserved
    # ------------------------------------------------------------------
    def test_spin0_invariant_moments2(self):
        """xx + yy is invariant under rotation."""
        m = self._m2(frame='ocs', rtp=RTP)
        ccs = m.ccs
        trace_before = (m.xx + m.yy).value
        trace_after = (ccs.xx + ccs.yy).value
        np.testing.assert_allclose(trace_after, trace_before, atol=1e-12)

    def test_spin0_invariant_moments4(self):
        """xxxx + 2*xxyy + yyyy is invariant under rotation."""
        m = self._m4(frame='ocs', rtp=RTP)
        ccs = m.ccs
        inv_before = (m.xxxx + 2 * m.xxyy + m.yyyy).value
        inv_after = (ccs.xxxx + 2 * ccs.xxyy + ccs.yyyy).value
        np.testing.assert_allclose(inv_after, inv_before, atol=1e-12)

    # ------------------------------------------------------------------
    # 90-degree rotation: known analytic result for 2nd moments
    # ------------------------------------------------------------------
    def test_90deg_rotation_moments2(self):
        """Under 90-deg CCS rotation: xx' = yy, yy' = xx, xy' = -xy."""
        rtp = Angle(np.pi / 2, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        ccs = m.ccs
        np.testing.assert_allclose(ccs.xx.value, m.yy.value, atol=1e-12)
        np.testing.assert_allclose(ccs.yy.value, m.xx.value, atol=1e-12)
        np.testing.assert_allclose(ccs.xy.value, -m.xy.value, atol=1e-12)

    # ------------------------------------------------------------------
    # type and frame are preserved correctly
    # ------------------------------------------------------------------
    def test_rotation_returns_same_type(self):
        m = self._m2(frame='ocs', rtp=RTP)
        assert type(m.ccs) is type(m)

    def test_rotation_sets_frame(self):
        m = self._m2(frame='ocs', rtp=RTP)
        assert m.ccs.frame == 'ccs'
        assert m.ccs.ocs.frame == 'ocs'

    def test_rotation_preserves_rtp(self):
        m = self._m2(frame='ocs', rtp=RTP)
        assert m.ccs.rtp is RTP


# ===================================================================
# Spots.compute_moments
# ===================================================================


def _make_spots_single(n_ray: int = 200, rtp: Angle = RTP) -> Spots:
    """Single field-point spot in CCS frame."""
    rng = np.random.default_rng(7)
    dx = rng.normal(size=n_ray) * u.micron
    dy = rng.normal(0.5, 1.5, size=n_ray) * u.micron
    vig = np.zeros(n_ray, dtype=bool)
    field = FieldCoords(x=0.5 * u.deg, y=-0.3 * u.deg, frame='ocs', rtp=rtp)
    return Spots(dx=dx, dy=dy, vignetted=vig, field=field, frame='ccs', rtp=rtp)


def _make_spots_batched(n_field: int = 4, n_ray: int = 200, rtp: Angle = RTP) -> Spots:
    """Batched (n_field, n_ray) spots in CCS frame."""
    rng = np.random.default_rng(13)
    dx = rng.normal(size=(n_field, n_ray)) * u.micron
    dy = rng.normal(0.5, 1.5, size=(n_field, n_ray)) * u.micron
    vig = np.zeros((n_field, n_ray), dtype=bool)
    field = FieldCoords(
        x=rng.uniform(-1, 1, n_field) * u.deg,
        y=rng.uniform(-1, 1, n_field) * u.deg,
        frame='ocs', rtp=rtp,
    )
    return Spots(dx=dx, dy=dy, vignetted=vig, field=field, frame='ccs', rtp=rtp)


class TestSpotsComputeMoments:
    # ------------------------------------------------------------------
    # Basic: correct type, frame, field, rtp
    # ------------------------------------------------------------------
    def test_returns_moments_instance(self):
        sp = _make_spots_single()
        m = sp.compute_moments(order=2)
        assert isinstance(m, Moments)

    def test_returns_moments2(self):
        sp = _make_spots_single()
        assert isinstance(sp.compute_moments(order=2), Moments[2])

    def test_returns_moments3(self):
        sp = _make_spots_single()
        assert isinstance(sp.compute_moments(order=3), Moments[3])

    def test_returns_moments4(self):
        sp = _make_spots_single()
        assert isinstance(sp.compute_moments(order=4), Moments[4])

    def test_frame_propagated(self):
        sp = _make_spots_single()
        assert sp.compute_moments(order=2).frame == sp.frame

    def test_rtp_propagated(self):
        sp = _make_spots_single()
        assert sp.compute_moments(order=2).rtp is sp.rtp

    def test_field_propagated(self):
        sp = _make_spots_single()
        assert sp.compute_moments(order=2).field is sp.field

    def test_order_too_low_raises(self):
        sp = _make_spots_single()
        with pytest.raises(ValueError):
            sp.compute_moments(order=0)

    def test_units_are_micron_power_order(self):
        sp = _make_spots_single()
        for order in (2, 3, 4):
            m = sp.compute_moments(order=order)
            names = [
                ''.join(p)
                for p in itertools.combinations_with_replacement('xy', order)
            ]
            for name in names:
                assert getattr(m, name).unit == u.micron**order

    def test_vignetted_rays_excluded(self):
        """Moments with half the rays vignetted differ from using all rays."""
        rng = np.random.default_rng(42)
        n_ray = 100
        dx = rng.normal(size=n_ray) * u.micron
        dy = rng.normal(size=n_ray) * u.micron
        vig_none = np.zeros(n_ray, dtype=bool)
        vig_half = np.zeros(n_ray, dtype=bool)
        vig_half[:50] = True
        field = FieldCoords(x=0.0 * u.deg, y=0.0 * u.deg)
        sp_full = Spots(dx=dx, dy=dy, vignetted=vig_none, field=field)
        sp_half = Spots(dx=dx, dy=dy, vignetted=vig_half, field=field)
        m_full = sp_full.compute_moments(order=2)
        m_half = sp_half.compute_moments(order=2)
        assert not np.isclose(m_full.xx.value, m_half.xx.value)

    # ------------------------------------------------------------------
    # Batched (n_field, n_ray) spots
    # ------------------------------------------------------------------
    def test_batched_moment_fields_are_arrays(self):
        sp = _make_spots_batched(n_field=4)
        m = sp.compute_moments(order=2)
        assert m.xx.shape == (4,)
        assert m.yy.shape == (4,)
        assert m.xy.shape == (4,)

    def test_batched_agrees_with_single_per_field(self):
        """Batched compute_moments matches looping over single field-point spots."""
        sp = _make_spots_batched(n_field=3, n_ray=300)
        m_batched = sp.compute_moments(order=2)
        for i in range(3):
            m_single = sp[i].compute_moments(order=2)
            for name in ('xx', 'xy', 'yy'):
                np.testing.assert_allclose(
                    getattr(m_batched, name)[i].value,
                    getattr(m_single, name).value,
                    rtol=1e-12,
                    err_msg=f"batched vs single mismatch for {name} at field {i}",
                )

    def test_batched_vignetted_excluded(self):
        """Per-field vignetting is correctly handled in batched mode."""
        rng = np.random.default_rng(99)
        n_field, n_ray = 3, 100
        dx = rng.normal(size=(n_field, n_ray)) * u.micron
        dy = rng.normal(size=(n_field, n_ray)) * u.micron
        vig = np.zeros((n_field, n_ray), dtype=bool)
        # Vignette half the rays only for field point 1
        vig[1, :50] = True
        field = FieldCoords(
            x=np.zeros(n_field) * u.deg,
            y=np.zeros(n_field) * u.deg,
        )
        sp_vig = Spots(dx=dx, dy=dy, vignetted=vig, field=field)
        sp_none = Spots(dx=dx, dy=dy, vignetted=np.zeros_like(vig), field=field)
        m_vig = sp_vig.compute_moments(order=2)
        m_none = sp_none.compute_moments(order=2)
        # Field points 0 and 2 are unaffected
        np.testing.assert_allclose(m_vig.xx[0].value, m_none.xx[0].value)
        np.testing.assert_allclose(m_vig.xx[2].value, m_none.xx[2].value)
        # Field point 1 should differ
        assert not np.isclose(m_vig.xx[1].value, m_none.xx[1].value)

    # ------------------------------------------------------------------
    # Key consistency test: compute in CCS rotate to OCS  ==
    #                       rotate spots to OCS then compute
    # ------------------------------------------------------------------
    @pytest.mark.parametrize('order', [2, 3, 4, 5])
    def test_moments_rotate_consistent_with_spots_rotate_single(self, order):
        """Single field point: moments_ccs.ocs == spots_ocs.compute_moments."""
        sp_ccs = _make_spots_single(n_ray=500)
        assert sp_ccs.frame == 'ccs'

        m_via_rotate = sp_ccs.compute_moments(order=order).ocs
        m_via_spots = sp_ccs.ocs.compute_moments(order=order)

        assert m_via_rotate.frame == 'ocs'
        assert m_via_spots.frame == 'ocs'

        moment_names = [
            ''.join(p)
            for p in itertools.combinations_with_replacement('xy', order)
        ]
        for name in moment_names:
            np.testing.assert_allclose(
                getattr(m_via_rotate, name).value,
                getattr(m_via_spots, name).value,
                rtol=1e-10,
                err_msg=(
                    f"order={order}, moment {name}: "
                    "moments_ccs.ocs != spots_ocs.compute_moments"
                ),
            )

    @pytest.mark.parametrize('order', [2, 3, 4, 5])
    def test_moments_rotate_consistent_with_spots_rotate_batched(self, order):
        """Batched field points: moments_ccs.ocs == spots_ocs.compute_moments."""
        sp_ccs = _make_spots_batched(n_field=4, n_ray=500)
        assert sp_ccs.frame == 'ccs'

        m_via_rotate = sp_ccs.compute_moments(order=order).ocs
        m_via_spots = sp_ccs.ocs.compute_moments(order=order)

        assert m_via_rotate.frame == 'ocs'
        assert m_via_spots.frame == 'ocs'

        moment_names = [
            ''.join(p)
            for p in itertools.combinations_with_replacement('xy', order)
        ]
        for name in moment_names:
            np.testing.assert_allclose(
                getattr(m_via_rotate, name).value,
                getattr(m_via_spots, name).value,
                rtol=1e-10,
                err_msg=(
                    f"order={order}, moment {name}: "
                    "batched moments_ccs.ocs != spots_ocs.compute_moments"
                ),
            )


# ===================================================================
# Moments.spin and Moments2 properties
# ===================================================================


class TestMoments2Properties:
    """Tests for T, e1, e2 on Moments2."""

    def test_T_value(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        assert m.T.value == pytest.approx(5.0)
        assert m.T.unit == u.mm**2

    def test_e1_value(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        assert m.e1.value == pytest.approx(-1.0 / 5.0)

    def test_e2_value(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        assert m.e2.value == pytest.approx(2.0 / 5.0)

    def test_e1_e2_dimensionless(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        assert m.e1.unit == u.dimensionless_unscaled
        assert m.e2.unit == u.dimensionless_unscaled


class TestMomentsSpin:
    """Tests for the generic spin(n, m) decomposition."""

    # ------------------------------------------------------------------
    # Spin-0 components equal known expressions
    # ------------------------------------------------------------------
    def test_spin00_moments2(self):
        """spin(0, 0) for order 2 is xx + yy."""
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        np.testing.assert_allclose(m.spin(0, 0).value, (m.xx + m.yy).value, atol=1e-12)

    def test_spin22_moments2(self):
        """spin(2, 2) for order 2 is xx - yy."""
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        np.testing.assert_allclose(m.spin(2, 2).value, (m.xx - m.yy).value, atol=1e-12)

    def test_spin2_neg2_moments2(self):
        """spin(2, -2) for order 2 is 2*xy."""
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        np.testing.assert_allclose(m.spin(2, -2).value, (2 * m.xy).value, atol=1e-12)

    def test_spin00_moments4(self):
        """spin(0, 0) for order 4 is xxxx + 2*xxyy + yyyy."""
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
        )
        expected = (m.xxxx + 2 * m.xxyy + m.yyyy).value
        np.testing.assert_allclose(m.spin(0, 0).value, expected, atol=1e-12)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def test_invalid_spin_raises(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        with pytest.raises(ValueError, match="Invalid spin"):
            m.spin(1, 1)  # wrong parity for order 2

    def test_invalid_m_raises(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        with pytest.raises(ValueError, match="m must be"):
            m.spin(2, 0)  # m must be +/-2

    def test_spin0_m_nonzero_raises(self):
        m = Moments2(xx=1.0 * u.mm**2, xy=0.0 * u.mm**2, yy=1.0 * u.mm**2)
        with pytest.raises(ValueError, match="m must be 0"):
            m.spin(0, 1)

    # ------------------------------------------------------------------
    # Spin-n invariance under 2pi/n rotations
    # ------------------------------------------------------------------
    def test_spin2_invariant_under_180deg(self):
        """Spin-2 components of order-2 moments are invariant under 180-deg rotation."""
        rtp = Angle(np.pi, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (2, -2):
            np.testing.assert_allclose(
                m_rot.spin(2, mm).value,
                m.spin(2, mm).value,
                atol=1e-12,
                err_msg=f"spin(2, {mm}) not invariant under 180-deg rotation",
            )

    def test_spin0_invariant_under_180deg(self):
        """Spin-0 component of order-2 moments is invariant under 180-deg rotation."""
        rtp = Angle(np.pi, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        np.testing.assert_allclose(
            m_rot.spin(0, 0).value, m.spin(0, 0).value, atol=1e-12,
        )

    def test_spin3_invariant_under_120deg(self):
        """Spin-3 components of order-3 moments are invariant under 120-deg rotation."""
        rtp = Angle(2 * np.pi / 3, unit=u.rad)
        m = Moments3(
            xxx=1.0 * u.mm**3, xxy=2.0 * u.mm**3,
            xyy=3.0 * u.mm**3, yyy=4.0 * u.mm**3,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (3, -3):
            np.testing.assert_allclose(
                m_rot.spin(3, mm).value,
                m.spin(3, mm).value,
                atol=1e-12,
                err_msg=f"spin(3, {mm}) not invariant under 120-deg rotation",
            )

    def test_spin1_invariant_under_360deg(self):
        """Spin-1 components of order-3 moments are invariant under 360-deg rotation."""
        rtp = Angle(2 * np.pi, unit=u.rad)
        m = Moments3(
            xxx=1.0 * u.mm**3, xxy=2.0 * u.mm**3,
            xyy=3.0 * u.mm**3, yyy=4.0 * u.mm**3,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (1, -1):
            np.testing.assert_allclose(
                m_rot.spin(1, mm).value,
                m.spin(1, mm).value,
                atol=1e-12,
                err_msg=f"spin(1, {mm}) not invariant under 360-deg rotation",
            )

    def test_spin4_invariant_under_90deg(self):
        """Spin-4 components of order-4 moments are invariant under 90-deg rotation."""
        rtp = Angle(np.pi / 2, unit=u.rad)
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (4, -4):
            np.testing.assert_allclose(
                m_rot.spin(4, mm).value,
                m.spin(4, mm).value,
                atol=1e-12,
                err_msg=f"spin(4, {mm}) not invariant under 90-deg rotation",
            )

    def test_spin2_invariant_under_180deg_order4(self):
        """Spin-2 components of order-4 moments are invariant under 180-deg rotation."""
        rtp = Angle(np.pi, unit=u.rad)
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (2, -2):
            np.testing.assert_allclose(
                m_rot.spin(2, mm).value,
                m.spin(2, mm).value,
                atol=1e-12,
                err_msg=f"spin(2, {mm}) of order-4 not invariant under 180-deg rotation",
            )

    def test_spin0_invariant_under_arbitrary_rotation(self):
        """Spin-0 components are invariant under any rotation angle."""
        rtp = Angle(1.234, unit=u.rad)
        m2 = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        m4 = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
            frame='ocs', rtp=rtp,
        )
        np.testing.assert_allclose(
            m2.ccs.spin(0, 0).value, m2.spin(0, 0).value, atol=1e-12,
        )
        np.testing.assert_allclose(
            m4.ccs.spin(0, 0).value, m4.spin(0, 0).value, atol=1e-12,
        )

    def test_spin_not_invariant_under_wrong_angle(self):
        """Spin-2 components should NOT be invariant under 90-deg rotation."""
        rtp = Angle(np.pi / 2, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        # spin-2 is NOT invariant under 90 deg (only 180 deg)
        assert not np.allclose(m_rot.spin(2, 2).value, m.spin(2, 2).value, atol=1e-12)

    # ------------------------------------------------------------------
    # Batched moments
    # ------------------------------------------------------------------
    def test_spin_batched(self):
        """spin() works on batched moments from compute_moments."""
        sp = _make_spots_batched(n_field=4, n_ray=500)
        m2 = sp.compute_moments(order=2)
        # spin(0,0) should be array of length 4
        s00 = m2.spin(0, 0)
        assert s00.value.shape == (4,)
        np.testing.assert_allclose(s00.value, (m2.xx + m2.yy).value, atol=1e-12)
