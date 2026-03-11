"""Unit tests for FieldCoords, Spots, and Zernikes."""
from __future__ import annotations

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import Angle

import galsim

from datatypes import FieldCoords, Spots, Zernikes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_field(
    n: int = 3,
    frame: str = "ocs",
    rtp: Angle | None = None,
    unit: u.Unit = u.deg,
) -> FieldCoords:
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.5, 1.5, n) * unit
    y = rng.uniform(-1.5, 1.5, n) * unit
    return FieldCoords(x=x, y=y, frame=frame, rtp=rtp)


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
        np.testing.assert_allclose(rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12)
        np.testing.assert_allclose(rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12)
        assert rt.frame == "ocs"

    def test_ccs_then_ocs_roundtrip(self):
        fc = _make_field(frame="ccs", rtp=RTP)
        rt = fc.ocs.ccs
        np.testing.assert_allclose(rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12)
        np.testing.assert_allclose(rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12)
        assert rt.frame == "ccs"

    def test_zero_rtp_is_identity(self):
        fc = _make_field(frame="ocs", rtp=Angle(0, unit=u.rad))
        ccs = fc.ccs
        np.testing.assert_allclose(ccs.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12)
        np.testing.assert_allclose(ccs.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12)

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
) -> Spots:
    rng = np.random.default_rng(12)
    dx = rng.normal(size=(n_field, n_ray)) * u.um
    dy = rng.normal(size=(n_field, n_ray)) * u.um
    vig = np.zeros((n_field, n_ray), dtype=bool)
    field = _make_field(n_field, frame="ocs", rtp=rtp)
    return Spots(
        dx=dx,
        dy=dy,
        vignetted=vig,
        field=field,
        wavelength=622.0 * u.nm,
        frame=frame,
        rtp=rtp,
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
    R_outer: float = 4.18,
    R_inner: float = 2.5,
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
        zk = _make_zernikes(R_outer=4.18, R_inner=2.5)
        assert zk.eps == pytest.approx(2.5 / 4.18)

    def test_eps_zero_outer(self):
        field = _make_field(1)
        zk = Zernikes(coefs=np.zeros(5) * u.um, field=field, R_outer=0.0)
        assert zk.eps == 0.0


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
        zk = _make_zernikes(jmax=10, R_outer=4.18, R_inner=2.5)
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
