"""Tests for FieldCoords."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import FieldCoords

from .utils import RTP, _make_field, roundtrip_asdf, roundtrip_asdf_ctx

class TestFieldCoordsConstruction:
    def test_scalar_promoted_to_1d(self):
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg)
        assert fc.x.ndim == 1
        assert fc.y.ndim == 1

    def test_1d_stays_1d(self):
        fc = _make_field(5)
        assert fc.x.ndim == 1
        assert len(fc) == 5  # 1-D -> 5 field points

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
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, rtp=RTP)
        fp = fc.focal_plane.angle
        np.testing.assert_allclose(
            fp.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            fp.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_roundtrip_focal_plane_angle(self):
        fc = FieldCoords(x=1.0 * u.mm, y=2.0 * u.mm, rtp=RTP)
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


class TestFieldCoordsEDCSDVCS:
    def test_edcs_noop_when_already_edcs(self):
        fc = _make_field(frame="edcs", rtp=RTP)
        assert fc.edcs is fc
        assert fc.frame == "edcs"

    def test_dvcs_noop_when_already_dvcs(self):
        fc = _make_field(frame="dvcs", rtp=RTP)
        assert fc.dvcs is fc
        assert fc.frame == "dvcs"

    def test_ccs_from_edcs_relabels(self):
        fc = _make_field(frame="edcs", rtp=RTP)
        ccs = fc.ccs
        assert ccs.frame == "ccs"
        np.testing.assert_allclose(ccs.x, fc.x)
        np.testing.assert_allclose(ccs.y, fc.y)

    def test_edcs_from_ccs_relabels(self):
        fc = _make_field(frame="ccs", rtp=RTP)
        edcs = fc.edcs
        assert edcs.frame == "edcs"
        np.testing.assert_allclose(edcs.x, fc.x)
        np.testing.assert_allclose(edcs.y, fc.y)

    def test_edcs_from_ocs(self):
        fc = _make_field(frame="ocs", rtp=RTP)
        edcs = fc.edcs
        # Should match ccs, but with frame 'edcs'
        ccs = fc.ccs
        assert edcs.frame == "edcs"
        np.testing.assert_allclose(edcs.x, ccs.x)
        np.testing.assert_allclose(edcs.y, ccs.y)

    def test_dvcs_from_edcs(self):
        fc = _make_field(frame="edcs", rtp=RTP)
        dvcs = fc.dvcs
        assert dvcs.frame == "dvcs"
        np.testing.assert_allclose(dvcs.x, fc.y)
        np.testing.assert_allclose(dvcs.y, fc.x)

    def test_dvcs_from_ccs(self):
        fc = _make_field(frame="ccs", rtp=RTP)
        dvcs = fc.dvcs
        assert dvcs.frame == "dvcs"
        np.testing.assert_allclose(dvcs.x, fc.y)
        np.testing.assert_allclose(dvcs.y, fc.x)

    def test_edcs_then_dvcs_then_edcs_roundtrip(self):
        fc = _make_field(frame="edcs", rtp=RTP)
        dvcs = fc.dvcs
        edcs2 = dvcs.edcs
        assert edcs2.frame == "edcs"
        np.testing.assert_allclose(edcs2.x, fc.x)
        np.testing.assert_allclose(edcs2.y, fc.y)

    def test_dvcs_then_edcs_then_dvcs_roundtrip(self):
        fc = _make_field(frame="dvcs", rtp=RTP)
        edcs = fc.edcs
        dvcs2 = edcs.dvcs
        assert dvcs2.frame == "dvcs"
        np.testing.assert_allclose(dvcs2.x, fc.x)
        np.testing.assert_allclose(dvcs2.y, fc.y)

    def test_ocs_from_dvcs(self):
        fc = _make_field(frame="dvcs", rtp=RTP)
        ocs = fc.ocs
        dvcs2 = ocs.dvcs
        assert dvcs2.frame == "dvcs"
        np.testing.assert_allclose(dvcs2.x, fc.x)
        np.testing.assert_allclose(dvcs2.y, fc.y)


class TestFieldCoordsFrameCase:
    def test_frame_coerced_to_lower(self):
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="OCS")
        assert fc.frame == "ocs"
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="CCS")
        assert fc.frame == "ccs"
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="EDCS")
        assert fc.frame == "edcs"
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="DVCS")
        assert fc.frame == "dvcs"

    def test_mixed_case_frame(self):
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="oCs")
        assert fc.frame == "ocs"
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="cCs")
        assert fc.frame == "ccs"
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="EdCs")
        assert fc.frame == "edcs"
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="DvCs")
        assert fc.frame == "dvcs"


class TestFieldCoordsMultiStepRoundtrip:
    """Multi-step roundtrips mixing frame and angle/focal-plane conversions."""

    def test_ccs_angle_to_dvcs_fp_to_ccs_angle(self):
        # CCS angle -> DVCS angle -> DVCS fp -> CCS fp -> CCS angle
        fc = _make_field(frame="ccs", rtp=RTP)
        rt = fc.dvcs.focal_plane.ccs.angle
        assert rt.frame == "ccs"
        assert rt.space == "angle"
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_ocs_angle_to_ccs_fp_to_dvcs_angle_to_ocs(self):
        # OCS angle -> CCS angle -> CCS fp -> DVCS fp -> DVCS angle -> OCS angle
        fc = _make_field(frame="ocs", rtp=RTP)
        rt = fc.ccs.focal_plane.dvcs.angle.ocs
        assert rt.frame == "ocs"
        assert rt.space == "angle"
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_dvcs_fp_to_ocs_angle_to_dvcs_fp(self):
        # DVCS fp -> DVCS angle -> OCS angle -> OCS fp -> DVCS fp
        fc = FieldCoords(
            x=np.array([10.0, -5.0, 20.0]) * u.mm,
            y=np.array([-15.0, 8.0, 3.0]) * u.mm,
            frame="dvcs", rtp=RTP,
        )
        rt = fc.angle.ocs.focal_plane.dvcs
        assert rt.frame == "dvcs"
        assert rt.space == "focal_plane"
        np.testing.assert_allclose(
            rt.x.to_value(u.mm), fc.x.to_value(u.mm), atol=1e-10
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.mm), fc.y.to_value(u.mm), atol=1e-10
        )

    def test_ocs_angle_fp_angle_roundtrip(self):
        # OCS angle -> OCS fp -> OCS angle
        fc = _make_field(frame="ocs", rtp=RTP)
        rt = fc.focal_plane.angle
        assert rt.frame == "ocs"
        assert rt.space == "angle"
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_ccs_fp_to_ocs_angle_to_edcs_fp(self):
        # CCS fp -> CCS angle -> OCS angle -> EDCS angle -> EDCS fp -> CCS fp
        fc = FieldCoords(
            x=np.array([12.0, -7.0]) * u.mm,
            y=np.array([-3.0, 18.0]) * u.mm,
            frame="ccs", rtp=RTP,
        )
        rt = fc.angle.ocs.edcs.focal_plane.ccs
        assert rt.frame == "ccs"
        assert rt.space == "focal_plane"
        np.testing.assert_allclose(
            rt.x.to_value(u.mm), fc.x.to_value(u.mm), atol=1e-10
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.mm), fc.y.to_value(u.mm), atol=1e-10
        )

    def test_dvcs_angle_through_all_frames_and_spaces(self):
        # DVCS angle -> CCS angle -> CCS fp -> OCS fp -> OCS angle -> DVCS angle
        fc = _make_field(frame="dvcs", rtp=RTP)
        rt = fc.ccs.focal_plane.ocs.angle.dvcs
        assert rt.frame == "dvcs"
        assert rt.space == "angle"
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-12
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-12
        )

    def test_double_space_roundtrip_with_frame_hops(self):
        # OCS angle -> DVCS fp -> CCS angle -> OCS fp -> DVCS angle -> OCS angle
        fc = _make_field(frame="ocs", rtp=RTP)
        rt = fc.dvcs.focal_plane.ccs.angle.ocs.focal_plane.dvcs.angle.ocs
        assert rt.frame == "ocs"
        assert rt.space == "angle"
        np.testing.assert_allclose(
            rt.x.to_value(u.deg), fc.x.to_value(u.deg), atol=1e-10
        )
        np.testing.assert_allclose(
            rt.y.to_value(u.deg), fc.y.to_value(u.deg), atol=1e-10
        )


# ---------------------------------------------------------------------------
# ASDF round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("asdf", reason="asdf not installed") is None,
    reason="asdf not installed",
)
class TestFieldCoordsAsdf:
    asdf = pytest.importorskip("asdf")

    def test_roundtrip_angle_no_rtp(self):
        fc = FieldCoords(x=1.5 * u.deg, y=-0.5 * u.deg)
        rt = roundtrip_asdf_ctx(fc)
        assert rt.frame == "ocs"
        assert rt.rtp is None
        np.testing.assert_allclose(rt.x.to_value(u.deg), fc.x.to_value(u.deg))
        np.testing.assert_allclose(rt.y.to_value(u.deg), fc.y.to_value(u.deg))

    def test_roundtrip_with_rtp(self):
        rtp = Angle(0.3, unit=u.rad)
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="ocs", rtp=rtp)
        rt = roundtrip_asdf_ctx(fc)
        assert rt.frame == "ocs"
        assert isinstance(rt.rtp, Angle)
        np.testing.assert_allclose(rt.rtp.rad, rtp.rad)

    def test_roundtrip_focal_plane(self):
        fc = FieldCoords(x=10.0 * u.mm, y=-5.0 * u.mm)
        rt = roundtrip_asdf_ctx(fc)
        assert rt.space == "focal_plane"
        np.testing.assert_allclose(rt.x.to_value(u.mm), fc.x.to_value(u.mm))
        np.testing.assert_allclose(rt.y.to_value(u.mm), fc.y.to_value(u.mm))

    def test_roundtrip_batched(self):
        x = np.linspace(-1.0, 1.0, 5) * u.deg
        y = np.zeros(5) * u.deg
        fc = FieldCoords(x=x, y=y)
        rt = roundtrip_asdf_ctx(fc)
        assert rt.x.shape == (5,)
        np.testing.assert_allclose(rt.x.to_value(u.deg), fc.x.to_value(u.deg))
        np.testing.assert_allclose(rt.y.to_value(u.deg), fc.y.to_value(u.deg))

    def test_roundtrip_frame_ccs(self):
        rtp = Angle(0.5, unit=u.rad)
        fc = FieldCoords(x=0.5 * u.deg, y=0.5 * u.deg, frame="ccs", rtp=rtp)
        rt = roundtrip_asdf_ctx(fc)
        assert rt.frame == "ccs"

    def test_roundtrip_preserves_units(self):
        fc = FieldCoords(x=1800.0 * u.arcsec, y=900.0 * u.arcsec)
        rt = roundtrip_asdf_ctx(fc)
        np.testing.assert_allclose(
            rt.x.to_value(u.arcsec), fc.x.to_value(u.arcsec)
        )


# ---------------------------------------------------------------------------
# Same round-trips via the installed entry-point (no config_context)
# ---------------------------------------------------------------------------

from .conftest import requires_starsharp_asdf  # noqa: E402


@requires_starsharp_asdf
class TestFieldCoordsAsdfEntryPoint:
    """Re-runs a representative subset using the auto-discovered extension."""

    def test_roundtrip_angle_no_rtp(self):
        fc = FieldCoords(x=1.5 * u.deg, y=-0.5 * u.deg)
        rt = roundtrip_asdf(fc)
        assert rt.frame == "ocs"
        assert rt.rtp is None
        np.testing.assert_allclose(rt.x.to_value(u.deg), fc.x.to_value(u.deg))
        np.testing.assert_allclose(rt.y.to_value(u.deg), fc.y.to_value(u.deg))

    def test_roundtrip_with_rtp(self):
        rtp = Angle(0.3, unit=u.rad)
        fc = FieldCoords(x=1.0 * u.deg, y=2.0 * u.deg, frame="ocs", rtp=rtp)
        rt = roundtrip_asdf(fc)
        assert isinstance(rt.rtp, Angle)
        np.testing.assert_allclose(rt.rtp.rad, rtp.rad)

    def test_roundtrip_focal_plane(self):
        fc = FieldCoords(x=10.0 * u.mm, y=-5.0 * u.mm)
        rt = roundtrip_asdf(fc)
        assert rt.space == "focal_plane"
        np.testing.assert_allclose(rt.x.to_value(u.mm), fc.x.to_value(u.mm))
        np.testing.assert_allclose(rt.y.to_value(u.mm), fc.y.to_value(u.mm))

