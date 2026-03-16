"""Tests for FieldCoords."""
from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import FieldCoords
from .utils import RTP, _make_wcs, _make_field


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
