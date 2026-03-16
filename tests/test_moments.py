"""Tests for Moments, Moments2, Moments3, Moments4."""
from __future__ import annotations

import itertools

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import FieldCoords, Moments, Moments2, Moments3, Moments4
from .utils import RTP, _make_spots_batched


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

    def test_ocs_noop(self):
        m = self._m2(frame='ocs', rtp=RTP)
        assert m.ocs is m

    def test_ccs_noop(self):
        m = self._m2(frame='ccs', rtp=RTP)
        assert m.ccs is m

    def test_ccs_without_rtp_raises(self):
        m = self._m2(frame='ocs', rtp=None)
        with pytest.raises(ValueError, match='rtp'):
            m.ccs

    def test_ocs_without_rtp_raises(self):
        m = self._m2(frame='ccs', rtp=None)
        with pytest.raises(ValueError, match='rtp'):
            m.ocs

    @pytest.mark.parametrize('make', ['_m2', '_m3', '_m4'])
    def test_zero_rotation_identity(self, make):
        m = getattr(self, make)(frame='ocs', rtp=Angle(0, unit=u.rad))
        ccs = m.ccs
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

    def test_spin0_invariant_moments2(self):
        m = self._m2(frame='ocs', rtp=RTP)
        ccs = m.ccs
        np.testing.assert_allclose(
            (ccs.xx + ccs.yy).value, (m.xx + m.yy).value, atol=1e-12
        )

    def test_spin0_invariant_moments4(self):
        m = self._m4(frame='ocs', rtp=RTP)
        ccs = m.ccs
        inv_before = (m.xxxx + 2 * m.xxyy + m.yyyy).value
        inv_after = (ccs.xxxx + 2 * ccs.xxyy + ccs.yyyy).value
        np.testing.assert_allclose(inv_after, inv_before, atol=1e-12)

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


class TestMoments2Properties:
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
    def test_spin00_moments2(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        np.testing.assert_allclose(m.spin(0, 0).value, (m.xx + m.yy).value, atol=1e-12)

    def test_spin22_moments2(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        np.testing.assert_allclose(m.spin(2, 2).value, (m.xx - m.yy).value, atol=1e-12)

    def test_spin2_neg2_moments2(self):
        m = Moments2(xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2)
        np.testing.assert_allclose(m.spin(2, -2).value, (2 * m.xy).value, atol=1e-12)

    def test_spin00_moments4(self):
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
        )
        expected = (m.xxxx + 2 * m.xxyy + m.yyyy).value
        np.testing.assert_allclose(m.spin(0, 0).value, expected, atol=1e-12)

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

    def test_spin2_invariant_under_180deg(self):
        rtp = Angle(np.pi, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (2, -2):
            np.testing.assert_allclose(
                m_rot.spin(2, mm).value, m.spin(2, mm).value, atol=1e-12,
                err_msg=f"spin(2, {mm}) not invariant under 180-deg rotation",
            )

    def test_spin0_invariant_under_180deg(self):
        rtp = Angle(np.pi, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        np.testing.assert_allclose(
            m.ccs.spin(0, 0).value, m.spin(0, 0).value, atol=1e-12,
        )

    def test_spin3_invariant_under_120deg(self):
        rtp = Angle(2 * np.pi / 3, unit=u.rad)
        m = Moments3(
            xxx=1.0 * u.mm**3, xxy=2.0 * u.mm**3,
            xyy=3.0 * u.mm**3, yyy=4.0 * u.mm**3,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (3, -3):
            np.testing.assert_allclose(
                m_rot.spin(3, mm).value, m.spin(3, mm).value, atol=1e-12,
                err_msg=f"spin(3, {mm}) not invariant under 120-deg rotation",
            )

    def test_spin1_invariant_under_360deg(self):
        rtp = Angle(2 * np.pi, unit=u.rad)
        m = Moments3(
            xxx=1.0 * u.mm**3, xxy=2.0 * u.mm**3,
            xyy=3.0 * u.mm**3, yyy=4.0 * u.mm**3,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (1, -1):
            np.testing.assert_allclose(
                m_rot.spin(1, mm).value, m.spin(1, mm).value, atol=1e-12,
                err_msg=f"spin(1, {mm}) not invariant under 360-deg rotation",
            )

    def test_spin4_invariant_under_90deg(self):
        rtp = Angle(np.pi / 2, unit=u.rad)
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (4, -4):
            np.testing.assert_allclose(
                m_rot.spin(4, mm).value, m.spin(4, mm).value, atol=1e-12,
                err_msg=f"spin(4, {mm}) not invariant under 90-deg rotation",
            )

    def test_spin2_invariant_under_180deg_order4(self):
        rtp = Angle(np.pi, unit=u.rad)
        m = Moments4(
            xxxx=1.0 * u.mm**4, xxxy=0.5 * u.mm**4, xxyy=0.25 * u.mm**4,
            xyyy=0.5 * u.mm**4, yyyy=1.0 * u.mm**4,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        for mm in (2, -2):
            np.testing.assert_allclose(
                m_rot.spin(2, mm).value, m.spin(2, mm).value, atol=1e-12,
                err_msg=f"spin(2, {mm}) of order-4 not invariant under 180-deg rotation",
            )

    def test_spin0_invariant_under_arbitrary_rotation(self):
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
        rtp = Angle(np.pi / 2, unit=u.rad)
        m = Moments2(
            xx=2.0 * u.mm**2, xy=1.0 * u.mm**2, yy=3.0 * u.mm**2,
            frame='ocs', rtp=rtp,
        )
        m_rot = m.ccs
        assert not np.allclose(m_rot.spin(2, 2).value, m.spin(2, 2).value, atol=1e-12)

    def test_spin_batched(self):
        """spin() works on batched moments from compute_moments."""
        sp = _make_spots_batched(n_field=4, n_ray=500)
        m2 = sp.compute_moments(order=2)
        s00 = m2.spin(0, 0)
        assert s00.value.shape == (4,)
        np.testing.assert_allclose(s00.value, (m2.xx + m2.yy).value, atol=1e-12)
