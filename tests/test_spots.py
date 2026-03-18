"""Tests for Spots."""
from __future__ import annotations

import itertools

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import FieldCoords, Moments, Spots
from .utils import RTP, _make_wcs, _make_field, _make_spots, _make_spots_single, _make_spots_batched


class TestSpotsConstruction:
    def test_scalar_promoted(self):
        fc = FieldCoords(x=1.0 * u.deg, y=0.5 * u.deg)
        sp = Spots(
            dx=0.1 * u.um,
            dy=0.2 * u.um,
            vignetted=np.array(False),
            field=fc,
        )
        assert sp.dx.ndim == 2
        assert sp.dy.ndim == 2
        assert sp.vignetted.ndim == 2

    def test_2d_shape_preserved(self):
        sp = _make_spots(4, 100, rtp=RTP)
        assert sp.dx.shape == (4, 100)
        assert len(sp) == 4

    def test_1d_promoted_to_2d(self):
        fc = FieldCoords(x=1.0 * u.deg, y=0.5 * u.deg)
        sp = Spots(
            dx=np.ones(10) * u.um,
            dy=np.ones(10) * u.um,
            vignetted=np.zeros(10, dtype=bool),
            field=fc,
        )
        assert sp.dx.ndim == 2
        assert sp.dx.shape == (1, 10)
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
        assert s.dx.ndim == 2
        assert s.dx.shape == (1, 20)
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


class TestSpotsComputeMoments:
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
        vig[1, :50] = True
        field = FieldCoords(
            x=np.zeros(n_field) * u.deg,
            y=np.zeros(n_field) * u.deg,
        )
        sp_vig = Spots(dx=dx, dy=dy, vignetted=vig, field=field)
        sp_none = Spots(dx=dx, dy=dy, vignetted=np.zeros_like(vig), field=field)
        m_vig = sp_vig.compute_moments(order=2)
        m_none = sp_none.compute_moments(order=2)
        np.testing.assert_allclose(m_vig.xx[0].value, m_none.xx[0].value)
        np.testing.assert_allclose(m_vig.xx[2].value, m_none.xx[2].value)
        assert not np.isclose(m_vig.xx[1].value, m_none.xx[1].value)

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


class TestSpotsEDCSDVCS:
    def test_edcs_noop_when_already_edcs(self):
        sp = _make_spots(frame="edcs", rtp=RTP)
        assert sp.edcs is sp
        assert sp.frame == "edcs"

    def test_dvcs_noop_when_already_dvcs(self):
        sp = _make_spots(frame="dvcs", rtp=RTP)
        assert sp.dvcs is sp
        assert sp.frame == "dvcs"

    def test_ccs_from_edcs_relabels(self):
        sp = _make_spots(frame="edcs", rtp=RTP)
        ccs = sp.ccs
        assert ccs.frame == "ccs"
        np.testing.assert_allclose(ccs.dx, sp.dx)
        np.testing.assert_allclose(ccs.dy, sp.dy)

    def test_edcs_from_ccs_relabels(self):
        sp = _make_spots(frame="ccs", rtp=RTP)
        edcs = sp.edcs
        assert edcs.frame == "edcs"
        np.testing.assert_allclose(edcs.dx, sp.dx)
        np.testing.assert_allclose(edcs.dy, sp.dy)

    def test_edcs_from_ocs(self):
        sp = _make_spots(frame="ocs", rtp=RTP)
        edcs = sp.edcs
        ccs = sp.ccs
        assert edcs.frame == "edcs"
        np.testing.assert_allclose(edcs.dx, ccs.dx)
        np.testing.assert_allclose(edcs.dy, ccs.dy)

    def test_dvcs_from_edcs(self):
        sp = _make_spots(frame="edcs", rtp=RTP)
        dvcs = sp.dvcs
        assert dvcs.frame == "dvcs"
        np.testing.assert_allclose(dvcs.dx, sp.dy)
        np.testing.assert_allclose(dvcs.dy, sp.dx)

    def test_dvcs_from_ccs(self):
        sp = _make_spots(frame="ccs", rtp=RTP)
        dvcs = sp.dvcs
        assert dvcs.frame == "dvcs"
        np.testing.assert_allclose(dvcs.dx, sp.dy)
        np.testing.assert_allclose(dvcs.dy, sp.dx)

    def test_edcs_then_dvcs_then_edcs_roundtrip(self):
        sp = _make_spots(frame="edcs", rtp=RTP)
        dvcs = sp.dvcs
        edcs2 = dvcs.edcs
        assert edcs2.frame == "edcs"
        np.testing.assert_allclose(edcs2.dx, sp.dx)
        np.testing.assert_allclose(edcs2.dy, sp.dy)

    def test_dvcs_then_edcs_then_dvcs_roundtrip(self):
        sp = _make_spots(frame="dvcs", rtp=RTP)
        edcs = sp.edcs
        dvcs2 = edcs.dvcs
        assert dvcs2.frame == "dvcs"
        np.testing.assert_allclose(dvcs2.dx, sp.dx)
        np.testing.assert_allclose(dvcs2.dy, sp.dy)

    def test_ocs_from_dvcs(self):
        sp = _make_spots(frame="dvcs", rtp=RTP)
        ocs = sp.ocs
        dvcs2 = ocs.dvcs
        assert dvcs2.frame == "dvcs"
        np.testing.assert_allclose(dvcs2.dx, sp.dx)
        np.testing.assert_allclose(dvcs2.dy, sp.dy)


class TestSpotsFrameCase:
    def test_frame_coerced_to_lower(self):
        sp = _make_spots(frame="OCS", rtp=RTP)
        assert sp.frame == "ocs"
        sp = _make_spots(frame="CCS", rtp=RTP)
        assert sp.frame == "ccs"
        sp = _make_spots(frame="EDCS", rtp=RTP)
        assert sp.frame == "edcs"
        sp = _make_spots(frame="DVCS", rtp=RTP)
        assert sp.frame == "dvcs"

    def test_mixed_case_frame(self):
        sp = _make_spots(frame="oCs", rtp=RTP)
        assert sp.frame == "ocs"
        sp = _make_spots(frame="cCs", rtp=RTP)
        assert sp.frame == "ccs"
        sp = _make_spots(frame="EdCs", rtp=RTP)
        assert sp.frame == "edcs"
        sp = _make_spots(frame="DvCs", rtp=RTP)
        assert sp.frame == "dvcs"

    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame must be one of"):
            _make_spots(frame="notaframe", rtp=RTP)
