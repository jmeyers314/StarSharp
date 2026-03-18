"""Tests for Zernikes."""

from __future__ import annotations

import astropy.units as u
import galsim
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import FieldCoords, Zernikes

from .utils import RTP, _make_field


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

    def test_jmax_inferred_from_shape(self):
        coefs = np.zeros(23) * u.um
        field = _make_field(1)
        zk = Zernikes(coefs=coefs, field=field)
        assert zk.jmax == 22

    def test_scalar_coefs_promoted(self):
        field = _make_field(1)
        zk = Zernikes(coefs=1.0 * u.um, field=field)
        assert zk.coefs.ndim == 2

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
        assert s.coefs.ndim == 2
        assert s.coefs.shape == (1, zk.jmax + 1)
        assert s.rtp is RTP

    def test_getitem_slice(self):
        zk = _make_zernikes(n_field=5, rtp=RTP)
        s = zk[1:4]
        assert s.coefs.shape[0] == 3
        assert s.R_outer == zk.R_outer

    def test_frozen(self):
        zk = _make_zernikes()
        with pytest.raises(AttributeError):
            zk.coefs = np.zeros(10) * u.um


class TestZernikesToGalsim:
    def test_single_field(self):
        zk = _make_zernikes(n_field=1, jmax=10, R_outer=4.18 << u.m, R_inner=2.5 << u.m)
        gz = zk.to_galsim()
        assert isinstance(gz, galsim.zernike.Zernike)
        assert gz.R_outer == 4.18
        assert gz.R_inner == 2.5
        np.testing.assert_allclose(gz.coef, zk.coefs.squeeze().to_value(u.um))

    def test_multi_field_requires_idx(self):
        zk = _make_zernikes(n_field=3, jmax=10)
        with pytest.raises((IndexError, TypeError)):
            zk.to_galsim()  # idx=None on multi-field should fail

    def test_multi_field_with_idx(self):
        zk = _make_zernikes(n_field=3, jmax=10)
        gz = zk.to_galsim(idx=1)
        np.testing.assert_allclose(gz.coef, zk.coefs[1].to_value(u.um))

    def test_unit_conversion(self):
        zk = _make_zernikes(jmax=10)
        gz_nm = zk.to_galsim(unit=u.nm)
        gz_um = zk.to_galsim(unit=u.um)
        np.testing.assert_allclose(gz_nm.coef, gz_um.coef * 1000, atol=1e-10)
