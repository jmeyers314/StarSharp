"""Tests for DoubleZernikes."""
from __future__ import annotations

import astropy.units as u
import galsim
import numpy as np
import pytest
from astropy.coordinates import Angle

from StarSharp.datatypes import DoubleZernikes
from .utils import RTP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg
PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m


def _make_dz(
    kmax: int = 10,
    jmax: int = 22,
    frame: str = "ocs",
    rtp: Angle | None = None,
    batch_shape: tuple[int, ...] = (),
) -> DoubleZernikes:
    rng = np.random.default_rng(77)
    coefs = rng.standard_normal(batch_shape + (kmax + 1, jmax + 1))
    # Set unused slices to zero
    coefs[..., 0, :] = 0.0
    coefs[..., :, :4] = 0.0

    return DoubleZernikes(
        coefs=coefs << u.um,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        pupil_outer=PUPIL_OUTER,
        pupil_inner=PUPIL_INNER,
        jmax=jmax,
        kmax=kmax,
        frame=frame,
        rtp=rtp,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDoubleZernikesConstruction:
    def test_jmax_kmax_inferred(self):
        dz = _make_dz(kmax=8, jmax=15)
        assert dz.kmax == 8
        assert dz.jmax == 15
        assert dz.coefs.shape == (9, 16)

    def test_atleast_2d(self):
        # 1-D coefs should be promoted to 2-D
        coefs = np.ones(11) * u.um
        dz = DoubleZernikes(
            coefs=coefs,
            field_outer=FIELD_OUTER,
            field_inner=FIELD_INNER,
            pupil_outer=PUPIL_OUTER,
            pupil_inner=PUPIL_INNER,
        )
        assert dz.coefs.ndim == 2

    def test_explicit_jmax_kmax(self):
        dz = _make_dz(kmax=6, jmax=10)
        assert dz.jmax == 10
        assert dz.kmax == 6

    def test_eps(self):
        dz = _make_dz()
        assert dz.eps == pytest.approx(0.612, rel=1e-4)

    def test_frozen(self):
        dz = _make_dz()
        with pytest.raises(AttributeError):
            dz.coefs = np.zeros((11, 23)) * u.um

    def test_default_frame_ocs(self):
        dz = _make_dz()
        assert dz.frame == "ocs"

    def test_wavelength_optional(self):
        dz = _make_dz()
        assert dz.wavelength is None
        dz_wl = DoubleZernikes(
            coefs=np.ones((5, 10)) * u.um,
            field_outer=FIELD_OUTER,
            field_inner=FIELD_INNER,
            pupil_outer=PUPIL_OUTER,
            pupil_inner=PUPIL_INNER,
            wavelength=620 * u.nm,
        )
        assert dz_wl.wavelength == 620 * u.nm


# ---------------------------------------------------------------------------
# batch_shape and __len__
# ---------------------------------------------------------------------------

class TestDoubleZernikesShape:
    def test_batch_shape_unbatched(self):
        dz = _make_dz(kmax=6, jmax=10)
        assert dz.batch_shape == ()

    def test_batch_shape_1d_batch(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(4,))
        assert dz.batch_shape == (4,)
        assert dz.coefs.shape == (4, 7, 11)

    def test_batch_shape_2d_batch(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(3, 4))
        assert dz.batch_shape == (3, 4)
        assert dz.coefs.shape == (3, 4, 7, 11)

    def test_len_unbatched(self):
        dz = _make_dz(kmax=6, jmax=10)
        # shape[0] == kmax+1 == 7
        assert len(dz) == 7

    def test_len_batched(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(7,))
        assert len(dz) == 7


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

class TestDoubleZernikesGetitem:
    def test_getitem_int_batch(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(4,))
        s = dz[2]
        assert s.coefs.shape == (7, 11)
        assert s.field_outer == dz.field_outer
        assert s.pupil_outer == dz.pupil_outer

    def test_getitem_slice_batch(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(6,))
        s = dz[1:4]
        assert s.coefs.shape == (3, 7, 11)

    def test_getitem_preserves_metadata(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(4,), rtp=RTP, frame="ocs")
        s = dz[0]
        assert s.jmax == dz.jmax
        assert s.kmax == dz.kmax
        assert s.rtp is dz.rtp
        assert s.frame == dz.frame


# ---------------------------------------------------------------------------
# Frame rotation (ocs / ccs)
# ---------------------------------------------------------------------------

class TestDoubleZernikesFrameRotation:
    def test_ocs_noop(self):
        dz = _make_dz(frame="ocs", rtp=RTP)
        assert dz.ocs is dz

    def test_ccs_noop(self):
        dz = _make_dz(frame="ccs", rtp=RTP)
        assert dz.ccs is dz

    def test_ocs_without_rtp_raises(self):
        dz = _make_dz(frame="ccs", rtp=None)
        with pytest.raises(ValueError, match="rtp"):
            dz.ocs

    def test_ccs_without_rtp_raises(self):
        dz = _make_dz(frame="ocs", rtp=None)
        with pytest.raises(ValueError, match="rtp"):
            dz.ccs

    def test_zero_rtp_identity(self):
        dz = _make_dz(frame="ocs", rtp=Angle(0, unit=u.rad))
        ccs = dz.ccs
        np.testing.assert_allclose(
            ccs.coefs.to_value(u.um), dz.coefs.to_value(u.um), atol=1e-12
        )

    def test_roundtrip_ocs_ccs_ocs(self):
        dz = _make_dz(frame="ocs", rtp=RTP)
        rt = dz.ccs.ocs
        assert rt.frame == "ocs"
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), dz.coefs.to_value(u.um), atol=1e-10
        )

    def test_roundtrip_ccs_ocs_ccs(self):
        dz = _make_dz(frame="ccs", rtp=RTP)
        rt = dz.ocs.ccs
        assert rt.frame == "ccs"
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), dz.coefs.to_value(u.um), atol=1e-10
        )

    def test_rotation_sets_frame(self):
        dz = _make_dz(frame="ocs", rtp=RTP)
        assert dz.ccs.frame == "ccs"
        assert dz.ccs.ocs.frame == "ocs"

    def test_rotation_preserves_rtp(self):
        dz = _make_dz(frame="ocs", rtp=RTP)
        assert dz.ccs.rtp is RTP

    def test_spin0_invariant_under_rotation(self):
        """The k,j ∈ {(1,1), (1, 4), (4, 1), (4, 4)} terms mode should be invariant
        under rotation."""
        dz = _make_dz(kmax=4, jmax=4, frame="ocs", rtp=RTP)
        idx = [1, 4]
        np.testing.assert_allclose(
            dz.ccs.coefs.to_value(u.um)[idx, idx],
            dz.coefs.to_value(u.um)[idx, idx],
            atol=1e-12,
        )

    def test_rotation_batched(self):
        dz = _make_dz(kmax=6, jmax=10, batch_shape=(3,), frame="ocs", rtp=RTP)
        rt = dz.ccs.ocs
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), dz.coefs.to_value(u.um), atol=1e-10
        )

    def test_rotation_against_single_zernikes(self):
        """Test that the rotation of the k=0 (field-independent) modes matches the Zernikes rotation."""
        from StarSharp.datatypes import Zernikes, FieldCoords

        dz = _make_dz(kmax=1, jmax=10, frame="ocs", rtp=RTP)
        zk = Zernikes(
            coefs=dz.coefs[0],
            field=FieldCoords(0.0*u.deg, 0.0*u.deg, frame="ocs", rtp=RTP),
            R_outer=PUPIL_OUTER,
            R_inner=PUPIL_INNER,
            wavelength=dz.wavelength,
            jmax=dz.jmax,
            frame=dz.frame,
            rtp=dz.rtp,
        )
        zk_ccs = zk.ccs
        dz_ccs = dz.ccs
        np.testing.assert_allclose(
            dz_ccs.coefs.to_value(u.um)[0],
            zk_ccs.coefs.to_value(u.um)[0],
            atol=1e-12,
        )

class TestDoubleZernikesSingleZernikes:
    def test_double_single_double_roundtrip(self):
        from StarSharp.datatypes import Zernikes, FieldCoords

        dz = _make_dz(kmax=3, jmax=11, frame="ocs", rtp=RTP)
        x = np.linspace(0.5, -0.5, 11) * u.deg
        y = np.linspace(-0.5, 0.5, 11) * u.deg
        xx, yy = np.meshgrid(x, y)
        field = FieldCoords(xx.ravel(), yy.ravel(), frame="ocs", rtp=RTP)
        zk = dz.single(field)
        assert isinstance(zk, Zernikes)
        assert zk.field == field
        rt = zk.double(kmax=3, field_outer=FIELD_OUTER)
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), dz.coefs.to_value(u.um), atol=1e-12
        )

    def test_batched_double_single_double_roundtrip(self):
        from StarSharp.datatypes import Zernikes, FieldCoords

        dz = _make_dz(kmax=3, jmax=11, frame="ocs", rtp=RTP, batch_shape=(2, 3))
        x = np.linspace(0.5, -0.5, 11) * u.deg
        y = np.linspace(-0.5, 0.5, 11) * u.deg
        xx, yy = np.meshgrid(x, y)
        field = FieldCoords(xx.ravel(), yy.ravel(), frame="ocs", rtp=RTP)
        zk = dz.single(field)
        assert isinstance(zk, Zernikes)
        assert zk.field == field
        rt = zk.double(kmax=3, field_outer=FIELD_OUTER)
        np.testing.assert_allclose(
            rt.coefs.to_value(u.um), dz.coefs.to_value(u.um), atol=1e-12
        )

# ---------------------------------------------------------------------------
# to_galsim
# ---------------------------------------------------------------------------

class TestDoubleZernikesToGalsim:
    def test_returns_galsim_object(self):
        dz = _make_dz(kmax=4, jmax=10)
        gz = dz.to_galsim()
        assert isinstance(gz, galsim.zernike.DoubleZernike)

    def test_to_galsim_with_idx(self):
        dz = _make_dz(kmax=4, jmax=10, batch_shape=(3,))
        gz = dz.to_galsim(idx=1)
        assert isinstance(gz, galsim.zernike.DoubleZernike)

    def test_to_galsim_radii(self):
        dz = _make_dz(kmax=4, jmax=10)
        gz = dz.to_galsim()
        assert gz.uv_outer == pytest.approx(PUPIL_OUTER.to_value(u.m))
        assert gz.xy_outer == pytest.approx(FIELD_OUTER.to_value(u.deg))
