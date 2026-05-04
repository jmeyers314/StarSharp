from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from StarSharp.io.butler import _table_to_zernikes, load_observed_zernikes

_NOLL_INDICES = np.array([4, 5, 6, 7, 8])
_N_PAIRS = 4


def _make_table(n=_N_PAIRS, rot_tel_pos=0.5):
    rng = np.random.default_rng(42)
    table = Table()
    table["zk_deviation_CCS"] = rng.standard_normal((n, len(_NOLL_INDICES))).astype(np.float32)
    table["thx_CCS_intra"] = rng.uniform(-0.01, 0.01, n)
    table["thy_CCS_intra"] = rng.uniform(-0.01, 0.01, n)
    table["thx_CCS_extra"] = rng.uniform(-0.01, 0.01, n)
    table["thy_CCS_extra"] = rng.uniform(-0.01, 0.01, n)
    table.meta["rotTelPos"] = rot_tel_pos
    table.meta["nollIndices"] = _NOLL_INDICES
    return table


class TestTableToZernikes:
    def test_coefs_shape(self):
        zk = _table_to_zernikes(_make_table())
        assert zk.coefs.shape == (2 * _N_PAIRS, max(_NOLL_INDICES) + 1)

    def test_frame_is_ccs(self):
        assert _table_to_zernikes(_make_table()).frame == "ccs"

    def test_r_outer(self):
        zk = _table_to_zernikes(_make_table())
        assert zk.R_outer.to_value(u.m) == pytest.approx(4.18)

    def test_r_inner(self):
        zk = _table_to_zernikes(_make_table())
        assert zk.R_inner.to_value(u.m) == pytest.approx(0.612 * 4.18)

    def test_rtp_set(self):
        assert _table_to_zernikes(_make_table()).rtp is not None

    def test_rtp_wrapped(self):
        zk = _table_to_zernikes(_make_table(rot_tel_pos=4.0))
        assert abs(zk.rtp.deg) <= 180.0

    def test_noll_indices_populated(self):
        table = _make_table()
        zk = _table_to_zernikes(table)
        coefs = zk.coefs.value
        np.testing.assert_allclose(
            coefs[:_N_PAIRS, _NOLL_INDICES],
            table["zk_deviation_CCS"],
        )

    def test_non_noll_indices_zero(self):
        zk = _table_to_zernikes(_make_table())
        coefs = zk.coefs.value
        zero_cols = np.setdiff1d(np.arange(coefs.shape[-1]), _NOLL_INDICES)
        assert np.all(coefs[:, zero_cols] == 0)

    def test_coefs_units_are_length(self):
        zk = _table_to_zernikes(_make_table())
        assert zk.coefs.unit.is_equivalent(u.m)

    def test_intra_extra_coefs_identical(self):
        zk = _table_to_zernikes(_make_table())
        coefs = zk.coefs.value
        np.testing.assert_array_equal(coefs[:_N_PAIRS], coefs[_N_PAIRS:])


class TestLoadObservedZernikesCache:
    def test_cache_miss_calls_butler(self, tmp_path):
        mock_butler = MagicMock()
        mock_butler.get.return_value = _make_table()
        with patch("StarSharp.io.butler._get_butler", return_value=mock_butler):
            load_observed_zernikes("repo", 20260415, 42, cache_dir=tmp_path)
        assert mock_butler.get.call_count == 1

    def test_cache_hit_skips_butler(self, tmp_path):
        mock_butler = MagicMock()
        mock_butler.get.return_value = _make_table()
        with patch("StarSharp.io.butler._get_butler", return_value=mock_butler):
            zk1 = load_observed_zernikes("repo", 20260415, 42, cache_dir=tmp_path)
            zk2 = load_observed_zernikes("repo", 20260415, 42, cache_dir=tmp_path)
        assert mock_butler.get.call_count == 1
        np.testing.assert_array_equal(zk1.coefs.value, zk2.coefs.value)

    def test_different_repo_different_cache_file(self, tmp_path):
        mock_butler = MagicMock()
        mock_butler.get.return_value = _make_table()
        with patch("StarSharp.io.butler._get_butler", return_value=mock_butler):
            load_observed_zernikes("repo_a", 20260415, 42, cache_dir=tmp_path)
            load_observed_zernikes("repo_b", 20260415, 42, cache_dir=tmp_path)
        assert mock_butler.get.call_count == 2

    def test_different_collections_different_cache_file(self, tmp_path):
        mock_butler = MagicMock()
        mock_butler.get.return_value = _make_table()
        with patch("StarSharp.io.butler._get_butler", return_value=mock_butler):
            load_observed_zernikes("repo", 20260415, 42, collections="col_a", cache_dir=tmp_path)
            load_observed_zernikes("repo", 20260415, 42, collections="col_b", cache_dir=tmp_path)
        assert mock_butler.get.call_count == 2

    def test_no_cache_dir_always_fetches(self):
        mock_butler = MagicMock()
        mock_butler.get.return_value = _make_table()
        with patch("StarSharp.io.butler._get_butler", return_value=mock_butler):
            load_observed_zernikes("repo", 20260415, 42)
            load_observed_zernikes("repo", 20260415, 42)
        assert mock_butler.get.call_count == 2
