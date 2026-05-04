from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from ..datatypes.field_coords import FieldCoords
from ..datatypes.zernikes import Zernikes

_DATASET_TYPE = "aggregateAOSVisitTableRaw"
_INSTRUMENT = "LSSTCam"
_R_OUTER = 4.18 * u.m
_R_INNER = 0.612 * _R_OUTER


@lru_cache(maxsize=4)
def _get_butler(repo: str, collections: str):
    from lsst.daf.butler import Butler
    return Butler(repo, collections=collections)


def _table_to_zernikes(table) -> Zernikes:
    noll_indices = table.meta["nollIndices"]
    rtp = Angle(table.meta["rotTelPos"] * u.radian).wrap_at(180 * u.degree)

    x = np.hstack([
        table["thx_CCS_intra"] * u.radian,
        table["thx_CCS_extra"] * u.radian,
    ])
    y = np.hstack([
        table["thy_CCS_intra"] * u.radian,
        table["thy_CCS_extra"] * u.radian,
    ])
    fc = FieldCoords(x, y, frame="ccs", rtp=rtp)

    n = len(table)
    jmax = int(np.max(noll_indices))
    zk_data = np.zeros((n, jmax + 1))
    zk_data[:, noll_indices] = table["zk_deviation_CCS"]
    # Duplicate rows so each field point (intra and extra) has Zernike coefficients.
    zk = np.vstack([zk_data, zk_data]) * u.um

    return Zernikes(
        zk,
        field=fc,
        R_outer=_R_OUTER,
        R_inner=_R_INNER,
        frame="ccs",
        rtp=rtp,
    )


def _cache_path(
    cache_dir: Path,
    repo: str,
    collections: str,
    day_obs: int,
    seq_num: int,
) -> Path:
    key = hashlib.md5(f"{repo}:{collections}".encode()).hexdigest()[:8]
    return cache_dir / f"zernikes_{key}_{day_obs}_{seq_num:05d}.asdf"


def load_observed_zernikes(
    repo: str,
    day_obs: int,
    seq_num: int,
    collections: str = "LSSTCam/runs/nightlyValidation",
    cache_dir: str | Path | None = None,
) -> Zernikes:
    """Load observed Zernike coefficients from the butler for a single visit.

    Parameters
    ----------
    repo : str
        Butler repository path or alias (e.g. ``"main"``).
    day_obs : int
        Observation day in YYYYMMDD format.
    seq_num : int
        Sequence number of the visit within the day.
    collections : str
        Butler collection to query.
    cache_dir : str or Path or None
        If given, cache the result as an ASDF file in this directory.
        Subsequent calls with the same arguments return the cached value
        without hitting the butler.
    """
    if cache_dir is not None:
        path = _cache_path(Path(cache_dir), repo, collections, day_obs, seq_num)
        if path.exists():
            with asdf.open(str(path)) as af:
                return af["zernikes"]

    butler = _get_butler(repo, collections)
    table = butler.get(
        _DATASET_TYPE,
        dataId={"day_obs": day_obs, "seq_num": seq_num, "instrument": _INSTRUMENT},
    )
    zernikes = _table_to_zernikes(table)

    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with asdf.AsdfFile({"zernikes": zernikes}) as af:
            af.write_to(str(path))

    return zernikes
