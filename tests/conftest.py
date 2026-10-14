import os
from pathlib import Path

import pytest

from .utils import _starsharp_asdf_registered

requires_starsharp_asdf = pytest.mark.skipif(
    not _starsharp_asdf_registered(),
    reason="StarSharp not registered as an asdf entry-point (pip install -e .[asdf])",
)


# Path to mtaos's normalization weights, used to check StarSharp's defaults
# against the operational values.  NOTE the filename is a typo: the file really
# uses an FWHM exponent of -0.5, not -0.15.  See
# test_normalizations_match_ts_ofc_v13.
_MTAOS_WEIGHTS_RELPATH = Path(
    "MTAOS/ofc/normalization_weights/range0.5_fwhm-0.15.yaml"
)


def ts_config_mttcs_weights() -> Path | None:
    """Return the path to mtaos's normalization weights, or None if not found.

    Looks at ``$TS_CONFIG_MTTCS_DIR`` first, then falls back to a sibling
    ``ts_config_mttcs`` checkout next to the StarSharp repo.
    """
    candidates = []
    env = os.environ.get("TS_CONFIG_MTTCS_DIR")
    if env:
        candidates.append(Path(env))
    # tests/ -> repo root -> sibling checkout
    candidates.append(Path(__file__).resolve().parents[2] / "ts_config_mttcs")

    for base in candidates:
        path = base / _MTAOS_WEIGHTS_RELPATH
        if path.is_file():
            return path
    return None


requires_ts_config_mttcs = pytest.mark.skipif(
    ts_config_mttcs_weights() is None,
    reason=(
        "ts_config_mttcs not found; set $TS_CONFIG_MTTCS_DIR or place a "
        "checkout alongside the StarSharp repo"
    ),
)


def _default_sensitivity_installed() -> bool:
    from importlib.resources import files
    return files("StarSharp").joinpath("data/sensitivity/3.14_i.asdf").is_file()


requires_default_sensitivity = pytest.mark.skipif(
    not _default_sensitivity_installed(),
    reason="Default sensitivity not installed (run bootstrap/make_default_sensitivity.py)",
)
