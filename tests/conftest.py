import pytest

from .utils import _starsharp_asdf_registered

requires_starsharp_asdf = pytest.mark.skipif(
    not _starsharp_asdf_registered(),
    reason="StarSharp not registered as an asdf entry-point (pip install -e .[asdf])",
)


def _default_sensitivity_installed() -> bool:
    from importlib.resources import files
    return files("StarSharp").joinpath("data/sensitivity/3.14_i.asdf").is_file()


requires_default_sensitivity = pytest.mark.skipif(
    not _default_sensitivity_installed(),
    reason="Default sensitivity not installed (run bootstrap/make_default_sensitivity.py)",
)
