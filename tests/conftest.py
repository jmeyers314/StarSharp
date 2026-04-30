import pytest

from .utils import _starsharp_asdf_registered

requires_starsharp_asdf = pytest.mark.skipif(
    not _starsharp_asdf_registered(),
    reason="StarSharp not registered as an asdf entry-point (pip install -e .[asdf])",
)
