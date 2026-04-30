"""Shared test helpers for the StarSharp.datatypes test suite."""

from __future__ import annotations

from functools import cache

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from StarSharp.datatypes import FieldCoords, Spots
from lsst.afw.cameraGeom import Camera

RTP = Angle(0.25, unit=u.rad)


@cache
def _load_camera() -> Camera:
    from lsst.obs.lsst import LsstCam
    return LsstCam().getCamera()


def _make_field(
    n: int = 3,
    frame: str = "ocs",
    rtp: Angle | None = None,
    unit: u.Unit = u.deg,
) -> FieldCoords:
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.5, 1.5, n) * unit
    y = rng.uniform(-1.5, 1.5, n) * unit
    return FieldCoords(x=x, y=y, frame=frame, rtp=rtp)


def _make_spots(
    n_field: int = 3,
    n_ray: int = 50,
    frame: str = "ccs",
    rtp: Angle | None = None,
    unit: u.Unit = u.um,
) -> Spots:
    rng = np.random.default_rng(12)
    dx = rng.normal(size=(n_field, n_ray)) * unit
    dy = rng.normal(size=(n_field, n_ray)) * unit
    vig = np.zeros((n_field, n_ray), dtype=bool)
    field = _make_field(n_field, frame="ocs", rtp=rtp)
    return Spots(
        dx=dx,
        dy=dy,
        vignetted=vig,
        field=field,
        wavelength=622.0 * u.nm,
        frame=frame,
        rtp=rtp,
    )


def _make_spots_single(n_ray: int = 200, rtp: Angle = RTP) -> Spots:
    """Single field-point spot in CCS frame."""
    rng = np.random.default_rng(7)
    dx = rng.normal(size=n_ray) * u.micron
    dy = rng.normal(0.5, 1.5, size=n_ray) * u.micron
    vig = np.zeros(n_ray, dtype=bool)
    field = FieldCoords(x=0.5 * u.deg, y=-0.3 * u.deg, frame="ocs", rtp=rtp)
    return Spots(dx=dx, dy=dy, vignetted=vig, field=field, frame="ccs", rtp=rtp)


def _make_spots_batched(n_field: int = 4, n_ray: int = 200, rtp: Angle = RTP) -> Spots:
    """Batched (n_field, n_ray) spots in CCS frame."""
    rng = np.random.default_rng(13)
    dx = rng.normal(size=(n_field, n_ray)) * u.micron
    dy = rng.normal(0.5, 1.5, size=(n_field, n_ray)) * u.micron
    vig = np.zeros((n_field, n_ray), dtype=bool)
    field = FieldCoords(
        x=rng.uniform(-1, 1, n_field) * u.deg,
        y=rng.uniform(-1, 1, n_field) * u.deg,
        frame="ocs",
        rtp=rtp,
    )
    return Spots(dx=dx, dy=dy, vignetted=vig, field=field, frame="ccs", rtp=rtp)
