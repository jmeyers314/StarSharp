"""StarSharp camera geometry.

Provides lightweight replacements for the lsst camera objects used by
StarSharp, loaded from bundled data files.  The coordinate-system sentinels
``FOCAL_PLANE`` and ``FIELD_ANGLE`` replace ``lsst.afw.cameraGeom`` constants
everywhere inside the package.
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.table import QTable

# Coordinate-system sentinels used as keys in getTransform / getCorners.
FOCAL_PLANE = "focal_plane"
FIELD_ANGLE = "field_angle"


class _RadialMapping:
    """Forward radial polynomial mapping.

    Evaluates  r_fp = f(r) = sum_k c_k * r^p_k  and returns
    (x_out, y_out) = (f(r) / r) * (x, y).
    """

    def __init__(self, powers, coefficients):
        self._powers = np.asarray(powers, dtype=float)
        self._coefficients = np.asarray(coefficients, dtype=float)

    def _eval(self, r):
        return sum(c * r ** p for c, p in zip(self._coefficients, self._powers))

    def _eval_deriv(self, r):
        return sum(c * p * r ** (p - 1) for c, p in zip(self._coefficients, self._powers))

    def applyForward(self, xy):
        x = np.asarray(xy[0], dtype=float)
        y = np.asarray(xy[1], dtype=float)
        r = np.hypot(x, y)
        r_out = self._eval(r)
        # limit as r -> 0 is coefficients[0] (coefficient of the r^1 term)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(r > 0, r_out / r, self._coefficients[0])
        return np.array([scale * x, scale * y])


class _InverseRadialMapping:
    """Inverse radial mapping solved by Newton-Raphson.

    Uses a fitted polynomial for the initial guess, then iterates
    ``r ← r - (f(r) - r_fp) / f'(r)`` using the forward mapping *fwd*
    to converge to machine precision.
    """

    def __init__(self, fwd: _RadialMapping, approx_powers, approx_coefficients,
                 n_iter: int = 6):
        self._fwd = fwd
        self._approx_powers = np.asarray(approx_powers, dtype=float)
        self._approx_coefficients = np.asarray(approx_coefficients, dtype=float)
        self._n_iter = n_iter

    def applyForward(self, xy):
        x = np.asarray(xy[0], dtype=float)
        y = np.asarray(xy[1], dtype=float)
        r_fp = np.hypot(x, y)

        # Initial guess from the approximate inverse polynomial
        r = sum(c * r_fp ** p
                for c, p in zip(self._approx_coefficients, self._approx_powers))

        # Newton-Raphson: solve f(r) = r_fp
        for _ in range(self._n_iter):
            f_r = self._fwd._eval(r)
            fp_r = self._fwd._eval_deriv(r)
            r = r - (f_r - r_fp) / fp_r

        # Inverse scale: r / r_fp; limit at r_fp=0 is 1 / f'(0) = 1 / a_0
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(r_fp > 0, r / r_fp, 1.0 / self._fwd._coefficients[0])
        return np.array([scale * x, scale * y])


class _Transform:
    """Shim with a getMapping() method, matching the lsst Transform API."""

    def __init__(self, mapping):
        self._mapping = mapping

    def getMapping(self):
        return self._mapping


class _Detector:
    """Minimal detector geometry.

    Corner objects returned by getCorners are plain 2-tuples ``(ccs_y, ccs_x)``
    matching the lsst FOCAL_PLANE convention where ``c[0]`` is CCS y and
    ``c[1]`` is CCS x.
    """

    def __init__(self, id, name, physical_type,
                 ccs_x_min, ccs_x_max, ccs_y_min, ccs_y_max):
        self._id = int(id)
        self._name = name
        self._physical_type = physical_type
        self._x0 = float(ccs_x_min)
        self._x1 = float(ccs_x_max)
        self._y0 = float(ccs_y_min)
        self._y1 = float(ccs_y_max)

    def getId(self):
        return self._id

    def getName(self):
        return self._name

    def getPhysicalType(self):
        return self._physical_type

    def getCorners(self, sys):
        x0, x1 = self._x0, self._x1
        y0, y1 = self._y0, self._y1
        return [(y0, x0), (y0, x1), (y1, x1), (y1, x0)]


class LsstCameraGeom:
    """Lightweight LSST camera geometry loaded from bundled data files.

    Implements the subset of the lsst Camera API used by StarSharp:
    ``getTransform``, ``__iter__``, and ``__len__``.

    Parameters
    ----------
    poly_path : path-like or None
        Path to the radial-transform YAML.  Defaults to the bundled
        ``StarSharp/data/camera/radial_transform.yaml``.
    det_path : path-like or None
        Path to the detector ECSV table.  Defaults to the bundled
        ``StarSharp/data/camera/detectors.ecsv``.
    """

    def __init__(self, poly_path=None, det_path=None):
        import yaml
        from importlib.resources import files

        if poly_path is None:
            poly_path = files("StarSharp").joinpath("data/camera/radial_transform.yaml")
        if det_path is None:
            det_path = files("StarSharp").joinpath("data/camera/detectors.ecsv")

        with open(poly_path) as f:
            poly = yaml.safe_load(f)

        self._fwd = _RadialMapping(
            poly["forward"]["powers"], poly["forward"]["coefficients"]
        )
        self._inv = _InverseRadialMapping(
            self._fwd,
            poly["inverse"]["powers"],
            poly["inverse"]["coefficients"],
        )

        table = QTable.read(str(det_path), format="ascii.ecsv")
        self._detectors = [
            _Detector(
                row["id"],
                row["name"],
                row["physical_type"],
                row["ccs_x_min"].to_value(u.mm),
                row["ccs_x_max"].to_value(u.mm),
                row["ccs_y_min"].to_value(u.mm),
                row["ccs_y_max"].to_value(u.mm),
            )
            for row in table
        ]

    def getTransform(self, from_sys, to_sys):
        if from_sys == FIELD_ANGLE and to_sys == FOCAL_PLANE:
            return _Transform(self._fwd)
        elif from_sys == FOCAL_PLANE and to_sys == FIELD_ANGLE:
            return _Transform(self._inv)
        else:
            raise ValueError(f"Unsupported transform: {from_sys!r} -> {to_sys!r}")

    def __iter__(self):
        return iter(self._detectors)

    def __len__(self):
        return len(self._detectors)


# ---------------------------------------------------------------------------
# Thin adapter for the official lsst Camera (use_lsst=True path)
# ---------------------------------------------------------------------------

class _LsstDetectorAdapter:
    """Wraps an lsst Detector to accept StarSharp's string sentinels."""

    def __init__(self, lsst_det):
        self._det = lsst_det

    def getId(self):
        return self._det.getId()

    def getName(self):
        return self._det.getName()

    def getPhysicalType(self):
        return self._det.getPhysicalType()

    def getCorners(self, sys):
        from lsst.afw.cameraGeom import FOCAL_PLANE as _FP, FIELD_ANGLE as _FA
        _smap = {FOCAL_PLANE: _FP, FIELD_ANGLE: _FA}
        return self._det.getCorners(_smap.get(sys, sys))


class _LsstCameraAdapter:
    """Wraps an lsst Camera to accept StarSharp's string sentinels."""

    def __init__(self, lsst_camera):
        self._cam = lsst_camera

    def getTransform(self, from_sys, to_sys):
        from lsst.afw.cameraGeom import FOCAL_PLANE as _FP, FIELD_ANGLE as _FA
        _smap = {FOCAL_PLANE: _FP, FIELD_ANGLE: _FA}
        return self._cam.getTransform(
            _smap.get(from_sys, from_sys),
            _smap.get(to_sys, to_sys),
        )

    def __iter__(self):
        return (_LsstDetectorAdapter(det) for det in self._cam)

    def __len__(self):
        return len(self._cam)
