from importlib.resources import files
from pathlib import Path
from typing import Literal
from functools import cache
import os

import astropy.units as u
import batoid
import numpy as np
import yaml
from astropy.coordinates import Angle
from astropy.table import QTable
from numpy.typing import NDArray

from ..datatypes import PointingModel, State, StateSchema
from .raytraced import RaytracedOpticalModel


LSSTBand = Literal["u", "g", "r", "i", "z", "y"]

# Maps normalised version key (leading "v" stripped) to batoid yaml prefix.
# "design" and "3.3" are permanent synonyms for the v3.3 as-design optics.
_OPTICS_VERSIONS: dict[str, str] = {
    "design": "LSST",
    "3.3":    "LSST",
    "3.12":   "Rubin_v3.12",
    "3.14":   "Rubin_v3.14",
}


def _optics_yaml(version: str, band: str) -> str:
    """Return the batoid yaml filename for *version* and *band*.

    The leading ``'v'`` is optional: ``'v3.12'`` and ``'3.12'`` are
    equivalent.  ``'design'`` is a permanent synonym for ``'3.3'``.
    """
    key = version.lstrip("v")
    prefix = _OPTICS_VERSIONS.get(key)
    if prefix is None:
        valid = sorted(_OPTICS_VERSIONS)
        raise ValueError(
            f"Unknown optics version {version!r}. "
            f"Valid values (leading 'v' optional): {valid}"
        )
    return f"{prefix}_{band}.yaml"


DEFAULT_WAVELENGTHS_NM: dict[LSSTBand, u.Quantity] = {
    "u": 370.9 * u.nm,
    "g": 476.7 * u.nm,
    "r": 619.4 * u.nm,
    "i": 753.9 * u.nm,
    "z": 866.8 * u.nm,
    "y": 973.9 * u.nm,
}


def default_schema() -> StateSchema:
    """Return the standard 50-DOF LSST AOS schema."""
    dof_names: list[str] = []
    for hpod in ["M2", "Cam"]:
        for dof in ["dz", "dx", "dy", "rx", "ry"]:
            dof_names.append(f"{hpod}_{dof}")
    for mirror in ["M1M3", "M2"]:
        for bmode in range(20):
            dof_names.append(f"{mirror}_b{bmode+1}")

    dof_units = np.array([u.um] * 50)
    dof_units[[3, 4, 8, 9]] = u.deg

    steps = np.array(
        [
            10.0,  # M2 dz [um]
            250.0,
            250.0,  # M2 dx, dy [um]
            10.0 / 3600.0,
            10.0 / 3600.0,  # M2 rx, ry [deg]
            10.0,  # Cam dz [um]
            500.0,
            500.0,  # Cam dx, dy [um]
            10.0 / 3600.0,
            10.0 / 3600.0,  # Cam rx, ry [deg]
        ]
        + [0.1] * 40,  # Bending modes [um]
        dtype=float,
    )

    return StateSchema(tuple(dof_names), tuple(dof_units), step=steps)


@cache
def default_camera():
    """Load the default LSST camera."""
    from lsst.obs.lsst import LsstCam

    return LsstCam().getCamera()


def default_pointing_model(
    schema: StateSchema | None = None,
    *,
    strict: bool = True,
    resource: str = "data/pointing_model/default.ecsv",
) -> PointingModel:
    """Load the packaged default pointing model.

    Parameters
    ----------
    schema : StateSchema or None
        Optional target schema. If provided, the loaded pointing model is
        aligned before being returned.
    strict : bool, optional
        Passed through to :meth:`PointingModel.aligned`.
    resource : str, optional
        Package-relative resource path under the ``StarSharp`` package.
    """
    table = QTable.read(files("StarSharp").joinpath(resource))
    pm = PointingModel.from_table(table)
    if schema is not None:
        pm = pm.aligned(schema, strict=strict)
    return pm


def default_normalizations(
    fwhm_exponent: float = 0.5,
    range_exponent: float = 0.5,
    *,
    schema: StateSchema | None = None,
) -> NDArray[np.floating]:
    """Return per-DOF normalization weights ``fwhm**fwhm_exponent * range**range_exponent``.

    Weights are read from ``fwhm.yaml`` and ``range.yaml``.  Both files contain
    a plain YAML list of 50 floats in the standard 50-DOF order defined by
    :func:`default_schema`.

    The files are resolved in this order:

    1. ``$TS_OFC_DIR/policy/normalization_weights/`` if the environment variable
       is set and the directory exists.
    2. The bundled ``StarSharp/data/normalization_weights/`` fallback.

    Parameters
    ----------
    fwhm_exponent, range_exponent : float
        Exponents applied to the FWHM-sensitivity and working-range weights
        respectively.  Defaults of ``(0.5, 0.5)`` give the geometric mean of both.
    schema : StateSchema or None
        Schema whose DOF count is used for validation.  Defaults to
        :func:`default_schema`.

    Returns
    -------
    norm : NDArray[np.floating], shape (n_dof,)
        Normalization vector suitable for passing to
        :meth:`StateSchema.with_svd`.
    """
    if schema is None:
        schema = default_schema()

    ts_ofc = os.environ.get("TS_OFC_DIR")
    if ts_ofc:
        candidate = Path(ts_ofc) / "policy" / "normalization_weights"
        if candidate.is_dir():
            weights_dir = candidate
        else:
            weights_dir = None
    else:
        weights_dir = None

    def _load(name: str) -> NDArray[np.floating]:
        if weights_dir is not None:
            text = (weights_dir / name).read_text()
        else:
            text = files("StarSharp").joinpath(f"data/normalization_weights/{name}").read_text()
        arr = np.asarray(yaml.safe_load(text), dtype=float)
        if arr.ndim != 1 or len(arr) != schema.n_dof:
            raise ValueError(f"{name}: expected {schema.n_dof} values, got {arr.shape}")
        return arr

    fwhm = _load("fwhm.yaml")
    rng = _load("range.yaml")
    return fwhm ** fwhm_exponent * rng ** range_exponent


def default_raytraced_model(
    *,
    band: LSSTBand = "r",
    version: str = "design",
    rtp: Angle = Angle("0 deg"),
    wavelength: u.Quantity | None = None,
    camera=None,
    state_schema: StateSchema | None = None,
    offset: State | None = None,
    pointing_model: PointingModel | None = None,
    use_default_pointing_model: bool = False,
    rtp_lookup=None,
) -> RaytracedOpticalModel:
    """Construct the canonical fiducial LSST ``RaytracedOpticalModel``.

    The returned model uses ``LSST_<band>.yaml`` and an ``LSSTBuilder``
    configured to match lsst ts_ofc conventions (OCS coords,
    degree-valued rotational DOFs, and no M2 bending-mode sign flip).

    Parameters
    ----------
    band : {'u', 'g', 'r', 'i', 'z', 'y'}, optional
        LSST band used to select the batoid yaml file.
    version : str, optional
        Optics model version.  A leading ``'v'`` is stripped before
        matching, so ``'v3.12'`` and ``'3.12'`` are equivalent.
        Valid values:

        * ``'design'`` or ``'3.3'`` — v3.3 as-design optics
          (``LSST_<band>.yaml``).
        * ``'3.12'`` — as-built v3.12 (``Rubin_v3.12_<band>.yaml``).
        * ``'3.14'`` — as-built v3.14 (``Rubin_v3.14_<band>.yaml``).

        Defaults to ``'design'``.
    rtp : Angle, optional
        Rotator position angle.
    wavelength : Quantity or None, optional
        Monochromatic tracing wavelength. Defaults to a band-specific value.
    camera : optional
        Camera geometry object. If None, the default LSST camera is loaded.
    state_schema : StateSchema or None, optional
        State schema for the model. Defaults to the standard 50-DOF LSST AOS
        schema.
    offset : State or None, optional
        Optional fixed AOS offset applied by the model.
    pointing_model : PointingModel or None, optional
        Optional explicit pointing model to attach.
    use_default_pointing_model : bool, optional
        If True, load and attach the packaged default pointing model aligned to
        the chosen state schema.
    rtp_lookup : RTPLookup or None, optional
        Optional RTP-indexed lookup table passed through to the model.
    """
    if wavelength is None:
        wavelength = DEFAULT_WAVELENGTHS_NM[band]
    if state_schema is None:
        state_schema = default_schema()
    if camera is None:
        camera = default_camera()

    if pointing_model is not None and use_default_pointing_model:
        raise ValueError(
            "Specify either pointing_model or use_default_pointing_model=True, not both"
        )
    if use_default_pointing_model:
        pointing_model = default_pointing_model(schema=state_schema, strict=True)

    from batoid_rubin import LSSTBuilder

    yaml_file = _optics_yaml(version, band)
    fiducial = batoid.Optic.fromYaml(yaml_file)
    builder = LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m2_bending_modes=False,
        dof_angle_units="degree",
    )

    return RaytracedOpticalModel(
        builder=builder,
        rtp=rtp,
        wavelength=wavelength,
        state_schema=state_schema,
        camera=camera,
        offset=offset,
        pointing_model=pointing_model,
        rtp_lookup=rtp_lookup,
    )


__all__ = [
    "LSSTBand",
    "DEFAULT_WAVELENGTHS_NM",
    "_OPTICS_VERSIONS",
    "default_schema",
    "default_camera",
    "default_normalizations",
    "default_pointing_model",
    "default_raytraced_model",
]