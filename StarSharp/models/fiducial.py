from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Literal
from functools import lru_cache
import os

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import Angle
from numpy.typing import NDArray

from ..datatypes import PointingModel, State, StateSchema
from ..models import RTPLookup
from ..utils import str_to_arr
from .raytraced import RaytracedOpticalModel


LSSTBand = Literal["u", "g", "r", "i", "z", "y"]

# Sentinel for "load the default" in default_raytraced_model.
class _DefaultType:
    """Singleton sentinel meaning 'load the packaged default'."""
    _instance: "_DefaultType | None" = None
    def __new__(cls) -> "_DefaultType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

_DEFAULT = _DefaultType()

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


def default_schema(
    use_dof: list[int] | str | None = None,
    n_keep: int | None = None,
    fwhm_exponent: float = 0.5,
    range_exponent: float = 0.5,
) -> StateSchema:
    """Return the standard 50-DOF LSST AOS schema.

    Parameters
    ----------
    use_dof : list[int], str, or None
        Active DOF indices.  A string is parsed as a comma-separated list of
        indices and/or inclusive ranges, e.g. ``"0-9,20,30-36"``.  If None,
        all 50 DOFs are active.
    n_keep : int or None
        If provided, compute SVD modes from the default sensitivity and return
        a schema with ``Vh`` set (enabling v-basis states).  Requires the
        default sensitivity to be installed under
        ``StarSharp/data/sensitivity/``.
    fwhm_exponent, range_exponent : float
        Passed to :func:`default_normalizations` when computing SVD modes.
        Only used when ``n_keep`` is not None.
    """
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

    if use_dof:
        use_dof = str_to_arr(use_dof)
    schema = StateSchema(tuple(dof_names), tuple(dof_units), step=steps, use_dof=use_dof)

    if n_keep is not None:
        sens = default_sensitivity()
        # Align sensitivity schema to the active DOF subset.  The gradient
        # has length n_dof=50 so f-basis validation still passes; with_svd
        # will call sens.x which slices gradient[use_dof] internally.
        if not np.array_equal(sens.schema.use_dof, schema.use_dof):
            restricted_schema = StateSchema(
                sens.schema.dof_names, sens.schema.dof_units,
                use_dof=schema.use_dof, step=sens.schema.step,
            )
            sens = replace(sens, schema=restricted_schema)
        norm = default_normalizations(
            fwhm_exponent=fwhm_exponent,
            range_exponent=range_exponent,
            schema=schema,
        )
        schema = schema.with_svd(sens, norm=norm, n_keep=n_keep)

    return schema


@lru_cache(maxsize=2)
def default_camera(use_lsst: bool = False):
    """Load the default LSST camera geometry.

    Parameters
    ----------
    use_lsst : bool
        If True, load the official ``lsst.obs.lsst.LsstCam`` (requires the
        lsst stack) and wrap it to accept StarSharp's coordinate-system
        sentinels.  If False (default), return the lightweight bundled
        ``LsstCameraGeom``.
    """
    if use_lsst:
        from lsst.obs.lsst import LsstCam
        from ..camera import _LsstCameraAdapter
        return _LsstCameraAdapter(LsstCam().getCamera())
    from ..camera import LsstCameraGeom
    return LsstCameraGeom()


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
    from astropy.table import QTable
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
    state_schema: StateSchema | None = None,
    use_dof: list[int] | str | None = None,
    n_keep: int | None = None,
    fwhm_exponent: float = 0.5,
    range_exponent: float = 0.5,
    offset: State | None = None,
    pointing_model: PointingModel | None | _DefaultType = _DEFAULT,
    rtp_lookup: RTPLookup | None | _DefaultType = _DEFAULT,
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
    state_schema : StateSchema or None, optional
        State schema for the model. Defaults to the standard 50-DOF LSST AOS
        schema.  Mutually exclusive with ``use_dof`` / ``n_keep``.
    use_dof : list[int], str, or None, optional
        Passed to :func:`default_schema`.  Ignored if ``state_schema`` is
        provided.
    n_keep : int or None, optional
        Passed to :func:`default_schema` to pre-compute SVD vmodes.  Ignored
        if ``state_schema`` is provided.
    fwhm_exponent, range_exponent : float, optional
        Passed to :func:`default_schema` / :func:`default_normalizations` for
        SVD mode computation.  Ignored if ``state_schema`` is provided.
    offset : State or None, optional
        Optional fixed AOS offset applied by the model.
    pointing_model : PointingModel, None, or _DEFAULT
        Pointing model to attach.  The default (``_DEFAULT``) loads the
        packaged default pointing model.  Pass ``None`` to use no pointing
        model, or supply an explicit :class:`PointingModel`.
    rtp_lookup : RTPLookup, None, or _DEFAULT
        RTP lookup table.  The default (``_DEFAULT``) loads the bundled
        table for the chosen version and band if one exists, otherwise
        ``None``.  Pass ``None`` to force no lookup table.
    """
    if wavelength is None:
        wavelength = DEFAULT_WAVELENGTHS_NM[band]
    if state_schema is None:
        state_schema = default_schema(
            use_dof=use_dof,
            n_keep=n_keep,
            fwhm_exponent=fwhm_exponent,
            range_exponent=range_exponent,
        )
    elif use_dof is not None or n_keep is not None:
        raise ValueError(
            "use_dof and n_keep are only supported when state_schema is None"
        )

    if pointing_model is _DEFAULT:
        pointing_model = default_pointing_model(schema=state_schema, strict=False)

    if rtp_lookup is _DEFAULT:
        from .rtp_lookup import RTPLookup
        key = version.lstrip("v")
        rtp_key = "3.3" if key == "design" else key  # "design" is an alias for 3.3
        rtp_resource = files("StarSharp").joinpath(f"data/rtp_lookup/v{rtp_key}_{band}.ecsv")
        rtp_lookup = RTPLookup.from_file(rtp_resource) if rtp_resource.is_file() else None

    from batoid_rubin import LSSTBuilder
    import batoid

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
    "default_sensitivity",
]


def default_sensitivity():
    """Load the default double-Zernike sensitivity (v3.14, i-band).

    Resolution order:

    1. **$TS_OFC_DIR official YAML** — if the environment variable is set and
       ``policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml`` exists,
       the official matrix is loaded (sign-flipped to match StarSharp
       convention) and used as the gradient.  Schema, nominal, and basis are
       taken from the bundled ASDF so that the returned object is fully typed.
    2. **Bundled ASDF** — ``StarSharp/data/sensitivity/3.14_i.asdf``.

    Returns
    -------
    Sensitivity[DoubleZernikes]
    """
    import asdf as _asdf

    from ..datatypes.double_zernikes import DoubleZernikes
    from ..datatypes.sensitivity import Sensitivity
    from ..io.asdf.extension import StarSharpExtension

    # Always load the bundled ASDF — it carries schema, nominal, basis, and
    # the gradient unit / metadata needed to interpret the official YAML.
    asdf_resource = files("StarSharp").joinpath("data/sensitivity/3.14_i.asdf")
    with _asdf.config_context() as cfg:
        cfg.add_extension(StarSharpExtension())
        with _asdf.open(str(asdf_resource)) as af:
            bundled: Sensitivity = af["sensitivity"]

    ts_ofc = os.environ.get("TS_OFC_DIR")
    if not ts_ofc:
        return bundled

    yaml_path = (
        Path(ts_ofc) / "policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml"
    )
    if not yaml_path.exists():
        return bundled

    with open(yaml_path) as f:
        raw = np.array(yaml.safe_load(f))      # (kmax_dim=31, jmax_dim=29, ndof=50)
    official_arr = -np.moveaxis(raw, -1, 0)    # (50, 31, 29), sign-flipped

    g = bundled.gradient
    official_gradient = DoubleZernikes(
        coefs=official_arr * g.coefs.unit,
        field_outer=g.field_outer,
        field_inner=g.field_inner,
        pupil_outer=g.pupil_outer,
        pupil_inner=g.pupil_inner,
        wavelength=g.wavelength,
        frame=g.frame,
        rtp=g.rtp,
    )
    return Sensitivity(
        gradient=official_gradient,
        schema=bundled.schema,
        nominal=bundled.nominal,
        basis=bundled.basis,
    )