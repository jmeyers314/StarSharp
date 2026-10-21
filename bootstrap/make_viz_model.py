#!/usr/bin/env python
"""Generate the packed binary model consumed by the browser visualization in ``viz/``.

Writes two files: a float32 blob (``model.f32``) holding every array
back-to-back, and a JSON sidecar (``model_meta.json``) describing each block's
offset, length, and shape.  ``viz/main.js`` fetches both and slices the blob
according to the sidecar, so the two must always be regenerated together.

What the viz does with this
---------------------------
The spot and donut blocks are a first-order model: the browser renders
``spot_xy + sum_i Sx[i] * dof[i]`` as the user drags DOF sliders.  The weight
and sensitivity blocks let the browser recompute the v-mode (SVD) basis live,
rather than baking in one choice of normalization.

Two independent choices are exposed to the browser, which is why there are
three weight blocks and two sensitivity blocks:

1. **Which FWHM normalization weights.**  ``fwhm_starsharp`` is measured by
   raytracing (``make_fwhm_weights.py``); ``fwhm_ts_ofc`` is ts_ofc's value,
   derived from their linearized double-Zernike sensitivity.  They agree to
   ~12% in the median.  ``range`` is shared.
2. **Which double-Zernike sensitivity feeds the SVD.**  ``wf_sens_starsharp``
   is raytraced fresh here at the requested version/band; ``wf_sens_ts_ofc`` is
   ts_ofc's operational matrix.

The browser applies the exponents itself (``range**a * fwhm**b``, both
user-editable), so only the raw weights are shipped.

Sign and unit conventions
-------------------------
Both sensitivity blocks are in um of wavefront per DOF unit (um for
translations and bending modes, *degrees* for rotations).

The only convention fix applied to the ts_ofc matrix is the **overall sign**:
it is negated on load, exactly as ``default_sensitivity`` does.  That alone
reconciles the two.

The residual per-DOF disagreement is real physics, not convention: ts_ofc's
matrix is a different optics version fit by a different method.

Nothing here depends on sensor boundaries
-----------------------------------------
Chip heights are excluded throughout (``include_chip_heights=False``): the viz
does not model per-detector height variation, and the WFS mean-field points sit
between two chips where no single height applies.

Focus is likewise passed *explicitly* on every call -- ``0`` for the science
field and both Zernike computations, ``-+1500 um`` for the intra/extra donuts.
Passing the piston explicitly removes any dependence on detector geometry, so
the science layer is uniformly in focus and the only donuts are the ones we
asked for.

Usage
-----
python make_viz_model.py                      # v3.14, r band
python make_viz_model.py --version 1000 --band i --kmax 20
python make_viz_model.py --science-nrad 3 --pupil-nrad 6   # fast, coarse
"""
import argparse
import json
import os
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import Angle

from StarSharp.models.fiducial import (
    _DEFAULT,
    default_normalizations,
    default_raytraced_model,
)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
VIZ_DIR = REPO_ROOT / "viz"

# Detector IDs of the wavefront sensor halves, used only to place the donut
# field points.  The defocus itself is passed explicitly below rather than
# inferred from these IDs -- see the focus note in the module docstring.
INTRA_DETS = [192, 196, 200, 204]
EXTRA_DETS = [191, 195, 199, 203]

# Canonical WFS defocus, and the explicit "in focus" value for science points.
FOCUS_OFFSET = 1500.0 * u.micron
NO_PISTON = 0.0 * u.micron

# Relative path of ts_ofc's operational sensitivity matrix.
TS_OFC_SENS_RELPATH = Path(
    "policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml"
)

# Bumped whenever the block layout changes incompatibly.  v7 was the last
# layout written by the standalone rubin_optical_model_visualization repo; v8
# splits the single fwhm_weights block into two sources and adds a second
# sensitivity block.
META_VERSION = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the packed model for the browser visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="3.14",
        help="Optics version: 'design'/'3.3'/'3.12'/'3.14'/'1000'")
    parser.add_argument("--band", default="r", help="LSST band (u/g/r/i/z/y)")
    parser.add_argument("--rtp", default="0 deg",
        help="Rotator position angle (astropy Angle string)")
    parser.add_argument("--kmax", type=int, default=30,
        help="Maximum field Noll index for the double-Zernike sensitivity. "
             "ts_ofc's matrix natively has kmax=30; other values truncate it "
             "or zero-pad it to match")
    parser.add_argument("--jmax", type=int, default=28,
        help="Maximum pupil Noll index")
    parser.add_argument("--field-outer", type=float, default=1.75,
        help="Outer field radius [deg]")
    parser.add_argument("--science-nrad", type=int, default=5,
        help="Radial rings in the science-field hexapolar grid")
    parser.add_argument("--pupil-nrad", type=int, default=12,
        help="Pupil sampling rings for science spot diagrams")
    parser.add_argument("--donut-nrad", type=int, default=20,
        help="Pupil sampling rings for donut spot diagrams")
    parser.add_argument("--dz-field-nrad", type=int, default=10,
        help="Radial rings in the field grid the double-Zernike sensitivity is "
             "fit over")
    parser.add_argument("--zk-rings", type=int, default=10,
        help="Pupil quadrature rings for each per-field Zernike computation")
    parser.add_argument("--outdir", default=None, metavar="DIR",
        help=f"Output directory (default: {VIZ_DIR})")
    parser.add_argument("--no-pointing-model", action="store_true",
        help="Disable the default pointing model")
    parser.add_argument("--no-rtp-lookup", action="store_true",
        help="Disable the default RTP lookup table")
    parser.add_argument("--spot-optic", default=None, metavar="YAML",
        help="DIAGNOSTIC: trace the spot and donut blocks through this batoid "
             "yaml instead of the one implied by --version/--band.  Intended "
             "for LSST_r_align_holes.yaml, whose artificial pupil "
             "obscurations spell '+X' and '+Y' on the pupil and so make the "
             "OCS orientation directly visible in the rendered spots.  "
             "Sensitivities and Zernikes still use the real optic, so the "
             "output is for orientation checks only -- do not ship it")
    return parser.parse_args()


def load_ts_ofc_sensitivity(kmax: int, jmax: int) -> np.ndarray:
    """Load ts_ofc's double-Zernike sensitivity, conformed to (50, kmax+1, jmax+1).

    The on-disk matrix is ``(31, 29, 50)`` -- kmax=30, jmax=28 -- in ts_ofc's
    sign convention.  It is negated to match StarSharp's (see the module
    docstring), then the field axis is truncated or zero-padded to the requested
    *kmax*.

    Raises
    ------
    RuntimeError
        If ``$TS_OFC_DIR`` is unset or the matrix is missing.
    """
    ts_ofc = os.environ.get("TS_OFC_DIR")
    if not ts_ofc:
        raise RuntimeError(
            "$TS_OFC_DIR is not set, so ts_ofc's sensitivity matrix cannot be "
            "located."
        )
    path = Path(ts_ofc) / TS_OFC_SENS_RELPATH
    if not path.is_file():
        raise RuntimeError(f"ts_ofc sensitivity matrix not found at {path}")

    with open(path) as f:
        raw = np.array(yaml.safe_load(f))  # (31, 29, 50)
    # (31, 29, 50) -> (50, 31, 29), negated into StarSharp's sign convention.
    arr = -np.moveaxis(raw, -1, 0)
    ndof, native_kmax1, native_jmax1 = arr.shape

    if native_jmax1 != jmax + 1:
        raise ValueError(
            f"ts_ofc matrix has jmax={native_jmax1 - 1}, but --jmax={jmax} was "
            f"requested; the pupil axis is not padded because a mismatch here "
            f"usually means the wrong matrix or the wrong --jmax."
        )

    want = kmax + 1
    if want == native_kmax1:
        return arr
    if want < native_kmax1:
        print(
            f"  truncating ts_ofc field axis {native_kmax1} -> {want}",
            flush=True,
        )
        return arr[:, :want, :]
    print(
        f"  zero-padding ts_ofc field axis {native_kmax1} -> {want}",
        flush=True,
    )
    out = np.zeros((ndof, want, native_jmax1), dtype=arr.dtype)
    out[:, :native_kmax1, :] = arr
    return out


def with_spot_optic(base, yaml_name: str):
    """Return a copy of *base* that ray-traces through a different batoid optic.

    Used by ``--spot-optic`` to swap in a diagnostic optics description (e.g.
    ``LSST_r_align_holes.yaml``, which carries artificial pupil obscurations
    spelling "+X" and "+Y") for the spot blocks only.

    Everything that is not the optic is carried over verbatim: rtp, wavelength,
    schema, pointing model, and the AOS offset.  Note that ``base.offset``
    *already* includes the RTP lookup contribution -- ``__init__`` folds the
    lookup into the offset -- so the clone is built with ``rtp_lookup=None``.
    Passing both would apply the lookup twice.
    """
    import batoid
    from batoid_rubin import LSSTBuilder

    from StarSharp.models.raytraced import RaytracedOpticalModel

    path = Path(yaml_name)
    # batoid.Optic.fromYaml resolves bare names against its own data dirs;
    # anything path-like is passed through as given.
    fiducial = batoid.Optic.fromYaml(
        str(path) if path.is_file() else yaml_name
    )
    builder = LSSTBuilder(
        fiducial,
        dof_coord_system="OCS",
        flip_m2_bending_modes=False,
        dof_angle_units="degree",
    )
    return RaytracedOpticalModel(
        builder=builder,
        rtp=base.rtp,
        wavelength=base.wavelength,
        state_schema=base.state_schema,
        offset=base.offset,
        pointing_model=base.pointing_model,
        rtp_lookup=None,  # already folded into base.offset
    )


def spots_arrays(sens):
    """Return (x0, y0, dx, dy) as plain arrays in micron, vignetted -> NaN."""
    x0 = sens.nominal.dx.to_value(u.micron).copy()
    y0 = sens.nominal.dy.to_value(u.micron).copy()
    vignetted = sens.nominal.vignetted
    x0[vignetted] = np.nan
    y0[vignetted] = np.nan
    return (
        x0,
        y0,
        sens.gradient.dx.to_value(u.micron),
        sens.gradient.dy.to_value(u.micron),
    )


def interleave(a, b):
    """Interleave two 1-D arrays as [a0, b0, a1, b1, ...] float32."""
    out = np.empty(a.size + b.size, dtype=np.float32)
    out[0::2] = a
    out[1::2] = b
    return out


def main():
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else VIZ_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    rtp = Angle(args.rtp)
    field_outer = args.field_outer * u.deg

    # ts_ofc's matrix is loaded up front: it needs no ray tracing, so failing
    # here costs the user seconds rather than the full generation run.
    print("Loading ts_ofc sensitivity matrix", flush=True)
    wf_sens_ts_ofc = load_ts_ofc_sensitivity(args.kmax, args.jmax)

    print(
        f"Building model: version={args.version!r}, band={args.band!r}, rtp={rtp}",
        flush=True,
    )
    model = default_raytraced_model(
        version=args.version,
        band=args.band,
        rtp=rtp,
        pointing_model=None if args.no_pointing_model else _DEFAULT,
        rtp_lookup=None if args.no_rtp_lookup else _DEFAULT,
    )
    schema = model.state_schema
    n_dof = schema.n_dof
    print(
        f"  pointing model: {model.pointing_model is not None}, "
        f"RTP lookup: {model.rtp_lookup is not None}, "
        f"offset |max| = {np.abs(model.offset.f.value).max():.4g}",
        flush=True,
    )

    # The spot blocks may be traced through a different optic than the
    # sensitivities; see --spot-optic.  Normally this is the same model.
    spot_model = model
    if args.spot_optic:
        spot_model = with_spot_optic(model, args.spot_optic)
        print(
            f"\n  *** DIAGNOSTIC BUILD: spots traced through "
            f"{args.spot_optic!r} ***\n"
            f"      Sensitivities and Zernikes still use "
            f"{args.version}/{args.band}, so the spot and sensitivity blocks "
            f"describe different optics.  Orientation checks only.\n",
            flush=True,
        )

    # ---------------------------------------------------------------- fields
    science_field = model.make_hex_field(
        outer=field_outer, nrad=args.science_nrad
    )
    intra_field = model.make_ccd_field(nx=1, types="ITL_WF", detnums=INTRA_DETS)
    extra_field = model.make_ccd_field(nx=1, types="ITL_WF", detnums=EXTRA_DETS)

    # ------------------------------------------------------- science spots
    print(
        f"Science spots: {science_field.nfield} field points, "
        f"pupil nrad={args.pupil_nrad}",
        flush=True,
    )
    science_sens = spot_model.spots_sensitivity(
        science_field,
        nrad=args.pupil_nrad,
        include_chip_heights=False,
        detector_piston=NO_PISTON,
        tqdm=tqdm,
    )
    x0, y0, dx, dy = spots_arrays(science_sens)
    nfield, nray = x0.shape
    N = nfield * nray

    # --------------------------------------------------------- donut spots
    # Focus is passed explicitly rather than inferred from detector ID, so the
    # two calls differ only in field and piston sign.
    print(f"Donut spots: intra, pupil nrad={args.donut_nrad}", flush=True)
    intra_sens = spot_model.spots_sensitivity(
        intra_field,
        nrad=args.donut_nrad,
        include_chip_heights=False,
        detector_piston=-FOCUS_OFFSET,
        tqdm=tqdm,
    )
    print(f"Donut spots: extra, pupil nrad={args.donut_nrad}", flush=True)
    extra_sens = spot_model.spots_sensitivity(
        extra_field,
        nrad=args.donut_nrad,
        include_chip_heights=False,
        detector_piston=+FOCUS_OFFSET,
        tqdm=tqdm,
    )
    ix0, iy0, idx_, idy_ = spots_arrays(intra_sens)
    ex0, ey0, edx, edy = spots_arrays(extra_sens)

    donut_x0 = np.concatenate([ix0, ex0], axis=0)
    donut_y0 = np.concatenate([iy0, ey0], axis=0)
    donut_dx = np.concatenate([idx_, edx], axis=1)
    donut_dy = np.concatenate([idy_, edy], axis=1)
    donut_nfield, donut_nray = donut_x0.shape
    donut_N = donut_nfield * donut_nray

    # ------------------------------------------------- corner zernikes (zk)
    # One point per WFS pair, at the mean of the two halves.
    corner_field = model.make_wfs_mean_field()
    print(f"Corner Zernikes: {corner_field.nfield} field points", flush=True)
    corner_sens = model.zernikes_sensitivity(
        corner_field,
        jmax=args.jmax,
        rings=args.zk_rings,
        include_chip_heights=False,
        detector_piston=NO_PISTON,
        tqdm=tqdm,
    )
    zk0_arr = corner_sens.nominal.coefs.to_value(u.nm)       # (nfield, jmax+1)
    dzk_arr = corner_sens.gradient.coefs.to_value(u.nm)      # (K, nfield, jmax+1)
    zk_corners, zk_terms = zk0_arr.shape
    # JS indexes dzk as (corner * nterms + term) * K + dof.
    dzk_flat = np.transpose(dzk_arr, (1, 2, 0)).ravel(order="C").astype(np.float32)
    zk0_flat = zk0_arr.ravel(order="C").astype(np.float32)

    # ------------------------------------- StarSharp double-Zernike sens
    print(
        f"StarSharp double-Zernike sensitivity: kmax={args.kmax}, "
        f"jmax={args.jmax}, field nrad={args.dz_field_nrad}, "
        f"pupil rings={args.zk_rings}",
        flush=True,
    )
    dz_field = model.make_hex_field(outer=field_outer, nrad=args.dz_field_nrad)
    dz_sens = model.double_zernikes_sensitivity(
        kmax=args.kmax,
        field_outer=field_outer,
        field=dz_field,
        jmax=args.jmax,
        rings=args.zk_rings,
        algorithm="zk",
        include_chip_heights=False,
        detector_piston=NO_PISTON,
        tqdm=tqdm,
    )
    wf_sens_starsharp = dz_sens.gradient.coefs.to_value(u.um)  # (50, kmax+1, jmax+1)
    if wf_sens_starsharp.shape != wf_sens_ts_ofc.shape:
        raise RuntimeError(
            "The two sensitivity blocks must have identical shape so the viz "
            f"can swap them: StarSharp {wf_sens_starsharp.shape} vs ts_ofc "
            f"{wf_sens_ts_ofc.shape}"
        )

    def flatten_sens(arr):
        """(ndof, kmax+1, jmax+1) -> (rows, ndof) float32 with rows = k*njmax + j."""
        return (
            np.transpose(arr, (1, 2, 0))
            .reshape(-1, arr.shape[0])
            .astype(np.float32)
        )

    ss_rows = flatten_sens(wf_sens_starsharp)
    ofc_rows = flatten_sens(wf_sens_ts_ofc)
    wf_rows, wf_cols = ss_rows.shape

    # --------------------------------------------------- normalization weights
    rng_w = default_normalizations(
        fwhm_exponent=0.0, range_exponent=1.0, schema=schema
    ).astype(np.float32)
    fwhm_ss = default_normalizations(
        fwhm_exponent=1.0, range_exponent=0.0, schema=schema, source="starsharp"
    ).astype(np.float32)
    fwhm_ofc = default_normalizations(
        fwhm_exponent=1.0, range_exponent=0.0, schema=schema, source="ts_ofc"
    ).astype(np.float32)

    # ------------------------------------------------------------- field coords
    sci_ocs = science_field.angle.ocs
    field_thx = sci_ocs.x.to_value(u.deg)
    field_thy = sci_ocs.y.to_value(u.deg)
    intra_ocs = intra_field.angle.ocs
    extra_ocs = extra_field.angle.ocs
    donut_thx = np.concatenate(
        [intra_ocs.x.to_value(u.deg), extra_ocs.x.to_value(u.deg)]
    )
    donut_thy = np.concatenate(
        [intra_ocs.y.to_value(u.deg), extra_ocs.y.to_value(u.deg)]
    )

    # ------------------------------------------------------------------- pack
    blocks: list[tuple[str, np.ndarray, dict]] = [
        ("spot_xy",
         interleave(x0.ravel(), y0.ravel()),
         {"encoding": "xy_interleaved", "units": "micron"}),
        ("Sx",
         dx.reshape(n_dof, N).ravel(order="C").astype(np.float32),
         {"shape": [n_dof, N], "order": "C", "units": "micron_per_dof_unit"}),
        ("Sy",
         dy.reshape(n_dof, N).ravel(order="C").astype(np.float32),
         {"shape": [n_dof, N], "order": "C", "units": "micron_per_dof_unit"}),
        ("field_xy",
         interleave(np.repeat(field_thx, nray), np.repeat(field_thy, nray)),
         {"encoding": "xy_interleaved", "units": "deg"}),
        ("donut_spot_xy",
         interleave(donut_x0.ravel(), donut_y0.ravel()),
         {"encoding": "xy_interleaved", "units": "micron"}),
        ("donut_Sx",
         donut_dx.reshape(n_dof, donut_N).ravel(order="C").astype(np.float32),
         {"shape": [n_dof, donut_N], "order": "C",
          "units": "micron_per_dof_unit"}),
        ("donut_Sy",
         donut_dy.reshape(n_dof, donut_N).ravel(order="C").astype(np.float32),
         {"shape": [n_dof, donut_N], "order": "C",
          "units": "micron_per_dof_unit"}),
        ("donut_field_xy",
         interleave(np.repeat(donut_thx, donut_nray),
                    np.repeat(donut_thy, donut_nray)),
         {"encoding": "xy_interleaved", "units": "deg"}),
        ("zk0",
         zk0_flat,
         {"shape": [zk_corners, zk_terms], "order": "C", "units": "nm"}),
        ("dzk",
         dzk_flat,
         {"shape": [zk_corners, zk_terms, n_dof], "order": "C",
          "units": "nm_per_dof_unit"}),
        ("range_weights",
         rng_w,
         {"shape": [n_dof], "order": "C", "units": "dof_unit"}),
        ("fwhm_starsharp_weights",
         fwhm_ss,
         {"shape": [n_dof], "order": "C", "units": "arcsec_per_dof_unit",
          "source": "starsharp"}),
        ("fwhm_ts_ofc_weights",
         fwhm_ofc,
         {"shape": [n_dof], "order": "C", "units": "arcsec_per_dof_unit",
          "source": "ts_ofc"}),
        ("wf_sens_starsharp",
         ss_rows.ravel(order="C"),
         {"shape": [wf_rows, wf_cols], "n_field_dz": args.kmax + 1,
          "n_pupil_zk": args.jmax + 1, "order": "C",
          "units": "micron_per_dof_unit", "source": "starsharp"}),
        ("wf_sens_ts_ofc",
         ofc_rows.ravel(order="C"),
         {"shape": [wf_rows, wf_cols], "n_field_dz": args.kmax + 1,
          "n_pupil_zk": args.jmax + 1, "order": "C",
          "units": "micron_per_dof_unit", "source": "ts_ofc",
          "conventions_applied": ["sign_negated"]}),
    ]

    layout = {}
    offset = 0
    for name, arr, extra in blocks:
        layout[name] = {
            "offset_f32": int(offset),
            "length_f32": int(arr.size),
            **extra,
        }
        offset += arr.size

    packed = np.concatenate([arr for _, arr, _ in blocks])
    assert packed.size == offset, "block offsets disagree with packed length"
    assert packed.dtype == np.float32, f"packed dtype drifted to {packed.dtype}"

    meta = {
        "version": META_VERSION,
        "dtype": "float32",
        "N": int(N),
        "K": int(n_dof),
        "donut_N": int(donut_N),
        "donut_nfield_intra": int(ix0.shape[0]),
        "donut_nfield_extra": int(ex0.shape[0]),
        "donut_nray": int(donut_nray),
        "donut_clock_angle_deg": 10.0,
        "donut_clock_radius_scale": 1.4,
        "provenance": {
            "optics_version": args.version,
            "band": args.band,
            "rtp_deg": float(rtp.to_value(u.deg)),
            "wavelength_nm": float(model.wavelength.to_value(u.nm)),
            "kmax": int(args.kmax),
            "jmax": int(args.jmax),
            "field_outer_deg": float(args.field_outer),
            "science_nrad": int(args.science_nrad),
            "pupil_nrad": int(args.pupil_nrad),
            "donut_nrad": int(args.donut_nrad),
            "dz_field_nrad": int(args.dz_field_nrad),
            "zk_rings": int(args.zk_rings),
            "pointing_model": model.pointing_model is not None,
            "rtp_lookup": model.rtp_lookup is not None,
            "include_chip_heights": False,
            "generator": "bootstrap/make_viz_model.py",
            # Non-null only for a --spot-optic diagnostic build, where the
            # spot blocks and the sensitivity blocks describe different optics.
            "spot_optic_override": args.spot_optic,
        },
        "dof_names": list(schema.dof_names),
        "dof_units": [str(x) for x in schema.dof_units],
        "fwhm_sources": ["starsharp", "ts_ofc"],
        "wf_sens_sources": ["starsharp", "ts_ofc"],
        "layout": layout,
    }

    model_path = outdir / "model.f32"
    meta_path = outdir / "model_meta.json"
    packed.tofile(model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    mb = packed.nbytes / 1024**2
    print(f"\nWrote {model_path} ({mb:.1f} MiB, {packed.size} float32)", flush=True)
    print(f"Wrote {meta_path} (layout version {META_VERSION})", flush=True)
    print(f"  N={N} ({nfield} field x {nray} ray), donut_N={donut_N}, K={n_dof}")
    print(f"  wf_sens: {wf_rows} x {wf_cols} per source")


if __name__ == "__main__":
    main()
