#!/usr/bin/env python
"""Focus scan: optimize DOFs over a range of RTP angles.

For each RTP angle the script:
  1. Builds a RaytracedOpticalModel (reusing a single batoid builder/camera).
  2. Evaluates the nominal (un-corrected) spot RMS.
  3. Optimizes the requested DOFs to minimize spot size.
  4. Evaluates the optimized spot RMS.

Results are saved to an ECSV table and a diagnostic PNG with one panel per
DOF plus a spot-RMS panel.  Warm-starting (previous angle's solution as the
initial guess) is used to improve convergence and limit false minima.

Usage examples
--------------
# design (v3.3) optics, r-band, default dofs
python focus_scan.py

# as-built v3.12 optics, i-band
python focus_scan.py --version 3.12 --band i

# 3-DOF tilt+focus
python focus_scan.py --dofs Cam_dz Cam_rx Cam_ry

# full hexapod scan
python focus_scan.py --dofs Cam_dz Cam_dx Cam_dy Cam_rx Cam_ry
"""

import argparse
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.table import QTable

from StarSharp.datatypes import StateFactory
from StarSharp.datatypes import PointingModel
from StarSharp.models import RaytracedOpticalModel, RTPLookup
from StarSharp.models.fiducial import default_raytraced_model, default_schema

# Derive available camera DOF names from the fiducial schema (no batoid load)
_FIDUCIAL_SCHEMA = default_schema()
_DOF_NAMES = _FIDUCIAL_SCHEMA.dof_names
_DEFAULT_DOFS = ["Cam_dz", "Cam_rx", "Cam_ry"]


def spot_rms(spots) -> float:
    """RMS spot radius (micron) over all unvignetted rays and field points."""
    w = ~spots.vignetted
    dx = spots.dx.to_value(u.micron)[w]
    dy = spots.dy.to_value(u.micron)[w]
    if len(dx) == 0:
        return np.nan
    return float(np.sqrt(np.mean(dx**2 + dy**2)))


def parse_detnums(detnums_str: str) -> list[int]:
    """Parse a string like '4..184:9' into a list of detector numbers."""
    detnums = []
    for part in detnums_str.split(","):
        if ".." in part:
            start_str, rest = part.split("..", 1)
            if ":" in rest:
                end_str, step_str = rest.split(":", 1)
                step = int(step_str)
            else:
                end_str = rest
                step = 1
            start = int(start_str)
            end = int(end_str)
            detnums.extend(range(start, end + 1, step))
        else:
            detnums.append(int(part))
    return detnums

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan RTP angles and optimize camera hexapod DOFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        default="3.3",
        help="Optics version: '3.3'/'3.12'/'3.14'",
    )
    parser.add_argument("--band", default="r", help="LSST band (u/g/r/i/z/y)")
    parser.add_argument(
        "--rtp-min", type=float, default=-90.0, help="Minimum RTP angle [deg]"
    )
    parser.add_argument(
        "--rtp-max", type=float, default=90.0, help="Maximum RTP angle [deg]"
    )
    parser.add_argument("--rtp-step", type=float, default=10.0, help="RTP step [deg]")
    parser.add_argument(
        "--mode",
        default="dx",
        choices=["dx", "var"],
        help="Optimization mode: 'dx' minimises raw displacements, 'var' minimises variance",
    )
    parser.add_argument(
        "--nrad", type=int, default=6, help="Pupil sampling rings for spot tracing"
    )
    parser.add_argument(
        "--dofs",
        nargs="+",
        default=_DEFAULT_DOFS,
        choices=_DOF_NAMES,
        metavar="DOF",
        help=(
            f"Camera hexapod DOFs to optimize. "
            f"Choices: {_DOF_NAMES}. "
            f"Default: {_DEFAULT_DOFS}"
        ),
    )
    parser.add_argument(
        "--nx", type=int, default=4, help="Grid points per CCD axis (nx*nx points per detector)"
    )
    parser.add_argument(
        "--detnums", type=str, default="4..184:9", help="Detector numbers to include (e.g. '4..184:9' for every 9th from 4 to 184)"
    )
    parser.add_argument(
        "--rtp-lookup",
        default=None,
        metavar="FILE",
        help="ECSV file with a pre-computed RTPLookup applied as a fixed offset before optimization",
    )
    parser.add_argument(
        "--pointing-model",
        default=None,
        metavar="FILE",
        help="ECSV file with a pointing model to apply during ray tracing",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ecsv filename (default: focus_scan_{version}_{band}.ecsv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate and resolve DOF names → indices/units (preserve order as given)
    seen = set()
    dof_names = []
    for name in args.dofs:
        if name in seen:
            raise ValueError(f"Duplicate DOF: {name!r}")
        seen.add(name)
        dof_names.append(name)

    rtp_angles = np.arange(
        args.rtp_min,
        args.rtp_max + 0.5 * args.rtp_step,
        args.rtp_step,
    )
    output_ecsv = Path(args.output or f"focus_scan_{args.version}_{args.band}.ecsv")
    output_png = output_ecsv.with_suffix(".png")

    print(f"Loading optics: version={args.version!r}, band={args.band!r}", flush=True)
    print(f"DOFs: {dof_names}", flush=True)

    rtp_lookup = None
    if args.rtp_lookup:
        print(f"Loading RTP lookup from {args.rtp_lookup!r}", flush=True)
        rtp_lookup = RTPLookup.from_file(args.rtp_lookup)

    pointing_model = None
    if args.pointing_model:
        print(f"Using pointing model from {args.pointing_model!r}", flush=True)
        pointing_model = PointingModel.from_table(QTable.read(args.pointing_model))

    ref_model = default_raytraced_model(
        version=args.version,
        band=args.band,
        rtp=Angle(0.0, unit=u.deg),
    )
    builder = ref_model.builder
    camera = ref_model.camera
    wavelength = ref_model.wavelength
    state_schema = ref_model.state_schema

    # Resolve DOF indices and units from the model's schema
    name_to_idx = {n: i for i, n in enumerate(state_schema.dof_names)}
    missing = [n for n in dof_names if n not in name_to_idx]
    if missing:
        raise ValueError(f"DOF(s) not found in schema: {missing}")
    use_dof = np.array([name_to_idx[n] for n in dof_names])
    units = [state_schema.dof_units[i] for i in use_dof]
    n_dofs = len(use_dof)

    dof_schema = replace(state_schema, use_dof=use_dof, Vh=None)
    sf = StateFactory(dof_schema)

    n_rtp = len(rtp_angles)
    results = np.full((n_rtp, n_dofs), np.nan)
    rms_nominal = np.full(n_rtp, np.nan)
    rms_optimized = np.full(n_rtp, np.nan)

    # Lookup DOFs not being optimized — carried through to the output table
    lookup_extra_names: list[str] = []
    lookup_extra_dofs: list[int] = []
    if rtp_lookup is not None:
        for name in rtp_lookup._dof_cols:
            if name not in dof_names and name in name_to_idx:
                lookup_extra_names.append(name)
                lookup_extra_dofs.append(name_to_idx[name])
    lookup_extra_units = [state_schema.dof_units[i] for i in lookup_extra_dofs]
    lookup_extra_results = np.full((n_rtp, len(lookup_extra_names)), np.nan)

    current_x = np.zeros(n_dofs)

    print(
        f"Scanning {n_rtp} RTP angles from {rtp_angles[0]:.1f} to {rtp_angles[-1]:.1f} deg, "
        f"step={args.rtp_step:.1f} deg, mode={args.mode!r}",
        flush=True,
    )

    for i, rtp_deg in enumerate(rtp_angles):
        print(f"[{i + 1}/{n_rtp}] RTP = {rtp_deg:+.1f} deg", end="  ", flush=True)

        rtp = Angle(rtp_deg, unit=u.deg)
        model = RaytracedOpticalModel(
            builder=builder,
            rtp=rtp,
            wavelength=wavelength,
            state_schema=state_schema,
            camera=camera,
            rtp_lookup=rtp_lookup,
            pointing_model=pointing_model,
        )

        detnums = parse_detnums(args.detnums)
        field = model.make_ccd_field(nx=args.nx, detnums=detnums)

        spots0 = model.spots(field=field, nrad=args.nrad)
        rms_nominal[i] = spot_rms(spots0)

        guess = sf.x(current_x)
        result = model.optimize(
            guess=guess,
            field=field,
            nrad=args.nrad,
            mode=args.mode,
        )

        x_vals = result.x.value
        if rtp_lookup is not None:
            lookup_state = rtp_lookup.state_at(rtp, state_schema)
            x_vals = x_vals + lookup_state.f.value[use_dof]
            if lookup_extra_dofs:
                lookup_extra_results[i] = lookup_state.f.value[lookup_extra_dofs]
        results[i] = x_vals
        current_x = result.x.value

        spots_opt = model.spots(field=field, state=result.f, nrad=args.nrad)
        rms_optimized[i] = spot_rms(spots_opt)

        dof_str = "  ".join(
            f"{name}={val:.4g}"
            for name, val in zip(dof_names, x_vals)
        )
        print(
            f"RMS: {rms_nominal[i]:.2f} → {rms_optimized[i]:.2f} um  {dof_str}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Save results table                                                   #
    # ------------------------------------------------------------------ #
    table = QTable()
    table["rtp"] = rtp_angles * u.deg
    for j, (name, unit) in enumerate(zip(dof_names, units)):
        table[name] = results[:, j] * unit
    for j, (name, unit) in enumerate(zip(lookup_extra_names, lookup_extra_units)):
        table[name] = lookup_extra_results[:, j] * unit
    table["rms_nominal"] = rms_nominal * u.micron
    table["rms_optimized"] = rms_optimized * u.micron

    table.write(output_ecsv, overwrite=True)
    print(f"\nSaved table to {output_ecsv}")

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = 2
        nrows = (n_dofs + 1 + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes = axes.ravel()

        unit_labels = [
            "um" if str(u_) == str(u.um) else "deg" for u_ in units
        ]
        for j, (name, unit_label) in enumerate(zip(dof_names, unit_labels)):
            ax = axes[j]
            ax.plot(rtp_angles, results[:, j], "o-", ms=4)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("RTP [deg]")
            ax.set_ylabel(f"{name} [{unit_label}]")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        ax = axes[n_dofs]
        ax.plot(rtp_angles, rms_nominal, "s--", ms=4, label="nominal", color="C1")
        ax.plot(rtp_angles, rms_optimized, "o-", ms=4, label="optimized", color="C0")
        ax.set_xlabel("RTP [deg]")
        ax.set_ylabel("Spot RMS [um]")
        ax.set_title("Spot RMS")
        ax.legend()
        ax.grid(True, alpha=0.3)

        for ax in axes[n_dofs + 1:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Focus scan — version={args.version!r}, band={args.band!r}, "
            f"mode={args.mode!r}, nrad={args.nrad}",
            fontsize=13,
        )
        fig.tight_layout()

        fig.savefig(output_png, dpi=150)
        print(f"Saved plot to {output_png}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
