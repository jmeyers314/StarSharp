#!/usr/bin/env python
"""Regenerate range.yaml (DOF working ranges) at full double precision.

These numbers are *not* computed by StarSharp -- they originate in ts_ofc and
are only transcribed here.  Requires ts_ofc to be importable; if it is not,
leave the committed file alone.

Provenance
----------
The 10 rigid-body strokes are hard-coded literals in
``ts_ofc/python/lsst/ts/ofc/ofc_data/base_ofc_data.py::default_rb_stroke``:

    [5900, 6700, 6700, 0.12, 0.12, 8700, 7600, 7600, 0.24, 0.24]

There is no more fundamental YAML for them.  Being exact decimals, the values
already round-trip through YAML bit-for-bit.

The 40 bending-mode ranges are *computed*:

    m1m3: (m1m3_force_range / 20) / max(|rot_mat_m1m3|, axis=0)
    m2:   (m2_force_range   / 20) / max(|rot_mat_m2|,   axis=0)

with ``m1m3_force_range = 67*2`` and ``m2_force_range = 22.5*2`` (also
hard-coded, ``base_ofc_data.py``), and ``rot_mat`` the bending-mode-to-force
influence matrices read from ts_ofc's ``policy/M1M3/M1M3_1um_156_force.yaml``
and ``policy/M2/M2_1um_72_force.yaml``.  Those two force files are the
fundamental data.

StarSharp previously stored these rounded to 8 decimals (0.45413045 vs
0.4541304520224734), a max relative error of 2.5e-6.  That is negligible for
normalization purposes but costs nothing to fix.

Usage
-----
python make_range_weights.py [--output PATH] [--no-install]
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "StarSharp" / "data" / "normalization_weights"
OUT_DIR = HERE / "normalization_weights"


def compute_range_weights() -> np.ndarray:
    """Return the 50 DOF working ranges, computed from ts_ofc at full precision."""
    try:
        from lsst.ts.ofc import BendModeToForce, OFCData
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "ts_ofc is not importable, so range.yaml cannot be regenerated.\n"
            "The committed file is already correct to ~8 significant figures; "
            "leave it alone unless you need full precision.\n"
            f"(import error: {exc})"
        )

    ofc_data = OFCData("lsst")

    m1m3_bending_range = ofc_data.m1m3_force_range / 20.0
    m2_bending_range = ofc_data.m2_force_range / 20.0

    m1m3_bmf = BendModeToForce("M1M3", ofc_data)
    m2_bmf = BendModeToForce("M2", ofc_data)

    return np.concatenate(
        (
            ofc_data.rb_stroke,
            m1m3_bending_range / np.max(np.abs(m1m3_bmf.rot_mat), axis=0),
            m2_bending_range / np.max(np.abs(m2_bmf.rot_mat), axis=0),
        )
    )


def write_yaml(ranges: np.ndarray, path: Path) -> None:
    from StarSharp.models.fiducial import default_schema

    schema = default_schema()
    if len(ranges) != schema.n_dof:
        raise ValueError(f"expected {schema.n_dof} ranges, got {len(ranges)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "# Allowed working range of each of the 50 AOS degrees of freedom.",
        "#",
        "# Units: um for the hexapod translations (M2/Cam dz, dx, dy), degrees for",
        "# the hexapod rotations (rx, ry), and um of surface deformation for the",
        "# M1M3 and M2 bending modes.",
        "#",
        "# PROVENANCE -- these values originate in ts_ofc and are transcribed here,",
        "# not computed by StarSharp.  Regenerate with",
        "# bootstrap/make_range_weights.py (requires ts_ofc importable).",
        "#",
        "# The 10 rigid-body strokes are hard-coded literals in ts_ofc",
        "# (ofc_data/base_ofc_data.py::default_rb_stroke); there is no upstream YAML",
        "# for them.  The 40 bending-mode ranges are computed as",
        "#",
        "#     (force_range / 20) / max(|rot_mat|, axis=0)",
        "#",
        "# with force_range = 67*2 N (M1M3) and 22.5*2 N (M2), and rot_mat the",
        "# bending-mode-to-force influence matrices from ts_ofc's",
        "# policy/M1M3/M1M3_1um_156_force.yaml and policy/M2/M2_1um_72_force.yaml.",
        "",
    ]
    for name, value in zip(schema.dof_names, ranges):
        lines.append(f"- {float(value)!r}  # {name}")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output path (default: {OUT_DIR / 'range.yaml'})")
    parser.add_argument("--no-install", action="store_true",
                        help="Do not copy the result into StarSharp/data/")
    args = parser.parse_args()

    out_path = args.output or (OUT_DIR / "range.yaml")
    write_yaml(compute_range_weights(), out_path)

    if not args.no_install:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dst = DATA_DIR / "range.yaml"
        shutil.copy2(out_path, dst)
        print(f"Installed {out_path.name} -> {dst}")


if __name__ == "__main__":
    sys.exit(main())
