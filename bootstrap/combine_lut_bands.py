#!/usr/bin/env python
"""Average DOF columns from focus-scan ECSV runs into a single RTPLookup table.

Finds every column that is present in all input files (excluding ``rtp``,
``rms_nominal``, and ``rms_optimized``), averages those columns across the
inputs at each RTP angle, and writes an ECSV suitable for use as an
RTPLookup.

All input tables must share the same RTP grid.

Usage examples
--------------
# Average Cam_dx/Cam_dy from the successful 3.14 5-DOF runs
python combine_lut_bands.py \\
    focus_scan/5/focus_scan_3.14_i.ecsv \\
    focus_scan/5/focus_scan_3.14_u.ecsv \\
    focus_scan/5/focus_scan_3.14_y.ecsv \\
    --output focus_scan/dxdy_3.14.ecsv

# Restrict to specific columns
python combine_lut_bands.py ... --cols Cam_dx Cam_dy
"""

import argparse

import astropy.units as u
import numpy as np
from astropy.table import QTable

_SKIP_COLS = {"rtp", "rms_nominal", "rms_optimized"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average DOF columns from focus-scan runs into an RTPLookup table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", metavar="FILE", help="Input ECSV files")
    parser.add_argument("--output", required=True, help="Output ECSV path")
    parser.add_argument(
        "--cols",
        nargs="+",
        default=None,
        metavar="COL",
        help="DOF columns to include (default: all shared DOF columns)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tables = [QTable.read(f) for f in args.inputs]

    # Verify all share the same RTP grid
    rtp_ref = tables[0]["rtp"].to_value(u.deg)
    for path, t in zip(args.inputs[1:], tables[1:]):
        rtp = t["rtp"].to_value(u.deg)
        if not np.allclose(rtp, rtp_ref):
            raise ValueError(f"{path} has a different RTP grid than {args.inputs[0]}")

    # Determine columns to average
    if args.cols is not None:
        dof_cols = args.cols
        for col in dof_cols:
            for path, t in zip(args.inputs, tables):
                if col not in t.colnames:
                    raise ValueError(f"{path!r} is missing requested column {col!r}")
    else:
        shared = set(tables[0].colnames)
        for t in tables[1:]:
            shared &= set(t.colnames)
        dof_cols = [c for c in tables[0].colnames if c in shared and c not in _SKIP_COLS]

    if not dof_cols:
        raise ValueError("No DOF columns to average")

    out = QTable()
    out["rtp"] = rtp_ref * u.deg
    for col in dof_cols:
        unit = tables[0][col].unit
        values = np.mean([t[col].to(unit).value for t in tables], axis=0)
        out[col] = values * unit
        print(f"  {col} [{unit}]: range [{values.min():.4g}, {values.max():.4g}]")

    out.write(args.output, overwrite=True)
    print(f"Wrote {args.output} ({len(out)} rows, {len(dof_cols)} DOF columns, averaged over {len(tables)} files)")


if __name__ == "__main__":
    main()
