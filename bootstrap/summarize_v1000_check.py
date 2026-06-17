#!/usr/bin/env python
"""Summarize the v1000-vs-v3.14 sufficiency check.

Reads the focus_scan output tables produced by check_v1000_vs_3.14.sh (each has
``rtp``, ``rms_nominal``, ``rms_optimized`` columns) and reports, per band and
overall, how much spot blur is left on the v1000 optic when the v3.14 lookup is
reused (rms_nominal) versus after a fresh residual fit (rms_optimized).

A small gap everywhere means the v3.14 lookup is sufficient for v1000.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from astropy.table import QTable


def band_of(path: Path) -> str:
    m = re.search(r"_([ugrizy])\.ecsv$", path.name)
    return m.group(1) if m else path.stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tables", nargs="+", help="focus_scan output ECSV files")
    args = ap.parse_args()

    all_gaps = []
    print(f"{'band':>4}  {'max_nom':>8}  {'max_opt':>8}  "
          f"{'max_gap':>8}  {'med_gap':>8}  {'rtp@maxgap':>10}")
    print("-" * 60)
    for path in sorted(map(Path, args.tables)):
        t = QTable.read(path)
        nom = t["rms_nominal"].to_value("micron")
        opt = t["rms_optimized"].to_value("micron")
        rtp = t["rtp"].to_value("deg")
        gap = nom - opt
        all_gaps.append(gap)
        i = int(np.nanargmax(gap))
        print(f"{band_of(path):>4}  {np.nanmax(nom):8.3f}  {np.nanmax(opt):8.3f}  "
              f"{np.nanmax(gap):8.3f}  {np.nanmedian(gap):8.3f}  {rtp[i]:10.1f}")

    g = np.concatenate(all_gaps)
    print("-" * 60)
    print(f"OVERALL  max gap = {np.nanmax(g):.3f} um   "
          f"median gap = {np.nanmedian(g):.3f} um   "
          f"(units: spot RMS, micron)")
    print("\nInterpretation: small gap (e.g. << optimized RMS) => v3.14 lookup "
          "is sufficient for v1000; large gap => bootstrap v1000 separately.")


if __name__ == "__main__":
    main()
