#!/usr/bin/env python
"""Produce the default sensitivity FITS and ASDF files and install them into StarSharp/data/sensitivity/.

This script runs compute_sensitivity.py (v3.14, i-band, zk algorithm) and
copies both outputs into StarSharp/data/sensitivity/.

Usage
-----
python make_default_sensitivity.py [--recompute]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "StarSharp" / "data" / "sensitivity"
SENS_DIR = HERE / "sensitivity"

STEM = "3.14_i"
FITS_PATH = SENS_DIR / f"{STEM}.fits"
ASDF_PATH = SENS_DIR / f"{STEM}.asdf"


def compute(recompute: bool = False) -> None:
    if FITS_PATH.exists() and ASDF_PATH.exists() and not recompute:
        print(f"Outputs already exist in {SENS_DIR}  (use --recompute to force)")
        return
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Computing sensitivity → {FITS_PATH}, {ASDF_PATH}")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "compute_sensitivity.py"),
            "--version", "3.14",
            "--band", "i",
            "--algorithm", "zk",
            "--output", str(FITS_PATH),
            "--output-asdf", str(ASDF_PATH),
        ],
        check=True,
    )


def install() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src in (FITS_PATH, ASDF_PATH):
        dst = DATA_DIR / src.name
        shutil.copy2(src, dst)
        print(f"Installed {src.name} → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation even if outputs already exist")
    args = parser.parse_args()

    compute(recompute=args.recompute)
    install()


if __name__ == "__main__":
    main()
