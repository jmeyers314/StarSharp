"""Refit DOF columns in a focus_scan ECSV against a + b*sin(RTP + phi),
sigma-clip outliers, refit, and optionally write a smoothed RTPLookup ECSV.

Columns listed with ``--cols`` are fit with a sinusoid; columns listed with
``--const-cols`` are fit with a constant (mean).  All other DOF columns are
passed through to the output using piecewise-linear interpolation, optionally
resampled to a different grid resolution.

RTP ranges can be manually excluded from the fit with ``--exclude-rtp``; excluded
points are shown in the plot but never used in the fit or sigma-clipping.

Usage
-----
    # Inspect a file, fit specific columns
    python refit_lut.py focus_scan/5/focus_scan_3.14_u.ecsv --cols Cam_dx Cam_dy --plot

    # Fit sin to tilt DOFs, write smoothed lookup table
    python refit_lut.py focus_scan/3/focus_scan_3.14_u.ecsv \\
        --cols Cam_rx Cam_ry --const-cols Cam_dz \\
        --output focus_scan/smooth_3.14_u.ecsv

    # Exclude a bad RTP range, tighter sigma clip, finer output grid
    python refit_lut.py focus_scan/5/focus_scan_3.14_u.ecsv \\
        --cols Cam_dx Cam_dy --exclude-rtp 10 40 --sigma 2 --step 1 --plot

    # Resample to 3 deg grid without fitting (no --cols or --const-cols)
    python refit_lut.py focus_scan/3/focus_scan_3.12_dz_r.ecsv \\
        --step 3 --output focus_scan/3/smooth_dz/dz_3.12_r.ecsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def _design_sin(rtp_rad: np.ndarray) -> np.ndarray:
    """Design matrix [1, sin(rtp), cos(rtp)], shape (n, 3)."""
    return np.column_stack([np.ones_like(rtp_rad), np.sin(rtp_rad), np.cos(rtp_rad)])


def _design_const(rtp_rad: np.ndarray) -> np.ndarray:
    """Design matrix [1], shape (n, 1)."""
    return np.ones((len(rtp_rad), 1))


def _ols(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def _predict(A: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return A @ coeffs


def _amplitude_phase(c: float, d: float) -> tuple[float, float]:
    """c*sin + d*cos  →  b*sin(x + phi).  Returns (b, phi_deg)."""
    return np.hypot(c, d), np.degrees(np.arctan2(d, c))


def _excluded_mask(rtp_deg: np.ndarray, exclude_ranges: list[list[float]]) -> np.ndarray:
    """Boolean mask, True where rtp_deg falls in any manually excluded range."""
    mask = np.zeros(len(rtp_deg), dtype=bool)
    for lo, hi in exclude_ranges:
        mask |= (rtp_deg >= lo) & (rtp_deg <= hi)
    return mask


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

_SKIP_COLS = {"rtp"}
_SKIP_PREFIX = "rms_"


def _dof_cols(tbl: QTable) -> list[str]:
    return [c for c in tbl.colnames
            if c not in _SKIP_COLS and not c.startswith(_SKIP_PREFIX)]


# ---------------------------------------------------------------------------
# Core fit for a single column
# Returns (coeffs, inlier_mask, excluded_mask) — all length n
# ---------------------------------------------------------------------------

def _fit_col(
    rtp_deg: np.ndarray,
    y: np.ndarray,
    use_const: bool,
    sigma: float,
    exclude_ranges: list[list[float]],
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one column.  Returns (coeffs, inlier, excluded) boolean masks."""
    rtp_rad = np.radians(rtp_deg)
    A_full = _design_const(rtp_rad) if use_const else _design_sin(rtp_rad)

    excluded = _excluded_mask(rtp_deg, exclude_ranges)
    candidate = ~excluded
    min_pts = 1 if use_const else 3

    if candidate.sum() < min_pts:
        print(f"WARNING: {label}: too few non-excluded points; using all data", file=sys.stderr)
        candidate = np.ones(len(y), dtype=bool)
        excluded = np.zeros(len(y), dtype=bool)

    # Initial fit on candidates
    A_cand = A_full[candidate]
    coeffs0 = _ols(A_cand, y[candidate])
    resid0 = y[candidate] - _predict(A_cand, coeffs0)

    # Sigma clip within candidates
    std0 = np.std(resid0)
    keep = np.abs(resid0) <= sigma * std0 if std0 > 0 else np.ones(candidate.sum(), dtype=bool)

    inlier = np.zeros(len(y), dtype=bool)
    inlier[np.where(candidate)[0][keep]] = True

    if inlier.sum() < min_pts:
        print(f"WARNING: {label}: too few inliers after sigma clipping; using all candidates",
              file=sys.stderr)
        inlier = candidate.copy()

    coeffs = _ols(A_full[inlier], y[inlier])
    return coeffs, inlier, excluded


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(
    path: str,
    tbl: QTable,
    sin_cols: list[str],
    const_cols: list[str],
    sigma: float,
    exclude_ranges: list[list[float]],
    show: bool = False,
) -> None:
    rtp_deg = np.asarray(tbl["rtp"])

    cols = sin_cols + const_cols
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 4 * len(cols)), squeeze=False)

    for ax, col in zip(axes[:, 0], cols):
        y = np.asarray(tbl[col])
        unit = tbl[col].unit if hasattr(tbl[col], "unit") else ""
        use_const = col in const_cols

        coeffs, inlier, excluded = _fit_col(rtp_deg, y, use_const, sigma, exclude_ranges,
                                            label=f"{path}:{col}")
        sigma_outlier = ~inlier & ~excluded
        n_excl = excluded.sum()
        n_out = sigma_outlier.sum()

        rtp_rad = np.radians(rtp_deg)
        A_full = _design_const(rtp_rad) if use_const else _design_sin(rtp_rad)
        resid = y - _predict(A_full, coeffs)

        rtp_dense = np.linspace(rtp_deg.min(), rtp_deg.max(), 500)
        A_dense = _design_const(np.radians(rtp_dense)) if use_const else _design_sin(np.radians(rtp_dense))
        fit_curve = _predict(A_dense, coeffs)

        if use_const:
            fit_label = f"fit: {coeffs[0]:.4f} (constant)"
        else:
            a, c, d = coeffs
            b, phi = _amplitude_phase(c, d)
            fit_label = f"fit: {a:.2f} + {b:.2f}·sin(RTP + {phi:.1f}°)"

        ax.scatter(rtp_deg[inlier], y[inlier], s=20, color="steelblue", label="inlier", zorder=3)
        if n_out:
            ax.scatter(rtp_deg[sigma_outlier], y[sigma_outlier], s=40, color="tomato", marker="x",
                       linewidths=1.5, label=f"σ-outlier (n={n_out})", zorder=4)
        if n_excl:
            ax.scatter(rtp_deg[excluded], y[excluded], s=40, color="gray", marker="^",
                       linewidths=1.5, label=f"excluded (n={n_excl})", zorder=4)
        ax.plot(rtp_dense, fit_curve, color="darkorange", lw=1.5, label=fit_label)

        ax.set_xlabel("RTP [deg]")
        ax.set_ylabel(f"{col} [{unit}]" if unit else col)
        title = f"{col}  ({'constant' if use_const else 'sin'} fit, σ-clip {sigma}σ"
        if n_out:
            title += f", {n_out} σ-outlier{'s' if n_out != 1 else ''}"
        if n_excl:
            title += f", {n_excl} excluded"
        ax.set_title(title + ")")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        rms = np.std(resid[inlier])
        if use_const:
            print(f"{col}: mean={coeffs[0]:.4f}  rms_resid={rms:.4f}  "
                  f"n_sigma_outlier={n_out}  n_excluded={n_excl}")
        else:
            a, c, d = coeffs
            b, phi = _amplitude_phase(c, d)
            print(f"{col}: a={a:.4f}  b={b:.4f}  phi={phi:.2f}°  rms_resid={rms:.4f}  "
                  f"n_sigma_outlier={n_out}  n_excluded={n_excl}")

    fig.suptitle(path, fontsize=10)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Output ECSV
# ---------------------------------------------------------------------------

def _write_output(
    out_path: str,
    tbl: QTable,
    sin_cols: list[str],
    const_cols: list[str],
    sigma: float,
    exclude_ranges: list[list[float]],
    step_deg: float,
) -> None:
    rtp_deg = np.asarray(tbl["rtp"])
    rtp_grid = np.arange(-90.0, 90.0 + 0.5 * step_deg, step_deg)
    rtp_rad_grid = np.radians(rtp_grid)

    out_tbl = QTable()
    out_tbl["rtp"] = rtp_grid * u.deg

    # Fitted columns: evaluate smooth model on output grid
    fitted_set = set(sin_cols) | set(const_cols)
    for col in fitted_set:
        if col not in tbl.colnames:
            continue
        y = np.asarray(tbl[col])
        unit = tbl[col].unit if hasattr(tbl[col], "unit") else u.dimensionless_unscaled
        use_const = col in const_cols
        coeffs, _, _ = _fit_col(rtp_deg, y, use_const, sigma, exclude_ranges, label=col)
        A = _design_const(rtp_rad_grid) if use_const else _design_sin(rtp_rad_grid)
        out_tbl[col] = _predict(A, coeffs) * unit

    # Passthrough columns: piecewise-linear interpolation onto output grid
    for col in _dof_cols(tbl):
        if col in fitted_set:
            continue
        y = np.asarray(tbl[col])
        unit = tbl[col].unit if hasattr(tbl[col], "unit") else u.dimensionless_unscaled
        out_tbl[col] = np.interp(rtp_grid, rtp_deg, y) * unit
        print(f"{col}: passthrough (piecewise-linear interpolation)")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_tbl.write(out_path, format="ascii.ecsv", overwrite=True)
    print(f"Wrote {len(rtp_grid)}-row lookup table → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="ECSV file from focus_scan")
    parser.add_argument("--sin-cols", nargs="+", default=[],
                        help="DOF columns to fit with a sinusoid")
    parser.add_argument("--const-cols", nargs="+", default=[], metavar="COL",
                        help="Columns to fit with a constant instead of a sinusoid (e.g. Cam_dz)")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Sigma threshold for outlier rejection (default: 3)")
    parser.add_argument("--exclude-rtp", nargs=2, type=float, action="append",
                        default=[], metavar=("LOW", "HIGH"),
                        help="Exclude RTP range [LOW, HIGH] deg from fit; can be repeated")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Write smoothed RTPLookup ECSV to this path")
    parser.add_argument("--step", type=float, default=3.0, metavar="DEG",
                        help="RTP grid step size for output ECSV in degrees (default: 3)")
    parser.add_argument("--plot", action="store_true",
                        help="Show interactive plots (default: off)")
    args = parser.parse_args()

    tbl = QTable.read(args.input)

    sin_cols = []
    for c in args.sin_cols:
        if c in tbl.colnames:
            sin_cols.append(c)
        else:
            print(f"WARNING: --sin-cols column not found in table: {c}", file=sys.stderr)
    const_cols = []
    for c in args.const_cols:
        if c in tbl.colnames:
            const_cols.append(c)
        else:
            print(f"WARNING: --const-cols column not found in table: {c}", file=sys.stderr)

    if sin_cols or const_cols:
        _plot(args.input, tbl, sin_cols, const_cols, args.sigma, args.exclude_rtp, show=args.plot)

    if args.output:
        _write_output(args.output, tbl, sin_cols, const_cols, args.sigma,
                      args.exclude_rtp, args.step)


if __name__ == "__main__":
    main()
