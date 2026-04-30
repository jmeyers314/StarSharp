"""Check consistency of double-Zernike sensitivity FITS files in a directory."""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits


def dz_norm(arr):
    """Characteristic amplitude of a double-Zernike sensitivity: sqrt(sum(coefs[1:, 4:]^2))."""
    return np.sqrt(np.sum(np.square(arr[1:, 4:])))


def load_official() -> np.ndarray | None:
    """Load the official ts_ofc sensitivity matrix.

    Returns array of shape (ndof, kmax_dim, jmax_dim) or None if TS_OFC_DIR
    is not set.
    """
    ts_ofc = os.environ.get("TS_OFC_DIR")
    if not ts_ofc:
        return None
    yaml_path = (
        Path(ts_ofc)
        / "policy/sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml"
    )
    with open(yaml_path) as f:
        raw = np.array(yaml.safe_load(f))   # (kmax_dim=31, jmax_dim=29, ndof=50)
    return np.moveaxis(raw, -1, 0)           # → (ndof=50, kmax_dim=31, jmax_dim=29)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path(__file__).parent / "sensitivity",
        type=Path,
        help="Directory containing sensitivity FITS files (default: ./sensitivity)",
    )
    parser.add_argument(
        "--compare",
        metavar="STEM",
        default="3.14_i",
        help="File stem to compare against the official matrix (default: 3.14_i)",
    )
    parser.add_argument(
        "--plot-dir",
        metavar="DIR",
        default="check_sensitivity",
        help="Directory to write per-DOF comparison plots (requires --compare and TS_OFC_DIR)",
    )
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.fits"))
    if not files:
        raise SystemExit(f"No .fits files found in {args.directory}")

    arrays = {}
    dof_names = None
    dof_units_str = None
    unit = kmax = jmax = None

    for path in files:
        with fits.open(path) as hdul:
            arrays[path.stem] = hdul[0].data.copy()
            if dof_names is None:
                unit = hdul[0].header.get("BUNIT", "?")
                kmax = hdul[0].header["KMAX"]
                jmax = hdul[0].header["JMAX"]
                dof_names = [n.strip() for n in hdul[1].data["dof_name"]]
                dof_units_str = [u.strip() for u in hdul[1].data["dof_unit"]]

    keys = sorted(arrays)
    stack = np.array([arrays[k] for k in keys])  # (nfiles, ndof, kmax+1, jmax+1)
    mean = np.mean(stack, axis=0)                 # (ndof, kmax+1, jmax+1)

    print(f"unit={unit}, kmax={kmax}, jmax={jmax}, files={len(keys)}\n")
    header = f"{'DOF':<30}  {'mean_norm':>10}  {'mean_diff%':>10}  " + "  ".join(f"{k:>12}" for k in keys)
    print(header)
    for i, name in enumerate(dof_names):
        mean_norm = dz_norm(mean[i])
        diff_norms = [dz_norm(arrays[k][i] - mean[i])/mean_norm for k in keys]
        mean_diff = np.mean(diff_norms)
        print(f"{name:<30}  {mean_norm:>10.6f}  {mean_diff*100:>10.3f}  " + "  ".join(f"{d*100:>12.3f}" for d in diff_norms))

    # ------------------------------------------------------------------
    # Compare one file against the official ts_ofc sensitivity matrix
    # ------------------------------------------------------------------
    official = load_official()
    if official is None:
        print("\n(Skipping official comparison: TS_OFC_DIR not set)")
        return

    compare_key = args.compare
    if compare_key not in arrays:
        print(f"\n(Skipping official comparison: '{compare_key}' not found in {args.directory})")
        return

    ours = arrays[compare_key]                       # (ndof, kmax+1, jmax+1)
    k_overlap = min(ours.shape[1], official.shape[1])
    off_clipped = -official[:, :k_overlap, :]        # (ndof, k_overlap, jmax+1); flip sign convention
    our_clipped = ours[:, :k_overlap, :]

    print(f"\n--- Comparison: {compare_key} vs official (k_overlap={k_overlap}, "
          f"note: angular DOFs in deg here, arcsec in official) ---\n")
    print(f"{'DOF':<30}  {'our_unit':>8}  {'norm_ours':>10}  {'norm_off':>10}  {'diff%':>8}")
    for i, (name, dof_unit) in enumerate(zip(dof_names, dof_units_str)):
        norm_ours = dz_norm(our_clipped[i])
        norm_off  = dz_norm(off_clipped[i])
        diff_norm = dz_norm(our_clipped[i] - off_clipped[i])
        if norm_ours > 0:
            diff_pct = diff_norm / norm_ours * 100
        else:
            diff_pct = float("nan")
        print(f"{name:<30}  {dof_unit:>8}  {norm_ours:>10.6f}  {norm_off:>10.6f}  {diff_pct:>8.2f}")

    if args.plot_dir is None:
        return

    import matplotlib.pyplot as plt

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_k, n_j = k_overlap, our_clipped.shape[2]
    # Plot range: k=1..15, j=4..28 (inclusive); origin=lower so k=1 at bottom
    extent = [3.5, 28.5, 0.5, 15.5]

    for i, (name, dof_unit) in enumerate(zip(dof_names, dof_units_str)):
        raw_ours = our_clipped[i, 1:16, 4:29]   # k=1..15, j=4..28
        raw_off  = off_clipped[i, 1:16, 4:29]

        # Normalise by dz_norm of ours (full matrix, not the plot slice)
        norm = dz_norm(our_clipped[i])
        if norm == 0:
            norm = 1.0
        mat_ours = raw_ours / norm
        mat_off  = raw_off  / norm
        diff     = mat_ours - mat_off
        diff_norm_val = dz_norm(our_clipped[i] - off_clipped[i]) / norm

        fig, axes = plt.subplots(1, 4, figsize=(19, 4), constrained_layout=True)
        fig.suptitle(
            f"{name}  ({dof_unit})  —  {compare_key} vs official"
            f"  (normalised by dz_norm;  diff dz_norm = {diff_norm_val:.3f})",
            fontsize=11,
        )

        kw = dict(extent=extent, aspect="auto", origin="lower", cmap="RdBu_r")

        im0 = axes[0].imshow(mat_ours, vmin=-1, vmax=1, **kw)
        axes[0].set_title(compare_key)

        im1 = axes[1].imshow(mat_off, vmin=-1, vmax=1, **kw)
        axes[1].set_title("official")

        im2 = axes[2].imshow(diff, vmin=-0.1, vmax=0.1, **kw)
        axes[2].set_title(f"{compare_key} − official")

        for ax, im in zip(axes[:3], [im0, im1, im2]):
            ax.set_xlabel("j  (field Zernike)")
            ax.set_ylabel("k  (pupil Zernike)")
            fig.colorbar(im, ax=ax, shrink=0.8, label="normalised")

        # 4th panel: text table of top-10 DZ coefficients by |official|
        n_rows, n_cols = mat_off.shape   # (15, 25): k=1..15, j=4..28
        flat_off  = mat_off.ravel()
        flat_ours = mat_ours.ravel()
        order = np.argsort(np.abs(flat_off))[::-1][:10]
        top_off   = flat_off[order]
        top_ours  = flat_ours[order]
        ks = order // n_cols + 1          # slice row → actual k
        js = order %  n_cols + 4          # slice col → actual j
        cum_rss = np.sqrt(np.cumsum(top_off ** 2))

        header = f"{'k':>2} {'j':>2}  {'cum_rss':>7}  {'official':>8}  {'ours':>8}  {'diff':>8}"
        rows = [header]
        for n in range(10):
            rows.append(
                f"{ks[n]:2d} {js[n]:2d}  {cum_rss[n]:7.4f}  {top_off[n]:+8.4f}"
                f"  {top_ours[n]:+8.4f}  {top_ours[n] - top_off[n]:+8.4f}"
            )
        axes[3].axis("off")
        axes[3].set_title("top 10 |official| DZs")
        axes[3].text(
            0.05, 0.95, "\n".join(rows),
            transform=axes[3].transAxes,
            va="top", ha="left",
            fontfamily="monospace", fontsize=8,
        )

        safe_name = name.replace("/", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=150)
        plt.close(fig)

    print(f"\nPlots written to {plot_dir}/")


if __name__ == "__main__":
    main()
