"""
Demo: solving wavefront sensor Zernikes for LSST-like WFS field positions.

Full workflow:

  1.  Build (or load) a DoubleZernikes gradient matrix.
  2.  Parse active DOFs and slice to x-basis Sensitivity.
  3.  Project DZ to the 8 WFS field positions to get a Sensitivity[Zernikes].
  4.  Use StateFactory to compute the SVD → Vh.
  5.  Attach Vh to the DZ sensitivity and convert to v-basis.
  6.  Build ZernikeSolver and solve.

Run with:
    python -m sandbox.wfs_zernike_demo

If TS_CONFIG_MTTCS_DIR is set the script loads the real MTAOS matrix;
otherwise it falls back to a synthetic matrix for standalone testing.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from StarSharp import (
    DoubleZernikes,
    FieldCoords,
    Sensitivity,
    State,
    StateFactory,
    Zernikes,
)
from StarSharp.solver import ZernikeSolver

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
JMAX = 28          # Noll indices 0 … 28
KMAX = 30          # field polynomial order for DZ
USE_DOF = "0-9,10-16,30-34"   # active M2/camera hexapod + M1M3 bending
NKEEP = 12         # retain 12 SVD modes
JMIN = 4           # drop piston (j=0), tip (j=1), tilt (j=2), focus (j=3)
N_DOF_FULL = 50    # total number of DOFs in the design matrix

PUPIL_OUTER = 4.18 * u.m
PUPIL_INNER = 4.18 * 0.612 * u.m
FIELD_OUTER = 1.75 * u.deg
FIELD_INNER = 0.0 * u.deg

# Approximate WFS positions (OCS frame, degrees)
# 4 corners + 4 edges, ~1.8° from field centre
RTP = Angle("20 deg")
WFS_X = np.array([ 1.176,  1.176, -1.176, -1.176,  0.0,   1.664,   0.0,  -1.664]) * u.deg
WFS_Y = np.array([ 1.176, -1.176,  1.176, -1.176,  1.664,  0.0,  -1.664,   0.0]) * u.deg
WFS_FIELD = FieldCoords(x=WFS_X, y=WFS_Y, frame="ocs", rtp=RTP)

# ─────────────────────────────────────────────────────────────────
# Load or synthesise gradient coefficients
# ─────────────────────────────────────────────────────────────────
def load_or_build_grad_coefs() -> tuple[np.ndarray, np.ndarray]:
    """Return (grad_coefs, norm) where grad_coefs has shape (N_DOF_FULL, kmax+1, jmax+1)."""
    ts_dir = os.environ.get("TS_CONFIG_MTTCS_DIR")
    if ts_dir:
        import yaml

        ofc_dir = Path(ts_dir) / "MTAOS/v13/ofc"
        sens_path = (
            ofc_dir / "sensitivity_matrix"
            / f"lsst_sensitivity_dz_{KMAX+1}_{JMAX+1}_{N_DOF_FULL}.yaml"
        )
        norm_path = ofc_dir / "normalization_weights" / "range-fwhm.yaml"

        with open(sens_path) as f:
            raw = np.array(yaml.safe_load(f))   # (kmax+1, jmax+1, ndof)
        with open(norm_path) as f:
            norm = np.array(yaml.safe_load(f))  # (ndof,)

        # raw shape: (kmax+1, jmax+1, ndof) → (ndof, kmax+1, jmax+1)
        grad_coefs = raw.transpose(2, 0, 1)
    else:
        print("TS_CONFIG_MTTCS_DIR not set — using synthetic sensitivity.")
        rng = np.random.default_rng(0)
        grad_coefs = rng.standard_normal((N_DOF_FULL, KMAX + 1, JMAX + 1))
        norm = np.ones(N_DOF_FULL)

    return grad_coefs, norm


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    grad_coefs, norm = load_or_build_grad_coefs()
    sf = StateFactory(
        grad_coefs.reshape(50, -1).T,
        norm=norm,
        use_dof=USE_DOF,
        nkeep=NKEEP,
    )
    gradient = DoubleZernikes(
        coefs=grad_coefs << u.um,
        field_outer=FIELD_OUTER,
        field_inner=FIELD_INNER,
        pupil_outer=PUPIL_OUTER,
        pupil_inner=PUPIL_INNER,
        frame="ocs",
        rtp=RTP,
    )
    sens = Sensitivity(
        gradient=gradient,
        basis="f",
        use_dof=sf.use_dof,
        n_dof=sf.n_dof,
        Vh=sf.Vh,
    )

    solver = ZernikeSolver(sens.v, jmin=JMIN)

    print(f"n_dof={sf.n_dof}, "
          f"n_active={len(sf.use_dof)}, "
          f"nkeep={sf.nkeep}")
    print(f"Singular values (first 5): {sf.S[:5].round(4)}")

    # ── Simulate a perturbation ──────────────────────────────────
    # Perturb along the first SVD mode (v-basis coefficient = 0.3)
    true_v = np.zeros(NKEEP)
    true_v[4] = 0.3
    true_state = sf.from_v(true_v)
    true_dzs = sens.predict(true_state)
    true_zks = true_dzs.single(WFS_FIELD)

    rng = np.random.default_rng(57721)
    # noise = np.zeros_like(true_zks.coefs.value) * u.um
    # noise[:, 4] += rng.normal(0.0, 0.1, size=8) * u.um
    noise = rng.normal(0.0, 0.1, size=true_zks.coefs.value.shape) * u.um
    observed = replace(
        true_zks,
        coefs=true_zks.coefs + noise
    )

    # ── Recover ─────────────────────────────────────────────────
    recovered = solver.solve(observed, mode="deviation")   # State in v-basis
    print(" mode      true       fit     resid")
    for i in range(len(recovered.value)):
        print(f"v[{i:2d}] {true_state.value[i]:9.5f} {recovered.value[i]:9.5f} {(true_state.value[i] - recovered.value[i]):9.5f}")

    print()
    print(" mode      true       fit     resid")
    for i in range(len(recovered.x.value)):
        print(f"x[{i:2d}] {true_state.x.value[i]:9.5f} {recovered.x.value[i]:9.5f} {(true_state.x.value[i] - recovered.x.value[i]):9.5f}")

    # What is the residual wavefront error in microns?
    recovered_dzs = sens.predict(recovered)
    recovered_zks = recovered_dzs.single(WFS_FIELD)
    residual_zks = replace(observed, coefs=observed.coefs - recovered_zks.coefs)
    residual_rms = np.mean(np.sqrt(np.sum(residual_zks.coefs[:, 4:]**2, axis=1)))
    print(f"\nResidual RMS wavefront error: {residual_rms:.3f} microns")


if __name__ == "__main__":
    main()

