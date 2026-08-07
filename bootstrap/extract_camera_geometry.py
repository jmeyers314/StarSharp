#!/usr/bin/env python
"""Extract LSST camera geometry and write to StarSharp/data/camera/.

Produces two files that let StarSharp operate without the lsst stack:

  StarSharp/data/camera/radial_transform.yaml
      Coefficients of a 1-D odd polynomial  r_fp = sum_k a_k * r^(2k+1)
      (r in radians, r_fp in mm) for field-angle → focal-plane conversion,
      plus the inverse polynomial r = sum_k b_k * r_fp^(2k+1).

  StarSharp/data/camera/detectors.ecsv
      Per-detector table with id, name, physical_type, and bounding-box
      corners in the CCS frame (mm).  Columns ccs_x_{min,max} and
      ccs_y_{min,max} match the convention used in field_coords.py /
      raytraced.py (CCS x = FOCAL_PLANE[1], CCS y = FOCAL_PLANE[0]).
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from astropy.table import QTable
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE
from lsst.obs.lsst import LsstCam

# ---------------------------------------------------------------------------
# Load camera
# ---------------------------------------------------------------------------
camera = LsstCam().getCamera()
fwd_mapping = camera.getTransform(FIELD_ANGLE, FOCAL_PLANE).getMapping()
inv_mapping = camera.getTransform(FOCAL_PLANE, FIELD_ANGLE).getMapping()

# ---------------------------------------------------------------------------
# Check azimuthal symmetry
# ---------------------------------------------------------------------------
r_check = np.deg2rad(1.5)  # a representative mid-field radius
n_az = 360
angles = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
thx_az = r_check * np.cos(angles)
thy_az = r_check * np.sin(angles)
fp_az = fwd_mapping.applyForward(np.array([thx_az, thy_az]))
r_fp_az = np.hypot(fp_az[0], fp_az[1])
asym_mm = r_fp_az.max() - r_fp_az.min()
print(f"Azimuthal r_fp variation at 1.5 deg field: {asym_mm:.4f} mm")
if asym_mm > 0.01:
    print("  WARNING: significant azimuthal asymmetry — polynomial approximation may be lossy")

# ---------------------------------------------------------------------------
# Fit forward radial polynomial: r_fp = sum_k a_k * r^(2k+1)
# ---------------------------------------------------------------------------
r_max = np.deg2rad(2.15)
r_sample = np.linspace(0, r_max, 2000)

# Evaluate along x-axis (thy = 0); result is aligned with x, so r_fp = fpx
fp_fwd = fwd_mapping.applyForward(np.array([r_sample, np.zeros_like(r_sample)]))
r_fp_sample = fp_fwd[0]  # mm

# Determine degree by checking residuals
for degree in range(1, 6):
    powers = 2 * np.arange(degree + 1) + 1  # 1, 3, 5, ..., 2*degree+1
    A = np.column_stack([r_sample ** p for p in powers])
    coeffs_fwd, *_ = np.linalg.lstsq(A, r_fp_sample, rcond=None)
    resid_max = np.max(np.abs(A @ coeffs_fwd - r_fp_sample))
    print(f"Forward poly degree {degree} (r^{powers[-1]}): max residual = {resid_max*1e3:.3f} μm")
    if resid_max < 1e-4:  # 0.1 μm is plenty
        break

print(f"Using degree {degree} forward polynomial (powers {powers.tolist()})")

# ---------------------------------------------------------------------------
# Fit inverse radial polynomial: r = sum_k b_k * r_fp^(2k+1)
# ---------------------------------------------------------------------------
r_fp_max = r_fp_sample[-1]
r_fp_inv = np.linspace(0, r_fp_max, 2000)

fp_inv = inv_mapping.applyForward(np.array([r_fp_inv, np.zeros_like(r_fp_inv)]))
r_inv_sample = fp_inv[0]  # radians

for degree_inv in range(1, 6):
    powers_inv = 2 * np.arange(degree_inv + 1) + 1
    A_inv = np.column_stack([r_fp_inv ** p for p in powers_inv])
    coeffs_inv, *_ = np.linalg.lstsq(A_inv, r_inv_sample, rcond=None)
    resid_max_inv = np.max(np.abs(A_inv @ coeffs_inv - r_inv_sample))
    resid_arcsec = np.rad2deg(resid_max_inv) * 3600
    print(f"Inverse poly degree {degree_inv} (r_fp^{powers_inv[-1]}): max residual = {resid_arcsec:.4f} arcsec")
    if resid_arcsec < 1e-4:  # 0.1 mas
        break

print(f"Using degree {degree_inv} inverse polynomial (powers {powers_inv.tolist()})")

# ---------------------------------------------------------------------------
# Round-trip check
# ---------------------------------------------------------------------------
r_rt = np.deg2rad(np.linspace(0.1, 2.0, 100))
A_rt = np.column_stack([r_rt ** p for p in powers])
r_fp_rt = A_rt @ coeffs_fwd
A_rt_inv = np.column_stack([r_fp_rt ** p for p in powers_inv])
r_back = A_rt_inv @ coeffs_inv
rt_err_arcsec = np.max(np.abs(np.rad2deg(r_back - r_rt))) * 3600
print(f"Round-trip error: {rt_err_arcsec:.4f} arcsec")

# ---------------------------------------------------------------------------
# Write polynomial YAML
# ---------------------------------------------------------------------------
out_dir = Path("StarSharp/data/camera")
out_dir.mkdir(parents=True, exist_ok=True)

poly_data = {
    "description": (
        "LSST camera radial polynomial transforms extracted from LsstCam. "
        "r_fp = sum_k a_k * r^powers[k]  (r in radians, r_fp in mm). "
        "The transform is applied per-component as fpx = r_fp/r * thx, etc."
    ),
    "forward": {
        "r_units": "radian",
        "r_fp_units": "mm",
        "powers": [int(p) for p in powers.tolist()],
        "coefficients": coeffs_fwd.tolist(),
    },
    "inverse": {
        "r_fp_units": "mm",
        "r_units": "radian",
        "powers": [int(p) for p in powers_inv.tolist()],
        "coefficients": coeffs_inv.tolist(),
    },
}

poly_path = out_dir / "radial_transform.yaml"
with open(poly_path, "w") as f:
    yaml.dump(poly_data, f, default_flow_style=False, sort_keys=False)
print(f"\nWrote {poly_path}")

# ---------------------------------------------------------------------------
# Extract detector table
# ---------------------------------------------------------------------------
# FOCAL_PLANE corners are Point2D objects; in lsst, corner[0] is the
# FOCAL_PLANE x-axis which equals DVCS x = CCS y, and corner[1] is
# FOCAL_PLANE y-axis which equals DVCS y = CCS x.  The code in
# field_coords.py and raytraced.py uses c[1] as CCS x and c[0] as CCS y.
rows = []
for det in camera:
    corners = det.getCorners(FOCAL_PLANE)
    ccs_xs = [c[1] for c in corners]   # FOCAL_PLANE[1] = CCS x
    ccs_ys = [c[0] for c in corners]   # FOCAL_PLANE[0] = CCS y
    rows.append({
        "id": det.getId(),
        "name": det.getName(),
        "physical_type": det.getPhysicalType(),
        "ccs_x_min": min(ccs_xs),
        "ccs_x_max": max(ccs_xs),
        "ccs_y_min": min(ccs_ys),
        "ccs_y_max": max(ccs_ys),
    })

table = QTable(rows=rows)
for col in ("ccs_x_min", "ccs_x_max", "ccs_y_min", "ccs_y_max"):
    table[col].unit = u.mm

det_path = out_dir / "detectors.ecsv"
table.write(det_path, format="ascii.ecsv", overwrite=True)
print(f"Wrote {det_path} ({len(table)} detectors)")
print(f"Physical types: {sorted(set(table['physical_type']))}")
