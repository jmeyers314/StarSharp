"""Construct a default pointing model by finite-differencing chief-ray shifts.

This script keeps the full pointing-model construction scaffold in place while
using the current fiducial `RaytracedOpticalModel` construction path.
"""

from __future__ import annotations

import argparse

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from StarSharp import (
    PointingModel,
    RaytracedOpticalModel,
    StateFactory,
)
from StarSharp.models.fiducial import default_raytraced_model


ARCSEC_PER_MICRON_FP = 0.02


def _spot_fp_displacements(
    spots_ref,
    spots_pert,
) -> np.ndarray:
    """Return per-field spot-centroid displacements in microns.

    Parameters
    ----------
    spots_ref, spots_pert : Spots
        Reference and perturbed spot diagrams with matching shape.

    Returns
    -------
    ndarray
        Array with shape ``(n_valid_fields, 2)`` containing per-field
        spot-centroid focal-plane displacements ``[dx_um, dy_um]``.
    """
    # Spots.dx/dy are relative to per-field centering references. Recover
    # absolute focal-plane ray positions by adding the field centers.
    x_ref_um = (
        spots_ref.ccs.dx.to_value(u.um)
        + spots_ref.field.ccs.x.to_value(u.um)[..., np.newaxis]
    )
    y_ref_um = (
        spots_ref.ccs.dy.to_value(u.um)
        + spots_ref.field.ccs.y.to_value(u.um)[..., np.newaxis]
    )
    x_pert_um = (
        spots_pert.ccs.dx.to_value(u.um)
        + spots_pert.field.ccs.x.to_value(u.um)[..., np.newaxis]
    )
    y_pert_um = (
        spots_pert.ccs.dy.to_value(u.um)
        + spots_pert.field.ccs.y.to_value(u.um)[..., np.newaxis]
    )

    # Use spots that are unvignetted in both reference and perturbed traces.
    valid = (~spots_ref.vignetted) & (~spots_pert.vignetted)
    ddx_um = x_pert_um - x_ref_um
    ddy_um = y_pert_um - y_ref_um

    disp_rows: list[np.ndarray] = []
    for ifield in range(spots_ref.nfield):
        mask = valid[ifield]
        if not np.any(mask):
            continue
        mean_dx = float(np.mean(ddx_um[ifield, mask]))
        mean_dy = float(np.mean(ddy_um[ifield, mask]))
        disp_rows.append(np.array([mean_dx, mean_dy], dtype=float))

    if not disp_rows:
        return np.empty((0, 2), dtype=float)
    return np.vstack(disp_rows)


def construct_pointing_model(
    band: str = "r",
    rtp: Angle = Angle("0 deg"),
) -> PointingModel:
    """Construct a default 2x50 pointing model matrix.

    The algorithm is:

    1. Trace reference spot diagrams at zero state.
    2. Perturb one DOF at a time using finite-difference step sizes.
    3. Re-trace spots and compute spot-centroid displacement per field.
     4. Compute mean/std centroid displacement across all valid fields.
     5. Divide by step size to get ``d(fp_shift)/d(state)``.

    Parameters
    ----------
    band : str, optional
        LSST band used to load the fiducial batoid optic yaml.
    rtp : Angle, optional
        Rotator position angle.
    Returns
    -------
    PointingModel
        Pointing model matrix in arcsec.

    Notes
    -----
    This is an iterative-development scaffold. It estimates per-DOF focal-plane
    motion by finite-differencing spot-diagram shifts. For convenience, the
    returned ``PointingModel`` is derived from these focal-plane shifts using
    the approximation ``0.02 arcsec / micron`` with aligned x/y axes.
    """
    model = default_raytraced_model(band=band, rtp=rtp)
    factory = StateFactory(model.state_schema)
    nominal_state = factory.zero("f")

    fc = model.make_ccd_field(detnums=list(range(90, 99)))

    print("Computing zero-state reference spots...")

    spots0 = model.spots(fc, state=nominal_state)
    steps = model.state_schema.step

    fp_matrix = np.zeros((2, model.state_schema.n_dof), dtype=float)
    fp_std_matrix = np.zeros((2, model.state_schema.n_dof), dtype=float)
    fp_resid_rms_matrix = np.zeros((2, model.state_schema.n_dof), dtype=float)
    fp_resid_max_matrix = np.zeros((2, model.state_schema.n_dof), dtype=float)

    print("Perturbing DOFs to build pointing matrix...")
    print("Reporting in focal-plane microns per state unit (um/um or um/deg)")

    for i in range(model.state_schema.n_dof):
        dof_name = model.state_schema.dof_names[i]
        dof_unit = model.state_schema.dof_units[i]
        unit_label = f"um/{dof_unit.to_string()}"
        step = steps[i]
        print(f"  DOF {i:02d} {dof_name:10s} step={step:g}")

        s = nominal_state.f.value.copy()
        s[i] += step
        state_i = factory.f(s)

        spots1 = model.spots(fc, state=state_i)
        dfp_spot = _spot_fp_displacements(spots0, spots1)
        dfp_mean = -np.nanmean(dfp_spot, axis=0)
        dfp_std = np.nanstd(dfp_spot, axis=0)
        fp_matrix[:, i] = dfp_mean / step
        fp_std_matrix[:, i] = dfp_std / step

        # Residuals of the fitted constant correction against each spot sample.
        target = -dfp_spot
        resid = target - dfp_mean[np.newaxis, :]
        fp_resid_rms_matrix[:, i] = np.sqrt(np.nanmean(resid**2, axis=0)) / step
        fp_resid_max_matrix[:, i] = np.nanmax(np.abs(resid), axis=0) / step

        # Per-unit-DOF coefficients (independent of finite-difference step size).
        mx = fp_matrix[0, i]
        my = fp_matrix[1, i]
        sx = fp_std_matrix[0, i]
        sy = fp_std_matrix[1, i]
        print(
            f"    mean({unit_label}): dx={mx:+.6g}, dy={my:+.6g}; "
            f"std({unit_label}): sx={sx:.3g}, sy={sy:.3g}"
        )

    print(
        "\nPer-DOF pointing fit summary "
        "(all values in focal-plane microns per state unit):"
    )
    num_w = 13
    header = (
        f"{'idx':>3}  {'dof_name':<12} {'unit':<9} "
        f"{'fp_dx_fit':>{num_w}} {'fp_dy_fit':>{num_w}} "
        f"{'sigma_x':>{num_w}} {'sigma_y':>{num_w}} "
        f"{'rms_res_x':>{num_w}} {'rms_res_y':>{num_w}} "
        f"{'max_res_x':>{num_w}} {'max_res_y':>{num_w}}"
    )
    print(header)
    print("-" * len(header))
    for i, dof_name in enumerate(model.state_schema.dof_names):
        dof_unit = model.state_schema.dof_units[i]
        unit_label = f"um/{dof_unit.to_string()}"
        fit_x = fp_matrix[0, i]
        fit_y = fp_matrix[1, i]
        sig_x = fp_std_matrix[0, i]
        sig_y = fp_std_matrix[1, i]
        rms_x = fp_resid_rms_matrix[0, i]
        rms_y = fp_resid_rms_matrix[1, i]
        max_x = fp_resid_max_matrix[0, i]
        max_y = fp_resid_max_matrix[1, i]
        print(
            f"{i:>3d}  {dof_name:<12} {unit_label:<9} "
            f"{fit_x:>{num_w}.6g} {fit_y:>{num_w}.6g} "
            f"{sig_x:>{num_w}.6g} {sig_y:>{num_w}.6g} "
            f"{rms_x:>{num_w}.6g} {rms_y:>{num_w}.6g} "
            f"{max_x:>{num_w}.6g} {max_y:>{num_w}.6g}"
        )

    # Convert from fp-micron sensitivity to pointing sensitivity using the
    # temporary approximation 0.02 arcsec per micron and aligned x/y axes.
    matrix_arcsec = fp_matrix * ARCSEC_PER_MICRON_FP * u.arcsec
    return PointingModel(schema=model.state_schema, matrix=matrix_arcsec)


def _validate_pointing_model(
    model: RaytracedOpticalModel,
    pm: PointingModel,
    *,
    n_mixed: int = 8,
    seed: int = 12345,
) -> None:
    """Report residual spot-centroid motion with derived pointing model applied.

    Runs per-DOF single perturbations and random linear combinations.
    """
    # fc = model.make_ccd_field(detnums=list(range(90, 99)))
    fc = model.make_ccd_field()
    factory = StateFactory(model.state_schema)
    steps = model.state_schema.step

    ref_spots = model.spots(fc, state=factory.zero("f"))
    corrected_model = RaytracedOpticalModel(
        builder=model.builder,
        rtp=model.rtp,
        wavelength=model.wavelength,
        state_schema=model.state_schema,
        camera=model.camera,
        offset=model.offset,
        pointing_model=pm,
    )

    def _run_case(name: str, state_vec: np.ndarray) -> tuple[float, float, float]:
        state = factory.f(state_vec)
        pert = corrected_model.spots(fc, state=state)
        disp = _spot_fp_displacements(ref_spots, pert)
        if disp.size == 0:
            return np.nan, np.nan, np.nan
        mag = np.hypot(disp[:, 0], disp[:, 1])
        return float(np.mean(mag)), float(np.std(mag)), float(np.max(mag))

    print("\nValidation: residual spot-centroid motion with derived PointingModel applied")
    print("Single-DOF states:")
    hdr = f"{'idx':>3}  {'dof_name':<12} {'mean_um':>12} {'std_um':>12} {'max_um':>12}"
    print(hdr)
    print("-" * len(hdr))
    for i, dof_name in enumerate(model.state_schema.dof_names):
        s = np.zeros(model.state_schema.n_dof, dtype=float)
        s[i] = steps[i]
        mean_um, std_um, max_um = _run_case(dof_name, s)
        print(f"{i:>3d}  {dof_name:<12} {mean_um:12.5g} {std_um:12.5g} {max_um:12.5g}")

    print("\nMixed linear-combination states:")
    mhdr = f"{'case':>4}  {'mean_um':>12} {'std_um':>12} {'max_um':>12}"
    print(mhdr)
    print("-" * len(mhdr))
    rng = np.random.default_rng(seed)
    for icase in range(n_mixed):
        coeff = rng.uniform(-1.0, 1.0, size=model.state_schema.n_dof)
        s = coeff * steps
        mean_um, std_um, max_um = _run_case(f"mix{icase}", s)
        print(f"{icase:>4d}  {mean_um:12.5g} {std_um:12.5g} {max_um:12.5g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Construct default pointing model via spot finite differences."
        )
    )
    parser.add_argument(
        "--band",
        type=str,
        default="r",
        help="LSST band used to load LSST_<band>.yaml (default: r)",
    )
    parser.add_argument(
        "--rtp",
        type=str,
        default="0 deg",
        help="Rotator angle parsable by astropy Angle (default: '0 deg')",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default=None,
        help="Optional filename to write pm.to_table() output (e.g. pointing_model.ecsv)",
    )
    args = parser.parse_args()

    pm = construct_pointing_model(
        band=args.band,
        rtp=Angle(args.rtp),
    )
    model = default_raytraced_model(band=args.band, rtp=Angle(args.rtp))
    _validate_pointing_model(model, pm)

    if args.output_table is not None:
        table = pm.to_table()
        table.write(args.output_table, overwrite=True)
        print(f"\nWrote pointing model table to {args.output_table}")

    print(pm)

