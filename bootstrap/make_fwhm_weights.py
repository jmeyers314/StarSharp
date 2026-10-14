#!/usr/bin/env python
"""Compute FWHM normalization weights by raytracing and install them into StarSharp/data/.

For each degree of freedom, the weight is the field-averaged quadrature
contribution to the PSF FWHM per unit DOF perturbation, in arcsec:

    w_i = sqrt(sum_k fieldweight_k * dFWHM_k^2) / s_i

where ``dFWHM_k`` is measured at field point ``k`` from the *difference of ray
displacements* between a perturbed and a nominal raytrace.

Why difference the rays rather than the spot widths
---------------------------------------------------
The obvious approach -- subtracting second moments, ``T(perturbed) - T(nominal)``
-- is wrong.  With a nonzero nominal wavefront ``W0``,

    T(d) = T0 + 2<W0', dW'> + |dW'|^2

so the difference retains a term *linear* in the perturbation ``d``, and
``sqrt(dT)/d`` diverges as ``d -> 0``.  Empirically that yields weights varying
by ~70% over a 16x range of step size.

The ray displacements themselves are linear in the perturbation amplitude, so
differencing *those* and then taking the second moment gives a step-independent
answer (verified to 7-8 significant figures over a 16x step range).

This is deliberately *not* the ts_ofc method.  ts_ofc evaluates its linearized
double-Zernike sensitivity and converts with
``lsst.ts.wep.utils.convertZernikesToPsfWidth``, which applies a linear
Zernike->FWHM gradient that its own docstring warns breaks down past ~0.20
arcsec RSS and "tends to overestimate the PSF degradation".  Raytracing measures
the real spot.  The two agree to ~12% in the median; see
``tests/test_fiducial.py::test_normalizations_match_ts_ofc_v13``.

Usage
-----
python make_fwhm_weights.py [--rings 5] [--spokes 6] [--field-radius 1.75]
                            [--nrad 10] [--step-scale 1.0] [--output PATH]
                            [--no-install]
"""
import argparse
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from galsim.zernike import gq_annulus_points

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "StarSharp" / "data" / "normalization_weights"
OUT_DIR = HERE / "normalization_weights"

# rms -> FWHM for a Gaussian.
FWHM_PER_RMS = 2.0 * np.sqrt(2.0 * np.log(2.0))


def make_quadrature_grid(
    rings: int, spokes: int, field_radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, weights)`` for a Gauss-Legendre grid on a circular field.

    Thin wrapper over :func:`galsim.zernike.gq_annulus_points` that normalizes
    the weights to sum to 1, so the field average is a weighted mean rather
    than an integral over the field area.  ts_ofc's
    ``generate_normalization_weights.py`` rolls its own equivalent grid; this
    produces the same points and (normalized) weights, so field-averaging is
    directly comparable.

    Parameters
    ----------
    rings : int
        Number of radial quadrature rings.
    spokes : int
        Number of azimuthal spokes.
    field_radius : float
        Field radius in degrees.
    """
    x, y, weights = gq_annulus_points(field_radius, 0.0, rings, spokes)
    return x, y, weights / weights.sum()


def compute_fwhm_weights(
    rings: int = 5,
    spokes: int = 6,
    field_radius: float = 1.75,
    nrad: int = 10,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Raytrace each DOF and return the FWHM weights, length ``n_dof``.

    Perturbation steps are ``default_schema().step`` times ``step_scale``.  The
    result is step-independent (see the module docstring), so ``step_scale``
    only trades raytrace dynamic range against nothing much.

    Returns
    -------
    weights : ndarray
        Field-averaged quadrature FWHM contribution per unit DOF, in arcsec
        per (um or deg) depending on the DOF.
    """
    from StarSharp.datatypes.field_coords import FieldCoords
    from StarSharp.datatypes.state import StateFactory
    from StarSharp.models.fiducial import default_raytraced_model, default_schema

    model = default_raytraced_model()
    schema = default_schema()
    factory = StateFactory(schema)

    fx, fy, field_weights = make_quadrature_grid(rings, spokes, field_radius)
    field = FieldCoords(
        x=fx * u.deg, y=fy * u.deg, frame="ocs", rtp=Angle(0.0 * u.deg)
    )

    def trace(state):
        # include_chip_heights=False: quadrature points landing in raft/chip
        # gaps have no detector to intersect, so every ray gets flagged
        # vignetted and the per-field mean becomes NaN.  Chip heights are
        # irrelevant to a field-averaged blur metric anyway.
        spots = model.spots(
            field, state=state, nrad=nrad, include_chip_heights=False
        ).angle
        return (
            spots.dx.to_value(u.arcsec),
            spots.dy.to_value(u.arcsec),
            spots.vignetted,
        )

    dx0, dy0, vig0 = trace(None)

    steps = np.asarray(schema.step, dtype=float) * step_scale
    weights = np.zeros(schema.n_dof, dtype=float)

    for idof in range(schema.n_dof):
        step = steps[idof]
        values = np.zeros(schema.n_dof)
        values[idof] = step
        dx1, dy1, vig1 = trace(factory.f(values))

        # Keep only rays that land in both traces.
        good = ~(vig0 | vig1)
        if not good.any(axis=-1).all():
            dead = np.flatnonzero(~good.any(axis=-1))
            raise RuntimeError(
                f"DOF {schema.dof_names[idof]}: field points {dead.tolist()} have "
                "no unvignetted rays; adjust the quadrature grid"
            )

        ddx = np.where(good, dx1 - dx0, np.nan)
        ddy = np.where(good, dy1 - dy0, np.nan)

        # Remove the centroid shift: it is a pointing change, not a blur.
        ddx = ddx - np.nanmean(ddx, axis=-1, keepdims=True)
        ddy = ddy - np.nanmean(ddy, axis=-1, keepdims=True)

        # Per-field variance summed over both axes, then per-axis rms -> FWHM.
        var = np.nanmean(ddx**2 + ddy**2, axis=-1)
        dfwhm = FWHM_PER_RMS * np.sqrt(var / 2.0)

        weights[idof] = np.sqrt(np.sum(field_weights * dfwhm**2)) / step

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        bad = [
            schema.dof_names[i]
            for i in np.flatnonzero(~np.isfinite(weights) | (weights <= 0))
        ]
        raise RuntimeError(f"Non-finite or non-positive weights for: {bad}")

    return weights


def write_yaml(
    weights: np.ndarray,
    path: Path,
    *,
    rings: int,
    spokes: int,
    field_radius: float,
    nrad: int,
    step_scale: float = 1.0,
) -> None:
    """Write *weights* as a YAML list with a provenance header."""
    from StarSharp.models.fiducial import default_schema

    schema = default_schema()
    path.parent.mkdir(parents=True, exist_ok=True)

    # The perturbation steps are just default_schema().step, so point at it
    # rather than listing 50 numbers.  The result is step-independent anyway
    # (see the method note above), so the values do not pin the output.
    step_note = (
        "" if step_scale == 1.0 else f" scaled by {step_scale}"
    )

    lines = [
        "---",
        "# FWHM-sensitivity normalization weights for the 50 AOS degrees of freedom.",
        "#",
        "# Each entry is the field-averaged quadrature contribution to the PSF FWHM",
        "# per unit DOF perturbation, in arcsec per um (translations, bending modes)",
        "# or arcsec per degree (rotations).  Larger means the DOF matters more for",
        "# image quality.",
        "#",
        "# PROVENANCE -- generated by bootstrap/make_fwhm_weights.py.  Measured by",
        "# raytracing the fiducial optical model and taking second moments of the",
        "# *difference* of ray displacements between perturbed and nominal traces.",
        "# Differencing rays (linear in perturbation amplitude) rather than spot",
        "# widths (quadratic) is what makes the result step-size independent.",
        "#",
        "# This is NOT ts_ofc's method: ts_ofc evaluates its linearized",
        "# double-Zernike sensitivity and converts via convertZernikesToPsfWidth,",
        "# which linearizes a second time.  The two agree to ~12% in the median.",
        "#",
        f"# method:       raytrace + 2nd moments of ray-displacement difference",
        f"# rings:        {rings}",
        f"# spokes:       {spokes}",
        f"# field_radius: {field_radius} deg",
        f"# nrad:         {nrad} (pupil hexapolar rings)",
        "# frame:        ocs, rtp = 0 deg",
        "# chip_heights: excluded (include_chip_heights=False)",
        f"# steps:        default_schema().step{step_note}",
        "#",
    ]

    for name, value in zip(schema.dof_names, weights):
        lines.append(f"- {float(value)!r}  # {name}")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rings", type=int, default=5,
                        help="Radial quadrature rings (default: 5)")
    parser.add_argument("--spokes", type=int, default=6,
                        help="Azimuthal spokes (default: 6)")
    parser.add_argument("--field-radius", type=float, default=1.75,
                        help="Field radius in degrees (default: 1.75)")
    parser.add_argument("--nrad", type=int, default=10,
                        help="Pupil hexapolar sampling rings (default: 10)")
    parser.add_argument("--step-scale", type=float, default=1.0,
                        help="Scale all DOF perturbation steps (default: 1.0). "
                             "Results should be insensitive to this.")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output path (default: {OUT_DIR / 'fwhm.yaml'})")
    parser.add_argument("--no-install", action="store_true",
                        help="Do not copy the result into StarSharp/data/")
    args = parser.parse_args()

    out_path = args.output or (OUT_DIR / "fwhm.yaml")

    print(
        f"Raytracing {args.rings}x{args.spokes} quadrature grid to "
        f"{args.field_radius} deg, nrad={args.nrad} ..."
    )
    weights = compute_fwhm_weights(
        rings=args.rings,
        spokes=args.spokes,
        field_radius=args.field_radius,
        nrad=args.nrad,
        step_scale=args.step_scale,
    )
    write_yaml(
        weights, out_path,
        rings=args.rings, spokes=args.spokes,
        field_radius=args.field_radius, nrad=args.nrad,
        step_scale=args.step_scale,
    )

    if not args.no_install:
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dst = DATA_DIR / "fwhm.yaml"
        shutil.copy2(out_path, dst)
        print(f"Installed {out_path.name} -> {dst}")


if __name__ == "__main__":
    sys.exit(main())
