#!/usr/bin/env python
"""Compute a double-Zernike sensitivity matrix and save to FITS and/or ASDF.

FITS output contains:
  - Primary HDU: gradient array of shape (ndof, kmax+1, jmax+1) in units of
    the sensitivity (waves per DOF unit, or arcsec per DOF unit depending on
    the Zernike normalization).
  - Extension 1 (BinTable): DOF names and units.

ASDF output serialises the full Sensitivity object (gradient, nominal, schema,
basis) via the StarSharp ASDF extension.

Usage examples
--------------
python compute_sensitivity.py --version 3.14 --band r
python compute_sensitivity.py --version 3.14 --band i --algorithm zk \\
    --output sensitivity/3.14_i.fits --output-asdf sensitivity/3.14_i.asdf
"""

import argparse

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import QTable

from StarSharp.datatypes import PointingModel
from StarSharp.models.fiducial import DEFAULT_WAVELENGTHS_NM, default_raytraced_model
from StarSharp.models import RTPLookup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute double-Zernike AOS sensitivity matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", default="3.3",
        help="Optics version: '3.3'/'3.12'/'3.14'")
    parser.add_argument("--band", default="r", help="LSST band (u/g/r/i/z/y)")
    parser.add_argument("--rtp", default="0 deg",
        help="Rotator position angle (astropy Angle string, e.g. '0 deg')")
    parser.add_argument("--kmax", type=int, default=15,
        help="Maximum field Noll index")
    parser.add_argument("--jmax", type=int, default=28,
        help="Maximum pupil Noll index")
    parser.add_argument("--field-outer", type=float, default=1.75,
        help="Outer field radius [deg]")
    parser.add_argument("--rings", type=int, default=10,
        help="Radial rings for Zernike field sampling")
    parser.add_argument("--algorithm", default="gq", choices=["gq", "ta", "zk"],
        help="Wavefront algorithm: 'gq' (Gaussian quadrature), 'ta' (transverse aberration), or 'zk' (Zernike fit)")
    parser.add_argument("--pointing-model", default=None, metavar="FILE",
        help="ECSV file with a pointing model to apply during ray tracing")
    parser.add_argument("--rtp-lookup", default=None, metavar="FILE",
        help="ECSV file with an RTPLookup applied as a fixed offset")
    parser.add_argument("--output", default=None,
        help="Output FITS path (default: sensitivity_{version}_{band}.fits)")
    parser.add_argument("--output-asdf", default=None, metavar="FILE",
        help="Also write a StarSharp ASDF file at this path (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or f"sensitivity_{args.version}_{args.band}.fits"

    pointing_model = None
    if args.pointing_model:
        pointing_model = PointingModel.from_table(QTable.read(args.pointing_model))
        print(f"Using pointing model from {args.pointing_model!r}", flush=True)

    rtp_lookup = RTPLookup.from_file(args.rtp_lookup) if args.rtp_lookup else None
    if rtp_lookup is not None:
        print(f"Using RTP lookup from {args.rtp_lookup!r}", flush=True)

    rtp = Angle(args.rtp)
    print(f"Building model: version={args.version!r}, band={args.band!r}, rtp={rtp}", flush=True)
    model = default_raytraced_model(
        version=args.version,
        band=args.band,
        rtp=rtp,
        pointing_model=pointing_model,
        rtp_lookup=rtp_lookup,
    )

    field = model.make_hex_field(outer=args.field_outer * u.deg, nrad=args.rings)

    print(
        f"Computing double-Zernike sensitivity: kmax={args.kmax}, jmax={args.jmax}, "
        f"field_outer={args.field_outer} deg, rings={args.rings}, algorithm={args.algorithm!r}",
        flush=True,
    )
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    sens = model.double_zernikes_sensitivity(
        kmax=args.kmax,
        field_outer=args.field_outer * u.deg,
        field=field,
        jmax=args.jmax,
        rings=args.rings,
        algorithm=args.algorithm,
        tqdm=tqdm,
        include_chip_heights=False,
    )

    # gradient shape: (ndof, kmax+1, jmax+1)
    gradient = sens.gradient.coefs  # Quantity
    array = gradient.value          # raw float array
    unit_str = gradient.unit.to_string()

    schema = model.state_schema
    wavelength = DEFAULT_WAVELENGTHS_NM[args.band]

    # Primary HDU: gradient array
    primary = fits.PrimaryHDU(array)
    hdr = primary.header
    hdr["VERSION"] = (args.version, "Optics version")
    hdr["BAND"] = (args.band, "LSST band")
    hdr["RTP"] = (float(rtp.deg), "Rotator position angle [deg]")
    hdr["WAVELEN"] = (float(wavelength.to_value(u.nm)), "Wavelength [nm]")
    hdr["KMAX"] = (args.kmax, "Maximum field Noll index")
    hdr["JMAX"] = (args.jmax, "Maximum pupil Noll index")
    hdr["NDOF"] = (array.shape[0], "Number of AOS DOFs")
    hdr["FOUTER"] = (args.field_outer, "Outer field radius [deg]")
    hdr["RINGS"] = (args.rings, "Radial rings for field sampling")
    hdr["ALGO"] = (args.algorithm, "Wavefront algorithm")
    hdr["BUNIT"] = (unit_str, "Physical unit of gradient values")
    hdr["COMMENT"] = "Axes: (ndof, kmax+1, jmax+1)"

    # Extension 1: DOF names and units table
    dof_table = fits.BinTableHDU.from_columns([
        fits.Column(name="dof_name", format="20A",
                    array=np.array(schema.dof_names)),
        fits.Column(name="dof_unit", format="10A",
                    array=np.array([unit.to_string() for unit in schema.dof_units])),
        fits.Column(name="dof_step", format="D",
                    array=schema.step),
    ], name="DOF_INFO")

    hdul = fits.HDUList([primary, dof_table])
    hdul.writeto(output, overwrite=True)
    print(f"Wrote {output}  shape={array.shape}  unit={unit_str!r}")

    if args.output_asdf is not None:
        import asdf
        from StarSharp.io.asdf.extension import StarSharpExtension

        with asdf.config_context() as cfg:
            cfg.add_extension(StarSharpExtension())
            with asdf.AsdfFile({"sensitivity": sens}) as af:
                af.write_to(args.output_asdf)
        print(f"Wrote {args.output_asdf}")


if __name__ == "__main__":
    main()
