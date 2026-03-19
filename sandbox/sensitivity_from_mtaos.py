import yaml
from pathlib import Path
import numpy as np
import os
import astropy.units as u

from StarSharp import Sensitivity, DoubleZernikes, StateFactory

directory = Path(os.environ["TS_CONFIG_MTTCS_DIR"]) / "MTAOS/v13/ofc/"
senspath = directory / "sensitivity_matrix" / "lsst_sensitivity_dz_31_29_50.yaml"
with open(senspath, "r") as f:
    sens = np.array(yaml.safe_load(f))
normpath = directory / "normalization_weights" / "range-fwhm.yaml"
with open(normpath, "r") as f:
    norm = np.array(yaml.safe_load(f))

sf = StateFactory(
    sens,
    norm=norm,
    use_dof="0-9,10-16,30-34",
    nkeep=12
)

gradient = DoubleZernikes(
    sens.transpose(2, 0, 1) * u.micron,
    field_outer=1.75 * u.deg,
    field_inner=0.0 * u.deg,
    pupil_outer=4.18 * u.m,
    pupil_inner=2.55 * u.m,
    frame="ocs",
)

sensitivity = Sensitivity(
    gradient=gradient,
)
sf2 = StateFactory(
    sensitivity,
    norm=norm,
    use_dof="0-9,10-16,30-34",
    nkeep=12
)
