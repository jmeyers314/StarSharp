from pathlib import Path

import astropy.units as u
import batoid
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.obs.lsst import LsstCam
from tqdm import tqdm

from StarSharp import RaytracedOpticalModel, StateFactory

fiducial = batoid.Optic.fromYaml("Rubin_v3.14_r.yaml")
builder = LSSTBuilder(
    fiducial,
    dof_coord_system="OCS",
    flip_m2_bending_modes=False,
    dof_angle_units="degree",
)
wavelength = 620 * u.nm
camera = LsstCam().getCamera()

sf3 = StateFactory(50, use_dof=[5, 8, 9])
offset = sf3.from_x([3.0, 0.005, 0.005])

rtp = Angle("0 deg")
model = RaytracedOpticalModel(
    builder=builder,
    rtp=rtp,
    wavelength=wavelength,
    camera=camera,
)

field = model.make_ccd_field(
    nx=1,
    types=("E2V", "ITL"),
    # detnums=list(range(4, 189, 9))
)

# Just calculate for specific dofs
sf22 = StateFactory(50, use_dof="0-9,10-16,30-34")
steps = sf22.from_x(model.steps[sf22.use_dof])


sens = model.zernikes_sensitivity(
    field=field,
    steps=steps,
    offset=offset,
)
fig, ax = plt.subplots(nrows=4, ncols=6, figsize=(10, 7))
for i in range(len(sens)):
    ax_i = ax.flat[i]
    z4 = sens[i].coefs[:, 4].to_value(u.micron)
    vmin, vmax = np.quantile(z4, [0.0, 1.0])
    im = ax_i.scatter(
        field.x.to_value(u.mm),
        field.y.to_value(u.mm),
        c=sens[i].coefs[:, 4].to_value(u.micron),
        vmin=vmin,
        vmax=vmax,
        cmap="bwr",
    )
    ax_i.set_title(f"DOF {sf22.use_dof[i]}")
plt.show()
