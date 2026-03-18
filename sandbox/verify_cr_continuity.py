import numpy as np
import batoid
from batoid_rubin import LSSTBuilder
from astropy.coordinates import Angle
from lsst.obs.lsst import LsstCam
from StarSharp import RaytracedOpticalModel, StateFactory
import astropy.units as u
from dataclasses import replace
from tqdm import trange
import matplotlib.pyplot as plt


fiducial = batoid.Optic.fromYaml("Rubin_v3.14_r.yaml")
# fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
builder = LSSTBuilder(
    fiducial,
    dof_coord_system="OCS",
    flip_m2_bending_modes=False,
    dof_angle_units="degree",
)
wavelength = 620 * u.nm
camera = LsstCam().getCamera()

rtp = Angle("20 deg")
model = RaytracedOpticalModel(
    builder=builder,
    rtp=rtp,
    wavelength=wavelength,
    camera=camera,
)

sf = StateFactory(50)
nominal_state = sf.from_f([0]*50)
steps = sf.from_f(model.steps)
field = model.make_ccd_field(nx=1)

# reference = "chief"
reference = "ring"
fig, axs = plt.subplots(nrows=5, ncols=10, figsize=(20, 10), constrained_layout=True)
nominal_sp = model.spots(state=nominal_state, field=field, nrad=2, reference=reference)
for i in trange(50):
    dval = np.zeros_like(steps.value)
    dval[i] = steps.value[i]
    dstate = replace(
        nominal_state,
        value=dval,
    )
    perturbed = model.spots(state=dstate, field=field, nrad=2, reference=reference)
    ax = axs.flat[i]
    ax.quiver(
        nominal_sp.field.x.to_value(u.mm),
        nominal_sp.field.y.to_value(u.mm),
        perturbed.field.x.to_value(u.mm) - nominal_sp.field.x.to_value(u.mm),
        perturbed.field.y.to_value(u.mm) - nominal_sp.field.y.to_value(u.mm),
    )
    ax.set_title(f"DOF {i}")
plt.show()
