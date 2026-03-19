from dataclasses import replace

import astropy.units as u
import batoid
import numpy as np
from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from lsst.obs.lsst import LsstCam
from tqdm import trange

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

rtp = Angle("20 deg")
model = RaytracedOpticalModel(
    builder=builder,
    rtp=rtp,
    wavelength=wavelength,
    camera=camera,
)

sf = StateFactory(50)
nominal_state = sf.from_f([0] * 50)
steps = sf.from_f(model.steps)
field = model.make_ccd_field(nx=1, detnums=np.arange(4, 189, 9))

nominal_sp = model.spots(state=nominal_state, field=field)
for i in trange(50):
    dval = np.zeros_like(steps.value)
    dval[i] = steps.value[i]
    dstate = replace(
        nominal_state,
        value=dval,
    )
    perturbed = model.spots(state=dstate, field=field)
    np.testing.assert_allclose(
        nominal_sp.px.to_value(u.mm),
        perturbed.px.to_value(u.mm),
        atol=1e-16,
    )
    np.testing.assert_allclose(
        nominal_sp.py.to_value(u.mm),
        perturbed.py.to_value(u.mm),
        atol=1e-16,
    )
    assert not np.allclose(
        nominal_sp.dx.to_value(u.mm),
        perturbed.dx.to_value(u.mm),
        atol=1e-16,
    )
    assert not np.allclose(
        nominal_sp.dy.to_value(u.mm),
        perturbed.dy.to_value(u.mm),
        atol=1e-16,
    )
