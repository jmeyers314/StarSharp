from pathlib import Path
import numpy as np
import batoid
from batoid_rubin import LSSTBuilder
from astropy.coordinates import Angle
from lsst.obs.lsst import LsstCam
from StarSharp import RaytracedOpticalModel, StateFactory
import astropy.units as u
import matplotlib.pyplot as plt


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

field = model.make_ccd_field(nx=1, detnums=np.arange(4, 189, 9))
sf = StateFactory(50)
nominal_state = sf.from_f([0]*50)
steps = sf.from_f(model.steps)

sens_gq = model.zernikes_sensitivity(
    field=field,
    steps=steps,
    algorithm="gq",
)

sens_ta = model.zernikes_sensitivity(
    field=field,
    steps=steps,
    algorithm="ta",
)

dof_dir = Path("gq_ta_sensitivity_comparison")
dof_dir.mkdir(exist_ok=True)

for idof in range(steps.value.size):
    fig, axs = plt.subplots(nrows=4, ncols=7, figsize=(25, 12), constrained_layout=True)
    for j in range(4, 29):
        ax = axs.flat[j-1]
        im = ax.scatter(
            field.x.to_value(u.mm),
            field.y.to_value(u.mm),
            c=sens_gq.gradient.coefs[idof, :, j].value,
            cmap="bwr",
            vmin=-np.max(np.abs(sens_gq.gradient.coefs[idof, :, j].value)),
            vmax=np.max(np.abs(sens_gq.gradient.coefs[idof, :, j].value)),
        )
        plt.colorbar(im, ax=ax)
    fig.delaxes(axs.flat[0])
    fig.delaxes(axs.flat[1])
    fig.delaxes(axs.flat[2])
    fig.delaxes(axs.flat[3])
    fig.suptitle(f"GQ sensitivity for DOF {idof}")
    fig.savefig(dof_dir / f"sensitivity_gq_dof{idof:02d}.png")
    plt.close(fig)

for idof in range(steps.value.size):
    fig, axs = plt.subplots(nrows=4, ncols=7, figsize=(25, 12), constrained_layout=True)
    for j in range(4, 29):
        ax = axs.flat[j-1]
        im = ax.scatter(
            field.x.to_value(u.mm),
            field.y.to_value(u.mm),
            c=sens_ta.gradient.coefs[idof, :, j].value,
            cmap="bwr",
            vmin=-np.max(np.abs(sens_ta.gradient.coefs[idof, :, j].value)),
            vmax=np.max(np.abs(sens_ta.gradient.coefs[idof, :, j].value)),
        )
        plt.colorbar(im, ax=ax)

    fig.delaxes(axs.flat[0])
    fig.delaxes(axs.flat[1])
    fig.delaxes(axs.flat[2])
    fig.delaxes(axs.flat[3])
    fig.suptitle(f"TA sensitivity for DOF {idof}")
    fig.savefig(dof_dir / f"sensitivity_ta_dof{idof:02d}.png")
    plt.close(fig)

for idof in range(steps.value.size):
    fig, axs = plt.subplots(nrows=4, ncols=7, figsize=(25, 12), constrained_layout=True)
    for j in range(4, 29):
        ax = axs.flat[j-1]
        dz = sens_gq.gradient.coefs[idof, :, j].value - sens_ta.gradient.coefs[idof, :, j].value
        im = ax.scatter(
            field.x.to_value(u.mm),
            field.y.to_value(u.mm),
            c=dz,
            cmap="bwr",
            vmin=-np.max(np.abs(dz)),
            vmax=np.max(np.abs(dz)),
        )
        plt.colorbar(im, ax=ax)
    fig.delaxes(axs.flat[0])
    fig.delaxes(axs.flat[1])
    fig.delaxes(axs.flat[2])
    fig.suptitle(f"Difference in sensitivity for DOF {idof}")
    fig.savefig(dof_dir / f"sensitivity_diff_dof{idof:02d}.png")
    plt.close(fig)
