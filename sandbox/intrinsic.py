from pathlib import Path

import astropy.units as u
import batoid
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from lsst.afw.cameraGeom import FOCAL_PLANE
from lsst.obs.lsst import LsstCam
from StarSharp import RaytracedOpticalModel, State
from tqdm import tqdm

fiducial = batoid.Optic.fromYaml("Rubin_v3.14_r.yaml")
builder = LSSTBuilder(
    fiducial,
    dof_coord_system="OCS",
    flip_m2_bending_modes=False,
    dof_angle_units="degree",
)
wavelength = 620 * u.nm
camera = LsstCam().getCamera()

angles = np.linspace(-90, 90, 37) * u.deg
frame_dir = Path("frames_intrinsic")
frame_dir.mkdir(exist_ok=True)

for i, rtp in enumerate(tqdm(angles, desc="Rendering frames")):
    rtp = Angle(rtp)
    model = RaytracedOpticalModel(
        builder=builder,
        rtp=rtp,
        wavelength=wavelength,
        camera=camera,
    )

    field = model.make_ccd_field(nx=4, types=("E2V", "ITL", "ITL_WF"))

    opt = model.optimize(
        State(np.zeros(50), basis="f", use_dof=[5, 8, 9], n_dof=50),
        field=model.make_ccd_field(nx=1),
        verbose=0,
        nrad=5,
        mode="size",
    )

    zk = model.zernikes(state=opt, field=field)

    fig, ax = plt.subplots(figsize=(7, 7))
    for det in camera:
        corners = det.getCorners(FOCAL_PLANE)
        corners = [(y, x) for x, y in corners]
        ax.plot(
            [c[0] for c in corners + [corners[0]]],
            [c[1] for c in corners + [corners[0]]],
            color="k",
            lw=1,
        )

    sc = ax.scatter(
        field.ccs.focal_plane.x.to_value(u.mm),
        field.ccs.focal_plane.y.to_value(u.mm),
        c=zk.coefs[:, 4].to_value(u.micron),
        cmap="bwr",
        s=10,
        vmin=-0.1,
        vmax=0.1,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"RTP = {rtp.to_value(u.deg):.1f} deg")
    fig.colorbar(sc, ax=ax, label="Z4 (micron)")

    fig.savefig(frame_dir / f"frame_{i:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
