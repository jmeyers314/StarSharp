import time

import astropy.units as u
import batoid
import numpy as np
from astropy.coordinates import Angle
from StarSharp.datatypes.state import StateFactory
from batoid_rubin import LSSTBuilder
from lsst.obs.lsst import LsstCam

from StarSharp.models.raytraced import RaytracedOpticalModel
from StarSharp.models.linear import LinearSpotModel
from StarSharp import State


class TestLinearSpotModel:
    def test_ctor(self):
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        builder = LSSTBuilder(
            telescope,
            dof_coord_system="OCS",
            flip_m2_bending_modes=False,
            dof_angle_units="degree"
        )
        camera = LsstCam().getCamera()
        model = RaytracedOpticalModel(
            builder,
            rtp = Angle("20 deg"),
            wavelength=620 * u.nm,
            camera=camera
        )
        fc = model.make_ccd_field(detnums=list(range(4, 189, 9))) # Raft centers
        spot_model = LinearSpotModel(
            raytraced=model,
            field=fc,
            use_dof="0-9,10-16,30-36",
        )

        rng = np.random.default_rng(4321)
        state = State(
            value=rng.normal(0.0, size=24) * model.steps[spot_model.use_dof],
            basis="x",
            use_dof=spot_model.use_dof,
            n_dof=50
        )
        mspots = model.spots(fc, state)
        lspots = spot_model.spots(state)

        # t0 = time.time()
        # for _ in range(5):
        #     mspots = model.spots(fc, state)
        # t1 = time.time()
        # print()
        # print(f"Time for 50 iterations: {t1 - t0} seconds")

        # t0 = time.time()
        # for _ in range(5):
        #     lspots = spot_model.spots(state)
        # t1 = time.time()
        # print(f"Time for 50 iterations: {t1 - t0} seconds")

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
        # for ax, mspot, lspot in zip(axs.flat, mspots, lspots):
        #     ax.scatter(
        #         mspot.dx[~mspot.vignetted], mspot.dy[~mspot.vignetted],
        #         color="C0", label="Raytraced", s=10
        #     )
        #     ax.scatter(
        #         lspot.dx[~lspot.vignetted], lspot.dy[~lspot.vignetted],
        #         color="C1", label="Linear", s=1
        #     )
        #     ax.set_aspect("equal")
        # plt.legend()
        # plt.show()

        m2 = mspots.moments()
        l2 = lspots.moments()

        np.testing.assert_allclose(
            m2.xx, l2.xx, rtol=5e-3, atol=5e-3
        )
        np.testing.assert_allclose(
            m2.yy, l2.yy, rtol=5e-3, atol=5e-3
        )
        np.testing.assert_allclose(
            m2.e1, l2.e1, rtol=0, atol=1e-3
        )
        np.testing.assert_allclose(
            m2.e2, l2.e2, rtol=0, atol=1e-3
        )
