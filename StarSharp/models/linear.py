from __future__ import annotations

from dataclasses import replace

import astropy.units as u
from attr import field
import galsim
import numpy as np
from astropy.coordinates import Angle
from lsst.afw.cameraGeom import Camera
from scipy.interpolate import RegularGridInterpolator

from ..datatypes import FieldCoords, Sensitivity, State, Zernikes, Spots
from .raytraced import RaytracedOpticalModel


class LinearSpotModel:
    def __init__(
        self,
        raytraced: RaytracedOpticalModel,
        field: FieldCoords,
        use_dof: str | int | None = None,
        spots_kwargs: dict | None = None,
    ):
        self.raytraced = raytraced
        self.field = field

        if isinstance(use_dof, str):
            dof_str = use_dof.replace(" ", "").strip()
            use_dof = []
            for part in dof_str.split(","):
                if "-" in part:
                    start, end = [int(p) for p in part.split("-")]
                    use_dof.extend(range(start, end + 1))
                else:
                    use_dof.append(int(part))
            use_dof = np.sort(use_dof)
        elif use_dof is None:
            use_dof = np.arange(50)

        self.use_dof = use_dof
        self.spots_kwargs = spots_kwargs or {}
        self._spots_sensitivity = None


    def spots(
        self,
        state: State,
    ) -> Spots:
        if self._spots_sensitivity is None:
            # Build a _full_ sensitivity map
            steps = self.raytraced.steps
            steps = State(
                value=steps[self.use_dof],
                basis="x",
                use_dof=self.use_dof,
                n_dof=50
            )

            self._spots_sensitivity = self.raytraced.spots_sensitivity(
                field=self.field,
                steps=steps,
                **self.spots_kwargs
            )

        return self._spots_sensitivity.predict(state)
