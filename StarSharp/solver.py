"""ZernikeSolver: least-squares estimation of telescope State from Zernikes."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import astropy.units as u
import numpy as np

from .datatypes import Sensitivity, State, Zernikes
from .datatypes.double_zernikes import DoubleZernikes


class ZernikeSolver:
    """Least-squares estimation of telescope State from Zernike measurements.

    Supports both ``Sensitivity[Zernikes]`` and ``Sensitivity[DoubleZernikes]``
    inputs.  For the latter, the sensitivity is projected onto *field* at
    solve time.

    Parameters
    ----------
    sensitivity : Sensitivity
        ``Sensitivity[Zernikes]`` or ``Sensitivity[DoubleZernikes]``.
    jmin : int, optional
        Minimum Noll index to include in the fit.  Columns
        ``0 … jmin-1`` are dropped from both the observations and the
        sensitivity before solving.  Default is ``0`` (use all terms).
        Set to ``4`` to discard piston, tip, and tilt.
    allow_narrow_zk : bool, optional
        If *True* (default), observations may have **fewer** Zernike
        terms than the sensitivity.  The sensitivity columns are
        truncated to match the observations.
        If *False*, a mismatch in this direction raises ``ValueError``.
    allow_narrow_sens : bool, optional
        If *True*, observations may have **more** Zernike terms than the
        sensitivity.  The observations are truncated to match the
        sensitivity.
        If *False* (default), a mismatch in this direction raises
        ``ValueError`` — extra observed modes would be silently ignored,
        which is usually a mistake.
    """

    def __init__(
        self,
        sensitivity: Sensitivity,
        jmin: int = 0,
        allow_narrow_zk: bool = True,
        allow_narrow_sens: bool = False,

    ):
        self.sensitivity = sensitivity
        self.jmin = jmin
        self.allow_narrow_zk = allow_narrow_zk
        self.allow_narrow_sens = allow_narrow_sens

    def solve(
        self,
        observed: Zernikes,
        mode: Literal["total", "deviation"] = "deviation",
    ) -> State:
        """Estimate the telescope State from observed Zernike coefficients.

        Parameters
        ----------
        observed : Zernikes
            Observed coefficients.  Shape ``(nfield, jmax+1)``.
            The field layout must match the sensitivity's field (same order,
            same frame).
        mode : ``"total"`` or ``"deviation"``
            Interpretation of *observed*:

            ``"total"``
                The observed wavefront includes the nominal (intrinsic)
                shape, i.e. ``obs ≈ nominal + gradient @ state``.  The
                stored nominal is subtracted before solving.
            ``"deviation"``
                The observed data is already the perturbation from the
                intrinsic wavefront, i.e. ``obs ≈ gradient @ state``.

        Returns
        -------
        State
            Estimated telescope state in the same basis as the sensitivity;
            call ``.x`` or ``.f`` to convert.
        """
        if mode not in ("total", "deviation"):
            raise ValueError(f"mode must be 'total' or 'deviation', got {mode!r}")

        # Work in ocs
        sensitivity = self.sensitivity.ocs
        observed = observed.ocs
        if isinstance(sensitivity.nominal, DoubleZernikes):
            nominal = sensitivity.nominal.single(observed.field.ocs)
            gradient = sensitivity.gradient.single(observed.field.ocs)
            sensitivity = replace(sensitivity, nominal=nominal, gradient=gradient)

        # Work in microns
        obs = observed.coefs.to_value(u.micron)
        sens = sensitivity.gradient.coefs.to_value(u.micron)
        nj_obs = obs.shape[-1]
        nj_sens = sens.shape[-1]

        if nj_obs < nj_sens:
            if not self.allow_narrow_zk:
                raise ValueError(
                    f"Observations have {nj_obs} Zernike terms but sensitivity has "
                    f"{nj_sens}. Set allow_narrow_zk=True to truncate the sensitivity."
                )
            sens = sens[..., :nj_obs]
        elif nj_obs > nj_sens:
            if not self.allow_narrow_sens:
                raise ValueError(
                    f"Sensitivity has {nj_sens} Zernike terms but observations have "
                    f"{nj_obs}. Set allow_narrow_sens=True to truncate the observations."
                )
            obs = obs[..., :nj_sens]

        nj = obs.shape[-1]  # final common width after narrow checks

        # Apply jmin: drop low-order modes from both obs and grad
        jmin = self.jmin
        if jmin >= nj:
            raise ValueError(
                f"jmin={jmin} is >= the number of Zernike terms ({nj}) after "
                "reconciling obs and sensitivity widths."
            )
        obs  = obs[..., jmin:]
        sens = sens[..., jmin:]

        if mode == "total":
            nom = sensitivity.nominal.coefs.to_value(u.micron)
            nom = nom[..., jmin:nj]
            obs = obs - nom

        x = np.linalg.lstsq(sens.reshape(sens.shape[0], -1).T, obs.ravel(), rcond=None)

        return State(
            value=x[0],
            basis=sensitivity.basis,
            use_dof=sensitivity.use_dof,
            n_dof=sensitivity.n_dof,
            Vh=sensitivity.Vh,
        )
