"""ZernikeSolver: least-squares estimation of telescope State from Zernikes."""

from dataclasses import replace
import warnings

import astropy.units as u
import numpy as np

from .datatypes import Sensitivity, State, Zernikes
from .datatypes.double_zernikes import DoubleZernikes
from .utils import str_to_arr


class ZernikeSolver:
    """Least-squares estimation of telescope State from Zernike measurements.

    Supports both ``Sensitivity[Zernikes]`` and ``Sensitivity[DoubleZernikes]``
    inputs.  For the latter, the sensitivity is projected onto *field* at
    solve time.

    Parameters
    ----------
    sensitivity : Sensitivity
        ``Sensitivity[Zernikes]`` or ``Sensitivity[DoubleZernikes]``.
    use_zk : list[int] or str, optional
        Noll indices to use from both the observations and the sensitivity.
        A string is parsed as a comma-separated list of indices and/or
        inclusive ranges (e.g. ``"4-28"``).  Both arrays are indexed by this
        selection; if any index is absent from either array an error is raised.
        Default is ``"4-28"``.
    """

    def __init__(
        self,
        sensitivity: Sensitivity,
        use_zk: "list[int] | str" = "4-28",
    ):
        self.sensitivity = sensitivity
        self.use_zk = str_to_arr(use_zk)

    def solve(
        self,
        observed: Zernikes,
    ) -> State:
        """Estimate the telescope State from observed Zernike coefficients.

        Parameters
        ----------
        observed : Zernikes
            Deviation from the intrinsic wavefront, i.e. ``obs ≈ gradient @ state``.
            Shape ``(nfield, jmax+1)``.  The field layout must match the
            sensitivity's field (same order, same frame).

        Returns
        -------
        State
            Estimated telescope state in the same basis as the sensitivity;
            call ``.x`` or ``.f`` to convert.
        """
        # Work in ocs
        sensitivity = self.sensitivity.ocs
        observed = observed.ocs
        if isinstance(sensitivity.nominal, DoubleZernikes):
            nominal = sensitivity.nominal.single(observed.field.ocs)
            gradient = sensitivity.gradient.single(observed.field.ocs)
            sensitivity = replace(sensitivity, nominal=nominal, gradient=gradient)

        # Check that observed.field matches gradient.field
        obs_field  = observed.field.ocs
        grad_field = sensitivity.gradient.field.ocs
        if obs_field.nfield != grad_field.nfield or not (
            np.allclose(obs_field.x.to_value(u.deg), grad_field.x.to_value(u.deg))
            and np.allclose(obs_field.y.to_value(u.deg), grad_field.y.to_value(u.deg))
        ):
            raise ValueError(
                f"observed.field ({obs_field.nfield} points) does not match "
                f"sensitivity gradient field ({grad_field.nfield} points)."
            )

        # Work in microns
        obs = observed.coefs.to_value(u.micron)
        sens = sensitivity.gradient.coefs.to_value(u.micron)
        nj_obs = obs.shape[-1]
        nj_sens = sens.shape[-1]

        # Validate and apply use_zk selection
        use_zk = self.use_zk
        missing_obs = use_zk[use_zk >= nj_obs]
        if len(missing_obs):
            raise ValueError(
                f"use_zk contains indices {missing_obs.tolist()} but observations "
                f"only have {nj_obs} Zernike terms (max valid index: {nj_obs - 1})."
            )
        missing_sens = use_zk[use_zk >= nj_sens]
        if len(missing_sens):
            raise ValueError(
                f"use_zk contains indices {missing_sens.tolist()} but sensitivity "
                f"only has {nj_sens} Zernike terms (max valid index: {nj_sens - 1})."
            )

        obs  = obs[..., use_zk]
        sens = sens[..., use_zk]

        A = sens.reshape(sens.shape[0], -1).T
        solution, _, rank, _ = np.linalg.lstsq(A, obs.ravel(), rcond=None)
        if rank < A.shape[1]:
            warnings.warn(
                f"ZernikeSolver design matrix is rank-deficient "
                f"({rank} < {A.shape[1]}); solution may be inaccurate.",
                stacklevel=2,
            )

        return State(
            value=solution,
            basis=sensitivity.basis,
            schema=sensitivity.schema,
        )
