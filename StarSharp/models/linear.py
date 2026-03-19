from __future__ import annotations

from dataclasses import replace

import astropy.units as u
import galsim
import numpy as np
from astropy.coordinates import Angle
from lsst.afw.cameraGeom import Camera
from scipy.interpolate import RegularGridInterpolator

from ..datatypes import FieldCoords, Sensitivity, State, Zernikes


class LinearOpticalModel:
    """Fast linear optical model using per-detector interpolation and
    DoubleZernike sensitivities.

    The nominal wavefront is stored on a regular per-detector grid and
    interpolated to arbitrary field positions.  The DOF-dependent
    perturbation is modelled via a :class:`~StarSharp.Sensitivity` whose
    gradient is stored as :class:`~StarSharp.DoubleZernikes` — smooth
    over the full field of view — so it can be evaluated at any field
    point without interpolation.

    Linear prediction:

    .. math::

        z(\\mathbf{x}, \\delta\\theta)
        = z_0(\\mathbf{x})
        + \\nabla_{\\theta}\\,z\\big|_{\\theta_0}
          \\cdot \\delta\\theta

    Parameters
    ----------
    nominal_zernikes : Zernikes
        Zernike coefficients on a per-detector grid (CCS frame).
        The ``field`` attribute must carry ``wcs`` and ``camera``.
    zernikes_sensitivity : Sensitivity[DoubleZernikes]
        Wavefront sensitivity whose gradient is in the OCS frame.
    rtp : Angle
        Rotator position angle.
    camera : Camera
        LSST camera geometry.
    """

    def __init__(
        self,
        nominal_zernikes: Zernikes,
        zernikes_sensitivity: Sensitivity,
        rtp: Angle,
        camera: Camera,
    ):
        self.rtp = rtp
        self.camera = camera
        self._nominal_zernikes = nominal_zernikes
        self._zernikes_sensitivity = zernikes_sensitivity

        # Build per-detector interpolators eagerly.
        self._zk_interps = self._build_zk_interpolators(nominal_zernikes)

    # ------------------------------------------------------------------
    # Field preparation
    # ------------------------------------------------------------------

    def _prepare_field(self, field: FieldCoords) -> FieldCoords:
        """Attach *rtp*, *camera*, and *wcs* from ``self`` if missing."""
        if field.rtp is None:
            field = replace(field, rtp=self.rtp)
        if field.camera is None:
            field = replace(field, camera=self.camera)
        if field.wcs is None and self._nominal_zernikes.field.wcs is not None:
            field = replace(field, wcs=self._nominal_zernikes.field.wcs)
        return field

    # ------------------------------------------------------------------
    # Interpolator construction
    # ------------------------------------------------------------------

    def _build_zk_interpolators(
        self,
        zk: Zernikes,
    ) -> dict[int, list[RegularGridInterpolator]]:
        """Build one ``RegularGridInterpolator`` per (detector, j) pair.

        Returns ``{detnum: [interp_j0, interp_j1, …]}``.
        """
        field_ccs = zk.field.focal_plane.ccs
        detnums = field_ccs.detnum
        coefs = zk.coefs.to_value(u.um)  # (nfield, jmax+1)
        x_mm = field_ccs.x.to_value(u.mm)
        y_mm = field_ccs.y.to_value(u.mm)

        interps: dict[int, list[RegularGridInterpolator]] = {}
        for det_id in np.unique(detnums):
            if det_id == -1:
                continue
            mask = detnums == det_id
            xi = x_mm[mask]
            yi = y_mm[mask]
            ci = coefs[mask]  # (npts, jmax+1)

            # Recover the regular grid axes
            xvals = np.unique(xi)
            yvals = np.unique(yi)

            if len(xvals) * len(yvals) != len(xi):
                raise ValueError(
                    f"Detector {det_id}: expected a regular grid of "
                    f"{len(xvals)}×{len(yvals)} points, got {len(xi)}"
                )

            njmax = ci.shape[-1]
            grid = np.full((len(xvals), len(yvals), njmax), np.nan)
            for xx, yy, cc in zip(xi, yi, ci):
                ix = np.searchsorted(xvals, xx)
                iy = np.searchsorted(yvals, yy)
                grid[ix, iy, :] = cc

            det_interps = []
            for j in range(njmax):
                det_interps.append(
                    RegularGridInterpolator(
                        (xvals, yvals),
                        grid[:, :, j],
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                )
            interps[int(det_id)] = det_interps

        return interps

    # ------------------------------------------------------------------
    # Nominal interpolation
    # ------------------------------------------------------------------

    def _interpolate_zernikes(self, field: FieldCoords) -> np.ndarray:
        """Interpolate nominal Zernike coefficients at *field*.

        Returns an array of shape ``(nfield, jmax+1)`` in microns.
        """
        fp_ccs = field.focal_plane.ccs
        x_mm = fp_ccs.x.to_value(u.mm).ravel()
        y_mm = fp_ccs.y.to_value(u.mm).ravel()
        detnums = fp_ccs.detnum.ravel()

        njmax = self._nominal_zernikes.jmax + 1
        result = np.full((len(x_mm), njmax), np.nan)

        for det_id, det_interps in self._zk_interps.items():
            mask = detnums == det_id
            if not np.any(mask):
                continue
            pts = np.column_stack([x_mm[mask], y_mm[mask]])
            for j, interp in enumerate(det_interps):
                result[mask, j] = interp(pts)

        return result.reshape(field.x.shape + (njmax,))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def zernikes(self, state: State, field: FieldCoords) -> Zernikes:
        """Evaluate the wavefront at arbitrary field positions.

        Interpolates the nominal wavefront on the per-detector grid and adds
        the linear DOF perturbation from the DoubleZernike sensitivity.

        Parameters
        ----------
        state : State
            AOS alignment perturbation (delta from the nominal).
        field : FieldCoords
            Query field coordinates (any frame / space).

        Returns
        -------
        Zernikes
            Predicted Zernikes in the OCS frame with units of micron.
        """
        field = self._prepare_field(field)

        # Nominal: per-detector interpolation (CCS), then rotate to OCS.
        nominal_um = self._interpolate_zernikes(field)  # (nfield, jmax+1) in µm

        jmax = self._nominal_zernikes.jmax
        rot_to_ocs = galsim.zernike.zernikeRotMatrix(jmax, -self.rtp.radian)
        nominal_ocs = nominal_um @ rot_to_ocs  # (nfield, jmax+1)

        # Sensitivity: DoubleZernike gradient evaluated at the query field,
        # then linearly combined with state weights.
        field_ocs = field.angle.ocs
        sens = self._zernikes_sensitivity
        weights = getattr(state, sens.steps.basis).value

        # sens.gradient is a DoubleZernikes with batch_shape = (ndof,)
        # .single(field) → Zernikes with coefs shape (ndof, nfield, jmax+1)
        grad_zk = sens.gradient.single(field_ocs)
        delta_um = np.einsum(
            "i...,i->...", grad_zk.coefs.to_value(u.um), weights
        )  # (nfield, jmax+1)

        coefs = (nominal_ocs + delta_um) * u.um

        return Zernikes(
            coefs=coefs,
            field=field_ocs,
            R_outer=self._nominal_zernikes.R_outer,
            R_inner=self._nominal_zernikes.R_inner,
            wavelength=self._nominal_zernikes.wavelength,
            frame="ocs",
            rtp=self.rtp,
        )

    def optimize(
        self,
        guess: State,
        field: FieldCoords,
        offset: State | None = None,
        jmin: int = 4,
    ) -> State:
        """Solve for the best-fit DOF perturbation via linear least squares.

        Minimises
        ``||nominal(field) + G @ delta_theta||^2`` over Zernike indices
        ``jmin … jmax`` at every field point.

        Parameters
        ----------
        guess : State
            Provides ``use_dof``, ``n_dof``, and ``Vh`` metadata.  Its
            value is ignored (the solve starts from zero).
        field : FieldCoords
            Field coordinates to fit.
        offset : State or None
            Fixed additive offset applied after the solve.
        jmin : int
            Lowest Noll index included in the fit (default: 4, skip piston/
            tip/tilt).

        Returns
        -------
        State
            Optimal perturbation in the same basis as the sensitivity steps.
        """
        field = self._prepare_field(field)
        field_ocs = field.angle.ocs

        # Nominal wavefront at the query field (OCS, µm)
        nominal_um = self._interpolate_zernikes(field)
        jmax = self._nominal_zernikes.jmax
        rot = galsim.zernike.zernikeRotMatrix(jmax, -self.rtp.radian)
        nominal_ocs = nominal_um @ rot  # (nfield, jmax+1)

        # Gradient matrix: (ndof, nfield, jmax+1)
        sens = self._zernikes_sensitivity
        grad_zk = sens.gradient.single(field_ocs)
        G = grad_zk.coefs.to_value(u.um)  # (ndof, nfield, jmax+1)

        # Flatten (nfield, jmax+1 slice) into a single vector.
        rhs = -nominal_ocs[:, jmin:].ravel()  # target: drive wavefront to zero
        ndof = G.shape[0]
        Gmat = G[:, :, jmin:].reshape(ndof, -1).T  # (nfield*(jmax+1-jmin), ndof)

        x, *_ = np.linalg.lstsq(Gmat, rhs, rcond=None)

        result = State(
            value=x,
            basis=sens.steps.basis,
            use_dof=guess.use_dof,
            n_dof=guess.n_dof,
            Vh=guess.Vh,
        )
        if offset is not None:
            result = result + offset
        return result
