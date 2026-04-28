"""Utility functions for StarSharp."""

import numpy as np
from numpy.typing import NDArray


def gaussian_quadrature_2d(
    nrings: int = 2,
    nphi: int = 6,
    cov: NDArray | None = None,
    center: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Deterministic weighted point set whose moments match a 2D Gaussian.

    Builds concentric rings of equally-spaced points whose radii and
    per-ring weights are set by Gauss–Laguerre quadrature on the radial
    variable `s = r²/2`.  All moments ``E[x^a y^b]`` of the target
    Gaussian are reproduced exactly through total degree

        D = min(nphi - 1, 4 * nrings - 2)

    For the default ``nrings=2, nphi=6`` (12 points) this gives D = 5,
    i.e. all moments through 5th order are exact.

    Parameters
    ----------
    nrings : int
        Number of concentric rings (default 2).
    nphi : int
        Points per ring (default 6).  Even values give point symmetry
        ``(x, y) ↔ (−x, −y)``.
    cov : (2, 2) array_like or None
        Target covariance matrix.  Defaults to the 2×2 identity
        (standard normal).  For a general covariance the standard-normal
        points are linearly transformed via the Cholesky factor.
    center : bool
        If *True*, prepend a point at the origin with weight zero.
        (The Gauss–Laguerre rule always assigns zero weight to r = 0
        because the radial density r·exp(−r²/2) vanishes there, but a
        centre point can still be useful as a structural reference.)

    Returns
    -------
    x, y : ndarray, shape ``(N,)``
        Sample positions, where ``N = nrings * nphi (+ 1 if center)``.
    w : ndarray, shape ``(N,)``
        Non-negative weights summing to 1.
    """
    # Gauss–Laguerre nodes t_j and weights omega_j for  ∫ f(t) exp(−t) dt
    # on [0, ∞).  An n-point rule integrates polynomials of degree ≤ 2n−1.
    t, omega = np.polynomial.laguerre.laggauss(nrings)

    # Map to radii: s = r²/2 = t  ⟹  r = √(2t)
    r = np.sqrt(2.0 * t)

    # Split each ring weight equally among its azimuthal points.
    w_per_point = omega / nphi

    # Uniform azimuthal grid on each ring
    theta = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    # (nrings, nphi) outer product → flattened
    x = (r[:, None] * np.cos(theta[None, :])).ravel()
    y = (r[:, None] * np.sin(theta[None, :])).ravel()
    w = np.repeat(w_per_point, nphi)

    if center:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])
        w = np.concatenate([[0.0], w])

    # Apply covariance transformation  x' = L @ x  where Σ = L Lᵀ
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        L = np.linalg.cholesky(cov)
        xy = L @ np.stack([x, y])
        x, y = xy[0], xy[1]

    return x, y, w
