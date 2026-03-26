from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class State:
    VALID_BASES = ("x", "f", "v")

    """Alignment state in one of three bases.

    Analogous to `FieldCoords` with its ``frame`` attribute:
    ``value`` holds the coefficient vector and ``basis`` identifies
    the representation.  Use the ``.x``, ``.f``, and ``.v``
    properties to convert between bases (each returns a new
    ``State``).

    Three bases
    -----------
    ``"f"`` — full DOF vector (length ``n_dof``).
        Inactive DOFs (not in ``use_dof``) are zero.
    ``"x"`` — active-DOF vector (length ``len(use_dof)``).
        The entries of the full vector selected by ``use_dof``.
    ``"v"`` — SVD-truncated orthogonal-basis vector (length ``nkeep``).
        Only available when ``Vh`` is set.
        The roundtrip ``x → v → x`` is lossy when
        ``Vh`` is not square, though ``v → x → v`` is lossless.

    Parameters
    ----------
    value : NDArray[np.floating]
        Coefficient vector in the basis given by ``basis``.
    basis : str
        One of ``"x"``, ``"f"``, or ``"v"``.
    use_dof : NDArray[np.integer] or None
        Indices of the active DOFs.  Required for conversions
        involving ``"x"`` or ``"v"`` bases.  When constructing
        in ``"f"`` basis with no conversions needed, may be omitted.
    n_dof : int or None
        Total number of DOFs.  Inferred from ``value`` when
        ``basis="f"``.  Required when constructing in ``"x"`` or
        ``"v"`` basis and converting to ``"f"``.
    Vh : NDArray[np.floating] or None
        Right singular vectors, shape ``(nkeep, len(use_dof))``.
        Required for any conversion involving the ``"v"`` basis.
    """

    value: NDArray[np.floating]
    basis: str = "x"
    use_dof: NDArray[np.integer] | None = None
    n_dof: int | None = None
    Vh: NDArray[np.floating] | None = None

    def __post_init__(self):
        object.__setattr__(self, "value", np.asarray(self.value, dtype=float))
        if self.basis not in self.VALID_BASES:
            raise ValueError(
                f"basis must be one of {self.VALID_BASES}, got {self.basis!r}"
            )
        if self.basis == "f":
            if self.n_dof is None:
                object.__setattr__(self, "n_dof", len(self.value))
            if len(self.value) != self.n_dof:
                raise ValueError(
                    f"Length of value ({len(self.value)}) does not match n_dof ({self.n_dof})"
                )
        if self.basis == "v" and self.Vh is None:
            raise ValueError("Vh must be set when constructing State with basis='v'")

    def _require(self, name: str):
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the State to use this conversion")
        return val

    @property
    def nkeep(self) -> int | None:
        """Number of SVD modes retained.  Only relevant when Vh is set."""
        if self.Vh is not None:
            return self.Vh.shape[0]
        return None

    @property
    def x(self) -> State:
        """State in the active-DOF (x) basis."""
        if self.basis == "x":
            return self
        if self.basis == "f":
            use_dof = self._require("use_dof")
            return replace(self, value=self.value[use_dof], basis="x")
        # basis == "v"
        Vh = self._require("Vh")
        return replace(self, value=self.value @ Vh, basis="x")

    @property
    def f(self) -> State:
        """State in the full-DOF (f) basis."""
        if self.basis == "f":
            return self
        xs = self.x  # go through x-basis
        use_dof = self._require("use_dof")
        n_dof = self._require("n_dof")
        fvalue = np.zeros(n_dof, dtype=float)
        fvalue[use_dof] = xs.value
        return replace(self, value=fvalue, basis="f")

    @property
    def v(self) -> State:
        """State in the SVD-truncated orthogonal (v) basis."""
        if self.basis == "v":
            return self
        Vh = self._require("Vh")
        xs = self.x  # go through x-basis
        return replace(self, value=xs.value @ Vh.T, basis="v")

    def __repr__(self) -> str:
        return f"State({self.value!r}, basis={self.basis!r})"

    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if (
            self.n_dof is not None
            and other.n_dof is not None
            and self.n_dof != other.n_dof
        ):
            raise ValueError("Cannot add States with different n_dof")

        if (
            self.use_dof is not None
            and other.use_dof is not None
            and np.array_equal(self.use_dof, other.use_dof)
        ):
            if (
                self.n_dof is not None
                and other.n_dof is not None
                and self.n_dof == other.n_dof
            ):
                if (
                    self.Vh is not None
                    and other.Vh is not None
                    and np.array_equal(self.Vh, other.Vh)
                ):
                    # Same use_dof, n_dof, and Vh: add in f-basis
                    return replace(
                        self,
                        value=self.f.value + other.f.value,
                        basis="f",
                    )

        return State(
            value=self.f.value + other.f.value,
            basis="f",
            n_dof=self.n_dof
        )


def _sensitivity_to_matrix(sens) -> np.ndarray:
    """Extract a ``(n_obs, n_dof)`` design matrix from a ``Sensitivity``.

    Works for any observable type that defines ``_sensitivity_fields``.
    For ``Sensitivity[Spots]``, vignetted rays (as recorded on the gradient)
    are excluded from the matrix.
    """
    from .spots import Spots

    gradient = sens.gradient
    obs_type = type(gradient)
    n_dof = gradient.batch_shape[0]

    # Build a column mask: True where we keep the observation.
    is_spots = isinstance(sens.gradient, Spots)
    if is_spots:
        valid = ~sens.gradient[0].vignetted.ravel()  # (n_field * n_ray,)
    else:
        valid = None

    chunks = []
    for f in obs_type._sensitivity_fields:
        arr = getattr(gradient, f)
        arr = arr.value if hasattr(arr, "value") else np.asarray(arr, dtype=float)
        arr = arr.reshape(n_dof, -1)  # (n_dof, n_flat_obs)
        if valid is not None:
            arr = arr[:, valid]  # (n_dof, n_valid_obs)
        chunks.append(arr)

    return np.concatenate(chunks, axis=1).T  # (total_obs, n_dof)


class StateFactory:
    """Factory for creating `State` objects with shared SVD context.

    Accepts the full sensitivity matrix ``A`` and computes the SVD
    on construction.  All ``State`` objects produced by this factory
    carry the ``use_dof``, ``n_dof``, and ``Vh`` needed for basis
    conversions.

    Parameters
    ----------
    A : array-like, int, or Sensitivity
        Design matrix.  If an array, should have shape ``(nobs, n_dof)``.
        If an *int* ``n``, an ``n × n`` identity is used.
        If a :class:`~StarSharp.Sensitivity`, the design matrix is extracted
        from the sensitivity fields of the gradient (via ``_sensitivity_fields``),
        flattened and concatenated to ``(nobs, n_dof)``.  For
        ``Sensitivity[Spots]``, vignetted rays are automatically excluded.
    norm : array-like of float
        Normalization factors for each DOF.  The SVD is computed on
        ``A @ np.diag(norm)``.
    use_dof : array-like of int
        Indices of the active DOFs.
    nkeep : int or None
        Number of SVD modes to retain.  Defaults to
        ``len(use_dof)`` (no truncation).
    """

    def __init__(
        self,
        A,
        norm: NDArray[np.floating] | None = None,
        use_dof: NDArray[np.integer] | str | None = None,
        nkeep: int | None = None,
    ):
        if isinstance(A, int):
            A = np.eye(A)
        else:
            # Lazy import to avoid circular dependency (sensitivity.py imports State).
            from .sensitivity import Sensitivity

            if isinstance(A, Sensitivity):
                A = _sensitivity_to_matrix(A)
        if use_dof is None:
            use_dof = np.arange(A.shape[-1])
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
        if norm is None:
            norm = np.ones(A.shape[-1], dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.use_dof = np.asarray(use_dof, dtype=int)
        self.n_dof = self.A.shape[-1]
        A_norm = self.A @ np.diag(norm)
        A_sliced = A_norm[..., self.use_dof].reshape(-1, len(self.use_dof))
        U, S, Vh = np.linalg.svd(A_sliced, full_matrices=False)
        self.U = U
        self.S = S
        self.full_Vh = Vh
        self.nkeep = nkeep if nkeep is not None else len(S)
        self.Vh = Vh[: self.nkeep] @ np.diag(norm[use_dof])
        self.Av = self.A[..., self.use_dof] @ self.Vh.T

    def from_x(self, value) -> State:
        """Create a State from active-DOF coefficients."""
        return State(
            value=value,
            basis="x",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
        )

    def from_f(self, value) -> State:
        """Create a State from full DOF coefficients."""
        return State(
            value=value,
            basis="f",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
        )

    def from_v(self, value) -> State:
        """Create a State from orthogonal-basis coefficients."""
        return State(
            value=value,
            basis="v",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
        )
