from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


VALID_BASES = ("x", "f", "v")


@dataclass(frozen=True)
class State:
    """Alignment state in one of three bases.

    Analogous to `FieldCoords` with its ``frame`` attribute:
    ``state`` holds the coefficient vector and ``basis`` identifies
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
        Only available when ``Vh`` (and optionally ``nkeep``) are set.
        The roundtrip ``x → v → x`` is lossy when
        ``nkeep < len(use_dof)``, though ``v → x → v`` is lossless.

    Parameters
    ----------
    state : NDArray[np.floating]
        Coefficient vector in the basis given by ``basis``.
    basis : str
        One of ``"x"``, ``"f"``, or ``"v"``.
    use_dof : NDArray[np.integer] or None
        Indices of the active DOFs.  Required for conversions
        involving ``"x"`` or ``"v"`` bases.  When constructing
        in ``"f"`` basis with no conversions needed, may be omitted.
    n_dof : int or None
        Total number of DOFs.  Inferred from ``state`` when
        ``basis="f"``.  Required when constructing in ``"x"`` or
        ``"v"`` basis and converting to ``"f"``.
    Vh : NDArray[np.floating] or None
        Right singular vectors, shape ``(len(use_dof), len(use_dof))``.
        Required for any conversion involving the ``"v"`` basis.
    nkeep : int or None
        Number of SVD modes retained.  Defaults to ``Vh.shape[0]``
        when ``Vh`` is provided.
    """

    state: NDArray[np.floating]
    basis: str = "x"
    use_dof: NDArray[np.integer] | None = None
    n_dof: int | None = None
    Vh: NDArray[np.floating] | None = None
    nkeep: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "state", np.asarray(self.state, dtype=float))
        if self.basis not in VALID_BASES:
            raise ValueError(f"basis must be one of {VALID_BASES}, got {self.basis!r}")
        if self.basis == "f" and self.n_dof is None:
            object.__setattr__(self, "n_dof", len(self.state))
        if self.basis == "v" and self.Vh is None:
            raise ValueError(
                "Vh (and nkeep) must be set when constructing State with basis='v'"
            )
        if self.Vh is not None and self.nkeep is None:
            object.__setattr__(self, "nkeep", self.Vh.shape[0])

    def _require(self, name: str):
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the State to use this conversion")
        return val

    @property
    def x(self) -> State:
        """State in the active-DOF (x) basis."""
        if self.basis == "x":
            return self
        if self.basis == "f":
            use_dof = self._require("use_dof")
            return State(
                state=self.state[use_dof],
                basis="x",
                use_dof=self.use_dof,
                n_dof=self.n_dof,
                Vh=self.Vh,
                nkeep=self.nkeep,
            )
        # basis == "v"
        Vh = self._require("Vh")
        nkeep = self._require("nkeep")
        return State(
            state=self.state @ Vh[:nkeep],
            basis="x",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
            nkeep=self.nkeep,
        )

    @property
    def f(self) -> State:
        """State in the full-DOF (f) basis."""
        if self.basis == "f":
            return self
        xs = self.x  # go through x-basis
        use_dof = self._require("use_dof")
        n_dof = self._require("n_dof")
        fstate = np.zeros(n_dof, dtype=float)
        fstate[use_dof] = xs.state
        return State(
            state=fstate,
            basis="f",
            use_dof=self.use_dof,
            n_dof=n_dof,
            Vh=self.Vh,
            nkeep=self.nkeep,
        )

    @property
    def v(self) -> State:
        """State in the SVD-truncated orthogonal (v) basis."""
        if self.basis == "v":
            return self
        Vh = self._require("Vh")
        nkeep = self._require("nkeep")
        xs = self.x  # go through x-basis
        return State(
            state=xs.state @ Vh[:nkeep].T,
            basis="v",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
            nkeep=self.nkeep,
        )

    def __repr__(self) -> str:
        return f"State({self.state!r}, basis={self.basis!r})"


class StateFactory:
    """Factory for creating `State` objects with shared SVD context.

    Accepts the full sensitivity matrix ``A`` and computes the SVD
    on construction.  All ``State`` objects produced by this factory
    carry the ``use_dof``, ``n_dof``, ``Vh``, and ``nkeep`` needed
    for basis conversions.

    Parameters
    ----------
    A : array-like
        Sensitivity matrix.  The last axis has length ``n_dof``.
        All other axes are flattened before SVD.
    use_dof : array-like of int
        Indices of the active DOFs.
    nkeep : int or None
        Number of SVD modes to retain.  Defaults to
        ``len(use_dof)`` (no truncation).
    """

    def __init__(
        self,
        A: NDArray[np.floating],
        use_dof: NDArray[np.integer],
        nkeep: int | None = None,
    ):
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
        self.A = np.asarray(A, dtype=float)
        self.use_dof = np.asarray(use_dof, dtype=int)
        self.n_dof = self.A.shape[-1]
        A_sliced = self.A[..., self.use_dof].reshape(-1, len(self.use_dof))
        U, S, Vh = np.linalg.svd(A_sliced, full_matrices=False)
        self.U = U
        self.S = S
        self.Vh = Vh
        self.nkeep = nkeep if nkeep is not None else len(S)

    def from_x(self, state) -> State:
        """Create a State from active-DOF coefficients."""
        return State(
            state=state,
            basis="x",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
            nkeep=self.nkeep,
        )

    def from_f(self, state) -> State:
        """Create a State from full DOF coefficients."""
        return State(
            state=state,
            basis="f",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
            nkeep=self.nkeep,
        )

    def from_v(self, state) -> State:
        """Create a State from orthogonal-basis coefficients."""
        return State(
            state=state,
            basis="v",
            use_dof=self.use_dof,
            n_dof=self.n_dof,
            Vh=self.Vh,
            nkeep=self.nkeep,
        )
