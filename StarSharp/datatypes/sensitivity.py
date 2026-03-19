from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .state import State


@dataclass(frozen=True)
class Sensitivity:
    VALID_BASES = ("x", "f", "v")

    """Linear sensitivity of an observable w.r.t. DOF perturbations.

    ``Sensitivity[T]`` wraps a *nominal* observable of type ``T`` and a
    *gradient* observable of the same type whose leading batch dimension
    indexes degrees of freedom.  The gradient stores
    ``(perturbed - nominal) / step`` for each DOF.

    Supports ``len()``, integer/slice indexing (into the DOF axis), and
    linear prediction via :meth:`predict`.

    Parameters
    ----------
    nominal : T
        The unperturbed observable.
    gradient : T
        Per-DOF derivatives.  The leading batch dimension (size ``ndof``)
        indexes degrees of freedom; remaining axes match *nominal*.
    steps : State
        Step sizes used for the finite differences.  Its ``basis`` defines
        the DOF representation in which the gradient is expressed.
    """

    gradient: object
    nominal: object | None = None
    basis: str = "x"
    use_dof: NDArray[np.integer] | None = None
    n_dof: int | None = None
    Vh: NDArray[np.floating] | None = None

    def __class_getitem__(cls, item):
        return cls

    def __post_init__(self):
        if self.nominal is None:
            # Set nominal to zeroed-out first gradient slice.
            # Copy broadcast fields (e.g. vignetted for Spots) unchanged.
            grad0 = self.gradient[0]
            updates = {
                f: np.zeros_like(getattr(grad0, f))
                for f in grad0._sensitivity_fields
            }
            for f in getattr(type(grad0), "_broadcast_fields", ()):
                updates[f] = getattr(grad0, f).copy()
            nominal = replace(grad0, **updates)
            object.__setattr__(self, "nominal", nominal)
        if self.basis not in self.VALID_BASES:
            raise ValueError(
                f"basis must be one of {self.VALID_BASES}, got {self.basis!r}"
            )
        if self.basis == "f" and self.n_dof is None:
            object.__setattr__(self, "n_dof", len(self.gradient))
        if self.basis == "v" and self.Vh is None:
            raise ValueError("Vh must be set when constructing State with basis='v'")

    def _require(self, name: str):
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the State to use this conversion")
        return val

    def __getitem__(self, idx):
        return self.gradient[idx]

    def predict(self, state: State):
        """Linear prediction: ``nominal + gradient @ state``.

        *state* is a perturbation (delta) expressed in any basis.  It is
        converted to the same basis as ``self.steps`` before combining.

        Returns a new instance of the wrapped datatype.
        """
        # TODO: check that state conversion attrs match self conversion attrs
        weights = getattr(state, self.basis).value
        if len(weights) != len(self.gradient):
            raise ValueError(f"Expected {len(self.gradient)} weights, got {len(weights)}")

        updates = {}
        for field_name in self.gradient._sensitivity_fields:
            grad = getattr(self.gradient, field_name)
            nom = getattr(self.nominal, field_name)
            delta = np.einsum("i...,i->...", grad.value, weights) * grad.unit
            updates[field_name] = nom + delta

        return replace(self.nominal, **updates)

    @classmethod
    def from_finite_differences(
        cls,
        nominal,
        perturbed_list: list,
        steps: State,
    ) -> Sensitivity:
        """Build a ``Sensitivity`` from nominal and perturbed observables.

        Parameters
        ----------
        nominal : T
            The unperturbed observable.
        perturbed_list : list[T]
            One perturbed observable per DOF, in the same order as
            ``steps.value``.
        steps : State
            Step sizes used for the finite differences.
        """
        n = len(perturbed_list)

        # Compute (perturbed - nominal) / step for each sensitivity field
        grad_arrays: dict[str, list] = {f: [] for f in nominal._sensitivity_fields}
        for perturbed, step in zip(perturbed_list, steps.value):
            for f in nominal._sensitivity_fields:
                diff = (getattr(perturbed, f) - getattr(nominal, f)) / step
                grad_arrays[f].append(diff.value)

        updates: dict[str, object] = {}
        for f in nominal._sensitivity_fields:
            unit = getattr(nominal, f).unit
            updates[f] = np.stack(grad_arrays[f]) * unit

        # Broadcast ancillary array fields (e.g. vignetted for Spots)
        for f in getattr(nominal, "_broadcast_fields", ()):
            val = getattr(nominal, f)
            updates[f] = np.broadcast_to(val, (n,) + val.shape).copy()

        gradient = replace(nominal, **updates)
        return cls(
            nominal=nominal,
            gradient=gradient,
            basis=steps.basis,
            use_dof=steps.use_dof,
            n_dof=steps.n_dof,
            Vh=steps.Vh,
        )

    def __repr__(self) -> str:
        tname = type(self.gradient).__name__
        return f"Sensitivity[{tname}](basis={self.basis!r})"

    # ------------------------------------------------------------------
    # Frame conversions
    # ------------------------------------------------------------------

    @property
    def frame(self) -> str:
        """Coordinate frame of the gradient and nominal (e.g. 'ocs', 'ccs')."""
        return self.gradient.frame

    def _apply_frame(self, frame: str) -> Sensitivity:
        """Return a new Sensitivity with gradient and nominal converted to *frame*."""
        return replace(
            self,
            gradient=getattr(self.gradient, frame),
            nominal=getattr(self.nominal, frame),
        )

    @property
    def ocs(self) -> Sensitivity:
        """Sensitivity with gradient and nominal in the OCS frame."""
        return self._apply_frame("ocs")

    @property
    def ccs(self) -> Sensitivity:
        """Sensitivity with gradient and nominal in the CCS frame."""
        return self._apply_frame("ccs")

    @property
    def dvcs(self) -> Sensitivity:
        """Sensitivity with gradient and nominal in the DVCS frame (Spots only)."""
        return self._apply_frame("dvcs")

    @property
    def edcs(self) -> Sensitivity:
        """Sensitivity with gradient and nominal in the EDCS frame (Spots only)."""
        return self._apply_frame("edcs")
