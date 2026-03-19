from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .state import State


@dataclass(frozen=True)
class Sensitivity:
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
    steps: State | None = None

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

    @property
    def ndof(self) -> int:
        return self.gradient.batch_shape[0]

    def __len__(self) -> int:
        return self.ndof

    def __getitem__(self, idx):
        return self.gradient[idx]

    def predict(self, state: State):
        """Linear prediction: ``nominal + gradient @ state``.

        *state* is a perturbation (delta) expressed in any basis.  It is
        converted to the same basis as ``self.steps`` before combining.

        Returns a new instance of the wrapped datatype.
        """
        weights = getattr(state, self.steps.basis).value
        if len(weights) != self.ndof:
            raise ValueError(f"Expected {self.ndof} weights, got {len(weights)}")

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
        ndof = len(perturbed_list)

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
            updates[f] = np.broadcast_to(val, (ndof,) + val.shape).copy()

        gradient = replace(nominal, **updates)
        return cls(nominal=nominal, gradient=gradient, steps=steps)

    def __repr__(self) -> str:
        tname = type(self.gradient).__name__
        return f"Sensitivity[{tname}](ndof={self.ndof})"
