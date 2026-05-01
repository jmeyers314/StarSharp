from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, TypeVar, get_args

import numpy as np

from .observable import SensitivityObservable
from .state import Basis, State, StateSchema


ObsT = TypeVar("ObsT", bound=SensitivityObservable)


def _assert_observable_protocol(obj, name: str = "observable") -> None:
    """Validate that *obj* satisfies the observable protocol required by Sensitivity."""
    cls = type(obj)

    if not hasattr(obj, "__len__"):
        raise TypeError(f"{name} must define __len__")
    if not hasattr(obj, "__getitem__"):
        raise TypeError(f"{name} must define __getitem__")

    fields = getattr(cls, "_sensitivity_fields", None)
    if not isinstance(fields, tuple) or not fields or any(
        not isinstance(f, str) for f in fields
    ):
        raise TypeError(
            f"{name} type must define class attribute _sensitivity_fields "
            "as a non-empty tuple[str, ...]"
        )

    bfields = getattr(cls, "_broadcast_fields", ())
    if not isinstance(bfields, tuple) or any(not isinstance(f, str) for f in bfields):
        raise TypeError(
            f"{name} type must define class attribute _broadcast_fields "
            "as tuple[str, ...] when present"
        )


@dataclass(frozen=True)
class Sensitivity(Generic[ObsT]):
    """Linear sensitivity of an observable w.r.t. DOF perturbations.

    ``Sensitivity[T]`` wraps a *nominal* observable of type ``T`` and a
    *gradient* observable of the same type whose leading batch dimension
    indexes degrees of freedom.  The gradient stores
    ``(perturbed - nominal) / step`` for each DOF.

    Parameters
    ----------
    gradient : T
        Per-DOF derivatives.  The leading batch dimension indexes DOFs;
        remaining axes match *nominal*.
    schema : StateSchema
        Shared metadata and conversion context.  Owns ``use_dof``,
        ``n_dof``, ``dof_names``, ``dof_units``, and optionally ``Vh``.
    nominal : T or None
        The unperturbed observable.  If None, inferred from the first
        gradient slice with zero-valued sensitivity fields.
    basis : {'f', 'x', 'v'}
        One of ``"f"``, ``"x"``, or ``"v"``.  Defaults to ``"f"``
        because ``from_finite_differences`` naturally produces a
        full-DOF gradient before any narrowing.

    Notes
    -----
    Basis narrowing (f → x → v) is supported.  Reverse reconstruction
    (v → x or v → f, x → f) is intentionally prohibited because
    projecting to v is lossy when ``n_keep < n_active``.

    Observable protocol
    -------------------
    The type ``T`` bound to ``ObsT`` must satisfy the following protocol:

    * ``T._sensitivity_fields`` — class-level tuple of field names whose
      values are ``Quantity`` arrays with a leading DOF batch dimension in
      the gradient.
    * ``T._broadcast_fields`` — class-level tuple of field names that are
      identical across all DOF slices (e.g. field coordinates).  Optional;
      defaults to ``()``.
    * ``T.__getitem__(i)`` — returns the observable slice for DOF index ``i``,
      used to infer ``nominal`` when it is not supplied explicitly.
    """

    gradient: ObsT
    schema: StateSchema
    nominal: ObsT | None = None
    basis: Basis = "f"

    def __post_init__(self):
        _assert_observable_protocol(self.gradient, name="gradient")
        if self.nominal is not None:
            _assert_observable_protocol(self.nominal, name="nominal")

        basis_values = get_args(Basis)
        if self.basis not in basis_values:
            raise ValueError(
                f"basis must be one of {basis_values}, got {self.basis!r}"
            )

        if not isinstance(self.schema, StateSchema):
            raise TypeError("schema must be a StateSchema")

        if self.basis == "f" and len(self.gradient) != self.schema.n_dof:
            raise ValueError(
                "f-basis Sensitivity gradient length must match schema.n_dof "
                f"(n_dof={self.schema.n_dof}), got {len(self.gradient)}"
            )

        if self.basis == "x" and len(self.gradient) != self.schema.n_active:
            raise ValueError(
                "x-basis Sensitivity gradient length must match schema.n_active "
                f"(len(use_dof)={self.schema.n_active}), got {len(self.gradient)}"
            )

        if self.basis == "v" and len(self.gradient) != self.schema.n_keep:
            raise ValueError(
                "v-basis Sensitivity gradient length must match schema.n_keep "
                f"(n_keep={self.schema.n_keep}), got {len(self.gradient)}"
            )

        if self.nominal is None:
            if len(self.gradient) == 0:
                raise ValueError("gradient must have at least one slice to infer nominal")
            grad0 = self.gradient[0]
            updates = {
                f: np.zeros_like(getattr(grad0, f))
                for f in type(grad0)._sensitivity_fields
            }
            for f in getattr(type(grad0), "_broadcast_fields", ()):
                updates[f] = getattr(grad0, f).copy()
            object.__setattr__(self, "nominal", replace(grad0, **updates))

    def __repr__(self) -> str:
        tname = type(self.gradient).__name__
        return f"Sensitivity[{tname}](basis={self.basis!r})"

    # ------------------------------------------------------------------
    # Metadata passthrough from schema
    # ------------------------------------------------------------------

    @property
    def use_dof(self):
        return self.schema.use_dof

    @property
    def n_dof(self) -> int:
        return self.schema.n_dof

    @property
    def n_active(self) -> int:
        return self.schema.n_active

    def _compatible_schema_for_basis(self, other: StateSchema, basis: Basis) -> bool:
        """Return True if *other* is compatible with this sensitivity's schema for *basis*.

        For f-basis, only ``dof_names`` and ``dof_units`` must match.
        For x-basis, ``use_dof`` must also match.
        For v-basis, ``Vh`` (shape and values) must additionally match.
        """
        if self.schema.dof_names != other.dof_names:
            return False
        if self.schema.dof_units != other.dof_units:
            return False
        if basis in ("x", "v") and not np.array_equal(self.schema.use_dof, other.use_dof):
            return False
        if basis == "v":
            if self.schema.Vh is None or other.Vh is None:
                return False
            if self.schema.Vh.shape != other.Vh.shape:
                return False
            if not np.array_equal(self.schema.Vh, other.Vh):
                return False
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, state: State) -> ObsT:
        """Linear prediction: ``nominal + gradient @ state``.

        Parameters
        ----------
        state : State
            Perturbation in the same basis as this sensitivity.

        Returns
        -------
        T
            New instance of the wrapped observable type.
        """
        if state.basis != self.basis:
            raise ValueError(
                "State basis must match Sensitivity basis for prediction "
                f"(got state.basis={state.basis!r}, sensitivity.basis={self.basis!r})"
            )

        if not self._compatible_schema_for_basis(state.schema, self.basis):
            raise ValueError(
                "State schema is not compatible with Sensitivity schema for "
                f"basis={self.basis!r}"
            )

        weights = state.value
        if len(weights) != len(self.gradient):
            raise ValueError(
                f"Expected {len(self.gradient)} weights for basis={self.basis!r}, "
                f"got {len(weights)}"
            )

        updates = {}
        for field_name in self.gradient._sensitivity_fields:
            grad = getattr(self.gradient, field_name)
            nom = getattr(self.nominal, field_name)
            delta = np.einsum("i...,i->...", grad.value, weights) * grad.unit
            updates[field_name] = nom + delta

        return replace(self.nominal, **updates)

    # ------------------------------------------------------------------
    # Basis narrowing: f → x → v only
    # ------------------------------------------------------------------

    @property
    def f(self) -> Sensitivity[ObsT]:
        """Sensitivity with gradient in the full-DOF (f) basis."""
        if self.basis == "f":
            return self
        raise ValueError(
            f"Reverse basis reconstruction ({self.basis!r} → 'f') is prohibited. "
            "Basis narrowing is one-way: f → x → v."
        )

    @property
    def x(self) -> Sensitivity[ObsT]:
        """Sensitivity with gradient in the active-DOF (x) basis."""
        if self.basis == "x":
            return self
        if self.basis == "v":
            raise ValueError(
                "Reverse basis reconstruction ('v' → 'x') is prohibited. "
                "Basis narrowing is one-way: f → x → v."
            )

        # self.basis == 'f': narrow by selecting active DOFs.
        schema = self.schema
        updates: dict = {}
        for field_name in type(self.gradient)._sensitivity_fields:
            q = getattr(self.gradient, field_name)
            updates[field_name] = q.value[schema.use_dof] * q.unit

        for field_name in getattr(type(self.gradient), "_broadcast_fields", ()):
            val = getattr(self.nominal, field_name)
            updates[field_name] = np.broadcast_to(
                val, (schema.n_active,) + val.shape
            ).copy()

        return replace(self, gradient=replace(self.gradient, **updates), basis="x")

    @property
    def v(self) -> Sensitivity[ObsT]:
        """Sensitivity with gradient in the SVD-truncated (v) basis."""
        if self.basis == "v":
            return self
        schema = self.schema
        if schema.Vh is None:
            raise ValueError("schema.Vh must be set to convert Sensitivity to 'v' basis")

        x_sens = self if self.basis == "x" else self.x
        updates: dict = {}
        for field_name in type(x_sens.gradient)._sensitivity_fields:
            q = getattr(x_sens.gradient, field_name)
            updates[field_name] = np.einsum("ij,j...->i...", schema.Vh, q.value) * q.unit

        for field_name in getattr(type(x_sens.gradient), "_broadcast_fields", ()):
            val = getattr(x_sens.nominal, field_name)
            updates[field_name] = np.broadcast_to(
                val, (schema.n_keep,) + val.shape
            ).copy()

        return replace(x_sens, gradient=replace(x_sens.gradient, **updates), basis="v")

    # ------------------------------------------------------------------
    # Construction from finite differences
    # ------------------------------------------------------------------

    @classmethod
    def from_finite_differences(
        cls,
        nominal: ObsT,
        perturbed_list: list[ObsT],
        steps: State,
    ) -> Sensitivity[ObsT]:
        """Build a Sensitivity from nominal and perturbed observables.

        Parameters
        ----------
        nominal : ObsT
            The unperturbed observable.
        perturbed_list : list[ObsT]
            One perturbed observable per DOF, in the same order as
            ``steps.value``.
        steps : State
            Step sizes.  ``steps.schema`` is propagated to the returned
            Sensitivity.

        Raises
        ------
        TypeError
            If ``nominal`` or any item in ``perturbed_list`` does not satisfy
            the observable protocol required by ``Sensitivity``, or if a
            perturbed item has a different concrete type than ``nominal``.
        ValueError
            If ``len(perturbed_list)`` does not equal ``len(steps.value)``.

        Returns
        -------
        Sensitivity[ObsT]
            Sensitivity object whose ``nominal`` is the input ``nominal``,
            whose ``gradient`` is finite-difference derivatives from
            ``perturbed_list``, and whose ``basis``/``schema`` are copied
            from ``steps``.
        """
        _assert_observable_protocol(nominal, name="nominal")

        for i, perturbed in enumerate(perturbed_list):
            _assert_observable_protocol(perturbed, name=f"perturbed_list[{i}]")
            if type(perturbed) is not type(nominal):
                raise TypeError(
                    "All perturbed observables must have the same concrete type as nominal"
                )

        if len(perturbed_list) != len(steps.value):
            raise ValueError(
                "perturbed_list length must match number of step coefficients: "
                f"got {len(perturbed_list)} vs {len(steps.value)}"
            )

        zero_idx = np.flatnonzero(steps.value == 0).tolist()
        if zero_idx:
            raise ValueError(
                f"All step sizes must be non-zero; got zero at indices {zero_idx}"
            )

        n = len(perturbed_list)
        grad_arrays: dict[str, list] = {f: [] for f in nominal._sensitivity_fields}
        for perturbed, step in zip(perturbed_list, steps.value):
            for f in nominal._sensitivity_fields:
                diff = (getattr(perturbed, f) - getattr(nominal, f)) / step
                grad_arrays[f].append(diff.value)

        updates: dict = {}
        for f in type(nominal)._sensitivity_fields:
            unit = getattr(nominal, f).unit
            updates[f] = np.stack(grad_arrays[f]) * unit

        for f in getattr(type(nominal), "_broadcast_fields", ()):
            val = getattr(nominal, f)
            updates[f] = np.broadcast_to(val, (n,) + val.shape).copy()

        gradient = replace(nominal, **updates)
        return cls(
            nominal=nominal,
            gradient=gradient,
            basis=steps.basis,
            schema=steps.schema,
        )

    # ------------------------------------------------------------------
    # Frame conversions (coordinate frame, not DOF basis)
    # ------------------------------------------------------------------

    @property
    def frame(self) -> str:
        """Coordinate frame of the gradient and nominal (e.g. 'ocs', 'ccs')."""
        return self.gradient.frame

    def _apply_frame(self, frame: str) -> Sensitivity[ObsT]:
        return replace(
            self,
            gradient=getattr(self.gradient, frame),
            nominal=getattr(self.nominal, frame),
        )

    @property
    def ocs(self) -> Sensitivity[ObsT]:
        return self._apply_frame("ocs")

    @property
    def ccs(self) -> Sensitivity[ObsT]:
        return self._apply_frame("ccs")

    @property
    def dvcs(self) -> Sensitivity[ObsT]:
        return self._apply_frame("dvcs")

    @property
    def edcs(self) -> Sensitivity[ObsT]:
        return self._apply_frame("edcs")


def _sensitivity_to_x_matrix(sens: Sensitivity[ObsT]) -> np.ndarray:
    """Extract a design matrix of shape ``(n_obs, n_active)`` from a Sensitivity.

    The input is narrowed to x-basis via ``sens = sens.x``.  This supports
    f-basis inputs and intentionally fails for v-basis inputs per the one-way
    basis narrowing rules.

    Works for any observable type that defines ``_sensitivity_fields``.
    For ``Sensitivity[Spots]``, vignetted rays are excluded from the rows.
    """
    from .spots import Spots

    sens = sens.x
    gradient = sens.gradient
    obs_type = type(gradient)
    n_active = gradient.batch_shape[0]

    is_spots = isinstance(gradient, Spots)
    valid = ~gradient[0].vignetted.ravel() if is_spots else None

    chunks = []
    for field_name in obs_type._sensitivity_fields:
        arr = getattr(gradient, field_name)
        arr = arr.value if hasattr(arr, "value") else np.asarray(arr, dtype=float)
        arr = arr.reshape(n_active, -1)  # (n_active, n_flat_obs)
        # Ray-level fields (e.g. dx/dy) match the vignette mask length;
        # field-level quantities (e.g. x0/y0) do not and should be kept intact.
        if valid is not None and arr.shape[1] == valid.size:
            arr = arr[:, valid]
        chunks.append(arr)

    return np.concatenate(chunks, axis=1).T  # (n_obs, n_active)
