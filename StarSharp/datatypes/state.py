from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, get_args

import astropy.units as u
import numpy as np
from numpy.typing import NDArray


Basis: TypeAlias = Literal["f", "x", "v"]


@dataclass(frozen=True)
class StateSchema:
    """Pure descriptor for the DOF structure shared by State and Sensitivity.

    Owns the DOF metadata and basis-conversion context but does not create
    State objects.  Constructible without a sensitivity matrix so callers
    can bootstrap workflows before sensitivity matrices exist.

    Parameters
    ----------
    dof_names : sequence of str
        Full-DOF names in f-basis order.
    dof_units : sequence of unit-like
        Full-DOF units in f-basis order.  Entries are parsed via
        ``astropy.units.Unit``.
    use_dof : sequence of int or None
        Active DOF indices for x-basis.  If None, all DOFs are active.
    step : ndarray or None
        Default finite-difference step sizes in f-basis order, shape
        ``(n_dof,)``. If None, defaults to all-ones.
    Vh : ndarray or None
        Mode mixing matrix for v-basis.  If None, v-basis is not supported.
        Must have shape (n_keep, n_active) where n_active = len(use_dof).
    S : ndarray or None
        Singular values from the SVD, shape (n_keep,).  Optional; stored for
        diagnostics but not used for basis conversion.
    U : ndarray or None
        Left singular vectors from the SVD, shape (n_obs, n_keep).  Optional;
        stored for diagnostics but not used for basis conversion.
    """

    dof_names: tuple[str, ...]
    dof_units: tuple[u.UnitBase, ...]
    use_dof: NDArray[np.integer] | None = None
    step: NDArray[np.floating] | None = None
    Vh: NDArray[np.floating] | None = None
    S: NDArray[np.floating] | None = None
    U: NDArray[np.floating] | None = None

    def __post_init__(self):
        names = tuple(str(n) for n in self.dof_names)
        units = tuple(u.Unit(unit) for unit in self.dof_units)
        if len(names) == 0:
            raise ValueError("dof_names must not be empty")
        if len(units) != len(names):
            raise ValueError(
                "dof_units must have same length as dof_names, "
                f"got {len(units)} vs {len(names)}"
            )

        if self.use_dof is None:
            use_dof = np.arange(len(names), dtype=int)
        else:
            use_dof = np.asarray(self.use_dof, dtype=int)
            if use_dof.ndim != 1:
                raise ValueError("use_dof must be a 1D index array")
            if np.any(use_dof < 0) or np.any(use_dof >= len(names)):
                raise ValueError("use_dof contains out-of-range indices")

        if self.step is None:
            step = np.ones(len(names), dtype=float)
        else:
            step = np.asarray(self.step, dtype=float)
            if step.ndim != 1 or step.shape[0] != len(names):
                raise ValueError(
                    f"step must have shape (n_dof={len(names)},), got {step.shape}"
                )

        if self.Vh is not None:
            Vh = np.asarray(self.Vh, dtype=float)
            if Vh.ndim != 2 or Vh.shape[1] != len(use_dof):
                raise ValueError(
                    f"Vh must have shape (n_keep, n_active={len(use_dof)}), "
                    f"got {Vh.shape}"
                )
        else:
            Vh = None

        n_keep = Vh.shape[0] if Vh is not None else None

        if self.S is not None:
            S = np.asarray(self.S, dtype=float)
            if S.ndim != 1:
                raise ValueError("S must be a 1D array of singular values")
            if n_keep is not None and S.shape[0] != n_keep:
                raise ValueError(
                    f"S must have length n_keep={n_keep}, got {S.shape[0]}"
                )
        else:
            S = None

        if self.U is not None:
            U = np.asarray(self.U, dtype=float)
            if U.ndim != 2:
                raise ValueError("U must be a 2D array of left singular vectors")
            expected_cols = n_keep if n_keep is not None else (S.shape[0] if S is not None else None)
            if expected_cols is not None and U.shape[1] != expected_cols:
                raise ValueError(
                    f"U must have shape (n_obs, n_keep={expected_cols}), got {U.shape}"
                )
        else:
            U = None

        object.__setattr__(self, "dof_names", names)
        object.__setattr__(self, "dof_units", units)
        object.__setattr__(self, "use_dof", use_dof)
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "Vh", Vh)
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "U", U)

    @property
    def n_dof(self) -> int:
        return len(self.dof_names)

    @property
    def n_active(self) -> int:
        return len(self.use_dof)

    @property
    def n_keep(self) -> int:
        if self.Vh is None:
            raise ValueError("n_keep is not defined: Vh has not been set on this schema")
        return self.Vh.shape[0]

    def with_svd(
        self,
        sensitivity,
        norm: NDArray[np.floating] | None = None,
        n_keep: int | None = None,
    ) -> StateSchema:
        """Return a new StateSchema with Vh computed from an x-basis Sensitivity.

        The SVD is computed on the design matrix ``A @ diag(norm)`` where ``A``
        has shape ``(n_obs, n_active)`` and is extracted from the x-basis
        sensitivity gradient.  ``Vh`` is stored as
        ``Vh_raw[:n_keep] @ diag(norm)`` so that v-basis coefficients have
        units of physical DOF perturbations (i.e. the norm is baked in).

        Parameters
        ----------
        sensitivity : Sensitivity
            An f- or x-basis sensitivity with a schema compatible with this
            one (same ``dof_names``, ``dof_units``, ``use_dof``).  v-basis
            is not accepted because basis narrowing is one-way.
        norm : array-like of float or None
            Per-active-DOF normalisation factors.  May have length
            ``n_active`` or ``n_dof`` (sliced by ``use_dof`` in the latter
            case).  Defaults to all-ones.
        n_keep : int or None
            Number of SVD modes to retain.  Defaults to ``n_active``
            (no truncation).

        Returns
        -------
        StateSchema
            A copy of this schema with ``Vh`` set.
        """
        # Lazy import to avoid circular dependency (sensitivity imports state).
        from .sensitivity import Sensitivity, _sensitivity_to_x_matrix

        if not isinstance(sensitivity, Sensitivity):
            raise TypeError("sensitivity must be a Sensitivity instance")
        if sensitivity.basis == "v":
            raise ValueError(
                "sensitivity must be in f- or x-basis for with_svd; "
                "v-basis is not accepted because basis narrowing is one-way"
            )

        other = sensitivity.schema
        if self.dof_names != other.dof_names:
            raise ValueError("sensitivity.schema.dof_names does not match this schema")
        if self.dof_units != other.dof_units:
            raise ValueError("sensitivity.schema.dof_units does not match this schema")
        if not np.array_equal(self.use_dof, other.use_dof):
            raise ValueError("sensitivity.schema.use_dof does not match this schema")

        A = _sensitivity_to_x_matrix(sensitivity)  # (n_obs, n_active)

        if norm is None:
            norm_x = np.ones(self.n_active, dtype=float)
        else:
            norm = np.asarray(norm, dtype=float)
            if norm.shape == (self.n_dof,):
                norm_x = norm[self.use_dof]
            elif norm.shape == (self.n_active,):
                norm_x = norm
            else:
                raise ValueError(
                    f"norm must have length n_active={self.n_active} or "
                    f"n_dof={self.n_dof}, got length {len(norm)}"
                )

        U_raw, S_raw, Vh_raw = np.linalg.svd(A @ np.diag(norm_x), full_matrices=False)

        if n_keep is None:
            n_keep = self.n_active
        if n_keep > Vh_raw.shape[0]:
            raise ValueError(
                f"n_keep={n_keep} exceeds the number of singular values "
                f"({Vh_raw.shape[0]})"
            )

        Vh_scaled = Vh_raw[:n_keep] @ np.diag(norm_x)
        U_keep = U_raw[:, :n_keep]  # (n_obs, n_keep)
        S_keep = S_raw[:n_keep]     # (n_keep,)
        return replace(self, Vh=Vh_scaled, S=S_keep, U=U_keep)


@dataclass(frozen=True)
class StateFactory:
    """Thin factory for creating `State` objects from a `StateSchema`.

    Parameters
    ----------
    schema : StateSchema
        DOF structure descriptor shared across State and Sensitivity objects.
    """

    schema: StateSchema

    # ------------------------------------------------------------------
    # Passthrough properties so callers can use factory.n_dof etc.
    # ------------------------------------------------------------------

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self.schema.dof_names

    @property
    def dof_units(self) -> tuple[u.UnitBase, ...]:
        return self.schema.dof_units

    @property
    def use_dof(self) -> NDArray[np.integer]:
        return self.schema.use_dof

    @property
    def Vh(self) -> NDArray[np.floating] | None:
        return self.schema.Vh

    @property
    def n_dof(self) -> int:
        return self.schema.n_dof

    @property
    def n_active(self) -> int:
        return self.schema.n_active

    @property
    def n_keep(self) -> int:
        return self.schema.n_keep

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def f(self, value) -> State:
        """Create a State in full-DOF (f) basis.

        Parameters
        ----------
        value : array-like, shape (n_dof,)
            State values for all DOFs.

        Returns
        -------
        State
        """
        return State(value=value, basis="f", schema=self.schema)

    def x(self, value) -> State:
        """Create a State in active-DOF (x) basis.

        Parameters
        ----------
        value : array-like, shape (n_active,)
            State values for the active DOFs selected by ``use_dof``.

        Returns
        -------
        State
        """
        return State(value=value, basis="x", schema=self.schema)

    def v(self, value) -> State:
        """Create a State in SVD mode (v) basis.

        Parameters
        ----------
        value : array-like, shape (n_keep,)
            State values in the truncated SVD mode basis.

        Returns
        -------
        State
        """
        if self.Vh is None:
            raise ValueError("schema.Vh must be set to construct a State in 'v' basis")
        return State(value=value, basis="v", schema=self.schema)

    def by_name(self, **kwargs) -> State:
        """Create a State from named coefficients.

        Accepted key patterns
        ---------------------
        1. Active x-basis DOF names only:
           Keys must be members of ``dof_names[use_dof]``.
        2. v-basis mode names only:
           Keys must match ``vmodeN`` where ``N`` is a 1-indexed integer.

        Mixing DOF names with ``vmodeN`` keys is not allowed.
        Any coefficient not explicitly provided is set to 0.0.
        """
        if not kwargs:
            return self.zero("x")

        active_names = tuple(self.dof_names[i] for i in self.use_dof)
        active_name_to_idx = {name: i for i, name in enumerate(active_names)}

        def _vmode_index(name: str) -> int | None:
            if not name.startswith("vmode"):
                return None
            suffix = name[5:]
            if not suffix.isdigit():
                return None
            idx = int(suffix)
            if idx < 1:
                return None
            return idx

        keys = tuple(kwargs.keys())
        is_active = {k: k in active_name_to_idx for k in keys}
        vmode_index = {k: _vmode_index(k) for k in keys}
        is_vmode = {k: vmode_index[k] is not None for k in keys}

        has_active = any(is_active.values())
        has_vmode = any(is_vmode.values())
        if has_active and has_vmode:
            raise ValueError(
                "Cannot mix active DOF names with vmodeN kwargs in StateFactory.by_name"
            )

        invalid = [k for k in keys if (not is_active[k] and not is_vmode[k])]
        if invalid:
            raise ValueError(
                "Invalid kwargs names for StateFactory.by_name: "
                f"{sorted(invalid)!r}. Expected active DOF names {active_names!r} "
                "or vmodeN keys (1-indexed)."
            )

        if has_vmode:
            if self.Vh is None:
                raise ValueError("schema.Vh must be set to construct a State in 'v' basis")

            vvalue = np.zeros(self.n_keep, dtype=float)
            for k, v in kwargs.items():
                mode_idx = vmode_index[k]
                if mode_idx > self.n_keep:
                    raise ValueError(
                        f"{k!r} is out of range for n_keep={self.n_keep}; "
                        f"valid keys are vmode1..vmode{self.n_keep}"
                    )
                vvalue[mode_idx - 1] = v
            return self.v(vvalue)

        xvalue = np.zeros(self.n_active, dtype=float)
        for k, v in kwargs.items():
            xvalue[active_name_to_idx[k]] = v
        return self.x(xvalue)

    def zero(self, basis: Basis = "f") -> State:
        """Return a zero-valued State in the requested basis.

        Parameters
        ----------
        basis : {'f', 'x', 'v'}, optional
            Basis for the returned state.  Defaults to ``'f'``.

        Returns
        -------
        State
        """
        if basis == "f":
            value = np.zeros(self.n_dof, dtype=float)
        elif basis == "x":
            value = np.zeros(self.n_active, dtype=float)
        elif basis == "v":
            if self.Vh is None:
                raise ValueError("schema.Vh must be set to create a zero v-basis state")
            value = np.zeros(self.n_keep, dtype=float)
        else:
            raise ValueError(f"basis must be one of ('x', 'f', 'v'), got {basis!r}")
        return State(value=value, basis=basis, schema=self.schema)


@dataclass(frozen=True)
class State:
    """Degree-of-freedom state.

    Parameters
    ----------
    value : array-like
        State values in the specified basis.  Shape must be (n_dof,) for
        f-basis, (n_active,) for x-basis, or (n_keep,) for v-basis.
    schema : StateSchema
        DOF descriptor providing metadata and basis-conversion context.
    basis : {'f', 'x', 'v'}, optional
        Basis of the state values.  ``'f'`` is the full-DOF basis, ``'x'``
        is the active-DOF basis, and ``'v'`` is the SVD mode basis defined
        by the schema's ``Vh`` matrix.  Default is ``'x'``.
    """

    value: NDArray[np.floating]
    schema: StateSchema
    basis: Basis = "x"

    def __post_init__(self):
        basis_values = get_args(Basis)
        if self.basis not in basis_values:
            raise ValueError(
                f"basis must be one of {basis_values}, got {self.basis!r}"
            )

        value = np.asarray(self.value, dtype=float)
        object.__setattr__(self, "value", value)

        if not isinstance(self.schema, StateSchema):
            raise TypeError(f"schema must be a StateSchema, got {type(self.schema).__name__}")

        if self.basis == "f" and len(value) != self.schema.n_dof:
            raise ValueError(
                f"f-basis value must have length {self.schema.n_dof}, got {len(value)}"
            )
        if self.basis == "x" and len(value) != self.schema.n_active:
            raise ValueError(
                f"x-basis value must have length {self.schema.n_active}, got {len(value)}"
            )
        if self.basis == "v":
            if self.schema.Vh is None:
                raise ValueError("schema.Vh must be set to construct a State in 'v' basis")
            if len(value) != self.schema.n_keep:
                raise ValueError(
                    f"v-basis value must have length {self.schema.n_keep}, got {len(value)}"
                )

    @property
    def n_dof(self) -> int:
        return self.schema.n_dof

    @property
    def use_dof(self) -> NDArray[np.integer]:
        return self.schema.use_dof

    @property
    def Vh(self) -> NDArray[np.floating] | None:
        return self.schema.Vh

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self.schema.dof_names

    @property
    def dof_units(self) -> tuple[u.UnitBase, ...]:
        return self.schema.dof_units

    @property
    def f(self) -> State:
        """Return this state in full-DOF basis."""
        if self.basis == "f":
            return self

        if self.basis == "v":
            return self.x.f

        schema = self.schema
        fvalue = np.zeros(schema.n_dof, dtype=float)
        fvalue[schema.use_dof] = self.value
        return State(value=fvalue, basis="f", schema=schema)

    @property
    def x(self) -> State:
        """Return this state in active-DOF basis."""
        if self.basis == "x":
            return self

        if self.basis == "v":
            return State(value=self.schema.Vh.T @ self.value, basis="x", schema=self.schema)

        schema = self.schema
        return State(value=self.value[schema.use_dof], basis="x", schema=schema)

    @property
    def v(self) -> State:
        """Return this state in v-basis."""
        if self.basis == "v":
            return self
        schema = self.schema
        if schema.Vh is None:
            raise ValueError("schema.Vh must be set to convert to v-basis")
        return State(value=schema.Vh @ self.x.value, basis="v", schema=schema)

    @property
    def f_names(self) -> tuple[str, ...]:
        """DOF names in full-DOF (f-basis) order."""
        return self.dof_names

    @property
    def f_units(self) -> tuple[u.UnitBase, ...]:
        """DOF units in full-DOF (f-basis) order."""
        return self.dof_units

    @property
    def x_names(self) -> tuple[str, ...]:
        """DOF names for the active DOFs selected by ``use_dof``."""
        return tuple(self.dof_names[i] for i in self.schema.use_dof)

    @property
    def x_units(self) -> tuple[u.UnitBase, ...]:
        """DOF units for the active DOFs selected by ``use_dof``."""
        return tuple(self.dof_units[i] for i in self.schema.use_dof)

    def _compatible_schema(self, other: State) -> bool:
        return (
            self.dof_names == other.dof_names
            and self.dof_units == other.dof_units
        )

    def __add__(self, other):
        """Add two States element-wise, promoting to f-basis.

        Both States must have compatible ``dof_names`` and ``dof_units``.
        The result is always returned in f-basis.  The result schema retains
        ``use_dof`` when both inputs share the same active set; it retains
        ``Vh`` only when both inputs share the same ``Vh``.
        """
        if not isinstance(other, State):
            return NotImplemented
        if not self._compatible_schema(other):
            raise ValueError("Cannot add State objects with incompatible schemas")

        value = self.f.value + other.f.value
        same_use_dof = np.array_equal(self.use_dof, other.use_dof)
        same_Vh = (
            (self.Vh is None and other.Vh is None)
            or (
                self.Vh is not None
                and other.Vh is not None
                and np.array_equal(self.Vh, other.Vh)
            )
        )

        if same_use_dof and same_Vh:
            schema = self.schema
        elif same_use_dof:
            schema = StateSchema(
                dof_names=self.dof_names,
                dof_units=self.dof_units,
                use_dof=self.use_dof,
                Vh=None,
            )
        else:
            schema = StateSchema(dof_names=self.dof_names, dof_units=self.dof_units)
        return State(value=value, basis="f", schema=schema)

    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        return State(value=self.value * scalar, basis=self.basis, schema=self.schema)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return (f"State(value={self.value!r}, basis={self.basis!r})")
