from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.table import QTable
from astropy.units import UnitConversionError
from astropy.units import Quantity

from .state import State, StateSchema


@dataclass(frozen=True)
class PointingModel:
    """Linear state-to-pointing mapping for raytrace field-angle inputs.

    Parameters
    ----------
    schema : StateSchema
        DOF schema describing the columns of ``matrix``.
    matrix : Quantity
        Shape ``(2, n_dof)`` matrix with angular units.  The first row maps
        state DOFs to ``dtheta_x`` and the second to ``dtheta_y``:
        ``[dtheta_x, dtheta_y] = matrix @ state.f.value``.
    """

    schema: StateSchema
    matrix: Quantity

    def __post_init__(self):
        if not isinstance(self.schema, StateSchema):
            raise TypeError("schema must be a StateSchema")

        # Only names/units matter for PointingModel compatibility.
        schema = StateSchema(
            dof_names=self.schema.dof_names,
            dof_units=self.schema.dof_units,
        )

        if not isinstance(self.matrix, Quantity):
            raise TypeError("matrix must be an astropy Quantity with angular units")

        arr = np.atleast_2d(self.matrix)
        if arr.shape != (2, schema.n_dof):
            raise ValueError(
                f"matrix must have shape (2, n_dof={schema.n_dof}), got {arr.shape}"
            )

        if arr.unit.physical_type != "angle":
            raise ValueError(
                "matrix must have angular units, "
                f"got physical type {arr.unit.physical_type!r}"
            )

        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "matrix", arr)

    def aligned(self, schema: StateSchema, strict: bool = True) -> PointingModel:
        """Return this model reindexed/rescaled to *schema*.

        Parameters
        ----------
        schema : StateSchema
            Target schema to align against.
        strict : bool, optional
            If True, every target DOF name must exist in this pointing model's
            schema. If False, missing DOFs are assigned zero response.
        """
        if not isinstance(schema, StateSchema):
            raise TypeError("schema must be a StateSchema")

        idx_by_name = {name: i for i, name in enumerate(self.schema.dof_names)}
        aligned = np.zeros((2, schema.n_dof), dtype=float) * self.matrix.unit
        missing: list[str] = []

        for j, (name, target_unit) in enumerate(zip(schema.dof_names, schema.dof_units)):
            i = idx_by_name.get(name)
            if i is None:
                missing.append(name)
                continue

            pm_unit = self.schema.dof_units[i]
            try:
                # state values are in target_unit; convert to PM column unit.
                scale = (1.0 * target_unit).to_value(pm_unit)
            except UnitConversionError as exc:
                raise ValueError(
                    f"Incompatible DOF units for {name!r}: "
                    f"target={target_unit}, pointing_model={pm_unit}"
                ) from exc

            aligned[:, j] = self.matrix[:, i] * scale

        if strict and missing:
            raise ValueError(
                "PointingModel schema does not cover target schema DOFs: "
                f"{missing!r}"
            )

        return PointingModel(schema=schema, matrix=aligned)

    def _delta(self, state: State) -> tuple[Quantity, Quantity]:
        """Return ``(dtheta_x, dtheta_y)`` assuming schema is already aligned."""
        dtheta = self.matrix.to_value(u.rad) @ state.f.value
        return dtheta[0] * u.rad, dtheta[1] * u.rad

    def delta(self, state: State, strict: bool = True) -> tuple[Quantity, Quantity]:
        """Return global ``(dtheta_x, dtheta_y)`` in radians for a given state."""
        aligned_pm = self.aligned(state.schema, strict=strict)
        return aligned_pm._delta(state)

    def to_table(self) -> QTable:
        """Convert pointing model to an astropy QTable.

        Returns
        -------
        QTable
            Table with one row per DOF and columns ``dof_name``, ``dof_unit``,
            ``dtheta_x``, and ``dtheta_y``.
        """
        return QTable(
            {
                "dof_name": self.schema.dof_names,
                "dof_unit": [unit.to_string() for unit in self.schema.dof_units],
                "dtheta_x": self.matrix[0],
                "dtheta_y": self.matrix[1],
            },
        )

    @classmethod
    def from_table(cls, table: QTable) -> PointingModel:
        """Create pointing model from an astropy QTable.

        Parameters
        ----------
        table : QTable
            Table with columns ``dof_name``, ``dof_unit``, ``dtheta_x``,
            and ``dtheta_y``.

        Returns
        -------
        PointingModel
        """
        required = ("dof_name", "dof_unit", "dtheta_x", "dtheta_y")
        if any(name not in table.colnames for name in required):
            raise ValueError(
                "Table must contain columns 'dof_name', 'dof_unit', "
                "'dtheta_x', and 'dtheta_y'"
            )

        dof_name = tuple(str(name) for name in table["dof_name"])
        dof_unit = tuple(u.Unit(unit) for unit in table["dof_unit"])
        dtheta_x = table["dtheta_x"]
        dtheta_y = table["dtheta_y"]

        if len(dtheta_x) != len(dof_name) or len(dtheta_y) != len(dof_name):
            raise ValueError(
                f"Columns must have same length, got dof_name={len(dof_name)}, "
                f"dtheta_x={len(dtheta_x)}, dtheta_y={len(dtheta_y)}"
            )

        schema = StateSchema(dof_names=dof_name, dof_units=dof_unit)
        matrix = np.vstack([dtheta_x, dtheta_y])
        return cls(schema=schema, matrix=matrix)
