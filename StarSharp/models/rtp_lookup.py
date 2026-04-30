"""RTP-indexed AOS offset lookup table.

Reads a focus-scan ECSV (produced by ``sandbox/focus_scan.py``) and provides
linearly-interpolated AOS state corrections at an arbitrary rotator position
angle.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.table import QTable

from ..datatypes import State, StateFactory, StateSchema


class RTPLookup:
    """Interpolated AOS offset as a function of rotator position angle.

    The lookup table is a QTable with an ``rtp`` column (Quantity, deg) and one
    column per AOS DOF (named to match the DOF names in a :class:`StateSchema`).
    Columns whose names are not found in the target schema are silently ignored,
    so a table produced with the 5-DOF set can be used with a model that only
    knows about 3 of those DOFs.

    Interpolation is linear with clamping at the boundary RTP values.

    Parameters
    ----------
    table : QTable
        Lookup table.  Must have an ``rtp`` column and at least one DOF column.
    """

    def __init__(self, table: QTable) -> None:
        if "rtp" not in table.colnames:
            raise ValueError("table must have an 'rtp' column")
        self._table = table.copy()
        self._table.sort("rtp")
        self._table["rtp_deg"] = self._table["rtp"].to_value(u.deg)
        _skip = {"rtp", "rtp_deg", "rms_nominal", "rms_optimized"}
        self._dof_cols = [col for col in self._table.colnames if col not in _skip]

    @classmethod
    def from_file(cls, path: str | Path) -> RTPLookup:
        """Load a lookup table from an ECSV file.

        Parameters
        ----------
        path : str or Path
            Path to the ECSV file produced by ``focus_scan.py``.
        """
        table = QTable.read(path)
        return cls(table)

    def state_at(self, rtp: Angle, schema: StateSchema) -> State:
        """Interpolate the AOS offset at *rtp* and return it as a State.

        Parameters
        ----------
        rtp : Angle
            Rotator position angle at which to evaluate the lookup.
        schema : StateSchema
            Target schema.  Must be a superset of the table's DOF columns —
            every DOF column in the lookup table must appear in *schema*, or a
            ``ValueError`` is raised.  DOFs in *schema* not present in the
            table are left at zero.

        Returns
        -------
        State
            Full f-basis State (``basis='f'``) compatible with *schema*.
        """
        rtp_deg = float(rtp.deg)
        rtp_arr = self._table["rtp_deg"].value

        i = np.clip(np.searchsorted(rtp_arr, rtp_deg, side="right") - 1, 0, len(rtp_arr) - 2)
        t = np.clip((rtp_deg - rtp_arr[i]) / (rtp_arr[i + 1] - rtp_arr[i]), 0.0, 1.0)

        dof_cols = self._dof_cols
        missing = [col for col in dof_cols if col not in schema.dof_names]
        if missing:
            raise ValueError(
                f"RTPLookup columns not found in schema: {missing}"
            )
        values = np.column_stack([
            self._table[col].to(schema.dof_units[schema.dof_names.index(col)]).value
            for col in dof_cols
        ])  # shape (n_rtp, n_cols)

        interp_vals = values[i] * (1 - t) + values[i + 1] * t
        return StateFactory(schema).by_name(**dict(zip(dof_cols, interp_vals))).f
