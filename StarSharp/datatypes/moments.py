from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity

from .field_coords import FieldCoords


def _moment_names(order: int) -> list[str]:
    """Return the ordered list of symmetric component names for a given order.

    E.g. order 2 → ['xx', 'xy', 'yy'], order 3 → ['xxx', 'xxy', 'xyy', 'yyy'].
    """
    return ["".join(p) for p in itertools.combinations_with_replacement("xy", order)]


@dataclass(frozen=True)
class Moments:
    """Container for 2D image moments of order n.

    Use ``Moments[n]`` to obtain the concrete class for orders 2, 3, and 4
    (``Moments2``, ``Moments3``, ``Moments4``), which accept named component
    kwargs and expose them as properties.  For arbitrary orders, construct
    directly with ``order`` and ``values``.

    Parameters
    ----------
    order : int
        Moment order (must be >= 1).
    values : Quantity
        Component array.  The first axis indexes the ``order + 1`` symmetric
        components in ``combinations_with_replacement("xy", order)`` order
        (e.g. xx, xy, yy for order 2).  Remaining axes are field / spatial
        dimensions.
    frame : str
        Coordinate frame: ``"ocs"``, ``"ccs"``, ``"dvcs"``, or ``"edcs"``.
    field : FieldCoords or None
        Field coordinates corresponding to the moments.
    rtp : Angle or None
        Rotation angle from OCS to CCS.  Required for frame conversions.
    """

    VALID_FRAMES = ("ocs", "ccs", "dvcs", "edcs")

    order: int
    values: Quantity
    frame: str = "ocs"
    field: FieldCoords | None = None
    rtp: Angle | None = None

    def __post_init__(self):
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        frame = self.frame.lower()
        object.__setattr__(self, "frame", frame)
        if frame not in self.VALID_FRAMES:
            raise ValueError(
                f"frame must be one of {self.VALID_FRAMES}, got {self.frame!r}"
            )
        values = np.atleast_1d(self.values)
        n_components = self.order + 1
        if values.shape[0] != n_components:
            raise ValueError(
                f"values.shape[0] must be {n_components} for order {self.order}, "
                f"got {values.shape[0]}"
            )
        object.__setattr__(self, "values", values)

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    @property
    def _moment_order(self) -> int:
        """Backward-compatible alias for ``order``."""
        return self.order

    @property
    def _moment_names(self) -> list[str]:
        """Ordered list of component names (e.g. ['xx', 'xy', 'yy'] for order 2)."""
        return _moment_names(self.order)

    # ------------------------------------------------------------------
    # Named component access for arbitrary orders
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Quantity:
        # Called only when normal attribute lookup fails.
        # Provides named access to moment components (e.g. .xx, .xy, .yy).
        try:
            order = object.__getattribute__(self, "order")
            values = object.__getattribute__(self, "values")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        names = _moment_names(order)
        if name in names:
            return values[names.index(name)]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Concrete-class lookup: Moments[2] → Moments2, etc.
    # ------------------------------------------------------------------

    @classmethod
    def __class_getitem__(cls, order: int) -> type:
        """Return the concrete moment class for a given order."""
        # Dict built lazily (Moments2/3/4 are defined after Moments).
        registry = {2: Moments2, 3: Moments3, 4: Moments4}
        return registry.get(order, cls)

    # ------------------------------------------------------------------
    # Internal constructor (bypasses named-kwargs __init__ of subclasses)
    # ------------------------------------------------------------------

    @classmethod
    def _create(
        cls,
        order: int,
        values: Quantity,
        frame: str,
        field: FieldCoords | None,
        rtp: Angle | None,
    ) -> Moments:
        """Construct from a pre-built values array.  Used by frame-conversion methods."""
        return cls(order=order, values=values, frame=frame, field=field, rtp=rtp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require(self, name: str) -> Any:
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Moments to use this property")
        return val

    def _tensor(self) -> np.ndarray:
        """Build the full (non-symmetric) moment tensor.

        Shape is ``(*field_shape, 2, ..., 2)`` with ``order`` trailing axes.
        """
        order = self.order
        names = _moment_names(order)
        name_to_idx = {n: i for i, n in enumerate(names)}
        field_shape = self.values.shape[1:]
        T = np.zeros(
            field_shape + (2,) * order,
            dtype=np.result_type(self.values.value, float),
        )
        for idx_tuple in itertools.product([0, 1], repeat=order):
            canonical = "".join(sorted("xy"[i] for i in idx_tuple))
            T[(...,) + idx_tuple] = self.values.value[name_to_idx[canonical]]
        return T

    # ------------------------------------------------------------------
    # Frame conversions
    # ------------------------------------------------------------------

    def _rot(self, angle: Angle, frame: str) -> Moments:
        """Return a new Moments with all components rotated by *angle*."""
        c = float(np.cos(angle.rad))
        s = float(np.sin(angle.rad))
        R = np.array([[c, s], [-s, c]])
        order = self.order

        T = self._tensor()
        T_rot = T.copy()
        for axis in range(order):
            axis_to_contract = -(order - axis)
            T_rot = np.tensordot(R, T_rot, axes=([1], [axis_to_contract]))
            T_rot = np.moveaxis(T_rot, 0, axis_to_contract)

        names = _moment_names(order)
        unit = self.values.unit
        new_values = np.stack(
            [T_rot[(...,) + tuple(0 if ch == "x" else 1 for ch in name)] for name in names],
            axis=0,
        ) * unit
        return type(self)._create(order, new_values, frame, self.field, self.rtp)

    def _swap(self, frame: str) -> Moments:
        """Return a new Moments with x and y swapped (reflection x <-> y)."""
        order = self.order
        names = _moment_names(order)
        name_to_idx = {n: i for i, n in enumerate(names)}
        indices = [
            name_to_idx["".join(sorted("y" if ch == "x" else "x" for ch in name))]
            for name in names
        ]
        return type(self)._create(order, self.values[indices], frame, self.field, self.rtp)

    def _relabel(self, frame: str) -> Moments:
        """Return an identical Moments with a different frame label."""
        return type(self)._create(self.order, self.values, frame, self.field, self.rtp)

    @property
    def ocs(self) -> Moments:
        """These moments in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self.ccs._rot(-rtp, "ocs")

    @property
    def ccs(self) -> Moments:
        """These moments in the CCS frame."""
        if self.frame == "ccs":
            return self
        if self.frame == "edcs":
            return self._relabel("ccs")
        if self.frame == "dvcs":
            return self._swap("ccs")
        rtp = self._require("rtp")
        return self._rot(rtp, "ccs")

    @property
    def edcs(self) -> Moments:
        """These moments in the EDCS frame (synonym for CCS, preserves name)."""
        if self.frame == "edcs":
            return self
        return self.ccs._relabel("edcs")

    @property
    def dvcs(self) -> Moments:
        """These moments in the DVCS frame (transpose of EDCS/CCS)."""
        if self.frame == "dvcs":
            return self
        return self.ccs._swap("dvcs")

    # ------------------------------------------------------------------
    # Spin decomposition
    # ------------------------------------------------------------------

    def spin_complex(self, m: int) -> Quantity:
        """Return the complex spin-m moment for this order-N Moments (m >= 0).

        Let N = self.order and z = x + i y. For a given spin m, define
        p = (N+m)/2 and q = (N-m)/2 (requires same parity). Then:

            M_m = < z^p zbar^q >

        Parameters
        ----------
        m : int
            Spin magnitude.  Must satisfy 0 <= m <= N with same parity as N.

        Returns
        -------
        Quantity (complex-valued)
        """
        N = self.order
        if m < 0:
            raise ValueError("m must be >= 0 for spin_complex()")
        if m > N or (m % 2) != (N % 2):
            raise ValueError(
                f"Invalid spin m={m} for order-{N} moments; "
                f"m must satisfy 0 <= m <= {N} with same parity as {N}"
            )

        p = (N + m) // 2
        q = (N - m) // 2

        T = self._tensor()
        unit = self.values.unit

        ez = np.array([1.0, 1.0j])
        ezbar = np.array([1.0, -1.0j])

        result = T
        for _ in range(p):
            result = np.tensordot(result, ez, axes=([-1], [0]))
        for _ in range(q):
            result = np.tensordot(result, ezbar, axes=([-1], [0]))

        return result * unit

    def spin_pair(self, m: int) -> tuple:
        """Return (cos, sin) components for spin m (m >= 0).

        For m=0, returns (scalar, None).
        For m>0, returns (Re(M_m), Im(M_m)).
        """
        M = self.spin_complex(m)
        if m == 0:
            return M.real, None
        return M.real, M.imag

    def spin_cos(self, m: int) -> Quantity:
        """Cosine (real) component of spin m (m >= 0)."""
        return self.spin_complex(m).real

    def spin_sin(self, m: int) -> Quantity:
        """Sine (imaginary) component of spin m (m > 0)."""
        if m <= 0:
            raise ValueError("m must be > 0 for spin_sin()")
        return self.spin_complex(m).imag

    def spin(self, m: int) -> Quantity:
        """Alias for spin_cos(m) when m >= 0, spin_sin(-m) when m < 0."""
        if m >= 0:
            return self.spin_cos(m)
        else:
            return self.spin_sin(-m)


# ---------------------------------------------------------------------------
# Concrete subclasses for orders 2, 3, 4
# ---------------------------------------------------------------------------

class Moments2(Moments):
    """Second-order moments.

    Components: ``xx``, ``xy``, ``yy``.

    Spin decomposition:
      - Spin-0 (scalar): ``xx + yy``
      - Spin-2 components: ``xx - yy``, ``2*xy``
    """

    def __init__(
        self,
        xx: Quantity,
        xy: Quantity,
        yy: Quantity,
        frame: str = "ocs",
        field: FieldCoords | None = None,
        rtp: Angle | None = None,
    ) -> None:
        unit = xx.unit
        values = np.stack([xx.to(unit).value, xy.to(unit).value, yy.to(unit).value]) * unit
        Moments.__init__(self, order=2, values=values, frame=frame, field=field, rtp=rtp)

    @classmethod
    def _create(cls, order, values, frame, field, rtp):
        obj = object.__new__(cls)
        Moments.__init__(obj, order=order, values=values, frame=frame, field=field, rtp=rtp)
        return obj

    @property
    def xx(self) -> Quantity:
        return self.values[0]

    @property
    def xy(self) -> Quantity:
        return self.values[1]

    @property
    def yy(self) -> Quantity:
        return self.values[2]

    @property
    def T(self) -> Quantity:
        """Trace: xx + yy."""
        return self.xx + self.yy

    @property
    def e1(self) -> float:
        """Ellipticity e1: (xx - yy) / T."""
        return ((self.xx - self.yy) / self.T).value

    @property
    def e2(self) -> float:
        """Ellipticity e2: 2 * xy / T."""
        return (2 * self.xy / self.T).value


class Moments3(Moments):
    """Third-order moments.

    Components: ``xxx``, ``xxy``, ``xyy``, ``yyy``.

    Spin decomposition (cos, sin from Re/Im of M_m):
      - Spin-1: Re<z^2 zbar> = xxx + xyy, Im<z^2 zbar> = xxy + yyy
      - Spin-3: Re<z^3> = xxx - 3*xyy, Im<z^3> = 3*xxy - yyy
    """

    def __init__(
        self,
        xxx: Quantity,
        xxy: Quantity,
        xyy: Quantity,
        yyy: Quantity,
        frame: str = "ocs",
        field: FieldCoords | None = None,
        rtp: Angle | None = None,
    ) -> None:
        unit = xxx.unit
        values = np.stack([
            xxx.to(unit).value,
            xxy.to(unit).value,
            xyy.to(unit).value,
            yyy.to(unit).value,
        ]) * unit
        Moments.__init__(self, order=3, values=values, frame=frame, field=field, rtp=rtp)

    @classmethod
    def _create(cls, order, values, frame, field, rtp):
        obj = object.__new__(cls)
        Moments.__init__(obj, order=order, values=values, frame=frame, field=field, rtp=rtp)
        return obj

    @property
    def xxx(self) -> Quantity:
        return self.values[0]

    @property
    def xxy(self) -> Quantity:
        return self.values[1]

    @property
    def xyy(self) -> Quantity:
        return self.values[2]

    @property
    def yyy(self) -> Quantity:
        return self.values[3]


class Moments4(Moments):
    """Fourth-order moments.

    Components: ``xxxx``, ``xxxy``, ``xxyy``, ``xyyy``, ``yyyy``.

    Spin decomposition:
      - Spin-0 (scalar): Re<z^2 zbar^2> = xxxx + 2*xxyy + yyyy
      - Spin-2: Re<z^3 zbar> = xxxx - yyyy, Im<z^3 zbar> = 2*(xxxy + xyyy)
      - Spin-4: Re<z^4> = xxxx - 6*xxyy + yyyy, Im<z^4> = 4*(xxxy - xyyy)
    """

    def __init__(
        self,
        xxxx: Quantity,
        xxxy: Quantity,
        xxyy: Quantity,
        xyyy: Quantity,
        yyyy: Quantity,
        frame: str = "ocs",
        field: FieldCoords | None = None,
        rtp: Angle | None = None,
    ) -> None:
        unit = xxxx.unit
        values = np.stack([
            xxxx.to(unit).value,
            xxxy.to(unit).value,
            xxyy.to(unit).value,
            xyyy.to(unit).value,
            yyyy.to(unit).value,
        ]) * unit
        Moments.__init__(self, order=4, values=values, frame=frame, field=field, rtp=rtp)

    @classmethod
    def _create(cls, order, values, frame, field, rtp):
        obj = object.__new__(cls)
        Moments.__init__(obj, order=order, values=values, frame=frame, field=field, rtp=rtp)
        return obj

    @property
    def xxxx(self) -> Quantity:
        return self.values[0]

    @property
    def xxxy(self) -> Quantity:
        return self.values[1]

    @property
    def xxyy(self) -> Quantity:
        return self.values[2]

    @property
    def xyyy(self) -> Quantity:
        return self.values[3]

    @property
    def yyyy(self) -> Quantity:
        return self.values[4]
