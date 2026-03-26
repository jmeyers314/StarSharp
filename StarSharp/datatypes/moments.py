from __future__ import annotations

import dataclasses
import itertools
from dataclasses import make_dataclass

import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity

from .field_coords import FieldCoords


class Moments:
    """Generic container for 2D image moments of order n.

    Use ``Moments[n]`` to obtain the concrete dataclass for a given order,
    e.g. ``Moments[2]``, ``Moments[3]``, ``Moments[4]``.

    Fields are named by all symmetric index combinations of 'x' and 'y',
    e.g. ``xx``, ``xy``, ``yy`` for order 2.

    All moment fields are `~astropy.units.Quantity` instances.
    Use ``frame`` to record the coordinate frame (``'ocs'`` or ``'ccs'``)
    and ``field`` to attach the corresponding `FieldCoords`.
    """

    _cache: dict = {}
    _specialized: dict = {}
    _moment_order: int | None = None  # overridden on concrete classes
    VALID_FRAMES = ("ocs", "ccs", "dvcs", "edcs")

    def __post_init__(self):
        # Coerce frame to lower case and validate
        if hasattr(self, "frame") and isinstance(self.frame, str):
            object.__setattr__(self, "frame", self.frame.lower())
        if hasattr(self, "frame") and self.frame not in self.VALID_FRAMES:
            raise ValueError(
                f"frame must be one of {self.VALID_FRAMES}, got {self.frame!r}"
            )

    @classmethod
    def specialize(cls, order: int):
        """Decorator to register a specialized implementation for an order."""

        def deco(subcls: type):
            cls._specialized[order] = subcls
            cls._cache[order] = subcls
            return subcls

        return deco

    @classmethod
    def __class_getitem__(cls, order: int) -> type:
        if order in cls._specialized:
            return cls._specialized[order]
        if order not in cls._cache:
            moment_names = [
                "".join(p) for p in itertools.combinations_with_replacement("xy", order)
            ]
            moment_fields = [(f, Quantity) for f in moment_names]
            meta_fields = [
                ("frame", str, dataclasses.field(default="ocs")),
                ("field", FieldCoords | None, dataclasses.field(default=None)),
                ("rtp", Angle | None, dataclasses.field(default=None)),
            ]
            cls._cache[order] = make_dataclass(
                f"Moments{order}",
                moment_fields + meta_fields,
                bases=(cls,),
                frozen=True,
                namespace={"_moment_order": order},
            )
        return cls._cache[order]

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Moments to use this property")
        return val

    def _tensor(self):
        """Build the full (non-symmetric) moment tensor as a plain ndarray."""
        order = self._moment_order
        sample = getattr(self, "x" * order).value
        leading_shape = np.shape(sample)
        T = np.zeros(leading_shape + (2,) * order, dtype=np.result_type(sample, float))
        for idx_tuple in itertools.product([0, 1], repeat=order):
            canonical = tuple(sorted(idx_tuple))
            name = "".join("xy"[i] for i in canonical)
            T[(...,) + idx_tuple] = getattr(self, name).value
        return T

    def _rot(self, angle: Angle, frame: str):
        """Return a new Moments with all moment components rotated by *angle*."""
        c = float(np.cos(angle.rad))
        s = float(np.sin(angle.rad))
        R = np.array([[c, s], [-s, c]])
        order = self._moment_order

        T = self._tensor()

        # Apply rotation: contract R along each index axis in turn (over last axes)
        T_rot = T.copy()
        for axis in range(order):
            axis_to_contract = -(order - axis)
            T_rot = np.tensordot(R, T_rot, axes=([1], [axis_to_contract]))
            T_rot = np.moveaxis(T_rot, 0, axis_to_contract)

        # Read back symmetric components, restoring units
        unit = getattr(self, "x" * order).unit  # e.g. self.xx.unit for order 2
        moment_names = [
            "".join(p) for p in itertools.combinations_with_replacement("xy", order)
        ]
        new_moments = {}
        for name in moment_names:
            idx = tuple(0 if ch == "x" else 1 for ch in name)
            new_moments[name] = T_rot[(...,) + idx] * unit
        return type(self)(**new_moments, frame=frame, field=self.field, rtp=self.rtp)

    def _swap(self, frame: str):
        """Return a new Moments with x and y swapped (reflection x↔y)."""
        order = self._moment_order
        moment_names = [
            "".join(p) for p in itertools.combinations_with_replacement("xy", order)
        ]
        new_moments = {}
        for name in moment_names:
            canonical_swapped = "".join(
                sorted("y" if ch == "x" else "x" for ch in name)
            )
            new_moments[name] = getattr(self, canonical_swapped)
        return type(self)(**new_moments, frame=frame, field=self.field, rtp=self.rtp)

    def _relabel(self, frame: str):
        """Return an identical Moments with a different frame label."""
        order = self._moment_order
        moment_names = [
            "".join(p) for p in itertools.combinations_with_replacement("xy", order)
        ]
        return type(self)(
            **{n: getattr(self, n) for n in moment_names},
            frame=frame,
            field=self.field,
            rtp=self.rtp,
        )

    @property
    def ocs(self):
        """These moments in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self.ccs._rot(-rtp, "ocs")

    @property
    def ccs(self):
        """These moments in the CCS frame (always frame='ccs')."""
        if self.frame == "ccs":
            return self
        if self.frame == "edcs":
            return self._relabel("ccs")
        if self.frame == "dvcs":
            return self._swap("ccs")
        rtp = self._require("rtp")
        return self._rot(rtp, "ccs")

    @property
    def edcs(self):
        """These moments in the EDCS frame (synonym for CCS, preserves name)."""
        if self.frame == "edcs":
            return self
        return self.ccs._relabel("edcs")

    @property
    def dvcs(self):
        """These moments in the DVCS frame (transpose of EDCS/CCS)."""
        if self.frame == "dvcs":
            return self
        return self.ccs._swap("dvcs")

    def spin_complex(self, m: int) -> Quantity:
        """Return the complex spin-m moment for this order-N Moments (m>=0).

        Let N = self._moment_order and z = x + i y. For a given spin m,
        define p=(N+m)/2 and q=(N-m)/2 (requires same parity). Then:

            M_m = < z^p zbar^q >

        Under rotation, M_m transforms by a phase exp(± i m theta) depending
        on rotation convention; this method just computes M_m from the tensor.

        Parameters
        ----------
        m : int
            Spin magnitude. Must satisfy 0 <= m <= N with same parity as N.

        Returns
        -------
        Quantity (complex-valued)
            Complex spin moment M_m with the same unit as the order-N moments.
        """
        N = self._moment_order
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
        unit = getattr(self, "x" * N).unit

        ez = np.array([1.0, 1.0j])
        ezbar = np.array([1.0, -1.0j])

        result = T
        for _ in range(p):
            result = np.tensordot(result, ez, axes=([-1], [0]))
        for _ in range(q):
            result = np.tensordot(result, ezbar, axes=([-1], [0]))

        return result * unit

    def spin_pair(self, m: int):
        """Return (cos, sin) components for spin m (m>=0).

        For m=0, returns (scalar, None).
        For m>0, returns:
            cos = Re(M_m), sin = Im(M_m)
        """
        M = self.spin_complex(m)
        if m == 0:
            return M.real, None
        return M.real, M.imag

    def spin_cos(self, m: int) -> Quantity:
        """Cosine (real) component of spin m (m>=0)."""
        return self.spin_complex(m).real

    def spin_sin(self, m: int) -> Quantity:
        """Sine (imag) component of spin m (m>0)."""
        if m <= 0:
            raise ValueError("m must be > 0 for spin_sin()")
        return self.spin_complex(m).imag

    def spin(self, m: int) -> Quantity:
        """Alias for spin_cos(m)."""
        if m >= 0:
            return self.spin_cos(m)
        else:
            return self.spin_sin(-m)


@Moments.specialize(2)
class Moments2(Moments[2]):
    """Second-order moments.

    Fields: ``xx``, ``xy``, ``yy``.

    Spin decomposition:
      - Spin-0 (scalar): ``xx + yy``
      - Spin-2 components: ``xx - yy``, ``2*xy``
    """

    @property
    def T(self):
        """Trace: xx + yy."""
        return self.xx + self.yy

    @property
    def e1(self):
        """Ellipticity e1: (xx - yy) / T."""
        return (self.xx - self.yy) / self.T

    @property
    def e2(self):
        """Ellipticity e2: 2 * xy / T."""
        return 2 * self.xy / self.T


@Moments.specialize(3)
class Moments3(Moments[3]):
    """Third-order moments.

    Fields: ``xxx``, ``xxy``, ``xyy``, ``yyy``.

    Spin decomposition (cos,sin from Re/Im of M_m):
      - Spin-1: Re<z^2 zbar> = xxx + xyy, Im<z^2 zbar> = xxy + yyy
      - Spin-3: Re<z^3> = xxx - 3*xyy, Im<z^3> = 3*xxy - yyy
    """


@Moments.specialize(4)
class Moments4(Moments[4]):
    """Fourth-order moments.

    Fields: ``xxxx``, ``xxxy``, ``xxyy``, ``xyyy``, ``yyyy``.

    Spin decomposition:
      - Spin-0 (scalar): Re<z^2 zbar^2> = xxxx + 2*xxyy + yyyy
      - Spin-2: Re<z^3 zbar> = xxxx - yyyy, Im<z^3 zbar> = 2*(xxxy + xyyy)
      - Spin-4: Re<z^4> = xxxx - 6*xxyy + yyyy, Im<z^4> = 4*(xxxy - xyyy)
    """