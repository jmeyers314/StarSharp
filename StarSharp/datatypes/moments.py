from __future__ import annotations

import dataclasses
import itertools
from dataclasses import make_dataclass
from typing import Optional

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

    Use the ``@Moments.specialize(order)`` decorator to register a
    specialized subclass for a given order.  Once registered,
    ``Moments[order]`` returns the specialized class.

    Examples
    --------
    >>> m2 = Moments[2](xx=1.0*u.mm**2, xy=0.0*u.mm**2, yy=1.0*u.mm**2)
    >>> m3 = Moments[3](xxx=0.0*u.mm**3, xxy=0.0*u.mm**3, xyy=0.0*u.mm**3, yyy=0.0*u.mm**3)
    """

    _cache: dict = {}
    _specialized: dict = {}
    _moment_order: int | None = None  # overridden on concrete classes

    def __post_init__(self):
        # Coerce frame to lower case
        if hasattr(self, 'frame') and isinstance(self.frame, str):
            object.__setattr__(self, 'frame', self.frame.lower())

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
                ''.join(p)
                for p in itertools.combinations_with_replacement('xy', order)
            ]
            moment_fields = [(f, Quantity) for f in moment_names]
            meta_fields = [
                ('frame', str, dataclasses.field(default='ocs')),
                ('field', Optional[FieldCoords], dataclasses.field(default=None)),
                ('rtp', Optional[Angle], dataclasses.field(default=None)),
            ]
            cls._cache[order] = make_dataclass(
                f"Moments{order}",
                moment_fields + meta_fields,
                bases=(cls,),
                frozen=True,
                namespace={'_moment_order': order},
            )
        return cls._cache[order]

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(
                f"{name} must be set on the Moments to use this property"
            )
        return val

    def _tensor(self):
        """Build the full (non-symmetric) moment tensor as a plain ndarray."""
        order = self._moment_order
        sample = getattr(self, 'x' * order).value
        is_batched = sample.shape != ()
        batch_shape = sample.shape if is_batched else ()
        T = np.zeros(batch_shape + (2,) * order, dtype=sample.dtype)
        for idx_tuple in itertools.product([0, 1], repeat=order):
            canonical = tuple(sorted(idx_tuple))
            name = ''.join('xy'[i] for i in canonical)
            if is_batched:
                T[(slice(None),) + idx_tuple] = getattr(self, name).value
            else:
                T[idx_tuple] = getattr(self, name).value
        return T

    def _rot(self, angle: Angle, frame: str):
        """Return a new Moments with all moment components rotated by *angle*."""
        c = float(np.cos(angle.rad))
        s = float(np.sin(angle.rad))
        R = np.array([[c, s], [-s, c]])
        order = self._moment_order
        is_batched = getattr(self, 'x' * order).value.shape != ()

        T = self._tensor()

        # Apply rotation: contract R along each index axis in turn (over last axes)
        T_rot = T.copy()
        for axis in range(order):
            axis_to_contract = -(order - axis)
            T_rot = np.tensordot(R, T_rot, axes=([1], [axis_to_contract]))
            T_rot = np.moveaxis(T_rot, 0, axis_to_contract)

        # Read back symmetric components, restoring units
        unit = getattr(self, 'x' * order).unit  # e.g. self.xx.unit for order 2
        moment_names = [
            ''.join(p)
            for p in itertools.combinations_with_replacement('xy', order)
        ]
        new_moments = {}
        for name in moment_names:
            idx = tuple(0 if ch == 'x' else 1 for ch in name)
            if is_batched:
                new_moments[name] = T_rot[(...,) + idx] * unit
            else:
                new_moments[name] = T_rot[idx] * unit
        return type(self)(**new_moments, frame=frame, field=self.field, rtp=self.rtp)

    @property
    def ocs(self):
        """These moments in the OCS frame."""
        if self.frame == 'ocs':
            return self
        rtp = self._require('rtp')
        return self._rot(-rtp, 'ocs')

    @property
    def ccs(self):
        """These moments in the CCS frame."""
        if self.frame == 'ccs':
            return self
        rtp = self._require('rtp')
        return self._rot(rtp, 'ccs')

    def spin(self, n, m):
        """Return the (n, m) spin-decomposed moment component.

        Uses the complex representation z = x + iy to decompose the
        symmetric moment tensor into spin components via
        M_{p,q} = <z^p zbar^q> where p = (order+n)/2, q = (order-n)/2.

        Parameters
        ----------
        n : int
            Spin quantum number.  Must be non-negative, <= order, and
            have the same parity as the moment order.
        m : int
            Selects the real (+n) or imaginary (-n) part.  Must be
            +n or -n (or 0 when n = 0).

        Returns
        -------
        Quantity
            The spin-decomposed moment component (unnormalized).
        """
        order = self._moment_order
        if n < 0 or n > order or (n % 2) != (order % 2):
            raise ValueError(
                f"Invalid spin n={n} for order-{order} moments; "
                f"n must satisfy 0 <= n <= {order} with same parity"
            )
        if n == 0 and m != 0:
            raise ValueError(f"For n=0, m must be 0, got m={m}")
        if n > 0 and m not in (n, -n):
            raise ValueError(f"For n={n}, m must be +/-{n}, got m={m}")

        p = (order + n) // 2
        q = (order - n) // 2

        T = self._tensor()
        unit = getattr(self, 'x' * order).unit

        ez = np.array([1, 1j])
        ezbar = np.array([1, -1j])

        result = T
        for _ in range(p):
            result = np.tensordot(result, ez, axes=([-1], [0]))
        for _ in range(q):
            result = np.tensordot(result, ezbar, axes=([-1], [0]))

        if m >= 0:
            return result.real * unit
        else:
            return result.imag * unit


@Moments.specialize(2)
class Moments2(Moments[2]):
    """Second-order moments with hard-coded coordinate transformations.

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
    """Third-order moments with hard-coded coordinate transformations.

    Fields: ``xxx``, ``xxy``, ``xyy``, ``yyy``.

    Spin decomposition:
      - Spin-1 components: ``xxx + xyy``, ``yyy + xxy``
      - Spin-3 components: ``xxx - 3*xyy``, ``yyy - 3*xxy``
    """


@Moments.specialize(4)
class Moments4(Moments[4]):
    """Fourth-order moments with hard-coded coordinate transformations.

    Fields: ``xxxx``, ``xxxy``, ``xxyy``, ``xyyy``, ``yyyy``.

    Spin decomposition:
      - Spin-0 (scalar): ``xxxx + 2*xxyy + yyyy``
      - Spin-2 components: ``xxxx - yyyy``, ``2*(xxxy + xyyy)``
      - Spin-4 components: ``xxxx - 6*xxyy + yyyy``, ``4*(xxxy - xyyy)``
    """
