from __future__ import annotations

import itertools
import dataclasses
from dataclasses import dataclass, make_dataclass
from typing import Optional

import astropy.units as u
import batoid
import galsim
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from galsim.fitswcs import GSFitsWCS
from lsst.afw.cameraGeom import FOCAL_PLANE, Camera
from numpy.typing import NDArray

VALID_FRAMES = ("ocs", "ccs")


@dataclass(frozen=True)
class FieldCoords:
    """Field coordinates in any supported frame and space.

    Parameters
    ----------
    x, y : Quantity
        Coordinate values with units.  Angular units (e.g. ``u.deg``,
        ``u.arcsec``) indicate field-angle space; length units
        (e.g. ``u.mm``) indicate focal-plane space.
    frame : str
        Coordinate frame: ``'ocs'`` (optical, default) or ``'ccs'``
        (camera).
    rtp : Angle or None
        Rotation angle from OCS to CCS frame.  Required for frame conversions.
    wcs : GSFitsWCS or None
        WCS for converting between field-angle and focal-plane space.  Required for
        space conversions.  WCS assumes field angles are in radians and focal-plane
        coordinates are in mm.
    camera : Camera or None
        Camera geometry for determining detector numbers.  Required for
        ``detnum`` property.
    """

    x: Quantity
    y: Quantity
    frame: str = "ocs"
    rtp: Optional[Angle] = None
    wcs: Optional[GSFitsWCS] = None
    camera: Optional[Camera] = None

    def __post_init__(self):
        if not isinstance(self.x, Quantity) or not isinstance(self.y, Quantity):
            raise TypeError("x and y must be astropy Quantity instances with units")
        object.__setattr__(self, "x", np.atleast_1d(self.x))
        object.__setattr__(self, "y", np.atleast_1d(self.y))
        if self.frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {VALID_FRAMES}, got {self.frame!r}")
        if not (self.x.unit.physical_type == self.y.unit.physical_type):
            raise ValueError(
                f"x and y must have compatible units, "
                f"got {self.x.unit} and {self.y.unit}"
            )
        if self.x.unit.physical_type not in ("angle", "length"):
            raise ValueError(
                f"units must be angular or length, "
                f"got physical type {self.x.unit.physical_type!r}"
            )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(
                f"{name} must be set on the FieldCoords to use this property"
            )
        return val

    @classmethod
    def from_telescope_wcs(
        cls,
        x: Quantity,
        y: Quantity,
        *,
        telescope: batoid.Optic,
        wavelength: Quantity,
        rtp: Optional[Angle] = None,
        **kwargs,
    ):
        if rtp is None:
            rtp = Angle("0 deg")
        rotated = telescope.withLocallyRotatedOptic("LSSTCamera", batoid.RotZ(rtp.rad))
        nrad = 20
        th_u, th_v = batoid.utils.hexapolar(
            np.deg2rad(2.0), nrad=nrad, naz=int(2 * np.pi * nrad)
        )
        rays = batoid.RayVector.fromFieldAngles(
            theta_x=th_u,
            theta_y=th_v,
            projection="gnomonic",
            optic=rotated,
            wavelength=wavelength.to_value(u.m),
        )
        rotated.trace(rays)
        wcs = galsim.FittedSIPWCS(
            rays.x * 1000, rays.y * 1000, th_u, th_v, order=3
        )  # Use mm <-> radians
        return cls(x, y, rtp=rtp, wcs=wcs, **kwargs)

    @property
    def space(self) -> str:
        """Inferred coordinate space: ``'angle'`` or ``'focal_plane'``."""
        if self.x.unit.physical_type == "angle":
            return "angle"
        return "focal_plane"

    def _rot(self, angle: Angle, frame: str) -> FieldCoords:
        """Return a new FieldCoords with x/y rotated by *angle*."""
        rx = self.x * np.cos(angle) + self.y * np.sin(angle)
        ry = -self.x * np.sin(angle) + self.y * np.cos(angle)
        return FieldCoords(
            x=rx,
            y=ry,
            frame=frame,
            rtp=self.rtp,
            wcs=self.wcs,
            camera=self.camera,
        )

    @property
    def ocs(self) -> FieldCoords:
        """This coordinate in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self._rot(-rtp, "ocs")

    @property
    def ccs(self) -> FieldCoords:
        """This coordinate in the CCS frame."""
        if self.frame == "ccs":
            return self
        rtp = self._require("rtp")
        return self._rot(rtp, "ccs")

    @property
    def focal_plane(self) -> FieldCoords:
        """This coordinate in focal-plane space (CCS frame)."""
        if self.space == "focal_plane":
            return self
        wcs = self._require("wcs")
        field = self.ocs
        fx, fy = wcs.toImage(
            field.x.to_value(u.radian),
            field.y.to_value(u.radian),
            units="radians",
        )
        fp_ccs = FieldCoords(
            x=fx << u.mm,
            y=fy << u.mm,
            frame="ccs",
            rtp=self.rtp,
            wcs=self.wcs,
            camera=self.camera,
        )
        return getattr(fp_ccs, self.frame)

    @property
    def angle(self) -> FieldCoords:
        """This coordinate in field-angle space (OCS frame)."""
        if self.space == "angle":
            return self
        wcs = self._require("wcs")
        fp = self.focal_plane
        fx, fy = wcs.toWorld(
            fp.x.to_value(u.mm),
            fp.y.to_value(u.mm),
            units="radians",
        )
        field_ocs = FieldCoords(
            x=fx << u.radian,
            y=fy << u.radian,
            frame="ocs",
            rtp=self.rtp,
            wcs=self.wcs,
            camera=self.camera,
        )
        return getattr(field_ocs, self.frame)

    @property
    def detnum(self) -> int | NDArray[np.integer]:
        """Detector number(s). Returns -1 for points off any detector."""
        camera = self._require("camera")
        fp = self.focal_plane.ccs
        x = np.atleast_1d(fp.x.to_value(u.mm))
        y = np.atleast_1d(fp.y.to_value(u.mm))

        result = np.full(x.shape, -1, dtype=int)
        for det in camera:
            corners = det.getCorners(FOCAL_PLANE)
            xs = [c[1] for c in corners]  # Transpose DVCS -> EDCS = CCS
            ys = [c[0] for c in corners]
            mask = (x >= min(xs)) & (x <= max(xs)) & (y >= min(ys)) & (y <= max(ys))
            result[mask] = det.getId()

        if np.isscalar(fp.x.value):
            return result.item()
        return result

    def __len__(self) -> int:
        if self.x.ndim == 1:
            return 1
        return self.x.shape[0]

    def __getitem__(self, idx: int | slice) -> FieldCoords:
        return FieldCoords(
            x=self.x[idx],
            y=self.y[idx],
            frame=self.frame,
            rtp=self.rtp,
            wcs=self.wcs,
            camera=self.camera,
        )

    def __repr__(self) -> str:
        return f"FieldCoords({self.x!r}, {self.y!r}, frame={self.frame!r})"


@dataclass(frozen=True)
class Spots:
    """Spot diagrams: ray intersection positions on the focal plane.

    May represent one or many field points.  When batched,
    ``dx`` and ``dy`` have shape ``(n_field, n_ray)``.

    Parameters
    ----------
    dx, dy : Quantity
        Ray positions relative to the centroid.
    vignetted : NDArray[bool]
        Vignetting mask.
    field : FieldCoords
        Field coordinate(s) at which the spot was computed.
    wavelength : Quantity or None
        Wavelength of the traced rays.
    frame : str
        Coordinate frame: ``'ocs'`` (optical, default) or ``'ccs'``
        (camera).
    rtp : Angle or None
        Rotation angle from OCS to CCS frame.  Required for frame conversions.
    """

    dx: Quantity
    dy: Quantity
    vignetted: NDArray[np.bool_]
    field: FieldCoords
    wavelength: Quantity | None = None
    frame: str = "ccs"
    rtp: Optional[Angle] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dx", np.atleast_1d(self.dx))
        object.__setattr__(self, "dy", np.atleast_1d(self.dy))
        object.__setattr__(self, "vignetted", np.atleast_1d(self.vignetted))
        if (
            self.rtp is not None
            and self.field.rtp is not None
            and not np.allclose(
                self.rtp.rad, self.field.rtp.rad  # type: ignore[union-attr]
            )
        ):
            raise ValueError(
                f"Spots.rtp ({self.rtp}) is inconsistent with "
                f"field.rtp ({self.field.rtp})"
            )

    def __len__(self) -> int:
        if self.dx.ndim == 1:
            return 1
        return self.dx.shape[0]

    def __getitem__(self, idx: int | slice) -> Spots:
        return Spots(
            dx=self.dx[idx],
            dy=self.dy[idx],
            field=self.field[idx],
            vignetted=self.vignetted[idx],
            wavelength=self.wavelength,
            frame=self.frame,
            rtp=self.rtp,
        )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Spots to use this property")
        return val

    def _rot(self, angle: Angle, frame: str) -> Spots:
        """Return a new Spots with dx/dy rotated by *angle*."""
        rdx = self.dx * np.cos(angle) + self.dy * np.sin(angle)
        rdy = -self.dx * np.sin(angle) + self.dy * np.cos(angle)
        return Spots(
            dx=rdx,
            dy=rdy,
            vignetted=self.vignetted,
            field=self.field,
            wavelength=self.wavelength,
            frame=frame,
            rtp=self.rtp,
        )

    @property
    def ocs(self) -> Spots:
        """This spot in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self._rot(-rtp, "ocs")

    @property
    def ccs(self) -> Spots:
        """This spot in the CCS frame."""
        if self.frame == "ccs":
            return self
        rtp = self._require("rtp")
        return self._rot(rtp, "ccs")

    def __repr__(self) -> str:
        return f"Spots(frame={self.frame!r})"

    def compute_moments(self, order: int = 2) -> Moments:
        """Compute 2d moments of the spot diagrams (excluding vignetted spots).

        Parameters
        ----------
        order : int
            Order of the moments to compute (default: 2).

        Returns
        -------
        Moments
            Moments of the spot diagram.
        """
        if order < 1:
            raise ValueError("Order must be at least 1")

        # Convert to micron as a working unit and copy only if necessary
        dx = self.dx.to_value(u.micron)
        if np.shares_memory(dx, self.dx.value):
            dx = dx.copy()
        dy = self.dy.to_value(u.micron)
        if np.shares_memory(dy, self.dy.value):
            dy = dy.copy()
        dx[self.vignetted] = np.nan
        dy[self.vignetted] = np.nan

        vals = {}
        for ny in range(order + 1):
            nx = order - ny
            name = "x" * nx + "y" * ny
            vals[name] = np.nanmean(
                dx ** nx * dy ** ny,
                axis=-1
            ) << (u.micron ** order)

        return Moments[order](**vals, frame=self.frame, field=self.field, rtp=self.rtp)


@dataclass(frozen=True)
class Zernikes:
    """Zernike coefficients at one or many field points.

    When batched, ``coefs`` has shape ``(..., jmax + 1)``.

    Parameters
    ----------
    coefs : Quantity
        Zernike coefficient array.  The last axis indexes Noll
        index *j* (0 … jmax).  Leading axes are batch dimensions.
    R_outer : float
        Outer radius of the annular pupil.
    R_inner : float
        Inner radius of the annular pupil.
    jmax : int or None
        Maximum Noll index.  Inferred from ``coefs`` if *None*.
    rtp : Angle or None
        Rotation angle from OCS to CCS frame.  Required for frame conversions.
    """

    coefs: Quantity
    field: FieldCoords
    R_outer: Quantity = None
    R_inner: Quantity = None
    wavelength: Quantity | None = None
    jmax: int | None = None
    frame: str = "ocs"
    rtp: Optional[Angle] = None

    def __post_init__(self):
        object.__setattr__(self, "coefs", np.atleast_1d(self.coefs))
        if self.jmax is None:
            object.__setattr__(self, "jmax", self.coefs.shape[-1] - 1)
        if (
            self.rtp is not None
            and self.field.rtp is not None
            and not np.allclose(
                self.rtp.rad, self.field.rtp.rad  # type: ignore[union-attr]
            )
        ):
            raise ValueError(
                f"Zernikes.rtp ({self.rtp}) is inconsistent with "
                f"field.rtp ({self.field.rtp})"
            )

    @property
    def eps(self) -> float:
        """Obscuration ratio R_inner / R_outer."""
        return (self.R_inner / self.R_outer).value

    def __len__(self) -> int:
        if self.coefs.ndim == 1:
            return 1
        return self.coefs.shape[0]

    def __getitem__(self, idx: int | slice) -> Zernikes:
        return Zernikes(
            coefs=self.coefs[idx],
            field=self.field[idx],
            R_outer=self.R_outer,
            R_inner=self.R_inner,
            wavelength=self.wavelength,
            jmax=self.jmax,
            frame=self.frame,
            rtp=self.rtp,
        )

    def _require(self, name: str):
        """Return the instance attribute or raise if not set."""
        val = getattr(self, name)
        if val is None:
            raise ValueError(f"{name} must be set on the Zernikes to use this property")
        return val

    def _rot(self, angle: Angle, frame: str) -> Zernikes:
        """Return a new Zernikes with the coefficients rotated by *angle*."""
        rot = galsim.zernike.zernikeRotMatrix(self.jmax, angle.radian)
        coefs_rot = self.coefs @ rot
        return Zernikes(
            coefs=coefs_rot,
            field=self.field,
            R_outer=self.R_outer,
            R_inner=self.R_inner,
            wavelength=self.wavelength,
            jmax=self.jmax,
            frame=frame,
            rtp=self.rtp,
        )

    @property
    def ocs(self) -> Zernikes:
        """This Zernikes in the OCS frame."""
        if self.frame == "ocs":
            return self
        rtp = self._require("rtp")
        return self._rot(-rtp, "ocs")

    @property
    def ccs(self) -> Zernikes:
        """This Zernikes in the CCS frame."""
        if self.frame == "ccs":
            return self
        rtp = self._require("rtp")
        return self._rot(rtp, "ccs")

    def to_galsim(
        self,
        idx: int | None = None,
        unit: u.Unit = u.um,
        radius_unit: u.Unit = u.m,
    ) -> galsim.zernike.Zernike:
        """Return a `galsim.zernike.Zernike` for one set of coefficients.

        Parameters
        ----------
        idx : int or None
            Index into the batch dimension.  Required when
            ``coefs`` has more than one dimension; ignored for a
            single (1-D) coefficient vector.
        unit : astropy.units.Unit
            Unit for the output coefficients (default: micron).
        """
        c = self.coefs if self.coefs.ndim == 1 else self.coefs[idx]
        if c.ndim != 1:
            raise TypeError("to_galsim() requires a single coefficient vector")
        return galsim.zernike.Zernike(
            c.to(unit).value,
            R_outer=self.R_outer.to_value(radius_unit),
            R_inner=self.R_inner.to_value(radius_unit),
        )

    def __repr__(self) -> str:
        return f"Zernikes({self.coefs!r}, jmax={self.jmax}, frame={self.frame!r})"


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
