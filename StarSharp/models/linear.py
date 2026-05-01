from ..datatypes import FieldCoords, Sensitivity, Spots, State, Zernikes
from ..utils import str_to_arr
from .raytraced import RaytracedOpticalModel


class LinearModel:
    """Linear (first-order) optical model wrapping a :class:`RaytracedOpticalModel`.

    On the first call to :attr:`spots_sensitivity` or
    :attr:`zernikes_sensitivity`, the corresponding sensitivity matrix is
    computed via finite differences on the underlying raytraced model and
    cached.  Subsequent calls use the cached sensitivity for fast linear
    prediction via :meth:`Sensitivity.predict`.

    Parameters
    ----------
    raytraced : RaytracedOpticalModel
        The underlying raytraced model used to compute sensitivities.
    field : FieldCoords
        Field points at which to evaluate the sensitivity.
    use_dof : list[int], str, or None
        Active DOF indices for sensitivity computation.  A string is parsed
        as a comma-separated list of indices and/or ranges (e.g.
        ``"0-9,10-16,30-36"``).  If None, the raytraced model's default
        active set is used.
    spots_kwargs : dict or None
        Extra keyword arguments forwarded to
        :meth:`RaytracedOpticalModel.spots_sensitivity`.
    zernikes_kwargs : dict or None
        Extra keyword arguments forwarded to
        :meth:`RaytracedOpticalModel.zernikes_sensitivity`.
    """

    def __init__(
        self,
        raytraced: RaytracedOpticalModel,
        field: FieldCoords,
        use_dof: list[int] | str | None = None,
        spots_kwargs: dict | None = None,
        zernikes_kwargs: dict | None = None,
    ):
        self.raytraced = raytraced
        self.field = field
        self.use_dof = str_to_arr(use_dof) if use_dof is not None else None
        self.spots_kwargs = spots_kwargs or {}
        self.zernikes_kwargs = zernikes_kwargs or {}
        self._spots_sensitivity: "Sensitivity | None" = None
        self._zernikes_sensitivity: "Sensitivity | None" = None

    @property
    def spots_sensitivity(self) -> Sensitivity:
        """Cached spot sensitivity matrix (computed on first access)."""
        if self._spots_sensitivity is None:
            self._spots_sensitivity = self.raytraced.spots_sensitivity(
                field=self.field,
                basis="x",
                use_dof=self.use_dof,
                **self.spots_kwargs,
            )
        return self._spots_sensitivity

    @property
    def zernikes_sensitivity(self) -> Sensitivity:
        """Cached Zernike sensitivity matrix (computed on first access)."""
        if self._zernikes_sensitivity is None:
            self._zernikes_sensitivity = self.raytraced.zernikes_sensitivity(
                field=self.field,
                basis="x",
                use_dof=self.use_dof,
                **self.zernikes_kwargs,
            )
        return self._zernikes_sensitivity

    def spots(self, state: State) -> Spots:
        """Linearly predict spot diagrams for *state*."""
        return self.spots_sensitivity.predict(state)

    def zernikes(self, state: State) -> Zernikes:
        """Linearly predict Zernike coefficients for *state*."""
        return self.zernikes_sensitivity.predict(state)
