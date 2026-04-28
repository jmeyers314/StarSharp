from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class SensitivityObservable(Protocol):
    """Structural protocol required by `Sensitivity`.

    Implementations provide one or more quantity-valued fields that are
    differentiated with respect to state DOFs, optional metadata fields that
    should be broadcast along a DOF axis, and indexing over the leading axis.
    """

    _sensitivity_fields: ClassVar[tuple[str, ...]]
    _broadcast_fields: ClassVar[tuple[str, ...]]

    def __len__(self) -> int: ...

    def __getitem__(self, idx): ...
