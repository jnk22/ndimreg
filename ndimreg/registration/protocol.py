"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .result import RegistrationResult

Dimension = Literal[2, 3]


class Registration(Protocol):
    """TODO."""

    def register(self, fixed: NDArray, moving: NDArray) -> RegistrationResult:
        """TODO."""
        ...

    @property
    def dim(self) -> Dimension:
        """TODO."""
        ...

    @property
    def name(self) -> str:
        """TODO."""
        ...
