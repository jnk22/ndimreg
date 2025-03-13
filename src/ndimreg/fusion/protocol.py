"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Fusion(Protocol):
    """TODO."""

    def fuse(self, *image: NDArray) -> NDArray:
        """TODO."""
        ...

    @property
    def name(self) -> str:
        """TODO."""
        ...
