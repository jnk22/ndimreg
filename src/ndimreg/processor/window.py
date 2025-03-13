"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from ndimreg.utils.image_operations import window

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class WindowFilter(BaseDataProcessor):
    """Apply window filter to image data."""

    def __init__(self, window_type: str) -> None:
        """TODO."""
        super().__init__(f"{self.__class__.__name__}({window_type})")
        self.__window_type = window_type

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        return [window(d, self.__window_type, d.shape) for d in data]
