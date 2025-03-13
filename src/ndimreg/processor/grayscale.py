"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from ndimreg.utils.image_operations import rgb2gray

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GrayscaleProcessor2D(BaseDataProcessor):
    """Grayscale image data."""

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        return [rgb2gray(d) if d.ndim > 2 else d for d in data]


class GrayscaleProcessor3D(BaseDataProcessor):
    """Grayscale image data."""

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        return [rgb2gray(d) if d.ndim > 3 else d for d in data]
