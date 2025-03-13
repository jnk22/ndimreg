"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from ndimreg.transform import transform

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ndimreg.transform import Transformation

    from .base import Dimension


class Zoomer(BaseDataProcessor):
    """Scale into/out of image data."""

    def __init__(self, factor: float, dim: Literal[2, 3]) -> None:
        """Initialize zoom out pre-processor.

        Parameters
        ----------
        factor
            Scaling factor.
        dim
            Image dimension for input data.
        """
        super().__init__(f"{self.__class__.__name__}({factor:.2f})")
        self.__factor: float = factor
        self.__dim: Dimension = dim

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        return [transform(d, dim=self.__dim, scale=self.__factor) for d in data]

    @override
    def backward(self, transformation: Transformation) -> Transformation:
        transformation.translation = tuple(
            np.array(transformation.translation) * 1 / self.__factor
        )

        return transformation
