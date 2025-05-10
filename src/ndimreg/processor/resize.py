from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from array_api_compat import array_namespace
from typing_extensions import override

from ndimreg.utils.image_operations import rescale_intensity

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ResizeProcessor(BaseDataProcessor):
    """Rescale/Normalize image data."""

    def __init__(self, *, upscale: bool = True, even: bool = True, G) -> None:
        """TODO."""
        super().__init__()

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        if self.group:
            xp = array_namespace(*data)
            in_range = xp.min([*data]), xp.max([*data])
        else:
            in_range = "image"

        return [rescale_intensity(d, in_range=in_range) for d in data]


def pad_or_downscale_matrix(
    matrix: NDArray,
    *,
    even: bool = False,
    upscale: bool = True,
    target_pow2: bool | None = None,
    cval: Any,
):
    """Pa or downscale a 2D numpy matrix to specified conditions.

    Parameters
    ----------
    - matrix: 2D numpy array to be padded or downscaled
    - even: If True, make sure dimensions are even (e.g., 71x70 -> 72x72)
    - upscale: If True, upscale to nearest odd dimensions (e.g., 71x70 -> 71x71),
              if False, downscale to even dimensions (e.g., 71x70 -> 70x70)
    - target: If specified, upscale or downscale to nearest power of 2 dimensions.

    Returns
    -------
    - Modified 2D numpy array
    """
    height, width = matrix.shape

    # Handle "even" parameter: ensuring both dimensions are even:
    if even:
        if height % 2 != 0:
            height += 1
        if width % 2 != 0:
            width += 1

    # Handle "upscale" and "downscale" based on the target size:
    if target_pow2:
        x = np.log2(max(height, width))
        next_power_of_2 = 2 ** int(np.ceil(x))
        prev_power_of_2 = 2 ** int(np.floor(x))

        new_size = next_power_of_2 if upscale else prev_power_of_2

        height = new_size
        width = new_size

    # Resize the matrix (padding or cropping) to the new dimensions
    new_matrix = np.zeros((height, width), dtype=matrix.dtype)

    new_matrix[: min(height, matrix.shape[0]), : min(width, matrix.shape[1])] = matrix[
        : min(height, matrix.shape[0]), : min(width, matrix.shape[1])
    ]

    return new_matrix
