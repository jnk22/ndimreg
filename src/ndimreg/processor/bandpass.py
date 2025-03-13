"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from ndimreg.utils.image_operations import difference_of_gaussians

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GaussianBandPassFilter(BaseDataProcessor):
    """Band-pass filter image data."""

    def __init__(
        self, low_sigma: float | None = None, high_sigma: float | None = None
    ) -> None:
        """TODO."""
        low_str = f"{low_sigma:.2f}" if low_sigma is not None else "None"
        high_str = f"{high_sigma:.2f}" if high_sigma is not None else "None"

        if low_sigma is None and high_sigma is None:
            msg = "At least one value of 'low_sigma' or 'high_sigma' is required"
            raise ValueError(msg)

        super().__init__(
            f"{self.__class__.__name__}(low_sigma={low_str},high_sigma={high_str})"
        )

        self.__low_sigma = low_sigma
        self.__high_sigma = high_sigma

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        return [
            difference_of_gaussians(d, self.__low_sigma, self.__high_sigma)
            for d in data
        ]
