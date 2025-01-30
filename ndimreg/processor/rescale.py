from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import get_namespace
from typing_extensions import override

from ndimreg.utils.image_operations import rescale_intensity

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RescaleProcessor(BaseDataProcessor):
    """Rescale/Normalize image data."""

    def __init__(self, *, group: bool = True) -> None:
        """TODO."""
        super().__init__()

        self.group: bool = group

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        if self.group:
            xp = get_namespace(*data)
            in_range = xp.min([*data]), xp.max([*data])
        else:
            in_range = "image"

        return [rescale_intensity(d, in_range=in_range) for d in data]
