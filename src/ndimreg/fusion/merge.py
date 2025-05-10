"""3D image registration with basic FFT shift using phase cross correlation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from array_api_compat import array_namespace
from typing_extensions import override

from .base import BaseFusion

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MergeFusion(BaseFusion):
    """Fuse 2D/3D images by merging all values with post-normalization.

    Capabilities
    ------------
    - Dimension: 2D/3D

    Limitations
    -----------
    Supports only grayscale images.
    """

    def __init__(self, *, alpha: float | None = None) -> None:
        """Initialize Merge Fusion.

        Parameters
        ----------
        alpha
            Alpha blending value for all images that are overlayed onto
            each other. If None, alpha will be dynamically determined as
            `alpha = 1 / len(images)`.
        """
        super().__init__()

        self.__alpha: float | None = alpha

    @override
    def _fuse(self, *images: NDArray, **_kwargs: Any) -> NDArray:
        xp = array_namespace(*images)

        alpha = self.__alpha or 1 / len(images)
        data = xp.sum(xp.asarray(images) * alpha, 0)

        return data / max_v if (max_v := xp.max(data)) > 1 else data
