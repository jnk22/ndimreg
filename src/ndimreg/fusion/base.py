"""Base fusion class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from loguru import logger

from ndimreg.utils import log_time

from .protocol import Fusion

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BaseFusion(ABC, Fusion):
    """TODO."""

    # TODO: Implement more methods (e.g., 'MaxFusion').

    @property
    def name(self) -> str:
        """TODO."""
        return self.__class__.__name__

    @log_time(print_func=logger.debug)
    def fuse(self, *images: NDArray, **kwargs: Any) -> NDArray:
        """Fuse images.

        Parameters
        ----------
        images
            Input images.
        **kwargs
            Additional keyword arguments passed to the fusion
            implementation.

        Returns
        -------
        NDArray
            Fused image.
        """
        # TODO: Ensure that input data uses same backend (error, upgrade, downgrade).
        # TODO: Make interface similar to registration (return stats, duration etc.).

        if len(images) < 2:
            msg = "At least 2 images are required for fusion"
            raise ValueError(msg)

        logger.debug(f"Fusing images with '{self.__class__.__name__}'")
        return self._fuse(*images, **kwargs)

    @abstractmethod
    def _fuse(self, *images: NDArray, **kwargs: Any) -> NDArray: ...
