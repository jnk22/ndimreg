"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _verify_image_2d(image: NDArray) -> None:
    if image.ndim != 2:
        msg = "Input must be a 2D image"
        raise ValueError(msg)

    if len(set(image.shape)) != 1:
        msg = "Input image must be a square"
        raise ValueError(msg)

    if len(image) % 2 != 0:
        msg = "Input image must have even sides"
        raise ValueError(msg)


def _verify_image_3d(image: NDArray) -> None:
    if image.ndim != 3:
        msg = "Input must be a 3D image"
        raise ValueError(msg)

    if len(set(image.shape)) != 1:
        msg = "Input image must be a cube"
        raise ValueError(msg)

    if len(image) % 2 != 0:
        msg = "Input image must have even sides"
        raise ValueError(msg)
