"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ndimreg.image import Image


def prepare_benchmark_image(
    image: Image,
    *,
    spacing: tuple[float, ...] | None = None,
    normalize: bool = False,
    max_pad: bool = False,
    safe_pad: bool = False,
    resize: int | None = None,
) -> Image:
    """TODO."""
    if spacing:
        image.apply_spacing(spacing, max_size=resize)
    if normalize:
        image.normalize()
    if max_pad:
        image.pad_equal_sides()
    if safe_pad:
        image.pad_safe_rotation(keep_shape=resize is None)
    if resize:
        image.resize_to_shape(resize)

    return image
