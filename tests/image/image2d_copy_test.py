"""Verify copying of `Image2D`."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from ndimreg.image import Image2D


def test_image2d_copy_data_remains_equal() -> None:
    """TODO."""
    image = Image2D(np.ones((10, 10)))
    assert image.shape == (10, 10)

    image_copy = image.copy()
    assert_array_equal(image_copy.data, image.data)
    assert image_copy.shape == image.shape


def test_image2d_copy_data_memory_location_changes() -> None:
    """TODO."""
    image = Image2D(np.ones((10, 10)))
    assert image.shape == (10, 10)

    image_copy = image.copy()
    assert image_copy.data is not image.data


def test_image2d_copy_appends_copy_suffix() -> None:
    """TODO."""
    image = Image2D(np.ones((1, 1)))
    assert image.copy().name == "image2d-copy"


def test_image2d_copy_allows_custom_name() -> None:
    """TODO."""
    image = Image2D(np.ones((1, 1)))
    image_copy_name = "custom-copy-name-123"

    assert image.copy(name=image_copy_name).name == image_copy_name


def test_image2d_copy_default_name() -> None:
    """TODO."""
    image = Image2D(np.ones((1, 1)))

    assert image.name == "image2d"


def test_image2d_copy_custom_name() -> None:
    """TODO."""
    image_name = "custom-image-name-123"
    image = Image2D(np.ones((1, 1)), name=image_name)

    assert image.name == image_name
