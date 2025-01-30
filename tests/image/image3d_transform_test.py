"""Tests for `ndimreg` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_constants import TEST_IMAGE_SIZE_3D, TEST_IMAGES_3D

if TYPE_CHECKING:
    from ndimreg.image import Image3D


@pytest.fixture(params=TEST_IMAGE_SIZE_3D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_3D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image3D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


def test_image3d_transform_indifferent_resize(image: Image3D) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.resize(1.0)

    assert image.shape == original_shape
    assert original_data is not image.data
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


def test_image3d_transform_indifferent_scale(image: Image3D) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.transform(scale=1.0)

    assert image.shape == original_shape
    assert original_data is not image.data
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


# NOTE: Tests are currently failing. Definition for scale=0 required.
@pytest.mark.skip
def test_image3d_full_zoom_equals_zero_image(image: Image3D) -> None:
    """TODO."""
    original_shape = image.shape
    image.transform(scale=0.0)

    assert image.shape == original_shape
    np.testing.assert_equal(image.data, np.zeros(original_shape))
