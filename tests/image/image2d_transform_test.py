"""Tests for `Image2D` translation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_constants import TEST_IMAGE_SIZE_2D, TEST_IMAGES_2D

if TYPE_CHECKING:
    from ndimreg.image import Image2D


@pytest.fixture(params=TEST_IMAGE_SIZE_2D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_2D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image2D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("inverse", [True, False], ids=lambda x: f"inverse={x}")
def test_image2d_transform_small_interpolation_effect_translation(
    image: Image2D, *, inverse: bool
) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.transform(translation=(0, 0), inverse=inverse)

    assert image.shape == original_shape
    assert original_data is not image.data

    # Due to interpolation it is possible that pixels that
    # are actual zeros become 'close to zero'. We allow this within very
    # a small tolerance.
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


@pytest.mark.parametrize("rotation", [0, 360, 720, -360], ids=lambda x: f"rot={x}")
@pytest.mark.parametrize("inverse", [True, False], ids=lambda x: f"inverse={x}")
def test_image2d_transform_small_interpolation_effect_rotation(
    image: Image2D, rotation: float, *, inverse: bool
) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.transform(rotation=rotation, inverse=inverse)

    assert image.shape == original_shape
    assert original_data is not image.data

    # Due to interpolation it is possible that pixels that
    # are actual zeros become 'close to zero'. We allow this within very
    # a small tolerance.
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


@pytest.mark.parametrize("inverse", [True, False], ids=lambda x: f"inverse={x}")
def test_image2d_transform_small_interpolation_effect_scale(
    image: Image2D, *, inverse: bool
) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.transform(scale=1.0, inverse=inverse)

    assert image.shape == original_shape
    assert original_data is not image.data

    # Due to interpolation it is possible that pixels that
    # are actual zeros become 'close to zero'. We allow this within very
    # a small tolerance.
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


def test_image2d_transform_indifferent_resize(image: Image2D) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.resize(1.0)

    assert image.shape == original_shape
    assert original_data is not image.data
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


def test_image2d_transform_indifferent_scale(image: Image2D) -> None:
    """TODO."""
    original_shape = image.shape
    original_data = image.data.copy()
    image.transform(scale=1.0)

    assert image.shape == original_shape
    assert original_data is not image.data
    np.testing.assert_allclose(image.data, original_data, atol=1e-15)


# NOTE: Tests are currently failing. Definition for scale=0 required.
@pytest.mark.skip
def test_image2d_full_zoom_equals_zero_image(image: Image2D) -> None:
    """TODO."""
    original_shape = image.shape
    image.transform(scale=0.0)

    assert image.shape == original_shape
    np.testing.assert_equal(image.data, np.zeros(original_shape))
