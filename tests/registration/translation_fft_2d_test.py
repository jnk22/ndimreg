"""Tests for FFT translation regisration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.spatial.distance import euclidean

from ndimreg.registration import TranslationFFT2DRegistration
from tests.test_constants import (
    TEST_IMAGE_SIZE_2D,
    TEST_IMAGES_2D,
    TEST_IMAGES_2D_DATA,
    TEST_SHIFTS_2D_EXTENDED,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ndimreg.image import Image2D

# TODO: Add more tests: Sub-pixel shifts.
# TODO: Add test with single value non-zero.


@pytest.fixture(params=TEST_IMAGE_SIZE_2D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_2D)
def image_2d(request: pytest.FixtureRequest, image_size: int) -> Image2D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=TEST_IMAGES_2D_DATA)
def image_2d_data(request: pytest.FixtureRequest, image_size: int) -> Image2D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.skip
def test_registration_2d_translationfft_register_with_empty_image(
    image_2d: Image2D,
) -> None:
    """Test registration of an input image with an empty image.

    The registration cannot succeed, but the result should be a
    rotation of 0, shift of 0, and scale of 1, i.e., no change.
    """
    registration = TranslationFFT2DRegistration()
    result = registration.register(image_2d.data, np.zeros_like(image_2d.data))

    assert result.transformation.translation == (0,) * 2
    assert result.transformation.rotation is None  # Rotation is not supported.
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.skip
def test_registration_2d_translationfft_equal_input_images(image_2d: Image2D) -> None:
    """Test for various types of images."""
    registration = TranslationFFT2DRegistration()
    result = registration.register(image_2d.data, image_2d.copy().data)

    assert result.transformation.translation == (0,) * 2
    assert result.transformation.rotation is None  # Rotation is not supported.
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((3, 3, 3)),
        np.zeros((3, 3)),
        np.zeros((10, 5)),
        np.ones((3, 3, 3)),
        np.ones((3, 3)),
        np.ones((10, 5)),
        np.array([[1]]),
        np.arange(16).reshape((4, 4)),
    ],
)
def test_registration_2d_translationfft_equal_input_images_with_different_shapes(
    data: NDArray,
) -> None:
    """Verify that translation is recovered when fixed and moving are equal.

    Images tested are:
    - 2D images without channels
    - 2D images with channels
    - 2D images with non-square shape

    The expected translation is (0, 0) for 2D images when both fixed and
    moving are equal.
    """
    result = TranslationFFT2DRegistration().register(data, data.copy())

    assert result.transformation.translation == (0,) * 2
    assert result.transformation.rotation is None  # Rotation is not supported.
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.parametrize("shift_y", TEST_SHIFTS_2D_EXTENDED, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_2D_EXTENDED, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("image_size", [64, 128], ids=lambda x: f"size={x}")
@pytest.mark.regression
def test_registration_2d_translationfft_shifts_rounded(
    image_2d_data: Image2D, input_shifts_2d_rounded: dict[str, int]
) -> None:
    """TODO."""
    image = image_2d_data
    shifts = tuple(input_shifts_2d_rounded.values())

    image_translated = image.copy(name="translated").transform(translation=shifts)

    registration = TranslationFFT2DRegistration()
    result = registration.register(image.data, image_translated.data)

    assert result.transformation.translation == shifts
    assert result.transformation.rotation is None  # Rotation is not supported.
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.parametrize("shift_y", TEST_SHIFTS_2D_EXTENDED, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_2D_EXTENDED, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("image_size", [64, 128], ids=lambda x: f"size={x}")
@pytest.mark.regression
def test_registration_2d_translationfft_shifts_subpixel(
    image_2d_data: Image2D, image_size: int, input_shifts_2d_subpixel: dict[str, int]
) -> None:
    """TODO."""
    image = image_2d_data
    shifts = tuple(input_shifts_2d_subpixel.values())

    image_translated = image.copy(name="translated").transform(translation=shifts)

    registration = TranslationFFT2DRegistration(upsample_factor=10)
    result = registration.register(image.data, image_translated.data)

    assert euclidean(result.transformation.translation, shifts) <= 0.01 * image_size
    assert result.transformation.rotation is None  # Rotation is not supported.
    assert result.transformation.scale is None  # Scaling is not supported.
