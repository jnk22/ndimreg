"""Test module for imreg_dft 2D registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.spatial.distance import euclidean

from ndimreg.registration import ImregDft2DRegistration
from tests.angle_utils import approx_rotation
from tests.test_constants import (
    TEST_IMAGE_SIZE_2D,
    TEST_IMAGES_2D,
    TEST_ROT_FULL_180_10,
    TEST_SHIFTS_2D,
)

if TYPE_CHECKING:
    from ndimreg.image import Image2D

# TODO: Add more tests: Sub-pixel shifts and rotations.
# TODO: Add test with single value non-zero.
# TODO: Add more tests: Scale inputs.


@pytest.fixture(params=TEST_IMAGE_SIZE_2D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_2D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image2D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.skip
def test_registration_2d_imregdft2d_register_with_empty_image(image: Image2D) -> None:
    """Test registration of an input image with an empty image.

    The registration cannot succeed, but the result should be a
    rotation of 0, shift of 0, and scale of 1, i.e., no change.
    """
    registration = ImregDft2DRegistration()
    result = registration.register(image.data, np.zeros_like(image.data))

    assert result.transformation.translation == pytest.approx((0.0,) * 2)
    assert result.transformation.rotation == approx_rotation(0)
    assert result.transformation.scale == pytest.approx(1)


@pytest.mark.skip
def test_registration_2d_imregdft2d_equal_input_images(image: Image2D) -> None:
    """Test unmodified images with `imregdft2DRegistration`.

    This test ensures that images that are identical to each other
    return a registration result with no rotation, no shift, and no scale.
    """
    result = ImregDft2DRegistration().register(image.data, image.copy().data)

    assert result.transformation.translation == pytest.approx((0.0,) * 2)
    assert result.transformation.rotation == approx_rotation(0)
    assert result.transformation.scale == pytest.approx(1)


@pytest.mark.parametrize("shift_y", TEST_SHIFTS_2D, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_2D, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("rotation", TEST_ROT_FULL_180_10, ids=lambda x: f"rot={x}")
@pytest.mark.parametrize("image_size", [256], ids=lambda x: f"size={x}")
@pytest.mark.regression
def test_registration_2d_imregdft2d_rotation_shift(
    astronaut_image_2d: Image2D,
    image_size: int,
    rotation: int,
    input_shifts_2d_rounded: dict[str, int],
) -> None:
    """TODO."""
    image = astronaut_image_2d
    shifts = tuple(input_shifts_2d_rounded.values())

    image_transformed = image.copy(name="transformed")
    image_transformed.transform(translation=shifts, rotation=rotation)

    registration = ImregDft2DRegistration()
    result = registration.register(image.data, image_transformed.data)

    assert euclidean(shifts, result.transformation.translation) <= 0.005 * image_size
    assert result.transformation.rotation == approx_rotation(rotation, abs=0.22)
    assert result.transformation.scale == pytest.approx(1, abs=0.005)  # 0.5%
