"""Test module for Keller ADF-based registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ndimreg.registration import Keller2DRegistration
from tests.angle_utils import approx_rotation
from tests.test_constants import (
    TEST_IMAGE_SIZE_2D,
    TEST_IMAGE_SIZE_VAR_64,
    TEST_IMAGES_2D,
    TEST_ROT_FULL_180_10,
    TEST_SHIFTS_2D,
)

if TYPE_CHECKING:
    from ndimreg.image import Image2D

# TODO: Add more tests: Sub-pixel shifts and rotations.
# TODO: Add test with single value non-zero.
# TODO: Add tests for shifted lion image (without rotation).
# WARNING: Tests with -171 (type: numpy, default) fail!


@pytest.fixture(params=TEST_IMAGE_SIZE_2D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_2D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image2D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.skip
@pytest.mark.parametrize(
    "rotation_normalization", [True, False], ids=lambda x: f"norm={x}"
)
@pytest.mark.parametrize("optimization", [True, False], ids=lambda x: f"opt={x}")
def test_registration_2d_keller2d_register_with_empty_image(
    image: Image2D, *, rotation_normalization: bool, optimization: bool, debug: bool
) -> None:
    """Test registration of an input image with an empty image.

    The registration cannot succeed, but the result should be a
    rotation of 0, shift of 0, and scale of 1, i.e., no change.
    """
    registration = Keller2DRegistration(
        rotation_optimization=optimization,
        rotation_normalization=rotation_normalization,
        debug=debug,
    )
    result = registration.register(image.data, np.zeros_like(image.data))

    assert result.transformation.rotation == 0
    assert result.transformation.translation == (0.0, 0.0)
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.skip
@pytest.mark.parametrize(
    "rotation_normalization", [True, False], ids=lambda x: f"norm={x}"
)
@pytest.mark.parametrize("optimization", [True, False], ids=lambda x: f"opt={x}")
def test_registration_2d_keller2d_equal_input_images(
    image: Image2D, *, rotation_normalization: bool, optimization: bool, debug: bool
) -> None:
    """Test unmodified images with `Keller2DRegistration`.

    This test ensures that images that are identical to each other
    return a registration result with no rotation, no shift, and no scale.
    """
    registration = Keller2DRegistration(
        rotation_optimization=optimization,
        rotation_normalization=rotation_normalization,
        debug=debug,
    )
    result = registration.register(image.data, image.copy().data)

    assert result.transformation.rotation == approx_rotation(0)
    assert result.transformation.translation == (0.0, 0.0)
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.parametrize("shift_y", TEST_SHIFTS_2D, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_2D, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("rotation", TEST_ROT_FULL_180_10, ids=lambda x: f"rot={x}")
@pytest.mark.parametrize(
    "rotation_normalization", [True, False], ids=lambda x: f"norm={x}"
)
@pytest.mark.parametrize(
    "image_size", [TEST_IMAGE_SIZE_VAR_64], ids=lambda x: f"size={x}"
)
@pytest.mark.regression
def test_registration_2d_keller2d_rotation_shift(
    f16_image_2d: Image2D,
    image_size: int,
    rotation: float,
    input_shifts_2d_rounded: dict[str, int],
    *,
    rotation_normalization: bool,
    debug: bool,
) -> None:
    """TODO."""
    image = f16_image_2d
    shifts = tuple(input_shifts_2d_rounded.values())

    image_transformed = image.copy(name="transformed")
    image_transformed.transform(translation=shifts, rotation=rotation)

    registration = Keller2DRegistration(
        rotation_normalization=rotation_normalization, debug=debug
    )
    result = registration.register(image.data, image_transformed.data)

    # The absolute tolerance is calculated based on the image size.
    # For rotation, we use the maximum possible rotation of arctan(2/N),
    # which is based on the interval for possible rotations in the paper.
    # 2x the inteval would be the exact interval for one rotation value.
    # Due to the rotation optimization, this could be reduced to 1.6.
    factor = 1.6 if rotation_normalization else 2

    rotation_tol = factor * np.rad2deg(np.arctan(1 / image_size))

    # Tested with 64x64 F16 iamge from ADF paper.

    assert result.transformation.rotation == approx_rotation(rotation, abs=rotation_tol)
    assert result.transformation.translation == shifts
    assert result.transformation.scale is None  # Scaling is not supported.
