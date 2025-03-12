"""Test module for Keller 3D volume registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ndimreg.registration import Keller3DRegistration
from ndimreg.utils.diffs import euler_xyz_to_quaternion_dist
from tests.test_constants import (
    TEST_IMAGE_SIZE_3D,
    TEST_IMAGE_SIZE_VAR_32,
    TEST_IMAGES_3D,
    TEST_SHIFTS_3D,
)

if TYPE_CHECKING:
    from ndimreg.image import Image3D

# TODO: Implement rotation checks on a sphere.
# TODO: Add more tests: Sub-pixel shifts and rotations.


@pytest.fixture(params=TEST_IMAGE_SIZE_3D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_3D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image3D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.skip
def test_registration_3d_keller3d_register_with_empty_image(
    image: Image3D, *, debug: bool
) -> None:
    """Test registration of an input image with an empty image.

    The registration cannot succeed, but the result should be a
    rotation of 0, shift of 0, and scale of 1, i.e., no change.
    """
    registration = Keller3DRegistration(debug=debug)
    result = registration.register(image.data, np.zeros_like(image.data))

    assert result.transformation.rotation == (0.0,) * 3
    assert result.transformation.translation == (0.0,) * 3
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.skip
def test_registration_3d_keller3d_equal_input_images(
    image: Image3D, *, debug: bool
) -> None:
    """Test unmodified images with `Keller3DRegistration`.

    This test ensures that images that are identical to each other
    return a registration result with no rotation, no shift, and no scale.
    """
    registration = Keller3DRegistration(debug=debug)

    result = registration.register(image.data, image.copy().data)

    assert result.transformation.rotation == (0.0,) * 3
    assert result.transformation.translation == (0.0,) * 3
    assert result.transformation.scale is None  # Scaling is not supported.


# TODO: Create tests for enabled rotation normalization.
@pytest.mark.parametrize("shift_z", TEST_SHIFTS_3D, ids=lambda x: f"tz={x}")
@pytest.mark.parametrize("shift_y", TEST_SHIFTS_3D, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_3D, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("rotation_z", [5, 25, 50, 80], ids=lambda x: f"rz={x}")
@pytest.mark.parametrize("rotation_y", [5, 25, 50, 80], ids=lambda x: f"ry={x}")
@pytest.mark.parametrize("rotation_x", [5, 25, 50, 80], ids=lambda x: f"rx={x}")
@pytest.mark.parametrize(
    "image_size", [TEST_IMAGE_SIZE_VAR_32], ids=lambda x: f"size={x}"
)
@pytest.mark.regression
def test_registration_3d_keller3d_rotation_shift(
    haase_image_3d_safe: Image3D,
    image_size: int,
    input_rotations_3d: dict[str, float],
    input_shifts_3d_rounded: dict[str, int],
    *,
    debug: bool,
) -> None:
    """TODO."""
    rotations = tuple(input_rotations_3d.values())
    shifts = tuple(input_shifts_3d_rounded.values())

    image_transformed = haase_image_3d_safe.copy(name="transformed")

    # Set 'clip=False' as there was one test that failed due to clipping
    # with a higher rotation error.
    image_transformed.transform(translation=shifts, rotation=rotations, clip=False)

    registration = Keller3DRegistration(debug=debug)
    result = registration.register(haase_image_3d_safe.data, image_transformed.data)

    quat_dist = euler_xyz_to_quaternion_dist(rotations, result.transformation.rotation)

    # TODO: Re-implement as approximation test as approx_rotation().
    assert result.transformation.translation == shifts  # Shifts are exact here.
    assert quat_dist <= 2.45 * np.arctan(1 / image_size)  # ~4.39° for 32x32x32
    assert result.transformation.scale is None  # Scaling is not supported.
