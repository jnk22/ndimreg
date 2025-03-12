"""Test module for Keller 3D volume registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import pytransform3d.rotations as pr

from ndimreg.registration import RotationAxis3DRegistration
from ndimreg.transform import AXIS_MAPPING
from ndimreg.transform.transformation import axis_rotation_matrix
from tests.angle_utils import approx_rotation
from tests.test_constants import (
    TEST_IMAGE_SIZE_VAR_32,
    TEST_IMAGES_3D,
    TEST_ROT_AXES_VAR,
    TEST_ROT_FULL_180_10,
    TEST_SHIFTS_3D,
)

if TYPE_CHECKING:
    from ndimreg.image import Image3D
    from ndimreg.transform import RotationAxis3D


@pytest.fixture(params=[4, 8, 16, 32])
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_3D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image3D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.skip
@pytest.mark.parametrize(
    "rotation_normalization", [True, False], ids=lambda x: f"norm={x}"
)
@pytest.mark.parametrize("axis", TEST_ROT_AXES_VAR, ids=lambda x: f"axis={x}")
@pytest.mark.parametrize("optimization", [True, False], ids=lambda x: f"opt={x}")
def test_registration_3d_rotationaxis3d_register_with_empty_image(
    image: Image3D,
    axis: RotationAxis3D,
    *,
    rotation_normalization: bool,
    optimization: bool,
    debug: bool,
) -> None:
    """Test registration of an input image with an empty image.

    The registration cannot succeed, but the result should be a
    rotation of 0, shift of 0, and scale of 1, i.e., no change.
    """
    registration = RotationAxis3DRegistration(
        axis,
        rotation_normalization=rotation_normalization,
        optimization=optimization,
        debug=debug,
    )
    result = registration.register(image.data, np.zeros_like(image.data))

    assert result.transformation.rotation == approx_rotation((0.0, 0.0, 0.0))
    assert result.transformation.translation == (0.0, 0.0, 0.0)
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.skip
@pytest.mark.parametrize(
    "rotation_normalization", [True, False], ids=lambda x: f"norm={x}"
)
@pytest.mark.parametrize("axis", TEST_ROT_AXES_VAR, ids=lambda x: f"axis={x}")
@pytest.mark.parametrize("optimization", [True, False], ids=lambda x: f"opt={x}")
def test_registration_3d_rotationaxis3d_equal_input_images(
    image: Image3D,
    axis: RotationAxis3D,
    *,
    rotation_normalization: bool,
    optimization: bool,
    debug: bool,
) -> None:
    """Test unmodified images with `RotationAxis3DRegistration`.

    This test ensures that images that are identical to each other
    return a registration result with no rotation, no shift, and no scale.
    """
    registration = RotationAxis3DRegistration(
        axis,
        rotation_normalization=rotation_normalization,
        optimization=optimization,
        debug=debug,
    )
    result = registration.register(image.data, image.copy().data)

    assert result.transformation.rotation == approx_rotation((0.0, 0.0, 0.0))
    assert result.transformation.translation == (0.0, 0.0, 0.0)
    assert result.transformation.scale is None  # Scaling is not supported.


@pytest.mark.parametrize("shift_z", TEST_SHIFTS_3D, ids=lambda x: f"tz={x}")
@pytest.mark.parametrize("shift_y", TEST_SHIFTS_3D, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_3D, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("rotation", TEST_ROT_FULL_180_10, ids=lambda x: f"rot={x}")
@pytest.mark.parametrize(
    "rotation_normalization", [True, False], ids=lambda x: f"norm={x}"
)
@pytest.mark.parametrize("axis", TEST_ROT_AXES_VAR, ids=lambda x: f"ax={x}")
@pytest.mark.parametrize(
    "image_size", [TEST_IMAGE_SIZE_VAR_32], ids=lambda x: f"size={x}"
)
@pytest.mark.regression
def test_registration_3d_rotationaxis3d_rotation_shift(
    haase_image_3d_safe: Image3D,
    image_size: int,
    rotation: float,
    axis: RotationAxis3D,
    input_shifts_3d_rounded: dict[str, float],
    *,
    rotation_normalization: bool,
    debug: bool,
) -> None:
    """TODO."""
    image = haase_image_3d_safe
    shifts = tuple(input_shifts_3d_rounded.values())

    rotation_matrix = axis_rotation_matrix(rotation, axis=axis, dim=3)
    image_transformed = image.copy(name="rotated")
    image_transformed.transform(translation=shifts, rotation=rotation_matrix)

    registration = RotationAxis3DRegistration(
        axis, rotation_normalization=rotation_normalization, debug=debug
    )
    result = registration.register(image.data, image_transformed.data)

    # To be comparable with the output Euler angles, the input rotation
    # on a single axis must be converted to a rotation matrix first and
    # then converted to the output Euler angle convention.
    basis = {0: 0, 1: 2, 2: 1}[AXIS_MAPPING[axis][1]]
    rotation_matrix = pr.active_matrix_from_angle(basis, np.deg2rad(-rotation))
    degrees = np.rad2deg(pr.intrinsic_euler_xyz_from_active_matrix(rotation_matrix))

    factor = 1.0 if rotation_normalization else 1.1
    rotation_tol = factor * np.rad2deg(np.arctan(1 / image_size))

    # TODO: Only validate single axis rotation (and/or quaternion distance).
    assert result.transformation.rotation == approx_rotation(degrees, abs=rotation_tol)
    assert result.transformation.translation == shifts  # Shifts are exact here.
    assert result.transformation.scale is None  # Scaling is not supported.
