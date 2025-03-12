"""Test module for Scikit 2D registration."""

from __future__ import annotations

import numpy as np
import pytest

from ndimreg.image import Image2D
from ndimreg.processor import GaussianBandPassFilter, WindowFilter
from ndimreg.registration import Scikit2DRegistration
from tests.angle_utils import approx_rotation
from tests.test_constants import (
    TEST_IMAGE_SIZE_2D,
    TEST_IMAGES_2D,
    TEST_ROT_FULL_180_10,
    TEST_SCALES_2D,
    TEST_SHIFTS_2D,
)

# TODO: Add more tests: Sub-pixel shifts and rotations.
# TODO: Add test with single value non-zero.


@pytest.fixture(params=TEST_IMAGE_SIZE_2D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture(params=TEST_IMAGES_2D)
def image(request: pytest.FixtureRequest, image_size: int) -> Image2D:  # noqa: ARG001
    """Fixture to provide different types of images."""
    return request.getfixturevalue(request.param)


@pytest.mark.skip
def test_registration_2d_scikit2d_register_with_empty_image(image: Image2D) -> None:
    """Test registration of an input image with an empty image.

    The registration cannot succeed, but the result should be a
    rotation of 0, shift of 0, and scale of 1, i.e., no change.
    """
    registration = Scikit2DRegistration()
    result = registration.register(image.data, np.zeros_like(image.data))

    assert result.transformation.rotation == approx_rotation(0)
    assert result.transformation.translation == pytest.approx((0.0,) * 2)
    assert result.transformation.scale == pytest.approx(1)


@pytest.mark.skip
def test_registration_2d_scikit2d_equal_input_images(image: Image2D) -> None:
    """Test unmodified images with `Scikit2DRegistration`.

    This test ensures that images that are identical to each other
    return a registration result with no rotation, no shift, and no scale.
    """
    result = Scikit2DRegistration().register(image.data, image.copy().data)

    assert result.transformation.rotation == approx_rotation(0)
    assert result.transformation.translation == pytest.approx((0.0,) * 2)
    assert result.transformation.scale == pytest.approx(1)


@pytest.mark.parametrize("shift_y", TEST_SHIFTS_2D, ids=lambda x: f"ty={x}")
@pytest.mark.parametrize("shift_x", TEST_SHIFTS_2D, ids=lambda x: f"tx={x}")
@pytest.mark.parametrize("scale", TEST_SCALES_2D, ids=lambda x: f"scale={x}")
@pytest.mark.parametrize("rotation", TEST_ROT_FULL_180_10, ids=lambda x: f"rot={x}")
@pytest.mark.parametrize("image_size", [512], ids=lambda x: f"size={x}")
@pytest.mark.regression
def test_registration_2d_scikit2d_rotation_shift_scale(
    astronaut_image_2d: Image2D,
    rotation: int,
    scale: float,
    input_shifts_2d_rounded: dict[str, int],
) -> None:
    """TODO."""
    image = astronaut_image_2d
    shifts = tuple(input_shifts_2d_rounded.values())

    image_transformed = image.copy(name="transformed")
    image_transformed.transform(translation=shifts, rotation=rotation, scale=scale)

    processors = [GaussianBandPassFilter(0, 1), WindowFilter("hann")]
    registration = Scikit2DRegistration(processors=processors)
    result = registration.register(image.data, image_transformed.data)

    # With scale=1.0, rotation works with abs=1, shifts with abs=1, and scale with 0.01.
    assert result.transformation.rotation == approx_rotation(rotation, abs=1.2)
    assert result.transformation.translation == pytest.approx(shifts, abs=3)
    assert result.transformation.scale == pytest.approx(scale, abs=0.016)  # 1.6%


@pytest.mark.regression
def test_registration_2d_scikit2d_scikit_image_example() -> None:
    """Verify that the scikit-image example registration example is working.

    The example shown in [1] is used to verify that the registration
    implementation is working as expected. The example uses a retina
    image from scikit-image datasets and applies a rotation and shift
    to the image. The registration result is then compared to the
    expected values.

    Notes
    -----
    The shifted imaged is slightly different from the example due to
    another implementation of the transformation functions.
    However, the registration result is even closer to the expected
    image transformation.

    References
    ----------
    .. [1] scikit-image team,
           "Using Polar and Log-Polar Transformations for Registration",
           skimage 0.23.0 documentation,
           https://scikit-image.org/docs/0.23.x/auto_examples/registration/plot_register_rotation.html
    """
    image = Image2D.from_skimage("retina")

    # See [1] for the input/expected values.
    shifts = (30, 15)
    rotation = 24
    scale = 1.4

    image_transformed = image.copy(name="transformed")
    image_transformed.transform(translation=shifts, rotation=rotation, scale=scale)

    # The example also uses a Gaussian band-pass filter and a Hann window.
    # We need to apply the same filters to the images as pre-processors.
    processors = [GaussianBandPassFilter(5, 20), WindowFilter("hann")]
    registration = Scikit2DRegistration(
        processors=processors,
        rotation_upsample_factor=10,
        rotation_normalization=False,
        rotation_disambiguate=False,
    )
    result = registration.register(image.data, image_transformed.data)

    assert result.transformation.rotation == approx_rotation(rotation, abs=0.1)
    assert result.transformation.translation == shifts
    assert result.transformation.scale == pytest.approx(scale, abs=0.001)
