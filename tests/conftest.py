"""TODO."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ndimreg.image import Image2D, Image3D

IMAGE_PATH_HAASE_MRT_3D = "data/3d/haase_mrt.tif"
IMAGE_PATH_TIMELAPSE_FUSED_3D = "data/3d/fused.tif"
IMAGE_PATH_ILLUMINATION_3D = "data/3d/illumination.tif"
IMAGE_PATH_F16_ADF_2D = "data/2d/f16_adf.png"

# TODO: Find a way to make images re-usable as 'session' fixtures.


def __strtobool(val: str) -> bool:
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return True

    if val in {"n", "no", "f", "false", "off", "0"}:
        return False

    msg = f"Invalid truth value: {val}"
    raise ValueError(msg)


@pytest.fixture
def debug() -> bool:
    """Debugging parameter for test registrations."""
    return __strtobool(os.getenv("DEBUG", "False"))


@pytest.fixture
def f16_image_2d(image_size: int) -> Image2D:
    """Return the F16 image."""
    return Image2D.from_path(IMAGE_PATH_F16_ADF_2D).resize_to_shape(image_size)


@pytest.fixture
def astronaut_image_2d(image_size: int) -> Image2D:
    """Return the astronaut image."""
    return Image2D.from_skimage("astronaut").resize_to_shape(image_size)


@pytest.fixture
def empty_image_2d(image_size: int) -> Image2D:
    """Return an empty (i.e., black) image."""
    return Image2D(np.zeros((image_size,) * 2), name="empty")


@pytest.fixture
def full_image_2d(image_size: int) -> Image2D:
    """Return a full (i.e., white) image."""
    return Image2D(np.ones((image_size,) * 2), name="full")


@pytest.fixture
def gradient_image_2d(image_size: int) -> Image2D:
    """Return a gradient image."""
    gradient_matrix = np.linspace(0, 1, image_size)
    return Image2D(np.meshgrid(gradient_matrix, gradient_matrix)[0], name="gradient")


@pytest.fixture
def random_image_2d(image_size: int) -> Image2D:
    """Return a random image."""
    rng = np.random.default_rng()
    return Image2D(rng.random((image_size,) * 2), name="random")


@pytest.fixture
def homogeneous_image_3x3() -> Image2D:
    """Return a 3x3 image with homogeneous pixel values."""
    return Image2D(np.full((3, 3), 0.5), name="homogeneous")


@pytest.fixture
def illumination_image_3d(image_size: int) -> Image3D:
    """Return the illumination test image."""
    return (
        Image3D.from_path(IMAGE_PATH_ILLUMINATION_3D)
        .pad_equal_sides()
        .resize_to_shape(image_size)
    )


@pytest.fixture
def haase_image_3d(image_size: int) -> Image3D:
    """Return the Haase MRT test image.

    Source: [clEsperanto/pyclesperanto_prototype](https://github.com/clEsperanto/pyclesperanto_prototype)
    """
    return (
        Image3D.from_path(IMAGE_PATH_HAASE_MRT_3D)
        .pad_equal_sides()
        .resize_to_shape(image_size)
    )


@pytest.fixture
def timelapse_fused_3d(image_size: int) -> Image3D:
    """TODO."""
    return (
        Image3D.from_path(IMAGE_PATH_TIMELAPSE_FUSED_3D)
        .normalize()
        .cut(low=0.2)
        .pad_equal_sides()
        .resize_to_shape(image_size)
    )


@pytest.fixture
def illumination_image_3d_safe(illumination_image_3d: Image3D) -> Image3D:
    """Return the illumination test image with rotation-safe padding."""
    return illumination_image_3d.pad_safe_rotation(keep_shape=True)


@pytest.fixture
def timelapse_fused_3d_safe(timelapse_fused_3d: Image3D) -> Image3D:
    """Return the illumination test image with rotation-safe padding."""
    return timelapse_fused_3d.pad_safe_rotation(keep_shape=True)


@pytest.fixture
def haase_image_3d_safe(haase_image_3d: Image3D) -> Image3D:
    """Return the Haase MRT test image with rotation-safe padding."""
    return haase_image_3d.pad_safe_rotation(keep_shape=True)


@pytest.fixture
def empty_image_3d(image_size: int) -> Image3D:
    """Return an empty (i.e., black) image."""
    return Image3D(np.zeros((image_size,) * 3), name="empty")


@pytest.fixture
def full_image_3d(image_size: int) -> Image3D:
    """Return a full (i.e., white) image."""
    return Image3D(np.ones((image_size,) * 3), name="full")


@pytest.fixture
def gradient_image_3d(image_size: int) -> Image3D:
    """Return a gradient image."""
    gradient_matrix = np.linspace(0, 1, image_size)
    return Image3D(
        np.meshgrid(gradient_matrix, gradient_matrix, gradient_matrix)[0],
        name="gradient",
    )


@pytest.fixture
def random_image_3d(image_size: int) -> Image3D:
    """Return a random image."""
    rng = np.random.default_rng()
    return Image3D(rng.random((image_size,) * 3), name="random")


@pytest.fixture
def input_shifts_3d(
    shift_x: float, shift_y: float, shift_z: float
) -> tuple[float, float, float]:
    """TODO."""
    return shift_x, shift_y, shift_z


@pytest.fixture
def input_shifts_3d_subpixel(
    image_size: int, input_shifts_3d: tuple[float, float, float]
) -> dict[str, float]:
    """TODO."""
    factor = image_size * 0.01
    return {k: v * factor for k, v in zip("xyz", input_shifts_3d, strict=True)}


@pytest.fixture
def input_shifts_3d_rounded(
    input_shifts_3d_subpixel: dict[str, float],
) -> dict[str, int]:
    """TODO."""
    return {k: round(v) for k, v in input_shifts_3d_subpixel.items()}


@pytest.fixture
def input_shifts_2d(shift_x: float, shift_y: float) -> tuple[float, float]:
    """TODO."""
    return shift_x, shift_y


@pytest.fixture
def input_shifts_2d_subpixel(
    image_size: int, input_shifts_2d: tuple[float, float]
) -> dict[str, float]:
    """TODO."""
    factor = image_size * 0.01
    return {k: v * factor for k, v in zip("xy", input_shifts_2d, strict=True)}


@pytest.fixture
def input_shifts_2d_rounded(
    input_shifts_2d_subpixel: dict[str, float],
) -> dict[str, int]:
    """TODO."""
    return {k: round(v) for k, v in input_shifts_2d_subpixel.items()}


@pytest.fixture
def input_rotations_3d(
    rotation_x: float, rotation_y: float, rotation_z: float
) -> dict[str, float]:
    """TODO."""
    return dict(zip("xyz", (rotation_x, rotation_y, rotation_z), strict=True))
