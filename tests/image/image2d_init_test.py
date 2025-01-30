"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from array_api_compat import is_cupy_array, is_numpy_array

try:
    import cupy as cp
except ImportError:
    cp = None

from ndimreg.image import Image2D
from tests.test_constants import TEST_IMAGE_SIZE_2D

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ndimreg.image.image import Device

AVAILABLE_DEVICES = ["cpu", "gpu"] if cp else ["cpu"]

CPU_DATA = ["random_image_2d_cpu_grayscale", "random_image_2d_cpu_rgb"]
GPU_DATA = ["random_image_2d_gpu_rgb", "random_image_2d_gpu_grayscale"]
ALL_DATA = [*CPU_DATA, *GPU_DATA] if cp else CPU_DATA

# TODO: Make device a fixture similar to 'image_size' parameter.


@pytest.fixture(params=TEST_IMAGE_SIZE_2D)
def image_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different image sizes."""
    return request.param


@pytest.fixture
def numpy_rng() -> Generator:
    """TODO."""
    return np.random.default_rng(seed=0)


@pytest.fixture
def cupy_rng() -> Generator:
    """TODO."""
    assert cp is not None
    return cp.random.default_rng(seed=0)


@pytest.fixture
def random_image_2d_cpu_grayscale(image_size: int, numpy_rng: Generator) -> NDArray:
    """TODO."""
    return numpy_rng.random((image_size,) * 2)


@pytest.fixture
def random_image_2d_cpu_rgb(image_size: int, numpy_rng: Generator) -> NDArray:
    """TODO."""
    return numpy_rng.random((image_size, image_size, 3))


@pytest.fixture
def random_image_2d_gpu_grayscale(image_size: int, cupy_rng: Generator) -> NDArray:
    """TODO."""
    return cupy_rng.random((image_size,) * 2)


@pytest.fixture
def random_image_2d_gpu_rgb(image_size: int, cupy_rng: Generator) -> NDArray:
    """TODO."""
    return cupy_rng.random((image_size, image_size, 3))


@pytest.fixture(params=CPU_DATA)
def image_data_cpu(request: pytest.FixtureRequest, image_size) -> NDArray:  # noqa: ANN001
    """TODO."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=GPU_DATA)
def image_data_gpu(request: pytest.FixtureRequest, image_size: int) -> NDArray:  # noqa: ARG001
    """TODO."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=ALL_DATA)
def image_data_anywhere(request: pytest.FixtureRequest, image_size: int) -> NDArray:  # noqa: ARG001
    """TODO."""
    return request.getfixturevalue(request.param)


def test_image_2d_init_data_is_on_cpu(image_data_anywhere: NDArray) -> None:
    """TODO."""
    assert Image2D(image_data_anywhere, device="cpu").device == "cpu"


@pytest.mark.skipif("cp is None")
def test_image_2d_init_data_is_on_gpu(image_data_anywhere: NDArray) -> None:
    """TODO."""
    assert Image2D(image_data_anywhere, device="gpu").device == "gpu"


@pytest.mark.skipif("cp is None")
def test_image_2d_init_cpu_data_is_on_gpu(image_data_cpu: NDArray) -> None:
    """TODO."""
    assert is_numpy_array(image_data_cpu)
    assert Image2D(image_data_cpu, device="gpu").device == "gpu"


@pytest.mark.parametrize("target_device", ["cpu", None])
def test_image_2d_init_cpu_data_is_on_cpu(
    image_data_cpu: NDArray, target_device: Device | None
) -> None:
    """TODO."""
    assert is_numpy_array(image_data_cpu)
    assert Image2D(image_data_cpu, device=target_device).device == "cpu"


@pytest.mark.skipif("cp is None")
def test_image_2d_init_gpu_data_is_on_cpu(image_data_gpu: NDArray) -> None:
    """TODO."""
    assert is_cupy_array(image_data_gpu)
    assert Image2D(image_data_gpu, device="cpu").device == "cpu"


@pytest.mark.skipif("cp is None")
@pytest.mark.parametrize("target_device", ["gpu", None])
def test_image_2d_init_gpu_data_is_on_gpu(
    image_data_gpu: NDArray, target_device: Device | None
) -> None:
    """TODO."""
    assert is_cupy_array(image_data_gpu)
    assert Image2D(image_data_gpu, device=target_device).device == "gpu"


def test_image_2d_from_device_to_same_device(image_data_anywhere: NDArray) -> None:
    """TODO."""
    image = Image2D(image_data_anywhere)

    previous_image_device = image.device
    image.to_device(previous_image_device)
    assert image.device == previous_image_device


@pytest.mark.skipif("cp is None")
@pytest.mark.parametrize(
    ("start_device", "end_device"), [("cpu", "gpu"), ("gpu", "cpu")]
)
def test_image_2d_from_cpu_to_gpu_device(
    image_data_anywhere: NDArray, start_device: Device, end_device: Device
) -> None:
    """TODO."""
    image = Image2D(image_data_anywhere, device=start_device)
    assert image.device == start_device

    image.to_device(end_device)
    assert image.device == end_device


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_image_2d_pad_safe_rotation(
    image_data_anywhere: NDArray, device: Device | None
) -> None:
    """TODO."""
    image = Image2D(image_data_anywhere, device=device)
    assert image.device == device

    initial_resolution = image.resolution
    image.pad_safe_rotation()
    assert image.resolution > initial_resolution


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_image_2d_pad_to_size_max(
    image_data_anywhere: NDArray, device: Device | None
) -> None:
    """TODO."""
    image = Image2D(image_data_anywhere, device=device)
    assert image.device == device

    maximum_side_length = max(image.resolution)
    image.pad_to_size(maximum_side_length)
    assert image.resolution == (maximum_side_length,) * 2


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_image_2d_pad_to_size_max_double(
    image_data_anywhere: NDArray, device: Device | None
) -> None:
    """TODO."""
    image = Image2D(image_data_anywhere, device=device)
    assert image.device == device

    double_maximum_side_length = max(image.resolution) * 2
    image.pad_to_size(double_maximum_side_length)
    assert image.resolution == (double_maximum_side_length,) * 2


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_image_2d_pad_equal_sides(
    image_data_anywhere: NDArray, device: Device | None
) -> None:
    """TODO."""
    image = Image2D(image_data_anywhere, device=device)
    assert image.device == device

    maximum_side_length = max(image.resolution)
    image.pad_equal_sides()
    assert image.resolution == (maximum_side_length,) * 2


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_image_2d_rgb_to_grayscale(
    random_image_2d_cpu_rgb: NDArray, device: Device | None
) -> None:
    """TODO."""
    image = Image2D(random_image_2d_cpu_rgb, device=device)
    assert image.device == device
    assert image.multichannel

    initial_resolution = image.resolution
    image.grayscale()

    assert image.resolution == initial_resolution
    assert not image.multichannel


@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_image_2d_grayscale_noop(
    random_image_2d_cpu_grayscale: NDArray, device: Device | None
) -> None:
    """TODO."""
    image = Image2D(random_image_2d_cpu_grayscale, device=device)
    assert image.device == device
    assert not image.multichannel

    initial_resolution = image.resolution
    image.grayscale()

    assert image.resolution == initial_resolution
    assert not image.multichannel
