"""Test module for PPFT2 code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ndimreg.image import Image2D
from ndimreg.registration.ppft import ppft2, ppft2_optimized, ppft2_vectorized
from ndimreg.registration.ppft.ppft2 import rppft2, rppft2_optimized, rppft2_vectorized

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.fixture
def f16_data(data_size: int) -> NDArray:
    """Return the F16 image data."""
    image = Image2D.from_path(Path("data/2d/f16_adf.png"))

    # PPFT2D only works with grayscale images.
    return image.resize_to_shape(data_size).grayscale().data


@pytest.fixture
def astronaut_data(data_size: int) -> NDArray:
    """Return the F16 image data."""
    image = Image2D.from_skimage("astronaut")

    # PPFT2D only works with grayscale images.
    return image.resize_to_shape(data_size).grayscale().data


@pytest.fixture
def empty_data(data_size: int) -> NDArray:
    """Return data with only zeros (i.e., black) image."""
    return np.zeros((data_size,) * 2)


@pytest.fixture
def full_data(data_size: int) -> NDArray:
    """Return data with only ones (i.e., white) image."""
    return np.ones((data_size,) * 2)


@pytest.fixture
def gradient_data(data_size: int) -> NDArray:
    """Return gradient image data."""
    gradient_matrix = np.linspace(0, 1, data_size)
    return np.meshgrid(gradient_matrix, gradient_matrix)[0]


@pytest.fixture
def random_data(data_size: int) -> NDArray:
    """Return random data."""
    return np.random.default_rng().random((data_size,) * 2)


@pytest.fixture(params=[4, 6, 8, 10, 12, 14, 16])
def data_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different data sizes."""
    return request.param


@pytest.fixture(
    params=[
        "gradient_data",
        "f16_data",
        "astronaut_data",
        "random_data",
        "empty_data",
        "full_data",
    ]
)
def data(request: pytest.FixtureRequest, data_size: int) -> NDArray:  # noqa: ARG001
    """Fixture to provide different types of data."""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("ppft2_func", [ppft2, ppft2_optimized])
def test_ppft2_returns_complex_dtype(ppft2_func: Callable, data: NDArray) -> None:
    """TODO."""
    assert ppft2_func(data).dtype == np.complex128


@pytest.mark.parametrize("ppft2_func", [ppft2, ppft2_optimized])
def test_ppft2_returns_correct_shape(ppft2_func: Callable, data: NDArray) -> None:
    """Verify that all PPFT2 generates two "pseudo-polar" outputs.

    PPFT2 generates two "pseudo-polar" outputs.
    Each has a shape of (2xN+1, N+1).
    All outputs are then combined into a single 3D matrix with the shape
    of (2, 2xN+1, N+1).
    """
    n = len(data)

    # We expect 2 sectors with each element having a shape of (2*n+1, n+1).
    # For n=4, the expected shape is (2, 9, 5).
    expected_shape = (2, 2 * n + 1, n + 1)

    assert ppft2_func(data).shape == expected_shape


@pytest.mark.parametrize("ppft2_func", [ppft2, ppft2_optimized])
def test_ppft2_sectors_symmetric_data(ppft2_func: Callable, data: NDArray) -> None:
    """Verfiy that all values in a fourier image are equal at its mirrored position."""
    # We only care for the magnitude of the data.
    result = np.abs(ppft2_func(data))

    # The returned PPFT2D has two fourier-transformed sectors,
    # therefore we need to compare both parts.
    np.testing.assert_allclose(np.flipud(result[0]), result[0], atol=1e-14)
    np.testing.assert_allclose(np.flipud(result[1]), result[1], atol=1e-14)


def test_ppft2_equals_rppft2(data: NDArray) -> None:
    """Compare the 'real' (and reduced) PPFT2D with the working PPFT2D."""
    actual = rppft2(data)

    # Output shape does not contain any symmetric data.
    expected = ppft2(data)[:, len(data) :]

    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_ppft2_equals_rppft2_vectorized(data: NDArray) -> None:
    """TODO."""
    # We add arbitrary data as vectorized input.
    expanded_data = np.stack([data, data * 0.5, data * 0.3, data * 0.1])

    actual = rppft2_vectorized(expanded_data)
    expected = np.array([rppft2(d) for d in expanded_data])

    np.testing.assert_equal(actual, expected)


def test_ppft2_equals_ppft2_vectorized(data: NDArray) -> None:
    """TODO."""
    # We add arbitrary data as vectorized input.
    expanded_data = np.stack([data, data * 0.5, data * 0.3, data * 0.1])

    actual = ppft2_vectorized(expanded_data)
    expected = np.array([ppft2(d) for d in expanded_data])

    np.testing.assert_equal(actual, expected)


def test_rppft2_equals_rppft2_optimized(data: NDArray) -> None:
    """TODO."""
    actual = rppft2_optimized(data)
    expected = rppft2(data)

    np.testing.assert_equal(actual, expected)


def test_ppft2_equals_ppft2_optimized(data: NDArray) -> None:
    """TODO."""
    actual = ppft2_optimized(data)
    expected = ppft2(data)

    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("dimension", [1, 3, 4, 5])
def test_ppft2_fails_for_non_2d_data(data_size: int, dimension: int) -> None:
    """TODO."""
    data = np.zeros((data_size,) * dimension)

    with pytest.raises(ValueError, match="Input must be a 2D image"):
        ppft2(data)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6, 7, 8])
def test_ppft2_fails_for_non_square_data(size: int) -> None:
    """TODO."""
    data = np.zeros((4, size))

    with pytest.raises(ValueError, match="Input image must be a square"):
        ppft2(data)


@pytest.mark.parametrize("size", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
def test_ppft2_fails_for_odd_sized_data(size: int) -> None:
    """TODO."""
    data = np.zeros((size,) * 2)

    with pytest.raises(ValueError, match="Input image must have even sides"):
        ppft2(data)
