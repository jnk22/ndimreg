"""Test module for ppft3 code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ndimreg.image import Image3D
from ndimreg.registration.ppft import ppft3, ppft3_vectorized

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def illumination_data(data_size: int) -> NDArray:
    """Return the illumination test image data."""
    image = Image3D.from_path(Path("data/3d/illumination.tif"))

    # PPFT3D only works with grayscale images.
    return image.resize_to_shape(data_size).grayscale().data


@pytest.fixture
def empty_data(data_size: int) -> NDArray:
    """Return data with only zeros (i.e., black) image."""
    return np.zeros((data_size,) * 3)


@pytest.fixture
def full_data(data_size: int) -> NDArray:
    """Return data with only ones (i.e., white) image."""
    return np.ones((data_size,) * 3)


@pytest.fixture
def gradient_data(data_size: int) -> NDArray:
    """Return gradient image data."""
    gradient_matrix = np.linspace(0, 1, data_size)
    return np.meshgrid(gradient_matrix, gradient_matrix, gradient_matrix)[0]


@pytest.fixture
def random_data(data_size: int) -> NDArray:
    """Return random data."""
    return np.random.default_rng().random((data_size,) * 3)


@pytest.fixture(params=[2, 4, 6, 8])
def data_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide different data sizes."""
    return request.param


@pytest.fixture(
    params=[
        "gradient_data",
        "illumination_data",
        "random_data",
        "empty_data",
        "full_data",
    ]
)
def data(request: pytest.FixtureRequest, data_size: int) -> NDArray:  # noqa: ARG001
    """Fixture to provide different types of data."""
    return request.getfixturevalue(request.param)


def test_ppft3_returns_complex_dtype(data: NDArray) -> None:
    """TODO."""
    assert ppft3(data).dtype == np.complex128


def test_ppft3_returns_correct_shape(data: NDArray) -> None:
    """Verify that all PPFT3 generates three "pseudo-polar" outputs.

    PPFT3 generates three "pseudo-polar" outputs.
    Each has a shape of (3xN+1, N+1, N+1).
    All outputs are then combined into a single 4D matrix with the shape
    of (3, 3xN+1, N+1, N+1).
    """
    n = len(data)

    # We expect 3 sectors with each one having a shape of (3*n+1, n+1, n+1).
    # For n=4, the expected shape is (3, 13, 5, 5).
    expected_shape = (3, 3 * n + 1, n + 1, n + 1)

    assert ppft3(data).shape == expected_shape


def test_ppft3_sectors_symmetric_data(data: NDArray) -> None:
    """Verfiy that all values in a fourier image are equal at its mirrored position."""
    # We only care for the magnitude of the data.
    sec1, sec2, sec3 = np.abs(ppft3(data))

    # The returned ppft3D has two fourier-transformed sectors,
    # therefore we need to compare both parts.
    assert sec1.shape == sec2.shape == sec3.shape

    assert_allclose(np.flipud(sec1), sec1)
    assert_allclose(np.flipud(sec2), sec2)
    assert_allclose(np.flipud(sec3), sec3)


def test_ppft3_equals_ppft3_vectorized(data: NDArray) -> None:
    """TODO."""
    actual = ppft3(data)
    expected = ppft3_vectorized(data)

    assert_allclose(actual, expected)


@pytest.mark.parametrize("dimension", [1, 2, 4, 5])
def test_ppft3_fails_for_non_3d_data(data_size: int, dimension: int) -> None:
    """TODO."""
    data = np.zeros((data_size,) * dimension)

    with pytest.raises(ValueError, match="Input must be a 3D image"):
        ppft3(data)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6, 7, 8])
def test_ppft3_fails_for_non_cube_data(size: int) -> None:
    """TODO."""
    data = np.zeros((4, size, size))

    with pytest.raises(ValueError, match="Input image must be a cube"):
        ppft3(data)


@pytest.mark.parametrize("size", [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
def test_ppft3_fails_for_odd_sized_data(size: int) -> None:
    """TODO."""
    data = np.zeros((size,) * 3)

    with pytest.raises(ValueError, match="Input image must have even sides"):
        ppft3(data)
