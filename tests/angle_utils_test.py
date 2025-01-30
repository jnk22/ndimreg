"""Test module for internal rotation approximation function `approx_rotation`.

The function `approx_rotation` is internally used for approximate
comparison of rotation angles.

To verify the test results that use this function, we must ensure that
the function itself is working as intended.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.angle_utils import (
    ApproxAngleScalar,
    ApproxAngleSequenceLike,
    approx_rotation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _pytest.python_api import ApproxBase
    from numpy.typing import NDArray

# TODO: Add more test cases.


@pytest.mark.parametrize(
    ("expected", "rel", "abs", "nan_ok", "expected_cls"),
    [
        (45.0, None, None, False, ApproxAngleScalar),
        ([45.0, 90.0], None, None, False, ApproxAngleSequenceLike),
    ],
)
def test_approx_rotation_instantiation(
    expected: float | Sequence[float],
    rel: float | None,
    abs: float | None,  # noqa: A002
    *,
    nan_ok: bool,
    expected_cls: type[ApproxBase],
) -> None:
    """Test the instantiation of the `approx_rotation` function."""
    # Call the function.
    approximator = approx_rotation(expected, rel, abs, nan_ok=nan_ok)

    # Check if the returned object is an instance of the expected class.
    assert isinstance(approximator, expected_cls)

    # Check if the attributes of the returned object are set correctly.
    assert approximator.expected == expected
    assert approximator.rel == rel
    assert approximator.abs == abs
    assert approximator.nan_ok == nan_ok


def test_approx_rotation_nan_ok() -> None:
    """TODO."""
    # Test with nan_ok=True
    approximator = approx_rotation(45.0, nan_ok=True)
    assert approximator.nan_ok is True


def test_approx_rotation_rel_abs() -> None:
    """TODO."""
    # Test with rel and abs specified
    rel_tol = 0.1
    abs_tol = 0.01

    approximator = approx_rotation(45.0, rel=rel_tol, abs=abs_tol)
    assert approximator.rel == rel_tol
    assert approximator.abs == abs_tol


@pytest.mark.parametrize("rotation", [-360, -270, -180, -90, 0, 90, 180, 270, 360])
def test_approx_rotation_singular_exact_match_numbers_1(rotation: float) -> None:
    """TODO."""
    assert rotation == approx_rotation(rotation)


@pytest.mark.parametrize(
    "rotation",
    [-187.5, -90.123, -45.00001, -12.3456, 0, 12.3456, 45.00001, 90.123, 187.5],
)
def test_approx_rotation_singular_exact_match_numbers_2(rotation: float) -> None:
    """TODO."""
    assert rotation == approx_rotation(rotation)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (-360, 0),
        (0, -360),
        (-359, 1),
        (1, -359),
        (359, -1),
        (-1, 359),
        (360, 0),
        (0, 360),
        (-360, 360),
        (360, -360),
        (-720, 360),
        (-136.0, 224.0),
        (224.0, -136.0),
    ],
)
def test_approx_rotation_singular_full_circle(actual: float, expected: float) -> None:
    """TODO."""
    assert actual == approx_rotation(expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(0.0, 1.0), (-0.5, 0.5), (0.1, 0.3), (-135.0, 224.0), (225.0, -136.0)],
)
def test_approx_rotation_tolerance(actual: float, expected: float) -> None:
    """TODO."""
    assert actual == approx_rotation(expected, abs=1)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.array([0]), np.array([0, 0, 0])),
        (np.array([0, 0, 0]), np.array([0])),
        (np.array([0]), np.array([])),
    ],
)
def test_approx_rotation_singular_full_circle_different_length(
    actual: float, expected: float
) -> None:
    """TODO."""
    assert actual != approx_rotation(expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.array([-360, 0]), np.array([0, -360])),
        (np.array([-359, 1]), np.array([1, -359])),
        (np.array([359, -1]), np.array([-1, 359])),
        (np.array([360, 0]), np.array([0, 360])),
    ],
)
def test_approx_rotation_singular_full_circle_sequence_succeeds(
    actual: NDArray, expected: NDArray
) -> None:
    """TODO."""
    assert actual == approx_rotation(expected)


@pytest.mark.parametrize(
    ("actual", "expected"), [(np.array([-360, 1]), np.array([1, -360]))]
)
def test_approx_rotation_singular_full_circle_sequence_fails(
    actual: NDArray, expected: NDArray
) -> None:
    """TODO."""
    assert actual != approx_rotation(expected)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(None, 45.0), ("45.0", 45.0), ([], 45.0), ({"angle": 45.0}, 45.0)],
)
def test_approx_rotation_invalid_input(actual: float, expected: float) -> None:
    """TODO."""
    assert actual != approx_rotation(expected)
