"""TODO."""

import math

import numpy as np
import pytest

from ndimreg.utils.diffs import angle_diff


# Test for scalar inputs in degrees
@pytest.mark.parametrize(
    ("angle1", "angle2", "expected"),
    [(30, 60, 30), (60, 30, 30), (350, 10, 20), (10, 350, 20), (180, -180, 0)],
)
def test_angle_diff_scalar_degrees(
    angle1: float, angle2: float, expected: float
) -> None:
    """TODO."""
    assert math.isclose(angle_diff(angle1, angle2), expected)


# Test for scalar inputs in radians
@pytest.mark.parametrize(
    ("angle1", "angle2", "expected"),
    [
        (np.pi / 6, np.pi / 3, np.pi / 6),
        (np.pi / 3, np.pi / 6, np.pi / 6),
        (np.pi, -np.pi, 0),
    ],
)
def test_angle_diff_scalar_radians(
    angle1: float, angle2: float, expected: float
) -> None:
    """TODO."""
    assert math.isclose(angle_diff(angle1, angle2, degrees=False), expected)


# Test for array inputs in degrees
@pytest.mark.parametrize(
    ("angle1", "angle2", "expected"),
    [(np.array([0, 30, 60]), np.array([30, 60, 90]), np.array([30, 30, 30]))],
)
def test_angle_diff_array_degrees(
    angle1: float, angle2: float, expected: float
) -> None:
    """TODO."""
    np.testing.assert_allclose(angle_diff(angle1, angle2), expected)


# Test for array inputs in radians
@pytest.mark.parametrize(
    ("angle1", "angle2", "expected"),
    [
        (
            np.array([0, np.pi / 6, np.pi / 3]),
            np.array([np.pi / 6, np.pi / 3, np.pi / 2]),
            np.array([np.pi / 6, np.pi / 6, np.pi / 6]),
        )
    ],
)
def test_angle_diff_array_radians(
    angle1: float, angle2: float, expected: float
) -> None:
    """TODO."""
    np.testing.assert_allclose(angle_diff(angle1, angle2, degrees=False), expected)


# Test for edge cases
@pytest.mark.parametrize(
    ("angle1", "angle2", "expected"), [(180, -180, 0), (170, 190, 20), (190, 170, 20)]
)
def test_angle_diff_edge_cases(angle1: float, angle2: float, expected: float) -> None:
    """TODO."""
    assert math.isclose(angle_diff(angle1, angle2), expected)


# Test different data types
@pytest.mark.parametrize(
    ("angle1", "angle2", "expected"),
    [
        (1, 2, 1),
        (np.float64(1), np.float64(2), 1),
        (1.5, 2.5, 1),
        (np.array([1]), np.array([2]), np.array([1])),
    ],
)
def test_angle_diff_different_types(
    angle1: float, angle2: float, expected: float
) -> None:
    """TODO."""
    assert np.isclose(angle_diff(angle1, angle2), expected)
