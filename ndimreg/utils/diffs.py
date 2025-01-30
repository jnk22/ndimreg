"""TODO."""

from typing import SupportsFloat, overload

import numpy as np
import pytransform3d.rotations as pr
from numpy.typing import NDArray

NumericType = int | float | np.integer | np.floating


@overload
def angle_diff(angle1: float, angle2: float, *, degrees: bool = True) -> float: ...


@overload
def angle_diff(
    angle1: NumericType, angle2: NumericType, *, degrees: bool = True
) -> SupportsFloat: ...


@overload
def angle_diff(
    angle1: NDArray, angle2: NDArray, *, degrees: bool = True
) -> NDArray: ...


def angle_diff(
    angle1: NumericType | NDArray,
    angle2: NumericType | NDArray,
    *,
    degrees: bool = True,
) -> SupportsFloat | NDArray:
    """TODO."""
    difference = angle2 - angle1

    # Normalize the difference to [-180, 180] for degrees or [-pi, pi] for radians.
    # TODO: Check whether 'pytransform3d.rotations.norm_angle' is more precise.
    if degrees:
        difference = (difference + 180) % 360 - 180
    else:
        difference = (difference + np.pi) % (2 * np.pi) - np.pi

    return abs(difference)


def translation_diff(
    translation1: tuple[float, ...] | NDArray, translation2: tuple[float, ...] | NDArray
) -> NDArray:
    """TODO."""
    if len(translation1) != len(translation2):
        msg = "Translation parameters must be of equal length"
        raise ValueError(msg)

    return abs(np.asarray(translation1) - np.asarray(translation2))


def scale_diff_rel(expected: float, actual: float) -> float:
    """Return scale difference as relative difference."""
    # TODO: Implement safe mode for potential division by zero.
    return (scale_diff_abs(expected, actual) / expected) * 100


def scale_diff_abs(scale1: float, scale2: float) -> float:
    """Return scale difference as absolute difference."""
    return abs(scale1 - scale2)


# TODO: Allow all rotation inputs (see transform() function input).
def euler_xyz_to_quaternion_dist(
    euler1: tuple[float, float, float],
    euler2: tuple[float, float, float],
    *,
    degrees: bool = True,
) -> float:
    """TODO."""
    euler1_rad = np.deg2rad(euler1) if degrees else euler1
    euler2_rad = np.deg2rad(euler2) if degrees else euler2
    q1 = pr.quaternion_from_euler(euler1_rad, 0, 1, 2, extrinsic=False)
    q2 = pr.quaternion_from_euler(euler2_rad, 0, 1, 2, extrinsic=False)
    return pr.quaternion_dist(q1, q2).item()


def euler_angles_diff(
    euler1: tuple[float, float, float],
    euler2: tuple[float, float, float],
    *,
    degrees: bool = True,
) -> NDArray:
    """TODO."""
    # NOTE: Should this be normalized/recalculated before to ensure proper ranges?
    return angle_diff(np.array(euler1), np.array(euler2), degrees=degrees)


def quaternion_dist(
    quat1: tuple[float, float, float, float], quat2: tuple[float, float, float, float]
) -> float:
    """TODO."""
    return pr.quaternion_dist(quat1, quat2).item()
