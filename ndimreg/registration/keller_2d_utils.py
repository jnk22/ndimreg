"""TODO."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Annotated, Literal, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import get_namespace
from loguru import logger
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from ndimreg.utils import fig_to_array

from .result import RegistrationDebugImage

if TYPE_CHECKING:
    from types import ModuleType

DType = TypeVar("DType", bound=np.generic)
DeltaMArray = Annotated[NDArray[DType], Literal["N"]]
OmegaArray = Annotated[NDArray[DType], Literal["N"]]


def omega_index(omega: OmegaArray) -> int:
    """TODO."""
    return get_namespace(omega).argmin(omega).item()


@overload
def omega_index_to_angle(index: float, n: int) -> float: ...


@overload
def omega_index_to_angle(index: NDArray, n: int) -> NDArray: ...


def omega_index_to_angle(index: float | NDArray, n: int) -> float | NDArray:
    """Convert an index to an angle based on the size of `n`.

    Parameters
    ----------
    index:
        A float or a numpy array of floats representing the index.
    n:
        An integer representing the size.

    Returns
    -------
    float | NDArray
        The angle as a float if `index` is a float, or a numpy array of
        floats if `index` is an NDArray.
    """
    return 2 * np.arctan2(2 * (index - n // 2), n)


def merge_sectors(m: NDArray, *, mask: NDArray | bool, xp: ModuleType) -> NDArray:
    """TODO."""
    # Using a mask for exluding points from a ray that exceed the
    # radial limit is supposed to remove high-frequency values.
    # Slicing the data starting from 'n' cuts off the mirrored data that
    # exists due to the Fourier Transform (PPFT2D).
    m1 = xp.where(mask, xp.nan, m[0])
    m2 = xp.where(mask, xp.nan, m[1])

    return xp.hstack((m1, m2[:, -2:0:-1])).T


def calculate_omega(delta_m: DeltaMArray) -> OmegaArray:
    """TODO."""
    n = len(delta_m) // 2
    return delta_m[:n] + delta_m[n:]


def calculate_delta_m(
    m1: NDArray, m2: NDArray, *, normalization: bool, xp: ModuleType
) -> DeltaMArray:
    """TODO."""
    return (
        __calculate_delta_m_normalized if normalization else __calculate_delta_m_default
    )(m1, m2, xp=xp)


@functools.lru_cache
def highpass_filter_mask(n: int) -> NDArray:
    """TODO."""
    return (np.arange(n + 1) * __generate_radial_sampling_intervals(n)[:, None] > n).T


def __calculate_delta_m_default(
    m1: NDArray, m2: NDArray, *, xp: ModuleType
) -> DeltaMArray:
    n = len(m1[0]) - 1
    rsi = __generate_radial_sampling_intervals(n)

    # We combine multiple radial sampling intervals as
    # [1.41, ..., 1, ..., 1.41, ..., 1, ..., 1.41) for all angles.
    # Note the last element being excluded as its respective value is
    # equivalent to the first element in the array.
    rsi_combined = xp.hstack((rsi, rsi[1:-1]))

    return xp.nansum(xp.abs(m1 - m2) * rsi_combined[:, None], axis=1)


def __calculate_delta_m_normalized(
    m1: NDArray, m2: NDArray, *, xp: ModuleType
) -> DeltaMArray:
    counts_1 = xp.sum(~xp.isnan(m1), axis=1)
    counts_2 = xp.sum(~xp.isnan(m2), axis=1)
    mean_1 = m1 - xp.nanmean(m1, axis=1, keepdims=True)
    mean_2 = m2 - xp.nanmean(m2, axis=1, keepdims=True)
    std_1 = xp.sqrt((1 / counts_1) * xp.nansum((m1 - mean_1) ** 2, axis=1))
    std_2 = xp.sqrt((1 / counts_2) * xp.nansum((m2 - mean_2) ** 2, axis=1))

    return xp.nansum((mean_1 - mean_2) ** 2, axis=1) / (std_1 * std_2)


@functools.lru_cache
def __generate_radial_sampling_intervals(n: int) -> NDArray:
    # Create the radial sampling intervals in ascending order: [1, ..., 1.41]
    rsi = np.sqrt(4 * ((np.arange(n // 2 + 1) / n) ** 2) + 1)

    # And return combined as [1.41, ..., 1, ..., 1.41].
    return np.array((*rsi[:0:-1], *rsi))


def __omega_indices(omega: OmegaArray) -> tuple[int, int, int]:
    """TODO."""
    n = len(omega)
    min_index = omega_index(omega)
    return (min_index - 1) % n, min_index, (min_index + 1) % n


def omega_index_optimized(omega: OmegaArray) -> float:
    """Optimizes the angle corresponding to the minimum value of the omega array.

    Arguments:
    ---------
    omega
        Array of omega values.

    Returns:
    -------
    float
        A float representing the optimized index in the omega array
        after local adjustment.
    """
    left_index, min_index, right_index = __omega_indices(omega)
    left_neigh, right_neigh = omega[(left_index, right_index),]

    if left_neigh < right_neigh:
        min_neigh, max_neigh = left_neigh, right_neigh
        sign = -1
    else:
        min_neigh, max_neigh = right_neigh, left_neigh
        sign = 1

    effect = np.divide(min_neigh, max_neigh).item()

    if math.isnan(effect) or math.isinf(effect):
        logger.warning("Max neighbor value is zero, potential issue with input data")
        return min_index

    peak_move = 0.5 * sign * (1 - effect)
    logger.debug(f"Shifted omega peak by {peak_move:.4f} at index {min_index}")

    return min_index + peak_move


def omega_index_optimized_debug(omega: OmegaArray) -> list[RegistrationDebugImage]:
    """Create debug output for optimized omega index.

    Arguments:
    ---------
    omega:
        Array of omega values.

    Returns:
    -------
    RegistrationDebugImage
        A debug image showing the omega peak optimization.
    """
    left_index, min_index, right_index = __omega_indices(omega)
    left_neigh, right_neigh = omega[(left_index, right_index),]
    min_neigh, max_neigh = sorted((left_neigh, right_neigh))

    if max_neigh == 0:
        msg = "Maximum neighbor value is zero (potential issue with input data)"
        logger.warning(msg)

        plt.text(0.5, 0.5, msg, color="red", ha="center", va="center")
        plt.axis("off")
        # TODO: Add non-optimized version.
        return [
            RegistrationDebugImage(
                fig_to_array(), "adf-index-optimization", dim=2, copy=False
            )
        ]

    sign = -1 if left_neigh < right_neigh else 1
    peak_move = sign * (1 - min_neigh / max_neigh) * 0.5

    n = len(omega)
    minimum_angle = omega_index_to_angle(min_index, n)
    optimized_angle = omega_index_to_angle(min_index + peak_move, n)
    optimized_shift = -np.rad2deg(minimum_angle - optimized_angle)

    coordinates = (-2, -1, 0, 1, 2)
    min_index_value = omega[min_index]
    values = [
        *np.linspace(left_neigh, min_index_value, 2, endpoint=False),
        *np.linspace(min_index_value, right_neigh, 3),
    ]
    labels = ["Left", "Left-Limit", "Minimum", "Right-Limit", "Right"]
    indices = np.linspace(min_index - 1, min_index + 1, 5)
    angles = omega_index_to_angle(indices, n)
    tick_labels = [
        f"{la}\n{dg:.2f}°" for la, dg in zip(labels, np.rad2deg(angles), strict=True)
    ]

    ha = "left" if peak_move < 0 else "right"
    text = rf"Shift: {optimized_shift:+.2f}° $\Rightarrow$ {np.rad2deg(optimized_angle):+.2f}°"
    offset_label = "Optimization Offset"
    line_shift = optimized_shift / np.rad2deg(abs(angles[1] - angles[2]))

    fig = plt.figure()
    plt.plot(coordinates, values, linestyle="-", color="b", label="Values")
    plt.grid(axis="x")
    plt.title("ADF Neighbor Optimization")
    plt.xticks(coordinates, tick_labels)
    plt.xlabel("Position/Degrees")
    plt.ylabel("Angular Difference Value")
    plt.axvline(
        line_shift, color="red", linestyle="--", linewidth=2, label=offset_label
    )
    plt.axvspan(-1, 1, color="green", alpha=0.3)
    plt.text(
        0.54 + line_shift / 6, 0.8, text, color="red", ha=ha, transform=fig.transFigure
    )
    plt.legend()
    plt.tight_layout()

    im1 = RegistrationDebugImage(
        fig_to_array(), "adf-index-optimization", dim=2, copy=False
    )

    fig = plt.figure()
    plt.plot(coordinates[::2], values[::2], linestyle="-", color="b", label="Values")
    plt.grid(axis="x")
    plt.title("ADF Neighbors")
    plt.xticks(coordinates[::2], tick_labels[::2])
    plt.xlabel("Position/Degrees")
    plt.ylabel("Angular Difference Value")
    plt.legend()
    plt.tight_layout()

    im2 = RegistrationDebugImage(
        fig_to_array(), "adf-index-optimization-before", dim=2, copy=False
    )

    return [im1, im2]


def omega_index_array_debug_wrapper(omega: NDArray) -> list[RegistrationDebugImage]:
    """TODO."""
    n = len(omega)
    min_indices = np.array(__omega_indices(omega))
    min_excerpt = omega[min_indices]

    return [
        __omega_index_array_debug(omega, np.arange(n), n, "omega-array"),
        __omega_index_array_debug(min_excerpt, min_indices, n, "omega-array-excerpt"),
    ]


def __omega_index_array_debug(
    omega: NDArray, omega_indices: NDArray, n: int, name: str
) -> RegistrationDebugImage:
    """TODO."""
    angles = np.array([omega_index_to_angle(x, n) for x in omega_indices])
    angles_flip = angles + np.pi
    angles, angles_flip = (np.rad2deg(pr.norm_angle(x)) for x in (angles, angles_flip))
    left_index, min_index, right_index = __omega_indices(omega)

    n = len(omega)
    fig_size = (max(8, n), 2)
    precision = ".2f"

    if n <= 4:
        font_size = 8
    elif n <= 16:
        font_size = 7
    else:
        font_size = 6

    _, ax = plt.subplots(figsize=fig_size)
    zipperator = zip(omega_indices, omega, angles, angles_flip, strict=True)
    for i, (index, value, angle, angle_flip) in enumerate(zipperator):
        if i == min_index:
            bg_color = "lightgreen"
        elif i in (left_index, right_index):
            bg_color = "lightblue"
        else:
            bg_color = "grey"

        ax.barh(0, 1, left=i, color=bg_color, edgecolor="black")
        ax.add_patch(Rectangle((i, -0.4), 0.2, 0.2, color="white", ec="black"))
        ax.text(
            i + 0.1,
            -0.3,
            str(index),
            ha="center",
            va="center",
            fontsize=font_size,
            color="black",
        )
        text_value = f"Value: {value:{precision}}"
        text_angle = rf"$\Rightarrow$ {angle:{precision}}° or {angle_flip:{precision}}°"
        text_full = f"{text_value}\n{text_angle}"
        ax.text(
            i + 0.5,
            0,
            text_full,
            ha="center",
            va="center",
            fontsize=font_size,
            color="black",
        )

    # Add arrows on left and right side to indicate circular array.
    # We use dotted lines if we only show an excerpt that does not
    # represent the whole omega array.
    arrow_props = {
        "arrowstyle": "<-",
        "linestyle": "dotted"
        if omega_indices[0] != 0 or omega_indices[-1] != n
        else "solid",
        "color": "black",
        "lw": 2,
    }
    ax.annotate("", xy=(-0.5, 0), xytext=(0, 0), arrowprops=arrow_props)
    ax.annotate("", xy=(n, 0), xytext=(n + 0.5, 0), arrowprops=arrow_props)

    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    return RegistrationDebugImage(fig_to_array(), name, dim=2, copy=False)
