"""TODO."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar, overload

import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import get_namespace
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from ppftpy import ppft2, rppft2
from scipy import fft

from ndimreg.utils import AutoScipyFftBackend, arr_as_img, fig_to_array, to_numpy_arrays

from .result import RegistrationDebugImage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import ModuleType


DType = TypeVar("DType", bound=np.generic)
DeltaMArray = Annotated[NDArray[DType], Literal["N"]]
OmegaArray = Annotated[NDArray[DType], Literal["N"]]


def _resolve_rotation(
    images: Iterable[NDArray],
    *,
    n: int,
    xp: ModuleType,
    vectorized: bool,
    normalized: bool,
    optimized: bool,
    highpass_filter: bool,
    is_complex: bool = False,
    apply_fft: bool = False,
) -> float:
    images = (im if i == 0 else xp.flip(im, axis=-1) for i, im in enumerate(images))

    if apply_fft:
        images = (fft.rfft(im, axis=0) for im in images)

    mask = __generate_mask(n, xp=xp) if highpass_filter else False
    ppft, idx = (ppft2, n) if is_complex or apply_fft else (rppft2, 0)
    ppft_kwargs = {"vectorized": vectorized, "scipy_fft": True}

    # TODO: Remove merging sectors (see Keller3DRegistration).
    magnitudes = (
        __merge_sectors(
            xp.where(mask, xp.nan, xp.abs(ppft(im, **ppft_kwargs)[..., idx:, :])), xp=xp
        )
        for im in images
    )

    delta_m_func = __delta_m_normalized if normalized else __delta_m_default
    rsi = __generate_radial_sampling_intervals(n, xp=xp)

    with AutoScipyFftBackend(xp):
        delta_m = delta_m_func(*magnitudes, xp=xp, rsi=rsi)

    omega = xp.atleast_2d(delta_m[..., :n] + delta_m[..., n:])
    min_omega = omega[xp.unravel_index(xp.argmin(omega), omega.shape)[0]]

    index_func = __index_optimized if optimized else __index_default
    omega_min_index = index_func(min_omega)

    return __omega_index_to_angle(omega_min_index, n)

    # TODO: Integrate debug output.
    # if self.debug:
    #     debug_images = [
    #         *self.__debug_output(*merged),
    #         *_omega_index_optimized_debug(to_numpy_array(omega)),
    #         *_omega_index_array_debug_wrapper(to_numpy_array(omega)),
    #         RegistrationDebugImage(moving_rotated, "re-rotated-moving", dim=2),
    #     ]
    # else:
    #     debug_images = None


@functools.lru_cache
def __generate_mask(n: int, *, xp: ModuleType) -> NDArray:
    rsi = __generate_radial_sampling_intervals(n, xp=xp)

    return (xp.arange(n + 1) * rsi[:, None] > n).T


@functools.lru_cache
def __generate_radial_sampling_intervals(n: int, xp: ModuleType) -> NDArray:
    rsi = xp.sqrt(4 * ((xp.arange(n // 2 + 1) / n) ** 2) + 1)

    # And return combined as [1.41, ..., 1, ..., 1.41].
    return xp.stack((*rsi[:0:-1], *rsi))


def __merge_sectors(m: NDArray, *, xp: ModuleType) -> NDArray:
    merged = xp.concatenate((m[..., 0, :, :], m[..., 1, :, -2:0:-1]), axis=-1)

    return xp.moveaxis(merged, -1, -2)


def __delta_m_default(
    m1: NDArray, m2: NDArray, *, xp: ModuleType, rsi: NDArray
) -> DeltaMArray:
    # We combine multiple radial sampling intervals as
    # [1.41, ..., 1, ..., 1.41, ..., 1, ..., 1.41) for all angles.
    # Note the last element being excluded as its respective value is
    # equivalent to the first element in the array.
    rsi_combined = xp.hstack((rsi, rsi[1:-1]))

    return xp.nansum(xp.abs(m1 - m2) * rsi_combined[:, None], axis=-1)


def __delta_m_normalized(
    m1: NDArray, m2: NDArray, *, xp: ModuleType, **_kwargs: Any
) -> DeltaMArray:
    counts_1 = xp.sum(~xp.isnan(m1), axis=-1)
    counts_2 = xp.sum(~xp.isnan(m2), axis=-1)
    mean_1 = m1 - xp.nanmean(m1, axis=-1, keepdims=True)
    mean_2 = m2 - xp.nanmean(m2, axis=-1, keepdims=True)
    std_1 = xp.sqrt((1 / counts_1) * xp.nansum((m1 - mean_1) ** 2, axis=-1))
    std_2 = xp.sqrt((1 / counts_2) * xp.nansum((m2 - mean_2) ** 2, axis=-1))

    return xp.nansum((mean_1 - mean_2) ** 2, axis=-1) / (std_1 * std_2)


def __omega_indices(omega: OmegaArray) -> tuple[int, int, int]:
    n = len(omega)
    min_index = __index_default(omega)
    return (min_index - 1) % n, min_index, (min_index + 1) % n


def __index_default(omega: OmegaArray) -> int:
    return get_namespace(omega).argmin(omega).item()


@overload
def __omega_index_to_angle(index: float, n: int) -> float: ...


@overload
def __omega_index_to_angle(index: NDArray, n: int) -> NDArray: ...


def __omega_index_to_angle(index: float | NDArray, n: int) -> float | NDArray:
    return 2 * np.arctan2(2 * (index - n // 2), n)


def __index_optimized(omega: OmegaArray) -> float:
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


def __omega_index_optimized_debug(omega: OmegaArray) -> list[RegistrationDebugImage]:
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
    minimum_angle = __omega_index_to_angle(min_index, n)
    optimized_angle = __omega_index_to_angle(min_index + peak_move, n)
    optimized_shift = -np.rad2deg(minimum_angle - optimized_angle)

    coordinates = (-2, -1, 0, 1, 2)
    min_index_value = omega[min_index]
    values = [
        *np.linspace(left_neigh, min_index_value, 2, endpoint=False),
        *np.linspace(min_index_value, right_neigh, 3),
    ]
    labels = ["Left", "Left-Limit", "Minimum", "Right-Limit", "Right"]
    indices = np.linspace(min_index - 1, min_index + 1, 5)
    angles = __omega_index_to_angle(indices, n)
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


def __omega_index_array_debug_wrapper(omega: NDArray) -> list[RegistrationDebugImage]:
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
    angles = np.array([__omega_index_to_angle(x, n) for x in omega_indices])
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


def __debug_omega_plots(omega_layers: NDArray) -> list[RegistrationDebugImage]:
    n = len(omega_layers[0])

    norm = omega_layers / np.linalg.norm(omega_layers, axis=1, keepdims=True)
    omega_layers_norm = norm * (1 / norm.max())

    min_val_index = np.unravel_index(np.argmin(omega_layers), omega_layers.shape)[0]

    row_mins = np.min(omega_layers, axis=1)
    row_maxs = np.max(omega_layers, axis=1)
    max_diff_index = np.argmax(row_maxs - row_mins)

    return [
        __debug_plot(omega_layers.T, n, "All"),
        __debug_plot(omega_layers.T, n, "All Log-Scaled", yscale="log"),
        __debug_plot(omega_layers_norm.T, n, "Normalized"),
        __debug_plot(omega_layers_norm.sum(0), n, "Normalized Sum"),
        __debug_plot(omega_layers[min_val_index], n, "Minimum Value"),
        __debug_plot(omega_layers[n // 2], n, "Middle Layer"),
        __debug_plot(omega_layers.sum(0), n, "Overall Sum"),
        __debug_plot(omega_layers[max_diff_index], n, "Max Difference"),
    ]


def __debug_plot(
    omega_layers: NDArray, n: int, name: str, *, yscale: str | ScaleBase = "linear"
) -> RegistrationDebugImage:
    plt.figure()
    plt.plot(omega_layers)
    plt.title(f"Angular Difference Function ({name})")
    plt.xlabel(r"$\theta$")
    plt.xticks([0, n - 1], ["0", r"$\pi / 2$"])
    plt.yscale(yscale)

    if omega_layers.ndim == 1:
        # TODO: Add degrees for minimum.
        plt.axvline(
            x=omega_layers.argmin().item(),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Minimum Value",
        )

    image_name = f"adf-function-{'-'.join(name.lower().split(' '))}"
    return RegistrationDebugImage(fig_to_array(), image_name, dim=2, copy=False)


def __debug_output(m1: NDArray, m2: NDArray) -> list[RegistrationDebugImage]:
    m1, m2 = to_numpy_arrays(m1, m2)

    m_images_data = (np.log1p(np.ma.array(m, mask=np.isnan(m))) for m in (m1, m2))
    m_images = (arr_as_img(m[:, ::-1].T, cmap="viridis") for m in m_images_data)
    debug_images = __build_debug_images(tuple(m_images), prefix="m-", dim=2)

    n = len(m1) - 1
    plot_settings = ((True, False), ("--", "-"), ("blue", "green"))
    delta_ms = [
        (_calculate_delta_m(m1, m2, normalization=norm, xp=np), norm, linestyle, color)
        for norm, linestyle, color in zip(*plot_settings, strict=True)
    ]
    omegas = [(_calculate_omega(dm), norm, ls, c) for dm, norm, ls, c in delta_ms]

    config_df = ("Difference Function", [0, n], [0, r"$\pi$"])
    for dm_input in [(delta_ms[0],), (delta_ms[1],), delta_ms]:
        for delta_m, norm, linestyle, color in dm_input:
            label = rf"$\Delta M{'_{N}' if norm else ''}(\theta)$"
            plt.plot(delta_m, label=label, linestyle=linestyle, color=color)

        suffix = "-combined" if len(dm_input) > 1 else f"-norm={dm_input[0][1]}"
        image_name = f"delta-m{suffix}"
        debug_images.append(__build_debug_plot(*config_df, image_name))

    config_adf = ("Angular Difference Function", [0, n // 2], [0, r"$\pi / 2$"])
    for o_input in [(omegas[0],), (omegas[1],), omegas]:
        for delta_m, norm, linestyle, color in o_input:
            label = rf"$ADF{'_{N}' if norm else ''}(\theta)$"
            plt.plot(delta_m, label=label, linestyle=linestyle, color=color)

        suffix = "-combined" if len(o_input) > 1 else f"-norm={o_input[0][1]}"
        image_name = f"adf-function{suffix}"
        debug_images.append(__build_debug_plot(*config_adf, image_name))

    return debug_images


def __build_debug_images(
    images: Sequence[NDArray],
    names: Sequence[str] = ("fixed", "moving"),
    *,
    dim: Literal[2, 3],
    prefix: str = "",
    suffix: str = "",
    copy: bool = True,
) -> list[RegistrationDebugImage]:
    xp = get_namespace(*images)
    images_corrected = (
        xp.fft.fftshift(xp.log1p(xp.abs(im))) if xp.iscomplexobj(im) else im
        for im in images
    )

    return [
        RegistrationDebugImage(data, f"{prefix}{name}{suffix}", dim=dim, copy=copy)
        for name, data in zip(names, images_corrected, strict=True)
    ]


def __build_debug_plot(
    title: str, ticks: list, labels: list, image_name: str
) -> RegistrationDebugImage:
    plt.title(title)
    plt.xlabel(r"$\theta$")
    plt.xticks(ticks, labels)
    plt.margins(0)
    plt.legend()
    plt.tight_layout()

    return RegistrationDebugImage(fig_to_array(), image_name, dim=2, copy=False)
