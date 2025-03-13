"""Utility functions for images."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from ndimreg.utils import to_numpy_array

from .timer import log_time

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

ContextType = Literal["paper", "notebook", "talk", "poster"]


@log_time(print_func=logger.debug)
def arr_as_img(data: NDArray, cmap: str | None = "gray") -> NDArray:
    """TODO."""
    # TODO: Compare/Merge with 'fig_to_array' performance/quality.
    with io.BytesIO() as buf:
        plt.imsave(buf, to_numpy_array(data), cmap=cmap)
        buf.seek(0)
        img = np.array(plt.imread(buf))

    plt.close()
    return img


@log_time(print_func=logger.debug)
def fig_as_img() -> NDArray:
    """TODO."""
    # Source: https://www.geeksforgeeks.org/save-plot-to-numpy-array-using-matplotlib/
    # TODO: Compare/Merge with 'fig_to_array' performance/quality.
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01)
        buf.seek(0)
        data = np.array(plt.imread(buf))

    plt.close()
    return data


@log_time(print_func=logger.debug)
def fig_to_array(fig: Figure | None = None) -> NDArray:
    """TODO."""
    # Source: https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    # TODO: Test whether https://stackoverflow.com/a/67823421/24321379 is better.
    fig = fig or plt.gcf()
    fig.canvas.draw()

    # Get the image as a numpy array from the buffer.
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]

    plt.close()
    return data


def set_matplotlib_context(
    context: ContextType = "notebook", font_scale: float = 1.0
) -> None:
    """Set matplotlib rcParams to mimic seaborn's set_context.

    Parameters
    ----------
        context (str): "paper", "notebook", "talk", or "poster"
        font_scale (float): Scaling factor for fonts and elements
    """
    contexts = {
        "paper": {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 0.8,
            "lines.markersize": 4,
        },
        "notebook": {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        },
        "talk": {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        },
        "poster": {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 3,
            "lines.markersize": 10,
        },
    }

    if context not in contexts:
        msg = "Context must be 'paper', 'notebook', 'talk', or 'poster'"
        raise ValueError(msg)

    params = {
        k: v * font_scale if "size" in k or "linewidth" in k else v
        for k, v in contexts[context].items()
    }

    plt.rcParams.update(params)
