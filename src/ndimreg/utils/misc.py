"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

CvalMode = Literal["min", "max", "mean"]


def array_to_shape(
    array: NDArray,
    *,
    mode: Literal["ceil", "floor", "round"] = "ceil",
    safe_mode: bool = True,
) -> tuple[int, ...]:
    """TODO."""
    match mode:
        case "ceil":
            func = np.ceil
        case "floor":
            func = np.floor
        case "round":
            func = np.round

    array = func(array).astype(int)

    if safe_mode:
        # Ensure that output shape has at least one pixel in each
        # dimension.
        array = np.maximum(np.ones(len(array), dtype=int), array)

    return tuple(array.tolist())


def calculate_cval(data: NDArray, cval: float | CvalMode) -> float:
    """TODO."""
    match cval:
        case "min":
            return data.min().item()
        case "max":
            return data.max().item()
        case "mean":
            return data.mean().item()

    return cval
