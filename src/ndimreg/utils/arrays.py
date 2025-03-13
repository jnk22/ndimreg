"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from array_api_compat import is_cupy_array

try:
    import cupy as cp
except ImportError:
    cp = None

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

Device = Literal["cpu", "gpu"]


def to_cupy_array(array: NDArray, *, copy: bool = False) -> NDArray:
    """TODO."""
    if not cp:
        msg = "CuPy is not available"
        raise ValueError(msg)

    if is_cupy_array(array):
        return array.copy() if copy else array

    # Copy is not required as a CPU-to-GPU transfer always copies data.
    return cp.asarray(array)


def to_cupy_arrays(*arrays: NDArray, copy: bool = False) -> Generator[NDArray]:
    """TODO."""
    yield from (to_cupy_array(arr, copy=copy) for arr in arrays)


def to_numpy_array(array: NDArray, *, copy: bool = False) -> NDArray:
    """TODO."""
    if is_cupy_array(array) and cp:
        # Copy is not required as a GPU-to-CPU transfer always copies data.
        return cp.asnumpy(array)

    return array.copy() if copy else array


def to_numpy_arrays(*arrays: NDArray, copy: bool = False) -> Generator[NDArray]:
    """TODO."""
    yield from (to_numpy_array(arr, copy=copy) for arr in arrays)


def to_device_array(
    array: NDArray, *, device: Device | None, copy: bool = False
) -> NDArray:
    """TODO."""
    match device:
        case "cpu":
            return to_numpy_array(array, copy=copy)
        case "gpu":
            return to_cupy_array(array, copy=copy)

    return array.copy() if copy else array


def to_device_arrays(
    *arrays: NDArray, device: Device | None, copy: bool = False
) -> Generator[NDArray]:
    """TODO."""
    yield from (to_device_array(arr, device=device, copy=copy) for arr in arrays)
