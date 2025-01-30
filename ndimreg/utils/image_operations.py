"""Wrapper module for CPU/GPU array functions.

Each function uses the implementation for either CPU or GPU, based on
the current device the data is on.

Execution Flow
--------------
For NVIDIA devices with CUDA support, the library `cucim` is used for
operations on the GPU. As `cucim` does not support AMD/ROCm devices,
operations will be temporarily executed on the CPU and the data will be
transfered back to the GPU afterwards.

CPU:
    Input (CPU) ---> Operations (CPU) ---> Output (CPU)
    (All operations occur entirely on the CPU)

NVIDIA GPU:
    Input (GPU) ---> Operations (GPU) ---> Output (GPU)
    (All operations occur entirely on the GPU)

AMD GPU:
    Input (GPU) ---> Data Transfer (GPU -> CPU) ---> Operations (CPU)
                ---> Data Transfer (CPU -> GPU) ---> Output (GPU)
    (Input resides on GPU, operations are performed on CPU, and results
    are transferred back to GPU)

Notes
-----
- Ensure that the appropriate device is selected before calling this function.
- For AMD GPUs, data transfer between the GPU and CPU may introduce latency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from array_api_compat import get_namespace, is_cupy_array
from loguru import logger
from scipy.ndimage import affine_transform as at_cpu
from skimage.color import rgb2gray as r2g_cpu
from skimage.exposure import rescale_intensity as ri_cpu
from skimage.filters import difference_of_gaussians as dog_cpu
from skimage.filters import window as w_cpu
from skimage.registration import phase_cross_correlation as pcc_cpu
from skimage.transform import resize_local_mean as rlm_cpu
from skimage.transform import warp_polar as wp_cpu
from skimage.util import img_as_float as iaf_cpu
from skimage.util import img_as_ubyte as iau_cpu

from .arrays import to_cupy_array, to_numpy_array

try:
    import cupy as cp
    from cupyx.scipy.ndimage import affine_transform as at_gpu
except ImportError:
    cp = None
    at_gpu = None

try:
    from cucim.skimage.color import rgb2gray as r2g_gpu
    from cucim.skimage.exposure import rescale_intensity as ri_gpu
    from cucim.skimage.filters import difference_of_gaussians as dog_gpu
    from cucim.skimage.filters import window as w_gpu
    from cucim.skimage.registration import phase_cross_correlation as pcc_gpu
    from cucim.skimage.transform import resize_local_mean as rlm_gpu
    from cucim.skimage.transform import warp_polar as wp_gpu
    from cucim.skimage.util import img_as_float as iaf_gpu
    from cucim.skimage.util import img_as_ubyte as iau_gpu
except ImportError:
    r2g_gpu = None
    ri_gpu = None
    dog_gpu = None
    w_gpu = None
    pcc_gpu = None
    rlm_gpu = None
    wp_gpu = None
    iaf_gpu = None
    iau_gpu = None

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

CUCIM_UNAVAILABLE_MSG = "cuCIM is not available, using CPU fallback method"

# TODO: Add loading/saving functions for image data.
# TODO: Add output device target as parameter?


def affine_transform(image: NDArray, inverse_map: NDArray, **kwargs: Any) -> NDArray:
    """TODO."""
    if is_cupy_array(image) and at_gpu:
        affine_transform = at_gpu
        inverse_map = to_cupy_array(inverse_map)
    else:
        affine_transform = at_cpu
        inverse_map = to_numpy_array(inverse_map)

    # We need to transfer the transformed data back to the original
    # image's backend in case of using the CPU-based transform on
    # data that uses it as fallback.
    xp = get_namespace(image)
    return xp.asarray(affine_transform(image, inverse_map, **kwargs))


def phase_cross_correlation(
    reference_image: NDArray, moving_image: NDArray, **kwargs: Any
) -> tuple[Any, float, float] | tuple[NDArray | NDArray[floating[Any]], Any, Any]:
    """TODO."""
    if is_cupy_array(reference_image) and is_cupy_array(moving_image) and cp:
        if pcc_gpu:
            pcc = pcc_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            reference_image = cp.asnumpy(reference_image)
            moving_image = cp.asnumpy(moving_image)
            pcc = pcc_cpu
    else:
        pcc = pcc_cpu

    return pcc(reference_image, moving_image, **kwargs)


def resize_local_mean(
    image: NDArray,
    output_shape: tuple[int, ...],
    grid_mode: bool = True,  # noqa: FBT001, FBT002
    preserve_range: bool = False,  # noqa: FBT001, FBT002
    **kwargs: Any,
) -> NDArray:
    """TODO."""
    xp = get_namespace(image)

    if is_cupy_array(image) and cp:
        if rlm_gpu:
            rlm = rlm_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            image = cp.asnumpy(image)
            rlm = rlm_cpu
    else:
        rlm = rlm_cpu

    return xp.asarray(rlm(image, output_shape, grid_mode, preserve_range, **kwargs))


def warp_polar(
    image: NDArray, center: tuple[int, int] | None = None, **kwargs: Any
) -> NDArray:
    """TODO."""
    xp = get_namespace(image)

    if is_cupy_array(image) and cp:
        if wp_gpu:
            wp = wp_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            image = cp.asnumpy(image)
            wp = wp_cpu
    else:
        wp = wp_cpu

    return xp.asarray(wp(image, center, **kwargs))


def rgb2gray(rgb: NDArray, **kwargs: Any) -> NDArray:
    """TODO."""
    xp = get_namespace(rgb)

    if is_cupy_array(rgb) and cp:
        if r2g_gpu:
            r2g = r2g_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            rgb = cp.asnumpy(rgb)
            r2g = r2g_cpu
    else:
        r2g = r2g_cpu

    return xp.asarray(r2g(rgb, **kwargs))


def rescale_intensity(
    data: NDArray,
    in_range: str | tuple[float, float] = "image",
    out_range: str | tuple[float, float] = "dtype",
) -> NDArray:
    """TODO."""
    xp = get_namespace(data)

    if is_cupy_array(data) and cp:
        if ri_gpu:
            ri = ri_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            data = cp.asnumpy(data)
            ri = ri_cpu
    else:
        ri = ri_cpu

    return xp.asarray(ri(data, in_range=in_range, out_range=out_range))  # type: ignore [reportArgumentType]


def difference_of_gaussians(
    image: NDArray,
    low_sigma: float | None,
    high_sigma: float | None = None,
    **kwargs: Any,
) -> NDArray:
    """TODO."""
    xp = get_namespace(image)

    if is_cupy_array(image) and cp:
        if dog_gpu:
            dog = dog_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            image = cp.asnumpy(image)
            dog = dog_cpu
    else:
        dog = dog_cpu

    return xp.asarray(dog(image, low_sigma, high_sigma, **kwargs))


def window(
    image: NDArray,
    window_type: str,
    shape: tuple[int, ...],
    warp_kwargs: Any | None = None,
) -> NDArray:
    """TODO."""
    xp = get_namespace(image)

    if is_cupy_array(image) and cp:
        if w_gpu:
            w = w_gpu
        else:
            logger.warning(CUCIM_UNAVAILABLE_MSG)
            image = cp.asnumpy(image)
            w = w_cpu
    else:
        w = w_cpu

    return xp.asarray(image * w(window_type, shape, warp_kwargs))


def img_as_ubyte(image: NDArray, force_copy: bool = False) -> NDArray:
    """TODO."""
    if is_cupy_array(image) and cp:
        if iau_gpu:
            img_as_ubyte = iau_gpu
        else:
            image = cp.asnumpy(image)
            img_as_ubyte = iau_cpu
    else:
        img_as_ubyte = iau_cpu

    return img_as_ubyte(image, force_copy)


def img_as_float(image: NDArray, force_copy: bool = False) -> NDArray:
    """TODO."""
    if is_cupy_array(image) and cp:
        if iaf_gpu:
            img_as_float = iaf_gpu
        else:
            image = cp.asnumpy(image)
            img_as_float = iaf_cpu
    else:
        img_as_float = iaf_cpu

    return img_as_float(image, force_copy)
