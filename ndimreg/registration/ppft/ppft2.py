"""Pseudo-Polar Fourier Transform in 2D.

This is an implementation of the PPFT2D [1]. Its implementation is inspired by
the original MATLAB implementation of the 3D version (PPFT3D) by Gil Shabat
(2017) [2], but rewritten entirely in Python.

References
----------
[1] A. Averbuch, R. Coifman, D. Donoho, M. Israeli, and J. Waldén, “Fast slant
        stack: A notion of radon transform for data in a cartesian grid which
        is rapidly computible, algebraically exact, geometrically faithful and
        invertible,” 2001.
[2] Gil Shabat (2025). 3D pseudo polar Fourier and Radon Transforms
        (https://www.mathworks.com/matlabcentral/fileexchange/61815-3d-pseudo-polar-fourier-and-radon-transforms),
        MATLAB Central File Exchange. Retrieved January 30, 2025.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from array_api_compat import get_namespace
from loguru import logger
from scipy import fft

from ndimreg.utils import log_time

from ._ppft import _calculate_nh_idx, _calculate_pq_pz, _calculate_rpq_rpz
from .verify_image import _verify_image_2d

if TYPE_CHECKING:
    from numpy.typing import NDArray

PADS: Final = ((0, 0), (0, 1))
PADS_VEC: Final = ((0, 0), (0, 0), (0, 1))

# TODO: Support np.fft as alternative to SciPy (for NumPy, CuPy, Dask, DPNP).
# NOTE: Using cache (especially on GPU) could fill up space... Investigate!
# NOTE: 'fft(axis=...)' consumes too much memory due to padded input.


@log_time(print_func=logger.info)
def ppft2(image: NDArray) -> NDArray:
    """Generate 2D pseudo-polar representation of input data.

    Parameters
    ----------
    image
        Input image. Shape must be NxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Two 2D pseudo-polar sectors based on input data.
    """
    xp = get_namespace(image)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_2d(image)

    n = len(image)
    pq, pz = _calculate_pq_pz(n, 2, xp)
    nx, npt3, idx = _calculate_nh_idx(n)

    image = xp.flipud(image)
    r_ims = (image, image.T)
    p_ims = (xp.pad(im, ((nx, nx + 1), (0, 0))) for im in r_ims)
    f_ims = (
        fft.fftshift(fft.fft(fft.ifftshift(p_im, axes=0), axis=0), axes=0)
        for p_im in p_ims
    )

    # TODO: Padding might not be optimal yet.
    pq_ims = (xp.pad(f_im, PADS) * pq for f_im in f_ims)
    pp_ims = (
        fft.ifft(fft.fft(pq_im, n=npt3, axis=1) * pz, axis=1)[:, idx] * pq
        for pq_im in pq_ims
    )

    # TODO: Benchmark and optimize: current/xp.fromiter/pre-allocate?
    return xp.array([pp_im[:, ::-1] for pp_im in pp_ims])


@log_time(print_func=logger.info)
def ppft2_optimized(image: NDArray) -> NDArray:
    """Generate 2D pseudo-polar representation of input data in parallel.

    Parameters
    ----------
    image
        Input image. Shape must be NxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Two 2D pseudo-polar sectors based on input data.
    """
    xp = get_namespace(image)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_2d(image)

    image = xp.flipud(image)

    n = len(image)
    pq, pz = _calculate_pq_pz(n, 2, xp)
    nx, npt3, idx = _calculate_nh_idx(n)

    r_ims = xp.array((image, image.T))
    p_ims = xp.pad(r_ims, ((0, 0), (nx, nx + 1), (0, 0)))
    f_ims = fft.fftshift(fft.fft(fft.ifftshift(p_ims, axes=1), axis=1), axes=1)

    pq_ims = xp.pad(f_ims, PADS_VEC) * pq
    pp_ims = fft.ifft(fft.fft(pq_ims, n=npt3, axis=2) * pz, axis=2)[:, :, idx] * pq

    return pp_ims[:, :, ::-1]


@log_time(print_func=logger.info)
def ppft2_vectorized(images: NDArray) -> NDArray:
    """Generate 2D pseudo-polar representation of input data in parallel.

    Parameters
    ----------
    image
        Input image. Shape must be NxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Two 2D pseudo-polar sectors based on input data.
    """
    # TODO: Optimize/Vectorize this as in ppft2_optimized.
    # TODO: Ensure correct input (i.e., multiple images as sequence/ndarray).
    # TODO: Verify that all arrays are using the same backend.
    xp = get_namespace(*images)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_2d(images[0])

    images = images[:, ::-1]

    n = len(images[0])
    pq, pz = _calculate_pq_pz(n, 2, xp)
    nx, npt3, idx = _calculate_nh_idx(n)

    r_ims = (images, images.transpose(0, 2, 1))
    p_ims = (xp.pad(im, ((0, 0), (nx, nx + 1), (0, 0))) for im in r_ims)
    f_ims = (
        fft.fftshift(fft.fft(fft.ifftshift(p_im, axes=1), axis=1), axes=1)
        for p_im in p_ims
    )

    pq_ims = (xp.pad(f_im, PADS_VEC) * pq for f_im in f_ims)
    pp_ims = [
        fft.ifft(fft.fft(pq_im, n=npt3, axis=2) * pz, axis=2)[:, :, idx] * pq
        for pq_im in pq_ims
    ]

    return xp.stack(pp_ims, axis=1)[:, :, :, ::-1]


def rppft2(image: NDArray) -> NDArray:
    """Generate 2D pseudo-polar representation of input data.

    Parameters
    ----------
    image
        Input image. Shape must be NxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Two 2D pseudo-polar sectors based on input data.
    """
    xp = get_namespace(image)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_2d(image)

    n = len(image)
    pq, pz = _calculate_rpq_rpz(n, 2, xp)
    nx, npt3, idx = _calculate_nh_idx(n)

    image = xp.flipud(image)
    r_ims = (image, image.T)
    p_ims = (xp.pad(im, ((nx, nx + 1), (0, 0))) for im in r_ims)
    f_ims = (fft.rfft(fft.ifftshift(p_im, axes=0), axis=0) for p_im in p_ims)

    # TODO: Try to remove symmetric half here as well.
    pq_ims = (xp.pad(fimx, PADS) * pq for fimx in f_ims)
    pp_ims = (
        fft.ifft(fft.fft(pq_im, n=npt3, axis=1) * pz, axis=1)[:, idx] * pq
        for pq_im in pq_ims
    )

    # TODO: Benchmark and optimize: current/xp.fromiter/pre-allocate?
    return xp.array([pp_im[:, ::-1] for pp_im in pp_ims])


@log_time(print_func=logger.info)
def rppft2_optimized(image: NDArray) -> NDArray:
    """Generate 2D pseudo-polar representation of input data in parallel.

    Parameters
    ----------
    image
        Input image. Shape must be NxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Two 2D pseudo-polar sectors based on input data.
    """
    xp = get_namespace(image)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_2d(image)

    image = xp.flipud(image)

    n = len(image)
    pq, pz = _calculate_rpq_rpz(n, 2, xp)
    nx, npt3, idx = _calculate_nh_idx(n)

    r_ims = xp.array((image, image.T))
    p_ims = xp.pad(r_ims, ((0, 0), (nx, nx + 1), (0, 0)))
    f_ims = fft.rfft(fft.ifftshift(p_ims, axes=1), axis=1)

    pq_ims = xp.pad(f_ims, PADS_VEC) * pq
    pp_ims = fft.ifft(fft.fft(pq_ims, n=npt3, axis=2) * pz, axis=2)[:, :, idx] * pq

    return pp_ims[:, :, ::-1]


@log_time(print_func=logger.info)
def rppft2_vectorized(images: NDArray) -> NDArray:
    """Generate 2D pseudo-polar representation of input data in parallel.

    Parameters
    ----------
    image
        Input image. Shape must be NxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Two 2D pseudo-polar sectors based on input data.
    """
    # TODO: Optimize/Vectorize this as in ppft2_optimized.
    # TODO: Ensure correct input (i.e., multiple images as sequence/ndarray).
    # TODO: Verify that all arrays are using the same backend.
    xp = get_namespace(*images)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_2d(images[0])

    images = images[:, ::-1]

    n = len(images[0])
    pq, pz = _calculate_rpq_rpz(n, 2, xp)
    nx, npt3, idx = _calculate_nh_idx(n)

    r_ims = (images, images.transpose(0, 2, 1))
    p_ims = (xp.pad(im, ((0, 0), (nx, nx + 1), (0, 0))) for im in r_ims)
    f_ims = (fft.rfft(fft.ifftshift(p_im, axes=1), axis=1) for p_im in p_ims)

    pq_ims = (xp.pad(f_im, PADS_VEC) * pq for f_im in f_ims)
    pp_ims = [
        fft.ifft(fft.fft(pq_im, n=npt3, axis=2) * pz, axis=2)[:, :, idx] * pq
        for pq_im in pq_ims
    ]

    return xp.stack(pp_ims, axis=1)[:, :, :, ::-1]
