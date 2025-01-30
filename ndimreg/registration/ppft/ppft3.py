"""Pseudo-Polar Fourier Transform in 3D.

This is an implementation of the PPFT3D [1]. Its implementation is inspired by
the original MATLAB implementation by Gil Shabat (2017) [2], but rewritten
entirely in Python.

References
----------
[1] A. Averbuch and Y. Shkolnisky. 3D Fourier based discrete Radon
        transform. Applied and Computational Harmonic Analysis, 15(1):33-69, 2003
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

from ._ppft import _calculate_pq_pz
from .verify_image import _verify_image_3d

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray

PADS: Final = ((0, 1), (0, 0))
PADS_VEC: Final = ((0, 0), (0, 1), (0, 0))

# TODO: This can probably be optimized to handle two images in one operation.
# TODO: Support np.fft as alternative to SciPy (for NumPy, CuPy, Dask, DPNP).
# TODO: Add a log_memory logger for returned data size (cmp. log_time).
# NOTE: Using cache (especially on GPU) could fill up space... Investigate!


@log_time(print_func=logger.info)
def ppft3(image: NDArray) -> NDArray:
    """Generate 3D pseudo-polar representation of input data.

    This is an implementation of [1].

    Parameters
    ----------
    image
        Input image. Shape must be NxNxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Three 3D pseudo-polar sectors based on input data.

    References
    ----------
    .. [1] Averbuch, A., Shkolnisky, Y.,
           "3D Fourier based discrete Radon transform,"
           Applied and Computational Harmonic Analysis , Vol. 15, No. 1,
           Elsevier BV, p. 33-69. :DOI:`10.1016/s1063-5203(03)00030-7`
    """
    xp = get_namespace(image)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_3d(image)

    n = len(image)
    r_ims = (xp.moveaxis(image, i, 0) for i in range(3))
    p_ims = (xp.pad(r_im, ((n, n + 1), (0, 0), (0, 0))) for r_im in r_ims)

    pq, pz = _calculate_pq_pz(n, 3, xp)
    f_ims = (
        fft.fftshift(fft.fft(fft.ifftshift(p_im, axes=0), axis=0), axes=0)
        for p_im in p_ims
    )
    pp_ims = (__pp_sector(f_im, pq, pz, xp) for f_im in f_ims)

    # TODO: Benchmark and optimize: current/xp.fromiter/pre-allocate?
    return xp.array(list(pp_ims))[:, :, ::-1, ::-1]


@log_time(print_func=logger.info)
def ppft3_vectorized(image: NDArray) -> NDArray:
    """Generate 3D pseudo-polar representation of input data.

    Parameters
    ----------
    image
        Input image. Shape must be NxNxN (i.e., grayscale) with even
        sides.

    Returns
    -------
    NDArray
        Three 3D pseudo-polar sectors based on input data.
    """
    # NOTE: This seems to be much faster on GPU for small images, but
    # requires more memory and therefore does not work for large images.
    xp = get_namespace(image)
    logger.info(f"Using backend: {xp.__name__}")

    _verify_image_3d(image)

    n = len(image)
    r_ims = (xp.moveaxis(image, i, 0) for i in range(3))
    p_ims = (xp.pad(r_im, ((n, n + 1), (0, 0), (0, 0))) for r_im in r_ims)

    pq, pz = _calculate_pq_pz(n, 3, xp)
    f_ims = (
        fft.fftshift(fft.fft(fft.ifftshift(p_im, axes=0), axis=0), axes=0)
        for p_im in p_ims
    )
    pp_ims = (__sec(__sec(f_im, pq, pz, xp), pq, pz, xp) for f_im in f_ims)

    # TODO: Benchmark and optimize: current/xp.fromiter/pre-allocate?
    return xp.array(list(pp_ims))[:, :, ::-1, ::-1]


def __sec(f_im: NDArray, pq: NDArray, pz: NDArray, xp: ModuleType) -> NDArray:
    xp1 = f_im.shape[2] + 1
    md = xp1 + xp1 // 2
    xq = pq[:, :, None]
    xz = pz[:, :, None]
    pq_im = xp.pad(xp.transpose(f_im, axes=(0, 2, 1)), PADS_VEC) * xq

    return fft.ifft(fft.fft(pq_im, xp1 * 3, axis=1) * xz, axis=1)[:, md : md + xp1] * xq


def __pp_sector(f_im: NDArray, pq: NDArray, pz: NDArray, xp: ModuleType) -> NDArray:
    xp = get_namespace(f_im, pq, pz)
    iterator = zip(f_im, pq, pz, strict=True)
    return xp.array(
        [__apply_qz(__apply_qz(f, q, z, xp), q, z, xp) for f, q, z in iterator]
    )


def __apply_qz(f: NDArray, q: NDArray, z: NDArray, xp: ModuleType) -> NDArray:
    f = f.T
    np1 = len(f) + 1
    idx = slice(md := np1 + np1 // 2, md + np1)

    fftx = fft.fft(xp.pad(f, PADS) * q[:, None], n=np1 * 3, axis=0)
    return fft.ifft(fftx * z[:, None], axis=0)[idx] * q[:, None]
