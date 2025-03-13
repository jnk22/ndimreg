"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING

from array_api_compat import get_namespace
from loguru import logger
from scipy import fft

from ndimreg.utils import AutoScipyFftBackend, log_time

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .result import RegistrationResult
    from .translation_fft_2d import TranslationFFT2DRegistration
    from .translation_fft_3d import TranslationFFT3DRegistration


@log_time(print_func=logger.info)
def resolve_shift(
    fixed: NDArray,
    moving: NDArray,
    registration: TranslationFFT2DRegistration | TranslationFFT3DRegistration,
    *,
    axes: tuple[int, int] = (0, 1),
) -> tuple[bool, tuple[float, ...], list[RegistrationResult]]:
    """Resolve shift and rotation ambguity for 2D and 3D images.

    Parameters
    ----------
    fixed
        Fixed input image.
    moving
        Moving image, will be used as-is and rotated around 180°.
    registration
        Translation FFT registration method to be used.
    axes
        Rotation axis, use default `(0, 1)` for 2D images.

    Returns
    -------
    tuple[bool, tuple[float, ...]]
        First element of tuple is True if 180°-rotated moving image has
        lower error (i.e., rotation must be flipped).
        Second element is the recovered shift.
    """
    # TODO: Test whether rotating FFT is equal to FFT2 of rotated image.
    # TODO: Test whether 'real' FFT can be used instead.
    # TODO: Use fallback instance for translation registration method.
    xp = get_namespace(fixed, moving)

    with AutoScipyFftBackend(xp):
        fixed_fft = fft.fftn(fixed)
        moving_fft = fft.fftn(moving)
        flipped_moving_fft = fft.fftn(xp.rot90(moving, 2, axes))

        result_rot = registration.register(fixed_fft, moving_fft)
        result_rot_flip = registration.register(fixed_fft, flipped_moving_fft)

    # The translation registration methods must return an error value
    # to evaluate whether the flipped or non-flipped rotation results
    # in a better result. As the API theoretically allows for 'None'
    # values to be returned, we raise an error in that case.
    if result_rot.error is None or result_rot_flip.error is None:
        err_msg = "Unexpected None value in translation estimation results"
        raise ValueError(err_msg)

    log_msg = f"Shift errors: 1) {result_rot.error:.2f}, 2) {result_rot_flip.error:.2f}"
    logger.debug(log_msg)

    if xp.isnan(xp.asarray((result_rot.error, result_rot_flip.error))).any():
        logger.warning("Phase cross correlation returned NaN, returning 0 shifts")
        return False, (0,) * len(fixed.shape), [result_rot, result_rot_flip]

    flip = result_rot_flip.error < result_rot.error
    return (
        bool(flip),
        (
            result_rot_flip.transformation.translation
            if flip
            else result_rot.transformation.translation
        ),
        [result_rot, result_rot_flip],
    )
