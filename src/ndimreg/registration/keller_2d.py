"""2D image registration using various approaches based on literature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from array_api_compat import array_namespace
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor2D
from ndimreg.transform import Transformation2D

from .base import BaseRegistration
from .keller_2d_utils import _resolve_rotation
from .result import RegistrationDebugImage, ResultInternal2D
from .shift_resolver import resolve_shift
from .translation_fft_2d import TranslationFFT2DRegistration

if TYPE_CHECKING:
    from numpy.typing import NDArray

# TODO: Optimize edge case handling (wrt. performance).


class Keller2DRegistration(BaseRegistration):
    """2D image registration using pseudo log-polar and FFT fourier transformation.

    This is an implementation of [1].

    Notes
    -----
    [1] references an algorithm for sub-pixel shift estimation,
    however we use the `phase_cross_correlation` from `scikit-image`
    instead, which uses another approach. Sub-pixel accuracy can be set
    by the `shift_upsample_factor` parameter.

    This algorithm has a runtime complexity of O(N^2 log N), with
    N = height or width in pixels.

    Capabilities
    ------------
    - Dimension: 2D
    - Translation: Yes
    - Rotation: Yes
    - Scale: No
    - Shear: No

    Limitations
    ------------
    - Images must be of same shape, i.e., NxN.
    - N must be even.
    - The paper shows only translations of up to 20 pixels on a 256x256
      image.

    References
    ----------
    .. [1] Keller, Y., Shkolnisky, Y., Averbuch, A.,
           "The Angular Difference Function and Its Application to Image Registration,"
           IEEE Transactions on Pattern Analysis and Machine Intelligence,
           Vol. 27, No. 6, pp. 969-976, 2005. :DOI:`10.1109/TPAMI.2005.128`
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        rotation_normalization: bool = True,
        rotation_optimization: bool = True,
        rotation_vectorized: bool = False,
        shift_normalization: bool = False,
        shift_disambiguate: bool = False,
        shift_upsample_factor: int = 1,
        highpass_filter: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the 2D Keller registration.

        Parameters
        ----------
        shift_normalization
            Whether to normalize the shift, by default False.
            In general, this should improvde the accuracy of the shift.
            However, it seems that it is currently broken as it
            leads to wrong results within error computation.
            See https://github.com/scikit-image/scikit-image/issues/7078
            for more information.
        shift_upsample_factor
            Upsample factor for the shift, by default 1.
            The upsample factor is used to increase the accuracy of the
            shift. The higher the factor, the more accurate the shift.
            However, it also increases the computation time.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor2D())

        self.__rotation_normalization: bool = rotation_normalization
        self.__rotation_optimization: bool = rotation_optimization
        self.__rotation_vectorized: bool = rotation_vectorized
        self.__highpass_filter: bool = highpass_filter

        self.__shift_registration = TranslationFFT2DRegistration(
            data_space="fourier",
            disambiguate=shift_disambiguate,
            normalization=shift_normalization,
            upsample_factor=shift_upsample_factor,
            debug=self.debug,
        )

    @property
    @override
    def dim(self) -> Literal[2]:
        return 2

    @override
    def _register(
        self, fixed: NDArray, moving: NDArray, **_kwargs: Any
    ) -> ResultInternal2D:
        images = (fixed, moving)
        xp = array_namespace(*images)

        rotation, debug_images = _resolve_rotation(
            images,
            n=len(fixed),
            xp=xp,
            vectorized=self.__rotation_vectorized,
            normalized=self.__rotation_normalization,
            optimized=self.__rotation_optimization,
            highpass_filter=self.__highpass_filter,
            is_complex=any(xp.isdtype(im.dtype, "complex floating") for im in images),
            debug=self.debug,
        )

        moving_rotated = self._transform(moving, rotation=rotation, degrees=False)

        if self.debug and debug_images:
            debug_images.append(
                RegistrationDebugImage(moving_rotated, "re-rotated-moving", dim=2)
            )

        flip_rotation, shift, shift_results = resolve_shift(
            fixed, moving_rotated, self.__shift_registration
        )
        rotation += xp.pi * flip_rotation

        tform = Transformation2D(translation=shift, rotation=np.rad2deg(-rotation))
        return ResultInternal2D(
            tform, sub_results=shift_results, debug_images=debug_images
        )
