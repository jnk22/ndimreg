"""2D image registration using log-polar warp for rotation recovery.

This is based on the example from scikit-image.
See https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from array_api_compat import array_namespace
from loguru import logger
from pytransform3d.rotations import norm_angle
from scipy import fft
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor2D
from ndimreg.transform import Transformation2D
from ndimreg.utils import AutoScipyFftBackend, arr_as_img
from ndimreg.utils.image_operations import warp_polar

from .base import BaseRegistration
from .result import ResultInternal2D
from .shift_resolver import resolve_shift
from .translation_fft_2d import TranslationFFT2DRegistration

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ndimreg.transform import InterpolationOrder


class Scikit2DRegistration(BaseRegistration):
    """2D image registration based on ``scikit-image``.

    This code is based on the example from scikit-image
    [Using Polar and Log-Polar Transformations for Registration](https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html#).

    This is a short example that uses log-polar transformation and
    Fourier transform to recover rotation and scale from two images.

    Capabilities
    ------------
    - Dimension: 2D
    - Translation: Yes
    - Rotation: Yes
    - Scale: Yes
    - Shear: No

    Limitations
    ----------
    - Scale change should be within 1.8-2.

    Source: https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html#some-notes-on-this-approach
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        shift_normalization: bool = False,
        shift_disambiguate: bool = False,
        shift_upsample_factor: int = 1,
        rotation_normalization: bool = False,
        rotation_disambiguate: bool = False,
        rotation_upsample_factor: int = 1,
        polar_transform_interpolation_order: InterpolationOrder = 0,
        **kwargs: Any,
    ) -> None:
        """TODO."""
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor2D())

        self._polar_transform_interpolation_order: InterpolationOrder = (
            polar_transform_interpolation_order
        )

        self.__rotation_registration = TranslationFFT2DRegistration(
            disambiguate=rotation_disambiguate,
            normalization=rotation_normalization,
            upsample_factor=rotation_upsample_factor,
            debug=self.debug,
        )

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
        xp = array_namespace(fixed, moving)
        fft_images = (xp.abs(fft.fftshift(fft.fft2(im))) for im in (fixed, moving))
        shape = fixed.shape
        radius = shape[0] // 8

        debug_images = [] if self.debug else None
        if debug_images is not None:
            # FIX: Zoom does not work as intended.
            with AutoScipyFftBackend(xp):
                fft_images = tuple(fft_images)
            slices = tuple(slice(c - radius, c + radius) for c in np.array(shape) // 2)
            f_im, m_im = (arr_as_img(im[slices], "magma") for im in fft_images)
            debug_images.extend(
                self._build_debug_images((f_im, m_im), prefix="fft-zoomed-", copy=False)
            )

        polar_warp_kwargs: Final = {
            "scaling": "log",
            "radius": radius,
            "output_shape": shape,
            "order": self._polar_transform_interpolation_order,
        }

        # Only use half of FFT.
        # TODO: Evaluate whether real FFT can be used instead.
        fft_half_limit = shape[0] // 2

        log_polar_images = (
            warp_polar(im, **polar_warp_kwargs)[:fft_half_limit] for im in fft_images
        )

        if debug_images is not None:
            with AutoScipyFftBackend(xp):
                log_polar_images = tuple(log_polar_images)
            f_im, m_im = (arr_as_img(im, "magma") for im in log_polar_images)
            debug_images.extend(
                self._build_debug_images((f_im, m_im), prefix="log-polar-", copy=False)
            )

        with AutoScipyFftBackend(xp):
            rotation_result = self.__rotation_registration.register(*log_polar_images)

        rotation_shifts = rotation_result.transformation.translation

        rotation = ((2 * xp.pi) / shape[0] * rotation_shifts[1]).item()
        scale = xp.exp(rotation_shifts[0] / (shape[1] / xp.log(radius))).item()

        if xp.isnan(scale):
            msg = "Registration failed due to scale=NaN, returning default values"
            logger.warning(msg)
            return ResultInternal2D()

        moving_rotated_scaled = self._transform(
            moving, rotation=rotation, scale=scale, degrees=False
        )
        flip_rotation, shift, shift_results = resolve_shift(
            fixed, moving_rotated_scaled, self.__shift_registration
        )
        rotation += xp.pi * flip_rotation

        tform = Transformation2D(
            translation=(shift[0], shift[1]),
            rotation=-np.rad2deg(norm_angle(rotation)).item(),
            scale=(1 / scale),
        )

        return ResultInternal2D(
            tform,
            sub_results=[rotation_result, *shift_results],
            debug_images=debug_images,
        )
