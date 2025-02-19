"""2D image registration using various approaches based on literature."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from array_api_compat import get_namespace
from matplotlib import pyplot as plt
from ppft_nd import rppft2
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor2D
from ndimreg.transform import Transformation2D
from ndimreg.utils import arr_as_img, fig_to_array, to_numpy_arrays
from ndimreg.utils.arrays import to_numpy_array
from ndimreg.utils.fft import AutoScipyFftBackend

from .base import BaseRegistration
from .keller_2d_utils import (
    calculate_delta_m,
    calculate_omega,
    highpass_filter_mask,
    merge_sectors,
    omega_index,
    omega_index_array_debug_wrapper,
    omega_index_optimized,
    omega_index_optimized_debug,
    omega_index_to_angle,
)
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

    def __init__(  # noqa: D417, PLR0913
        self,
        *,
        rotation_normalization: bool = True,
        rotation_optimization: bool = True,
        highpass_filter: bool = True,
        shift_normalization: bool = False,
        shift_disambiguate: bool = False,
        shift_upsample_factor: int = 1,
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
        xp = get_namespace(fixed, moving)
        n = len(fixed)

        images = (fixed, moving[:, ::-1])
        mask = xp.asarray(highpass_filter_mask(n)) if self.__highpass_filter else False

        with AutoScipyFftBackend(xp):
            magnitudes = xp.abs(rppft2(xp.asarray(images)))

        merged = (merge_sectors(m, mask=mask, xp=xp) for m in magnitudes)
        if self.debug:
            # Convert generator into re-usable tuple to keep for debug.
            merged = tuple(merged)

        omega = calculate_omega(
            calculate_delta_m(
                *merged, normalization=self.__rotation_normalization, xp=xp
            )
        )

        omega_min_index = (
            omega_index_optimized if self.__rotation_optimization else omega_index
        )(omega)

        rotation = omega_index_to_angle(omega_min_index, n)
        moving_rotated = self._transform(moving, rotation=rotation, degrees=False)

        flip_rotation, shift, shift_results = resolve_shift(
            fixed, moving_rotated, self.__shift_registration
        )
        rotation -= np.pi * flip_rotation

        if self.debug:
            debug_images = [
                *self.__debug_output(*merged),
                *omega_index_optimized_debug(to_numpy_array(omega)),
                *omega_index_array_debug_wrapper(to_numpy_array(omega)),
                RegistrationDebugImage(moving_rotated, "re-rotated-moving", dim=2),
            ]
        else:
            debug_images = None

        tform = Transformation2D(
            translation=(shift[0], shift[1]), rotation=np.rad2deg(-rotation)
        )
        return ResultInternal2D(
            tform, sub_results=shift_results, debug_images=debug_images
        )

    def __debug_output(self, m1: NDArray, m2: NDArray) -> list[RegistrationDebugImage]:
        m1, m2 = to_numpy_arrays(m1, m2)

        m_images_data = (np.log1p(np.ma.array(m, mask=np.isnan(m))) for m in (m1, m2))
        m_images = (arr_as_img(m[:, ::-1].T, cmap="viridis") for m in m_images_data)
        debug_images = self._build_debug_images(tuple(m_images), prefix="m-")

        n = len(m1) - 1
        plot_settings = ((True, False), ("--", "-"), ("blue", "green"))
        delta_ms = [
            (
                calculate_delta_m(m1, m2, normalization=norm, xp=np),
                norm,
                linestyle,
                color,
            )
            for norm, linestyle, color in zip(*plot_settings, strict=True)
        ]
        omegas = [(calculate_omega(dm), norm, ls, c) for dm, norm, ls, c in delta_ms]

        config_df = ("Difference Function", [0, n], [0, r"$\pi$"])
        for dm_input in [(delta_ms[0],), (delta_ms[1],), delta_ms]:
            for delta_m, norm, linestyle, color in dm_input:
                label = rf"$\Delta M{'_{N}' if norm else ''}(\theta)$"
                plt.plot(delta_m, label=label, linestyle=linestyle, color=color)

            suffix = "-combined" if len(dm_input) > 1 else f"-norm={dm_input[0][1]}"
            image_name = f"delta-m{suffix}"
            debug_images.append(self.__build_debug_plot(*config_df, image_name))

        config_adf = ("Angular Difference Function", [0, n // 2], [0, r"$\pi / 2$"])
        for o_input in [(omegas[0],), (omegas[1],), omegas]:
            for delta_m, norm, linestyle, color in o_input:
                label = rf"$ADF{'_{N}' if norm else ''}(\theta)$"
                plt.plot(delta_m, label=label, linestyle=linestyle, color=color)

            suffix = "-combined" if len(o_input) > 1 else f"-norm={o_input[0][1]}"
            image_name = f"adf-function{suffix}"
            debug_images.append(self.__build_debug_plot(*config_adf, image_name))

        return debug_images

    def __build_debug_plot(
        self, title: str, ticks: list, labels: list, image_name: str
    ) -> RegistrationDebugImage:
        plt.title(title)
        plt.xlabel(r"$\theta$")
        plt.xticks(ticks, labels)
        plt.margins(0)
        plt.legend()
        plt.tight_layout()

        return RegistrationDebugImage(fig_to_array(), image_name, dim=2, copy=False)
