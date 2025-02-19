"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import get_namespace
from loguru import logger
from matplotlib import pyplot as plt
from ppft_nd import ppft2
from scipy import fft
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor3D
from ndimreg.transform import AXIS_MAPPING, Transformation3D, rotate_axis
from ndimreg.utils import AutoScipyFftBackend, fig_to_array, to_numpy_array

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
from .result import RegistrationDebugImage, ResultInternal3D
from .shift_resolver import resolve_shift
from .translation_fft_3d import TranslationFFT3DRegistration

if TYPE_CHECKING:
    from matplotlib.scale import ScaleBase
    from numpy.typing import NDArray

    from ndimreg.transform import RotationAxis3D, RotationAxis3DIndex

# TODO: Optimize edge case handling (wrt. performance).
# TODO: Rename class (e.g., 'SingleAxisAngle3DRegistration').
# TODO: Allow input parameter to choose selection (mid layer, minimum, sum, ...).
# NOTE: Several flips/axis alignments are potentially redundant.

SRC: Final = (0, 1, 2)
DEST: Final = {0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}
FLIPS: Final = (..., (*(slice(None),) * 2, slice(None, None, -1)))


class RotationAxis3DRegistration(BaseRegistration):
    """Registration algorithm to recover rotation around a single axis."""

    def __init__(  # noqa: PLR0913
        self,
        axis: RotationAxis3D = "z",
        *,
        rotation_normalization: bool = True,
        rotation_optimization: bool = True,
        highpass_filter: bool = True,
        shift_normalization: bool = True,
        shift_disambiguate: bool = False,  # WARNING: Does not work on GPU.
        shift_upsample_factor: int = 1,
        **kwargs: Any,
    ) -> None:
        """TODO."""
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor3D())

        self.__rotation_axis: RotationAxis3DIndex = AXIS_MAPPING[axis][1]
        self.__rotation_normalization: bool = rotation_normalization
        self.__rotation_optimization: bool = rotation_optimization
        self.__highpass_filter: bool = highpass_filter

        self.__shift_registration = TranslationFFT3DRegistration(
            data_space="fourier",
            disambiguate=shift_disambiguate,
            normalization=shift_normalization,
            upsample_factor=shift_upsample_factor,
            debug=self.debug,
        )

    @property
    @override
    def dim(self) -> Literal[3]:
        return 3

    @override
    def _register(
        self, fixed: NDArray, moving: NDArray, **_kwargs: Any
    ) -> ResultInternal3D:
        xp = get_namespace(fixed, moving)

        images = (fixed, moving)
        images = (xp.moveaxis(im, SRC, DEST[self.__rotation_axis]) for im in images)
        images = (im[flip] for im, flip in zip(images, FLIPS, strict=True))

        n = len(fixed)
        mask = xp.asarray(highpass_filter_mask(n)) if self.__highpass_filter else False

        # PERF: This should be a vectorized operation instead.
        # NOTE: Real FFT used as only real image input data is expected.
        ps = (xp.abs(ppft2(fft.rfft(im, axis=0))[:, :, n:]) for im in images)
        merged = ((merge_sectors(p, mask=mask, xp=xp) for p in px) for px in ps)
        norm = self.__rotation_normalization

        with AutoScipyFftBackend(xp):
            omega_layers = xp.asarray(
                [
                    calculate_omega(calculate_delta_m(*mx, normalization=norm, xp=xp))
                    for mx in zip(*merged, strict=True)
                ]
            )

        # We then use eq. 4.4 to select the layer with the lowest value.
        # That layer then represents the best omega value which we use
        # for rotation estimation on a single axis.
        # TODO: Find out what what minimum resembles the one defined in paper.
        # See debug output for possible solutions.
        min_index = xp.unravel_index(xp.argmin(omega_layers), omega_layers.shape)[0]
        logger.debug(f"Minimum index: {min_index} (layer {min_index + 1}/{n})")

        omega = omega_layers[min_index]
        omega_min_index = (
            omega_index_optimized if self.__rotation_optimization else omega_index
        )(omega)

        rotation = omega_index_to_angle(omega_min_index, n)
        logger.debug(f"Recovered axis rotation: {xp.rad2deg(rotation):.2f}°")

        moving_rotated = rotate_axis(
            moving,
            rotation,
            axis=self.__rotation_axis,
            dim=3,
            degrees=False,
            clip=False,
            mode=self._transform_mode,
            interpolation_order=self._transform_interpolation_order,
        )

        axes = {0: (1, 2), 1: (0, 1), 2: (0, 2)}[self.__rotation_axis]
        flip_rotation, shift, shift_results = resolve_shift(
            fixed, moving_rotated, self.__shift_registration, axes=axes
        )
        rotation += np.pi * flip_rotation

        # Convert to correct Euler angles convention.
        basis = {0: 0, 1: 2, 2: 1}[self.__rotation_axis]
        rot_matrix = pr.active_matrix_from_angle(basis, rotation)
        angles = np.rad2deg(pr.intrinsic_euler_xyz_from_active_matrix(rot_matrix))
        logger.debug(f"Recovered angles: [{', '.join(f'{x:.2f}' for x in angles)}]")

        if self.debug:
            debug_images = [
                *self.__debug_omega_plots(to_numpy_array(omega_layers)),
                *omega_index_optimized_debug(to_numpy_array(omega)),
                *omega_index_array_debug_wrapper(to_numpy_array(omega)),
                RegistrationDebugImage(moving_rotated, "re-rotated-moving", dim=3),
            ]
        else:
            debug_images = None

        tform = Transformation3D(
            translation=(shift[0], shift[1], shift[2]), rotation=tuple(angles.tolist())
        )

        return ResultInternal3D(
            tform, sub_results=shift_results, debug_images=debug_images
        )

    def __debug_omega_plots(
        self, omega_layers: NDArray
    ) -> list[RegistrationDebugImage]:
        n = len(omega_layers[0])

        norm = omega_layers / np.linalg.norm(omega_layers, axis=1, keepdims=True)
        omega_layers_norm = norm * (1 / norm.max())

        min_val_index = np.unravel_index(np.argmin(omega_layers), omega_layers.shape)[0]

        row_mins = np.min(omega_layers, axis=1)
        row_maxs = np.max(omega_layers, axis=1)
        max_diff_index = np.argmax(row_maxs - row_mins)

        return [
            self.__debug_plot(omega_layers.T, n, "All"),
            self.__debug_plot(omega_layers.T, n, "All Log-Scaled", yscale="log"),
            self.__debug_plot(omega_layers_norm.T, n, "Normalized"),
            self.__debug_plot(omega_layers_norm.sum(0), n, "Normalized Sum"),
            self.__debug_plot(omega_layers[min_val_index], n, "Minimum Value"),
            self.__debug_plot(omega_layers[n // 2], n, "Middle Layer"),
            self.__debug_plot(omega_layers.sum(0), n, "Overall Sum"),
            self.__debug_plot(omega_layers[max_diff_index], n, "Max Difference"),
        ]

    @staticmethod
    def __debug_plot(
        omega_layers: NDArray, n: int, name: str, *, yscale: str | ScaleBase = "linear"
    ) -> RegistrationDebugImage:
        plt.figure()
        plt.plot(omega_layers)
        plt.title(f"Angular Difference Function ({name})")
        plt.xlabel(r"$\theta$")
        plt.xticks([0, n - 1], ["0", r"$\pi / 2$"])
        plt.yscale(yscale)

        if omega_layers.ndim == 1:
            # TODO: Add degrees for minimum.
            plt.axvline(
                x=omega_layers.argmin().item(),
                color="red",
                linestyle="--",
                linewidth=2,
                label="Minimum Value",
            )

        image_name = f"adf-function-{'-'.join(name.lower().split(' '))}"
        return RegistrationDebugImage(fig_to_array(), image_name, dim=2, copy=False)
