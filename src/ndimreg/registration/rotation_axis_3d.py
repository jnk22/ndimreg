"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import array_namespace
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor3D
from ndimreg.transform import AXIS_MAPPING, Transformation3D, rotate_axis

from .base import BaseRegistration
from .keller_2d_utils import _resolve_rotation
from .result import RegistrationDebugImage, ResultInternal3D
from .shift_resolver import resolve_shift
from .translation_fft_3d import TranslationFFT3DRegistration

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ndimreg.transform import RotationAxis3D, RotationAxis3DIndex

# TODO: Optimize edge case handling (wrt. performance).
# TODO: Rename class (e.g., 'SingleAxisAngle3DRegistration').

SRC: Final = (0, 1, 2)
DEST: Final = {0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}
SHIFT_AXES: Final = {0: (1, 2), 1: (0, 1), 2: (0, 2)}
ROTATION_BASIS: Final = {0: 0, 1: 2, 2: 1}


class RotationAxis3DRegistration(BaseRegistration):
    """Registration algorithm to recover rotation around a single axis."""

    def __init__(  # noqa: PLR0913
        self,
        axis: RotationAxis3D = "z",
        *,
        rotation_normalization: bool = True,
        rotation_optimization: bool = True,
        rotation_vectorized: bool = False,
        shift_normalization: bool = True,
        shift_disambiguate: bool = False,  # WARNING: Does not work on GPU.
        shift_upsample_factor: int = 1,
        highpass_filter: bool = True,
        **kwargs: Any,
    ) -> None:
        """TODO."""
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor3D())

        self.__rotation_axis: RotationAxis3DIndex = AXIS_MAPPING[axis][1]
        self.__rotation_normalization: bool = rotation_normalization
        self.__rotation_optimization: bool = rotation_optimization
        self.__rotation_vectorized: bool = rotation_vectorized
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
        # TODO: Support complex image input.
        images = (fixed, moving)
        xp = array_namespace(*images)

        rotation, debug_images = _resolve_rotation(
            (xp.moveaxis(im, SRC, DEST[self.__rotation_axis]) for im in images),
            n=len(fixed),
            xp=xp,
            vectorized=self.__rotation_vectorized,
            normalized=self.__rotation_normalization,
            optimized=self.__rotation_optimization,
            highpass_filter=self.__highpass_filter,
            apply_fft=True,
            debug=self.debug,
        )

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

        if self.debug and debug_images:
            debug_images.append(
                RegistrationDebugImage(moving_rotated, "re-rotated-moving", dim=3)
            )

        axes = SHIFT_AXES[self.__rotation_axis]
        flip_rotation, shift, shift_results = resolve_shift(
            fixed, moving_rotated, self.__shift_registration, axes=axes
        )
        rotation += xp.pi * flip_rotation

        basis = ROTATION_BASIS[self.__rotation_axis]
        rotation_matrix = pr.active_matrix_from_angle(basis, rotation)
        angles = np.rad2deg(
            pr.euler_from_matrix(rotation_matrix, 0, 1, 2, extrinsic=False)
        )

        tform = Transformation3D(translation=shift, rotation=tuple(angles))
        return ResultInternal3D(
            tform, sub_results=shift_results, debug_images=debug_images
        )
