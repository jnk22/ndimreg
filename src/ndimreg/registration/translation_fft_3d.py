"""3D image registration with basic FFT shift using phase cross correlation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor3D
from ndimreg.transform import Transformation3D
from ndimreg.utils.image_operations import phase_cross_correlation

from .base import BaseRegistration
from .result import ResultInternal3D

if TYPE_CHECKING:
    from numpy.typing import NDArray


# TODO: Test parameters 'disambiguate' and 'normalization'.
# WARNING: The returned error does not exactly match the peak value if
# using normalization. This is due to a bug in scikit-image, see
# https://github.com/scikit-image/scikit-image/issues/7078.


class TranslationFFT3DRegistration(BaseRegistration):
    """Register 3D image translation using FFT shift theorem.

    Capabilities
    ------------
    - Dimension: 3D
    - Translation: Yes
    - Rotation: No
    - Scale: No
    - Shear: No

    Limitations
    -----------
    None.
    """

    def __init__(
        self,
        *,
        normalization: bool = True,
        disambiguate: bool = False,
        upsample_factor: int = 1,
        data_space: Literal["real", "fourier"] = "real",
        **kwargs: Any,
    ) -> None:
        """TODO."""
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor3D())

        self.__data_space = data_space
        self.__normalization = normalization
        self.__disambiguate = disambiguate
        self.__upsample_factor = upsample_factor

    @property
    @override
    def dim(self) -> Literal[3]:
        return 3

    @override
    def _register(
        self, fixed: NDArray, moving: NDArray, **_kwargs: Any
    ) -> ResultInternal3D:
        shifts, error, _ = phase_cross_correlation(
            fixed,
            moving,
            space=self.__data_space,
            normalization="phase" if self.__normalization else None,
            disambiguate=self.__disambiguate,
            upsample_factor=self.__upsample_factor,
        )

        # x, y, and z are shifted in the opposite direction.
        tform = Transformation3D(translation=tuple(-np.roll(shifts, -1)))
        return ResultInternal3D(tform, error=error)
