"""2D image registration using imreg_dft library."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from array_api_compat import array_namespace, is_cupy_namespace
from imreg_dft.imreg import _similarity
from loguru import logger
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor2D
from ndimreg.transform import Transformation2D
from ndimreg.utils import to_numpy_arrays

from .base import BaseRegistration
from .result import ResultInternal2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ndimreg.transform import InterpolationOrder

# We create an alias  'np.bool' as the wrapped library 'imreg_dft' uses
# the deprecated (and now removed) type 'np.bool' which leads to an
# actual AttributeError. At some point, this should be solved upstream.
np.bool = bool


class ImregDft2DRegistration(BaseRegistration):
    """2D image registration using ``imreg_dft`` library.

    [imreg_dft](https://github.com/matejak/imreg_dft) is fork of
    [cgohle/imreg](https://github.com/cgohlke/imreg) and in stable
    state.

    Capabilities
    ------------
    - Dimension: 2D
    - Translation: Yes
    - Rotation: Yes
    - Scale: Yes
    - Shear: No

    Limitations
    ----------
    - Scale change must be less than 2.
    - No subpixel precision (but you can use resampling to get around this).

    Source: https://imreg-dft.readthedocs.io/en/latest/api.html#imreg-module
    """

    @property
    @override
    def dim(self) -> Literal[2]:
        return 2

    @override
    def __init__(
        self,
        numiter: int = 1,
        transform_interpolation_order: InterpolationOrder = 3,
        constraints: dict | None = None,
        filter_pcorr: int = 0,
        exponent: float | Literal["inf"] = "inf",
        reports: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the registration object.

        Parameters
        ----------
        numiter
            How many times to iterate when determining scale and rotation.
        order
            Order of approximation (when doing transformations).
            1 = linear, 3 = cubic etc.
        filter_pcorr
            Radius of a spectrum filter for translation detection
        exponent
            The exponent value used during processing. Refer to the docs
            for a thorough explanation. Generally, pass "inf" when
            feeling conservative. Otherwise, experiment, values below 5
            are not even supposed to work.
        constraints
            Specify preference of seeked values. Pass None (default) for
            no constraints, otherwise pass a dict with keys ``angle``,
            ``scale``, ``tx`` and/or ``ty`` (i.e. you can pass all, some
            of them or none of them, all is fine). The value of a key is
            supposed to be a mutable 2-tuple (e.g. a list), where the
            first value is related to the constraint center and the
            second one to softness of the constraint (the higher is the
            number, the more soft a constraint is).
            More specifically, constraints may be regarded as weights
            in form of a shifted Gaussian curve.
            However, for precise meaning of keys and values,
            see the documentation section :ref:`constraints`.
            Names of dictionary keys map to names of command-line
            arguments.
        reports
            Refer to the documentation of the imreg_dft library for
            more information.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor2D())

        self._imregdft_kwargs = {
            "numiter": numiter,
            "order": transform_interpolation_order,
            "constraints": constraints,
            "filter_pcorr": filter_pcorr,
            "exponent": exponent,
            "reports": reports,
        }

    @override
    def _register(
        self, fixed: NDArray, moving: NDArray, **_kwargs: Any
    ) -> ResultInternal2D:
        if is_cupy_namespace(array_namespace(fixed, moving)):
            msg = f"Registration method {self.name} does not support GPU device (CuPy), transfering data to CPU (NumPy)"
            logger.warning(msg)
            fixed, moving = to_numpy_arrays(fixed, moving)

        if np.array_equal(fixed, moving):
            # 'imreg_dft' does not work when registering identical
            # images sometimes. Therefore, we explicitly check for
            # equality, even though this is an edge case.
            return ResultInternal2D(Transformation2D.noop())

        try:
            with warnings.catch_warnings():
                # NOTE: The similarity registration/recovery from
                # imreg_dft uses deprecated SciPy function calls.
                # FIX: Address previous note.
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # We use '_similarity' directly instead 'similarity' as
                # this one does not generate and return a fused image.
                imreg_result = _similarity(fixed, moving, **self._imregdft_kwargs)

        except IndexError:
            # Index error happens when registering two identical images
            # or with an empty image.
            msg = (
                "Both images are equal or one is empty, returning 0 degrees and shifts"
            )
            logger.warning(msg)
            # TODO: Is returning default (None values) or an error more appropriate?
            return ResultInternal2D(Transformation2D.noop())

        result = SimpleNamespace(**imreg_result)
        tform = Transformation2D(
            translation=(-result.tvec[1], -result.tvec[0]),
            rotation=-result.angle,
            scale=(1 / result.scale),
        )

        return ResultInternal2D(tform)
