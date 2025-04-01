"""Base registration class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from array_api_compat import get_namespace
from loguru import logger
from scipy.fft import fftshift

from ndimreg.transform import Transformation, transform
from ndimreg.utils import Timer, log_time

from .protocol import Registration
from .result import RegistrationDebugImage, RegistrationResult2D, RegistrationResult3D

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ndimreg.processor import DataProcessor
    from ndimreg.transform import InterpolationOrder, TransformationMode

    from .protocol import Dimension
    from .result import RegistrationDebugImages, RegistrationResult, ResultInternal

T = TypeVar("T", bound=Transformation)


class BaseRegistration(ABC, Registration):
    """TODO."""

    def __init__(
        self,
        *,
        processors: Sequence[DataProcessor] | None = None,
        transform_interpolation_order: InterpolationOrder = 3,
        transform_mode: TransformationMode = "crop",
        debug: bool = False,
        **_kwargs: Any,
    ) -> None:
        """TODO."""
        if transform_mode in ("extend", "resize"):
            raise NotImplementedError

        self.debug: bool = debug

        self._processors: list[DataProcessor] = list(processors) if processors else []
        self._transform_interpolation_order: InterpolationOrder = (
            transform_interpolation_order
        )
        self._transform_mode: TransformationMode = transform_mode

    @property
    @abstractmethod
    def dim(self) -> Dimension:
        """TODO."""

    @property
    def name(self) -> str:
        """TODO."""
        return self.__class__.__name__

    @log_time(print_func=logger.debug)
    def register(
        self, fixed: NDArray, moving: NDArray, **kwargs: Any
    ) -> RegistrationResult:
        """Register images.

        Parameters
        ----------
        fixed
            Fixed image.
        moving
            Moving image.
        **kwargs
            Additional keyword arguments passed to the registration
            implementation.

        Returns
        -------
        RegistrationResult
            Result of the registration process. The result contains the
            transformations applied to the fixed image that result in
            the moving image.
        """
        # TODO: Allow for individual return types, e.g., stitched images.
        # TODO: Add resizer (NxN) to allow for different image sizes.
        # TODO: Implement size/image checks here?
        # TODO: Verify that input images are equal in size (if required).
        # TODO: Handle maximum possible precison (e.g., docstring, return value, ...).
        # TODO: Allow multiple moving images to be registered with fixed image.
        # TODO: Implement transformation representations for non-similarity transforms.
        # TODO: Implement/optimize 'auto-correct' for inputs (grayscale, resize, ...).
        # TODO: Define behavior on failing registrations, empty inputs, wrong size, ...
        # TODO: Update returned rotation/transformation output (e.g., matrix, ...).
        # TODO: Passthrough orders (interpolation, transformation) to sub-registrations.
        # TODO: Allow specific configuration (interpolation etc.) for sub-registrations.
        # TODO: Make non-cropping transformations (resize, extend) possible.
        # TODO: Temporarily raise RegistrationError here in try-except.
        # TODO: Properly implement re-scaling on 'resize' and 'downscale' modes.
        # TODO: Use 'precision' instead of 'upsample_factor' for translation recovery.
        # TODO: Implement hashes based on registration configuration.
        # TODO: Implement configuration classes for parameter inputs.
        # TODO: Ensure that input data uses same backend (error, upgrade, downgrade).
        # TODO: Make default auto-correct pre-processors optional via argument.
        # TODO: Use radians + rotation matrices (or quaternions) internally.
        # TODO: Allow custom padding parameter (cval) for extend/resize modes.
        # TODO: Improve total duration precision with a register-decorator.
        # TODO: Check FFT vs real-FFT for performance + note capabilities.
        # TODO: Use timers on pre-defined abstract methods instead.
        # PERF: Check for cachable functions within registration methods.

        with Timer(interval_name="input") as timer:
            logger.debug(f"Registering images with '{self.name}'")
            debug_images_input = self.__build_debug_images_guarded(fixed, moving)

            timer.start_interval("preprocessing")
            fixed_preprocessed, moving_preprocessed, debug_images_preprocesed = (
                self.__pre_process_data(fixed, moving)
            )

            timer.start_interval("registration")
            result = self._register(fixed_preprocessed, moving_preprocessed, **kwargs)

            timer.start_interval("postprocessing")
            tform_postprocessed = self.__post_process_result(result.transformation)
            logger.debug(
                f"Recovered transformation in {self.name}: {tform_postprocessed}"
            )

            timer.start_interval("result")
            debug_images: RegistrationDebugImages = {
                "input": debug_images_input,
                "preprocessing": debug_images_preprocesed,
                "registration": result.debug_images,
            }

            match self.dim:
                case 2:
                    result_class = RegistrationResult2D
                case 3:
                    result_class = RegistrationResult3D

        return result_class(
            self.name,
            tform_postprocessed,
            error=result.error,
            # Ignore the type warning as we have only used appropriate
            # interval names for supported duration types.
            duration={interval.name: interval.duration for interval in timer.intervals},  # type: ignore[reportAssignmentType]
            debug_images=debug_images,
            sub_results=result.sub_results,
        )

    @abstractmethod
    def _register(
        self, fixed: NDArray, moving: NDArray, **kwargs: Any
    ) -> ResultInternal: ...

    def _transform(self, data: NDArray, **kwargs: Any) -> NDArray:
        # Convenience wrapper to automatically use default
        # transformation parameters.
        tform_kwargs = {
            "dim": self.dim,
            "mode": self._transform_mode,
            "clip": False,
            "interpolation_order": self._transform_interpolation_order,
        }

        return transform(data, **(tform_kwargs | kwargs))

    def _build_debug_images(
        self,
        images: Sequence[NDArray],
        names: Sequence[str] = ("fixed", "moving"),
        *,
        prefix: str = "",
        suffix: str = "",
        copy: bool = True,
    ) -> list[RegistrationDebugImage]:
        xp = get_namespace(*images)
        logger.debug(f"Using namespace for debug images: {xp.__name__}")

        images_corrected = (
            fftshift(xp.log1p(xp.abs(im)))
            if xp.isdtype(im.dtype, "complex floating")
            else im
            for im in images
        )

        return [
            RegistrationDebugImage(
                data, f"{prefix}{name}{suffix}", dim=self.dim, copy=copy
            )
            for name, data in zip(names, images_corrected, strict=True)
        ]

    def __pre_process_data(
        self, fixed: NDArray, moving: NDArray
    ) -> tuple[NDArray, NDArray, list[RegistrationDebugImage] | None]:
        debug_images = [] if self.debug else None
        for processor in self._processors:
            logger.debug(f"Pre-processing images with '{processor.name}'")
            fixed, moving = processor.process(fixed, moving)

            if debug_images is not None:
                prefix = f"{processor.name}-"
                images = (fixed, moving)
                debug_images.extend(self._build_debug_images(images, prefix=prefix))

        return fixed, moving, debug_images

    def __post_process_result(self, transformation: T) -> T:
        for processor in reversed(self._processors):
            logger.debug(f"Post-processing result with '{processor.name}'")
            transformation = processor.backward(transformation)

        return transformation

    def __build_debug_images_guarded(
        self, fixed: NDArray, moving: NDArray
    ) -> list[RegistrationDebugImage] | None:
        return (
            self._build_debug_images(
                (fixed, moving), prefix="input-", suffix=f"-{self.name}"
            )
            if self.debug
            else None
        )
