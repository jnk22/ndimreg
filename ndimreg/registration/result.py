"""TODO."""

from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from ndimreg.transform import Transformation, Transformation2D, Transformation3D
from ndimreg.utils import to_numpy_array

if TYPE_CHECKING:
    from collections.abc import Collection

    from numpy.typing import NDArray

    from .protocol import Dimension

# TODO: Implement various 'no-op' results (e.g., equal image, ...)

T = TypeVar("T", bound=Transformation)


@dataclass(frozen=True, slots=True)
class RegistrationDebugImage:
    """TODO."""

    data: NDArray
    name: str

    _: KW_ONLY
    dim: Dimension

    copy: InitVar[bool] = True

    def __post_init__(self, copy: bool) -> None:
        """TODO."""
        object.__setattr__(self, "data", to_numpy_array(self.data, copy=copy))


RegistrationDebugStep = Literal["input", "preprocessing", "registration"]
RegistrationDebugImages = dict[
    RegistrationDebugStep, list[RegistrationDebugImage] | None
]

RegistrationDurationStep = Literal[
    "input", "preprocessing", "registration", "postprocessing", "result"
]
RegistrationDuration = dict[RegistrationDurationStep, float]


class ResultInternal(ABC, Generic[T]):
    """Generic registration result."""

    # TODO: Define error/possible non-successful error codes.
    # TODO: Rename with underscore to mark as internal?

    transformation: T
    debug_images: list[RegistrationDebugImage] | None
    error: float | None
    sub_results: list[RegistrationResult] | None


@dataclass(slots=True)
class ResultInternal2D(ResultInternal[Transformation2D]):
    """Internal 2D registration result."""

    transformation: Transformation2D
    debug_images: list[RegistrationDebugImage] | None = None
    error: float | None = None
    sub_results: list[RegistrationResult] | None = None


@dataclass(slots=True)
class ResultInternal3D(ResultInternal[Transformation3D]):
    """Internal 3D registration result."""

    transformation: Transformation3D
    debug_images: list[RegistrationDebugImage] | None = None
    error: float | None = None
    sub_results: list[RegistrationResult] | None = None


class RegistrationResult(ABC, Generic[T]):
    """Generic registration result."""

    registration_name: str
    transformation: T
    duration: RegistrationDuration
    error: float | None
    debug_images: RegistrationDebugImages
    sub_results: list[RegistrationResult] | None

    @property
    def total_duration(self) -> float:
        """Return total registration duration in seconds."""
        return sum(self.get_durations(depth=0)[0][1].values())

    def get_durations(
        self,
        step: Collection[RegistrationDurationStep]
        | RegistrationDurationStep
        | None = None,
        depth: int | None = None,
    ) -> list[tuple[str, RegistrationDuration]]:
        """TODO.

        Parameters
        ----------
        depth
            Maximum recursive depth of durations to fetch. Start depth
            for only highest level (i.e., the main registration) is 0.
            Defaults to None, which returns all nested durations.
        """
        # TODO: Return durations as nested dictionary (add information/id to each step).
        if depth is not None and depth < 0:
            return []

        if step is None:
            subset = self.duration
        else:
            keys = [step] if isinstance(step, str) else step
            subset: RegistrationDuration = {
                key: self.duration[key] for key in keys if key in self.duration
            }
        result_duration = self.registration_name, subset

        next_depth = depth if depth is None else depth - 1
        if not self.sub_results:
            return [result_duration]

        sub_results_durations = (
            sub.get_durations(step=step, depth=next_depth) for sub in self.sub_results
        )
        return [result_duration, *itertools.chain.from_iterable(sub_results_durations)]

    def get_debug_images(
        self,
        step: Collection[RegistrationDebugStep] | RegistrationDebugStep | None = None,
        *,
        dim: Dimension,
        depth: int | None = None,
    ) -> list[RegistrationDebugImage]:
        """TODO.

        Parameters
        ----------
        depth
            Maximum recursive depth of debug images to fetch. Start
            depth for only highest level (i.e., the main registration)
            is 0. Defaults to None, which returns all existing debug
            images.
        """
        # TODO: Return images as nested dictionary (add information/id to each step).
        if depth is not None and depth < 0:
            return []

        keys = [step] if isinstance(step, str) else step

        images = (
            image
            for key, available_images in self.debug_images.items()
            if available_images is not None and (not keys or key in keys)
            for image in available_images
            if image.dim == dim
        )

        next_depth = depth if depth is None else depth - 1
        if not self.sub_results:
            return list(images)

        sub_results_images = (
            sub.get_debug_images(dim=dim, step=step, depth=next_depth)
            for sub in self.sub_results
        )

        return [*images, *itertools.chain.from_iterable(sub_results_images)]


@dataclass(frozen=True, slots=True)
class RegistrationResult2D(RegistrationResult[Transformation2D]):
    """2D registration result."""

    registration_name: str
    transformation: Transformation2D

    _: KW_ONLY
    duration: RegistrationDuration
    error: float | None = None
    debug_images: RegistrationDebugImages
    sub_results: list[RegistrationResult] | None


@dataclass(frozen=True, slots=True)
class RegistrationResult3D(RegistrationResult[Transformation3D]):
    """3D registration result."""

    registration_name: str
    transformation: Transformation3D

    _: KW_ONLY
    duration: RegistrationDuration
    error: float | None = None
    debug_images: RegistrationDebugImages
    sub_results: list[RegistrationResult] | None
