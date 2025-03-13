"""Data loader for images."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from typing_extensions import override

from .image import Image
from .image2d import Image2D
from .image3d import Image3D

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

T = TypeVar("T", bound=Image)


class ImageLoader(Protocol):
    """TODO."""

    def __len__(self) -> int:
        """TODO."""
        ...

    def __iter__(self) -> Iterator[Image]:
        """TODO."""
        ...

    def __getitem__(self, idx: int) -> Image:
        """TODO."""
        ...


@dataclass(frozen=True, slots=True)
class _PathsImageLoader(Generic[T], ABC):
    """TODO."""

    image_paths: Sequence[Path]

    def __len__(self) -> int:
        """Return the number of image paths."""
        return len(self.image_paths)

    def __iter__(self) -> Iterator[T]:
        """Iterate over datasets and yield images."""
        yield from (self._from_path(path) for path in self.image_paths)

    def __getitem__(self, idx: int) -> T:
        """Get an image by index."""
        return self._from_path(self.image_paths[idx])

    @abstractmethod
    def _from_path(self, path: str | Path) -> T: ...


@dataclass(frozen=True, slots=True)
class Paths2DImageLoader(_PathsImageLoader[Image2D]):
    """Paths loader for 2D images."""

    @override
    def _from_path(self, path: str | Path) -> Image2D:
        return Image2D.from_path(path)


@dataclass(frozen=True, slots=True)
class Paths3DImageLoader(_PathsImageLoader[Image3D]):
    """Paths loader for 3D images."""

    @override
    def _from_path(self, path: str | Path) -> Image3D:
        return Image3D.from_path(path)


@dataclass(frozen=True, slots=True)
class _ScikitImageLoader(Generic[T], ABC):
    """Generic Scikit image loader."""

    dataset_names: Sequence[str]

    def __len__(self) -> int:
        """Return the number of datasets."""
        return len(self.dataset_names)

    def __iter__(self) -> Iterator[T]:
        """Iterate over datasets and yield images."""
        yield from (self._from_skimage(ds_name) for ds_name in self.dataset_names)

    def __getitem__(self, idx: int) -> T:
        """Get an image by index."""
        return self._from_skimage(self.dataset_names[idx])

    @abstractmethod
    def _from_skimage(self, dataset_name: str) -> T: ...


@dataclass(frozen=True, slots=True)
class Scikit2DImageLoader(_ScikitImageLoader[Image2D]):
    """Scikit loader for 2D images."""

    @override
    def _from_skimage(self, dataset_name: str) -> Image2D:
        return Image2D.from_skimage(dataset_name)


@dataclass(frozen=True, slots=True)
class Scikit3DImageLoader(_ScikitImageLoader[Image3D]):
    """Scikit loader for 3D images."""

    @override
    def _from_skimage(self, dataset_name: str) -> Image3D:
        return Image3D.from_skimage(dataset_name)
