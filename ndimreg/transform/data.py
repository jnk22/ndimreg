"""Module for registration results."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self


class Transformation(ABC):
    """Base class for affine transformation data."""

    translation: Any
    rotation: Any
    scale: Any

    def __repr__(self) -> str:
        """Return string representation."""
        translation = f"translation={self._build_str_repr(self.translation)}"
        rotation = f"rotation={self._build_str_repr(self.rotation)}"
        scale = f"scale={self._build_str_repr(self.scale)}"

        return f"{self.__class__.__name__}({translation}, {rotation}, {scale})"

    @staticmethod
    def _build_str_repr(values: tuple[float, ...] | float | None) -> str:
        if values is None:
            return "None"

        if isinstance(values, float | int):
            return f"{values:.2f}"

        return f"({', '.join(f'{v:.2f}' for v in values)})"


@dataclass(slots=True, kw_only=True, frozen=True, repr=False)
class Transformation2D(Transformation):
    """2D transformation input."""

    translation: tuple[float, float] | None = None
    rotation: float | None = None
    scale: float | None = None

    @classmethod
    def noop(cls) -> Self:
        """TODO."""
        return cls(translation=(0, 0), rotation=0, scale=1)


@dataclass(slots=True, kw_only=True, frozen=True, repr=False)
class Transformation3D(Transformation):
    """3D transformation input."""

    translation: tuple[float, ...] | None = None
    rotation: tuple[float, ...] | None = None
    scale: float | None = None

    @classmethod
    def noop(cls) -> Self:
        """TODO."""
        return cls(translation=(0, 0, 0), rotation=(0, 0, 0), scale=1)
