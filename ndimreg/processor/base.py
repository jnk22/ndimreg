"""Interface for image processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, TypeVar

from ndimreg.transform import Transformation

from .protocol import DataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T", bound=Transformation)

Dimension = Literal[2, 3]


class BaseDataProcessor(ABC, DataProcessor):
    """Base implementation for data processors."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize pre-processing function.

        Attributes
        ----------
        name
            Name of the pre-processing function for logging and user
            output. Defaults to the name of the class.
        """
        self._name: str = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Name of the processing function."""
        return self._name

    @abstractmethod
    def process(self, *data: NDArray) -> list[NDArray]:
        """Apply data processing.

        Parameters
        ----------
        data
            Data to process.
        """

    def backward(self, transformation: T) -> T:
        """Apply post-processing to resulting transformation object.

        Parameters
        ----------
        transformation
            Transformation as registration result to process.
        """
        return transformation
