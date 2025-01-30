"""Interface for image processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from ndimreg.transform import Transformation

if TYPE_CHECKING:
    from numpy.typing import NDArray


T = TypeVar("T", bound=Transformation)

Dimension = Literal[2, 3]


class DataProcessor(Protocol):
    """Interface for data processors.

    Attributes
    ----------
    name
        Name of the processing function for logging and user output.
    """

    @property
    def name(self) -> str:
        """Name of the processing function."""
        ...

    def process(self, *data: NDArray) -> list[NDArray]:
        """Apply data processing.

        Parameters
        ----------
        data
            Data to process.
        """
        ...

    def backward(self, transformation: T) -> T:
        """Apply post-processing to resulting transformation object.

        Parameters
        ----------
        transformation
            Transformation as registration result to process.
        """
        ...
