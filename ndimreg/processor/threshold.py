"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from array_api_compat import get_namespace
from typing_extensions import override

from .base import BaseDataProcessor

if TYPE_CHECKING:
    from numpy.typing import NDArray

ThresholdCutMode = Literal["min", "max", "mean"]


class ThresholdCutFilter(BaseDataProcessor):
    """Band-pass filter image data."""

    def __init__(
        self,
        *,
        low: float | None = None,
        high: float | None = None,
        cval: float | ThresholdCutMode = "min",
        group: bool = True,
    ) -> None:
        """TODO."""
        low_str = f"{low:.2f}" if low is not None else "None"
        high_str = f"{high:.2f}" if high is not None else "None"
        cval_str = f"{cval:.2f}" if isinstance(cval, float) else cval

        super().__init__(
            f"{self.__class__.__name__}(low={low_str},high={high_str},cval={cval_str})"
        )

        self.group: bool = group
        self.__low: float | None = low
        self.__high: float | None = high
        self.__cval: float | ThresholdCutMode = cval

    @override
    def process(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        if self.__low is None and self.__high is None:
            return [*data]

        return self.__process_group(*data) if self.group else self.__process_each(*data)

    def __process_group(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        xp = get_namespace(*data)

        match self.__cval:
            case "min":
                cval = xp.min(*data)
            case "max":
                cval = xp.max(*data)
            case "mean":
                cval = xp.mean(*data)
            case _:
                cval = self.__cval

        for d in data:
            d[(d < self.__low) | (d > self.__high)] = cval

        return [*data]

    def __process_each(self, *data: NDArray) -> list[NDArray]:
        """TODO."""
        for d in data:
            match self.__cval:
                case "min":
                    cval = d.min()
                case "max":
                    cval = d.max()
                case "mean":
                    cval = d.mean()
                case _:
                    cval = self.__cval

            d[(d < self.__low) | (d > self.__high)] = cval

        return [*data]
