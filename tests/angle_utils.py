"""Utility functions for testing."""

from __future__ import annotations

import math
from collections.abc import Sized
from typing import TYPE_CHECKING

import numpy as np
from _pytest.python_api import ApproxScalar, ApproxSequenceLike

from ndimreg.utils.diffs import angle_diff

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _pytest.python_api import ApproxBase
    from numpy.typing import NDArray

NumericType = int | float | np.integer | np.floating


class ApproxAngleScalar(ApproxScalar):
    """TODO."""

    def __init__(
        self,
        expected: object,
        rel: float | None = None,
        abs: float | None = None,  # noqa: A002
        *,
        nan_ok: bool = False,
        degrees: bool = True,
    ) -> None:
        """TODO."""
        super().__init__(expected, rel, abs, nan_ok=nan_ok)

        self.degrees: bool = degrees

    def __eq__(self, actual: object) -> bool:
        """TODO."""
        if not (
            isinstance(self.expected, NumericType) and isinstance(actual, NumericType)
        ):
            return NotImplemented

        if math.isnan(self.expected):
            return self.nan_ok and math.isnan(actual)

        if actual == self.expected:
            return True

        return angle_diff(actual, self.expected, degrees=self.degrees) <= self.tolerance


class ApproxAngleSequenceLike(ApproxSequenceLike):
    """TODO."""

    def __init__(
        self,
        expected: object,
        rel: float | None = None,
        abs: float | None = None,  # noqa: A002
        *,
        nan_ok: bool = False,
        degrees: bool = True,
    ) -> None:
        """TODO."""
        super().__init__(expected, rel, abs, nan_ok=nan_ok)

        self.degrees: bool = degrees

    def __eq__(self, actual: object) -> bool:
        """TODO."""
        if not isinstance(actual, Sized):
            return NotImplemented

        if len(actual) != len(self.expected):
            return False

        return all(
            a == self._approx_scalar(x) for a, x in self._yield_comparisons(actual)
        )

    def _approx_scalar(self, x: object) -> ApproxAngleScalar:
        return ApproxAngleScalar(x, rel=self.rel, abs=self.abs, degrees=self.degrees)


def approx_rotation(
    expected: NumericType | Sequence[NumericType] | NDArray,
    rel: float | None = None,
    abs: float | None = None,  # noqa: A002
    *,
    nan_ok: bool = False,
    degrees: bool = True,
) -> ApproxBase:
    """Test angles for approximate equality with respect to circular values.

    Notes
    -----
    This only supports decimal angles in degrees, either scalar or sequences.
    """
    if (
        hasattr(expected, "__getitem__")
        and isinstance(expected, Sized)
        and not isinstance(expected, (str | bytes))
    ):
        return ApproxAngleSequenceLike(
            expected, rel, abs, nan_ok=nan_ok, degrees=degrees
        )

    return ApproxAngleScalar(expected, rel, abs, nan_ok=nan_ok, degrees=degrees)
