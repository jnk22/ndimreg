"""TODO."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from ndimreg.utils.diffs import translation_diff


@pytest.mark.parametrize(
    ("translation1", "translation2", "expected_result"),
    [
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), np.array([0.0, 0.0, 0.0])),
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), np.array([3.0, 3.0, 3.0])),
        ((1.0, -2.0, 3.0), (4.0, -5.0, -6.0), np.array([3.0, 3.0, 9.0])),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])),
        ((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0), np.array([2.0, 4.0, 6.0])),
    ],
)
def test_translation_diff(
    translation1: tuple[float, ...],
    translation2: tuple[float, ...],
    expected_result: NDArray,
) -> None:
    """Verify that basic translation differences are calucalted.Calculate."""
    result = translation_diff(translation1, translation2)
    assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    ("translation1", "translation2"), [((1.0, 2.0), (1.0, 2.0, 3.0))]
)
def test_translation_diff_differing_length_raises_error(
    translation1: tuple[float, ...], translation2: tuple[float, ...]
) -> None:
    """TODO."""
    with pytest.raises(
        ValueError, match="Translation parameters must be of equal length"
    ):
        translation_diff(translation1, translation2)
