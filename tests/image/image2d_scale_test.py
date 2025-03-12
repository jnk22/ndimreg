"""Tests for `ndimreg` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

if TYPE_CHECKING:
    from ndimreg.image import Image2D


def test_image2d_scale_one(homogeneous_image_3x3: Image2D) -> None:
    """Verify that zoomed `Image2D` does not change with factor 1.0."""
    image_data = homogeneous_image_3x3.data.copy()
    homogeneous_image_3x3.transform(scale=1.0)

    # When zoom factor is 1.0, the image should not change.
    assert_allclose(homogeneous_image_3x3.data, image_data)
    assert homogeneous_image_3x3.shape == (3, 3)


@pytest.mark.parametrize("factor", [1e-10, 1e-5, 0.1, 0.3, 0.32, 0.33])
def test_image2d_scale_out_below_third_removes_edges(
    homogeneous_image_3x3: Image2D, factor: float
) -> None:
    """Verify that zoomed `Image2D` removes borders with factor below 0.5."""
    homogeneous_image_3x3.transform(scale=factor)
    assert homogeneous_image_3x3.shape == (3, 3)

    zoomed_out_image = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    assert_allclose(homogeneous_image_3x3.data, zoomed_out_image)


@pytest.mark.parametrize("factor", [1.01, 1.5, 2.0, 3.0])
def test_image2d_scale_into_homogenous_image_stays_equal(
    homogeneous_image_3x3: Image2D, factor: float
) -> None:
    """Verify that zoomed `Image2D` does not change with factor above 1.0."""
    image_data = homogeneous_image_3x3.data.copy()

    homogeneous_image_3x3.transform(scale=factor)
    assert homogeneous_image_3x3.shape == (3, 3)

    # When zoom factor is above 1.0, the image should not change.
    assert_allclose(homogeneous_image_3x3.data, image_data)


# NOTE: Tests are currently failing. Definition for scale=0 required.
@pytest.mark.skip
def test_image2d_scale_zero_maximum_scale_out_removes_content(
    homogeneous_image_3x3: Image2D,
) -> None:
    """Verify that zoomed `Image2D` removes all content for maximum factor 0.0."""
    homogeneous_image_3x3.transform(scale=0.0)
    assert homogeneous_image_3x3.shape == (3, 3)

    # When zoom factor is 0.0, i.e., the maximum possible 'zoom out
    # factor', all image content will be removed and therefore only
    # zeros remain.
    empty_image = np.zeros_like(homogeneous_image_3x3.data)
    assert_array_equal(homogeneous_image_3x3.data, empty_image)
