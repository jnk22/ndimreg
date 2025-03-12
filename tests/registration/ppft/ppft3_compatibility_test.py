"""Test module for PPFT3 code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat

from ndimreg.registration.ppft import ppft3, ppft3_vectorized

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("ppft3_func", [ppft3, ppft3_vectorized])
def test_ppft3_original_example_im2(ppft3_func: Callable) -> None:
    """Verify example input to be converted to proper output shape.

    This example is a test for a single test input for PPFT3 that has
    been manually compared to the output of the original MATLAB code.
    """
    data = np.ones((4, 4, 4)) * np.array([0.1, 0.2, 0.3, 0.4]).reshape(1, 1, 4)

    assert data.shape == (4, 4, 4)
    assert data.ndim == 3  # noqa: PLR2004
    assert data.dtype == np.float64
    result = ppft3_func(data)

    assert result.shape == (3, 13, 5, 5)

    assert result[0, 0, 0, 0] == pytest.approx(0.0462 - 0.0099j, rel=1e-3)
    assert_allclose(
        result[0, :, 0, 0],
        np.array(
            [
                0.0462 - 0.0099j,
                0.2890 - 0.0932j,
                0.1509 - 0.0658j,
                0.0269 - 0.0328j,
                1.6290 - 1.8533j,
                9.3580 - 4.7693j,
                16.0000 + 0.0000j,
                9.3580 + 4.7693j,
                1.6290 + 1.8533j,
                0.0269 + 0.0328j,
                0.1509 + 0.0658j,
                0.2890 + 0.0932j,
                0.0462 + 0.0099j,
            ]
        ),
        rtol=1e-2,
    )

    # TODO: Fix tests (might be due to wrong flipping within PP assignment).
    assert_allclose(
        result[0, 0, :, 0],
        np.array(
            [
                0.0462 - 0.0099j,
                -0.0317 - 0.0179j,
                -0.1316 - 0.3820j,
                0.0140 - 0.0337j,
                -0.0425 + 0.0207j,
            ]
        ),
        rtol=1e-2,
    )


@pytest.mark.parametrize("ppft3_func", [ppft3, ppft3_vectorized])
def test_ppft3_original_example_im2_ones(ppft3_func: Callable) -> None:
    """Verify example input to be converted to proper output shape.

    This example is a test for a single test input for PPFT3 that has
    been manually compared to the output of the original MATLAB code.
    """
    # Define the array in Python using NumPy
    data = np.ones((4,) * 3) * np.ones(4).reshape(1, 1, 4)
    assert data.shape == (4, 4, 4)
    assert data.ndim == 3  # noqa: PLR2004
    assert data.dtype == np.float64

    result = ppft3_func(data)
    assert result.shape == (3, 13, 5, 5)

    assert_allclose(
        result[0, :, 0, 0],
        np.array(
            [
                0.0364 - 0.0959j,
                1.0597 - 0.5562j,
                0.5079 + 0.1252j,
                -0.0267 - 0.0387j,
                0.9817 - 8.0853j,
                30.4410 - 26.9684j,
                64.0000 + 0.0000j,
                30.4410 + 26.9684j,
                0.9817 + 8.0853j,
                -0.0267 + 0.0387j,
                0.5079 - 0.1252j,
                1.0597 + 0.5562j,
                0.0364 + 0.0959j,
            ]
        ),
        rtol=1e-3,
    )

    assert_allclose(
        result[0, 0, :, 0],
        np.array(
            [
                0.0364 - 0.0959j,
                -0.0700 + 0.0368j,
                -0.8511 - 0.2098j,
                -0.0449 - 0.0651j,
                -0.0124 + 0.1018j,
            ]
        ),
        rtol=1e-3,
    )


# TODO: Read dynamically if possible.
@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("ppft3_func", [ppft3, ppft3_vectorized])
def test_ppft3_matlab_implementation_compatibility_fastppft3(
    ppft3_func: Callable, n: int
) -> None:
    """Verify that PPFT3 results match the MATLAB implementation.

    This ensures compatibility between the MATLAB and Python
    implementations.

    All generated files have been created within MATLAB using the
    following code:
    https://github.com/ShkolniskyLab/PPFT3D/blob/main/radon3/tests/testppft3.m

    ````matlab
    im = rand(n,n,n);
    %save("random_" + n + ".in.mat", "im")
    ...
    pp2 = ppft3(im);
    %save("random_" + n + ".out.mat", "pp2")
    """
    data_path = Path("tests/registration/ppft/ppft3_data")

    in_file = data_path / f"random_{n}.in.mat"
    out_file = data_path / f"random_{n}.out.mat"

    assert in_file.exists()
    assert out_file.exists()

    expected = loadmat(out_file)["pp2"]
    actual = ppft3_func(loadmat(in_file)["im"])

    assert_allclose(actual, expected, rtol=1e-12)
