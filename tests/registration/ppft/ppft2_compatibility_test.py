"""Test module for PPFT2D code.

Data generation for compatibility tests:

```matlab
function generatePPFTData

generatePPFTDataRandom(@OptimizedPPFT, "random_ppft2_")
generatePPFTDataRandom(@slowPPFT, "random_slowppft2_")
generatePPFTDataImage(@OptimizedPPFT, "f16_ppft2", im2double(rgb2gray(imread("f16_adf.png"))))
generatePPFTDataImage(@slowPPFT, "f16_slowppft2", im2double(rgb2gray(imread("f16_adf.png"))))

function generatePPFTDataRandom(fun, prefix)

for n = 2:7
    ns = 2 ^ n;
    disp("Generating PPFT2 data for random image with size " + ns + "...");
    im = rand(ns, ns);
    [pp_sector1, pp_sector2] = fun(im);
    pp = zeros(2, ns * 2 + 1, ns + 1);
    pp(1, :, :) = pp_sector1;
    pp(2, :, :) = pp_sector2;
    save(prefix + ns + ".in.mat", "im");
    save(prefix + ns + ".out.mat", "pp");
end

function generatePPFTDataImage(fun, prefix, im)

im_size = size(im)
im_size = im_size(1)
disp("Generating PPFT2 data input image");
[pp_sector1, pp_sector2] = fun(im);
pp = zeros(2, im_size * 2 + 1, im_size + 1);
pp(1, :, :) = pp_sector1;
pp(2, :, :) = pp_sector2;
save(prefix + ".in.mat", "im");
save(prefix  + ".out.mat", "pp");
```

All generated files have been created within MATLAB using the following
code:
https://github.com/ShkolniskyLab/ppft2D/blob/main/radon3/tests/testppft2.m
"""  # noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.io import loadmat

from ndimreg.registration.ppft import ppft2, ppft2_optimized

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("ppft2_func", [ppft2, ppft2_optimized])
def test_ppft2_matlab_compatibility_ppft2_f16_image(ppft2_func: Callable) -> None:
    """Verify that PPFT2D results match the MATLAB implementation."""
    data_path = Path("tests/registration/ppft/ppft2_data")

    expected = loadmat(data_path / "f16_ppft2.out.mat")["pp"]
    actual = ppft2_func(loadmat(data_path / "f16_ppft2.in.mat")["im"])

    np.testing.assert_allclose(actual, expected, rtol=1e-11)


# TODO: Read dynamically if possible.
@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("ppft2_func", [ppft2, ppft2_optimized])
def test_ppft2_matlab_compatibility_ppft2_random_data(
    ppft2_func: Callable, n: int
) -> None:
    """Verify that PPFT2D results match the MATLAB implementation."""
    data_path = Path("tests/registration/ppft/ppft2_data")

    in_file = data_path / f"random_ppft2_{n}.in.mat"
    out_file = data_path / f"random_ppft2_{n}.out.mat"

    assert in_file.exists()
    assert out_file.exists()

    expected = loadmat(out_file)["pp"]
    actual = ppft2_func(loadmat(in_file)["im"])

    np.testing.assert_allclose(actual, expected, rtol=1e-12)
