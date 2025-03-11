# ndimreg

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ndimreg)](https://github.com/jnk22/ndimreg/blob/main/pyproject.toml)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/ndimreg)](https://github.com/jnk22/ndimreg/blob/main/pyproject.toml)
[![ci](https://github.com/jnk22/ndimreg/actions/workflows/ci.yml/badge.svg)](https://github.com/jnk22/ndimreg/actions/workflows/ci.yml)
[![ReadtheDocs](https://readthedocs.org/projects/ndimreg/badge/?version=latest)](https://ndimreg.readthedocs.io)
[![codecov](https://codecov.io/github/jnk22/ndimreg/graph/badge.svg?token=5EJL318F1D)](https://codecov.io/github/jnk22/ndimreg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI - License](https://img.shields.io/pypi/l/ndimreg)](https://github.com/jnk22/ndimreg/blob/main/LICENSE)

This is a library and CLI for image registrations and image transformations
supporting 2D and 3D images.

Its main goal is to provide a collection of fast and optimized image
registration algorithms with an easy-to-use API.
Additionally, a comprehensive benchmarking suite can be used to compare various
registration methods and configurations.

## CLI Usage

Use [uv](https://github.com/astral-sh/uv) to run directly from your terminal:

```bash
uvx ndimreg
```

Or install via PIP first:

```bash
pip install ndimreg
ndimreg
```

_Run `ndimreg --help` to see all available commands._

### Extra Backends

The following additional device backends and FFT backends (CPU) are available
as extras:

| Backend                                           | Type                          | Extra      |
| ------------------------------------------------- | ----------------------------- | ---------- |
| [pyFFTW](https://github.com/pyFFTW/pyFFTW)        | FFT Backend (CPU)             | `pyfftw`   |
| [mkl_fft](https://github.com/IntelPython/mkl_fft) | FFT Backend (CPU)             | `mkl`      |
| [CuPy](https://github.com/cupy/cupy)              | GPU Support (NVIDIA, CUDA-12) | `cuda12`   |
| [CuPy](https://github.com/cupy/cupy)              | GPU Support (NVIDIA, CUDA-11) | `cuda11`   |
| [CuPy](https://github.com/cupy/cupy)              | GPU Support (AMD, ROCm 5.0)   | `rocm-5-0` |
| [CuPy](https://github.com/cupy/cupy)              | GPU Support (AMD, ROCm 4.3)   | `rocm-4-3` |

To install an extra backend, use `pip install ndimreg[extra]`, (e.g., `pip install ndimreg[cuda12]` for NVIDIA support with CUDA-12).

### Benchmarks

You can perform extensive benchmarks for various registrations, images, and
configurations with `ndimreg benchmark`.

To see all possible input variations, run `ndimreg benchmark 2d --help` or
`ndimreg benchmark 3d --help` respectively.

All outputs are generated as CSV and JSON data.

## Library Usage

### Setting the (CPU) FFT Backend

`ndimreg` respects your `scipy.fft` backend.

Set a global FFT backend for all following registrations:

```python
import mkl_fft._scipy_fft_backend as mkl_fft
from scipy import fft

# Set the global FFT backend to `mkl_fft` which will then be used
# during registration. You can use any backend that supports SciPy FFT.
fft.set_global_backend(mkl_fft)

# ...setup registration method and load images...

result = registration.register(fixed_image, moving_image)
```

Set a temporary FFT backend as context manager:

```python
import mkl_fft._scipy_fft_backend as mkl_fft
from scipy import fft

# ...setup registration method and load images...

# Temporarily set the FFT backend to `mkl_fft` which will then be used
# during registration. You can use any backend that supports SciPy FFT.
# The global/default backend will not be changed or overwritten.
with fft.set_backend(mkl_fft):
    result = registration.register(fixed_image, moving_image)
```

For more information, check out [SciPy: Discrete Fourier transforms: Backend control](https://docs.scipy.org/doc/scipy/reference/fft.html#backend-control)

### Installation With GPU Support

The following GPUs are supported:

- **NVIDIA (CUDA)**: Install as `pip install ndimreg[cuda12]`.
- **AMD (ROCm)**: Install as `pip install ndimreg[rocm-5-0]`.

### Choosing the FFT Backend

You can choose between the following CPU FFT backends by setting the
`--fft-backend` parameter:

| Backend                                           | Parameter           | Extra    |
| ------------------------------------------------- | ------------------- | -------- |
| [scipy](https://github.com/scipy/scipy)           | `scipy` _(default)_ | --       |
| [pyFFTW](https://github.com/pyFFTW/pyFFTW)        | `pyfftw`            | `pyfftw` |
| [mkl_fft](https://github.com/IntelPython/mkl_fft) | `mkl`               | `mkl`    |

To install an extra FFT backend such as `mkl`, install `ndimreg` as `pip install ndimreg[mkl]`.

**Note:** _When registering data that is on the GPU, FFT operations will always
use the `CuPy` backend!_

## Examples

**Image Source:** [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) (originally uploaded to [NASA Great Images database](https://flic.kr/p/r9qvLn))

### Translation Recovery (2D)

| Original                                                                          | Transformed                                                                             | Recovered                                                                          | Fused                                                                       |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| <img alt="Original Image" src="./docs/img/translation2d/astronaut-original.png"/> | <img alt="Transformed Image" src="./docs/img/translation2d/astronaut-transformed.png"/> | <img alt="Recoverd Image" src="./docs/img/translation2d/astronaut-recovered.png"/> | <img alt="Fused Image" src="./docs/img/translation2d/astronaut-fused.png"/> |

```console
ndimreg transform 2d \
  --method translation-2d \
  --options '{"upsample_factor": 10}' \
  --image-datasets astronaut \
  --resize 256 \
  --translations-absolute \
  --translation 12.8 25.6
```

```text
Expected: Transformation2D(translation=(12.80, 25.60), rotation=None, scale=None)
Recovered: Transformation2D(translation=(12.90, 25.70), rotation=None, scale=None)
Duration: 12.29 ms
```

<details>
<summary><strong>Click here to show the sample code!</strong></summary>
<br>

You can download the full sample [here](./examples/translation2d_recovery.py)!

```python
from ndimreg.fusion import MergeFusion
from ndimreg.image import Image2D
from ndimreg.registration import TranslationFFT2DRegistration
from ndimreg.transform import Transformation2D
from ndimreg.utils import format_time

# Load a test 2D image from 'scikit-image' and resize it to 256x256.
original = Image2D.from_skimage("astronaut").resize_to_shape(256)

# Copy the original image and transform it.
transformation = Transformation2D(translation=(12.8, 25.6))
transformed = original.copy().transform(transformation=transformation)

# Use the registration method with 'upsample_factor=10' for a translation
# precision of 0.1 to register the original image with the transformed image.
registration = TranslationFFT2DRegistration(upsample_factor=10)
result = original.register(registration, transformed)[0]

# We now transform the previously modified image with the recovered
# transformation output and fuse it with the original image.
recovered = transformed.copy().transform(
    transformation=result.transformation, inverse=True
)
fused = original.fuse(MergeFusion(), recovered)
```

</details>

### Translation and Rotation Recovery (2D)

| Original                                                                     | Transformed                                                                        | Recovered                                                                     | Fused                                                                  |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| <img alt="Original Image" src="./docs/img/keller2d/astronaut-original.png"/> | <img alt="Transformed Image" src="./docs/img/keller2d/astronaut-transformed.png"/> | <img alt="Recoverd Image" src="./docs/img/keller2d/astronaut-recovered.png"/> | <img alt="Fused Image" src="./docs/img/keller2d/astronaut-fused.png"/> |

```console
ndimreg transform 2d \
  --method keller-adf-2d \
  --options '{"shift_upsample_factor": 10}' \
  --image-datasets astronaut \
  --resize 256 \
  --translations-absolute \
  --translation 12.8 25.6 \
  --rotation 22
```

```text
Expected: Transformation2D(translation=(12.80, 25.60), rotation=22.00, scale=None)
Recovered: Transformation2D(translation=(12.80, 25.70), rotation=22.10, scale=None)
Duration: 187.90 ms
```

<details>
<summary><strong>Click here to show the sample code!</strong></summary>
<br>

You can download the full sample [here](./examples/keller2d_recovery.py)!

```python
from ndimreg.fusion import MergeFusion
from ndimreg.image import Image2D
from ndimreg.registration import Keller2DRegistration
from ndimreg.transform import Transformation2D
from ndimreg.utils import format_time

# Load a test 2D image from 'scikit-image' and resize it to 256x256.
original = Image2D.from_skimage("astronaut").resize_to_shape(256)

# Copy the original image and transform it.
transformation = Transformation2D(translation=(12.8, 25.6), rotation=22)
transformed = original.copy().transform(transformation=transformation)

# Use the registration method with 'shift_upsample_factor=10' for a translation
# precision of 0.1 to register the original image with the transformed image.
registration = Keller2DRegistration(shift_upsample_factor=10)
result = original.register(registration, transformed)[0]

# We now transform the previously modified image with the recovered
# transformation output and fuse it with the original image.
recovered = transformed.copy().transform(
    transformation=result.transformation, inverse=True
)
fused = original.fuse(MergeFusion(), recovered)
```

</details>

### Translation, Rotation, and Scale Recovery (2D)

| Original                                                                       | Transformed                                                                          | Recovered                                                                       | Fused                                                                    |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| <img alt="Original Image" src="./docs/img/imregdft2d/astronaut-original.png"/> | <img alt="Transformed Image" src="./docs/img/imregdft2d/astronaut-transformed.png"/> | <img alt="Recoverd Image" src="./docs/img/imregdft2d/astronaut-recovered.png"/> | <img alt="Fused Image" src="./docs/img/imregdft2d/astronaut-fused.png"/> |

```console
ndimreg transform 2d \
    --method imregdft-2d \
    --image-datasets astronaut \
    --resize 256 \
    --translations-absolute \
    --translation 12.8 25.6 \
    --rotation 22 \
    --scale 1.1
```

```text
Expected: Transformation2D(translation=(12.80, 25.60), rotation=22.00, scale=1.10)
Recovered: Transformation2D(translation=(13.09, 25.42), rotation=22.01, scale=1.10)
Duration: 92.09 ms
```

<details>
<summary><strong>Click here to show the sample code!</strong></summary>
<br>

You can download the full sample [here](./examples/imregdft2d_recovery.py)!

```python
# Load a test 2D image from 'scikit-image' and resize it to 256x256.
original = Image2D.from_skimage("astronaut").resize_to_shape(256)

# Copy the original image and transform it.
transformation = Transformation2D(translation=(12.8, 25.6), rotation=22, scale=1.1)
transformed = original.copy().transform(transformation=transformation)

# Use the default options for the 'imreg_dft' wrapped registration method
# to register the original image with the transformed image.
registration = ImregDft2DRegistration()
result = original.register(registration, transformed)[0]

# We now transform the previously modified image with the recovered
# transformation output and fuse it with the original image.
recovered = transformed.copy().transform(
    transformation=result.transformation, inverse=True
)
fused = original.fuse(MergeFusion(), recovered)
```

</details>

### Translation, Rotation, and Scale Recovery (2D) -- 2

| Original                                                                     | Transformed                                                                        | Recovered                                                                     | Fused                                                                  |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| <img alt="Original Image" src="./docs/img/scikit2d/astronaut-original.png"/> | <img alt="Transformed Image" src="./docs/img/scikit2d/astronaut-transformed.png"/> | <img alt="Recoverd Image" src="./docs/img/scikit2d/astronaut-recovered.png"/> | <img alt="Fused Image" src="./docs/img/scikit2d/astronaut-fused.png"/> |

```console
ndimreg transform 2d \
    --method scikit-2d \
    --options '{"shift_upsample_factor": 10}' \
    --image-datasets astronaut \
    --resize 256 \
    --translations-absolute \
    --translation 12.8 25.6 \
    --rotation 22 \
    --scale 1.1 \
    --bandpass 0 1 \
    --window hann
```

```text
Expected: Transformation2D(translation=(12.80, 25.60), rotation=22.00, scale=1.10)
Recovered: Transformation2D(translation=(13.20, 26.00), rotation=21.09, scale=1.08)
Duration: 86.50 ms
```

<details>
<summary><strong>Click here to show the sample code!</strong></summary>
<br>

You can download the full sample [here](./examples/scikit2d_recovery.py)!

```python
from ndimreg.fusion import MergeFusion
from ndimreg.image import Image2D
from ndimreg.processor import GaussianBandPassFilter, WindowFilter
from ndimreg.registration import Scikit2DRegistration
from ndimreg.transform import Transformation2D
from ndimreg.utils import format_time

# Load a test 2D image from 'scikit-image' and resize it to 256x256.
original = Image2D.from_skimage("astronaut").resize_to_shape(256)

# Copy the original image and transform it.
transformation = Transformation2D(translation=(12.8, 25.6), rotation=22, scale=1.1)
transformed = original.copy().transform(transformation=transformation)

# Use the registration method with 'shift_upsample_factor=10' for a translation
# precision of 0.1 to register the original image with the transformed image.
# This method also requires pre-processing with a bandpass and window filter.
pre_processors = [GaussianBandPassFilter(0, 1), WindowFilter("hann")]
registration = Scikit2DRegistration(shift_upsample_factor=10, processors=pre_processors)
result = original.register(registration, transformed)[0]

# We now transform the previously modified image with the recovered
# transformation output and fuse it with the original image.
recovered = transformed.copy().transform(
    transformation=result.transformation, inverse=True
)
fused = original.fuse(MergeFusion(), recovered)
```

</details>

### Translation (3D)

...

### Translation and Axis Rotation Recovery (3D)

...

### Translation and Full Rotation Recovery (3D)

...

## Features

### Available Registration Methods

| Name                           | Dimension |  T  |  R  |  S  | Description                                                                           |
| ------------------------------ | :-------: | :-: | :-: | :-: | ------------------------------------------------------------------------------------- |
| `Scikit2DRegistration`         |    2D     | ✅  | ✅  | ✅  | Based on **scikit-image** example[^1].                                                |
| `ImregDft2DRegistration`       |    2D     | ✅  | ✅  | ✅  | Wrapper around [**imreg_dft**](https://github.com/matejak/imreg_dft) library.         |
| `Keller2DRegistration`         |    2D     | ✅  | ✅  | ❌  | Implementation of [10.1109/TPAMI.2005.128](https://doi.org/10.1109/TPAMI.2005.128).   |
| `TranslationFFT2DRegistration` |    2D     | ✅  | ❌  | ❌  | Translation recovery using _Fourier Shift Theorem_.                                   |
| `TranslationFFT3DRegistration` |    3D     | ✅  | ❌  | ❌  | Translation recovery using _Fourier Shift Theorem_.                                   |
| `Keller3DRegistration`         |    3D     | ✅  | ✅  | ❌  | Implementation of [10.1109/TSP.2006.881217](https://doi.org/10.1109/tsp.2006.881217). |
| `RotationAxis3DRegistration`   |    3D     | ✅  | ✅  | ❌  | Part of `Keller3DRegistration` for single axis rotation recovery.                     |

T: Translation\
R: Rotation\
S: Scale

[^1]: <https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html>

### Image Transformations and Utilities

| Feature   | State | Backend(s)                                                          |
| --------- | :---: | ------------------------------------------------------------------- |
| Translate |  ✅   | scipy/CuPy: `[cupyx.]scipy.ndimage.affine_transform()`              |
| Rotate    |  ✅   | scipy/CuPy: `[cupyx.]scipy.ndimage.affine_transform()`              |
| Scale     |  ✅   | scipy/CuPy: `[cupyx.]scipy.ndimage.affine_transform()`              |
| Resize    |  ✅   | scikit-image/cuCIM: `[cucim.]skimage.transform.resize_local_mean()` |
| Pad       |  🚧   | --                                                                  |
| Normalize |  ✅   | scikit-image/cuCIM: `[cucim.]skimage.exposure.rescale_intensity()`  |
| Grayscale |  ✅   | scikit-image/cuCIM: `[cucim.]skimage.color.rgb2gray()`              |
| Clip      |  ✅   | --                                                                  |
| Cut       |  ✅   | --                                                                  |
| Copy      |  ✅   | numpy: `numpy.copy()`                                               |
| Save      |  ✅   | scikit-image: `skimage.io.imsave()`                                 |
| Show      |  ✅   | matplotlib (2D), napari (3D)                                        |

✅ Feature exists.\
🚧 Feature is WIP.

## Backends

### Registration

- [NumPy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [scikit-image](https://github.com/scikit-image/scikit-image)
- Optional: [CuPy](https://github.com/cupy/cupy) +
  [cuCIM](https://github.com/rapidsai/cucim) _(GPU support on Nvidia/AMD)_
- Optional: [pyFFTW](https://github.com/pyFFTW/pyFFTW)
- Optional: [mkl_fft](https://github.com/IntelPython/mkl_fft)

_The following backends can be added as alternatives and/or to support more
architectures in the future:_

- [ ] [DPNP](https://github.com/IntelPython/dpnp)
- [ ] [clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype)
- [ ] [PyTorch](https://github.com/pytorch/pytorch)
- [ ] [Dask](https://github.com/dask/dask)
- [ ] [xarray](https://github.com/pydata/xarray)

### Transformation

- [NumPy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [scikit-image](https://github.com/scikit-image/scikit-image)
- Optional: [CuPy](https://github.com/cupy/cupy) +
  [cuCIM](https://github.com/rapidsai/cucim) _(GPU support on Nvidia/AMD)_

### Interoperability

- Interoperability for general array operations is implemented via the
  [Python array API standard](https://github.com/data-apis/array-api).
- Interoperability for FFT operations is implemented via [scipy's backend API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.set_backend.html).
