"""TODO."""

# WARNING: Do not import outside of commands due to slow import time!

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Final

from loguru import logger
from scipy import fft

from ndimreg.image import Image2D, Image3D
from ndimreg.registration import (
    ImregDft2DRegistration,
    Keller2DRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    Scikit2DRegistration,
    TranslationFFT2DRegistration,
    TranslationFFT3DRegistration,
)
from ndimreg.utils import get_fft_backend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ndimreg.image import Image
    from ndimreg.processor import DataProcessor
    from ndimreg.registration import BaseRegistration, RegistrationResult
    from ndimreg.utils.fft import CpuFftBackend

    from ._types import (
        BandpassFilter,
        Dimension,
        RegistrationMethod2D,
        RegistrationMethod3D,
        WindowFilter,
        Zoom,
    )

REGISTRATION_METHODS_2D: Final[dict[RegistrationMethod2D, type[BaseRegistration]]] = {
    "keller-adf-2d": Keller2DRegistration,
    "scikit-2d": Scikit2DRegistration,
    "imregdft-2d": ImregDft2DRegistration,
    "translation-2d": TranslationFFT2DRegistration,
}
REGISTRATION_METHODS_3D: Final[dict[RegistrationMethod3D, type[BaseRegistration]]] = {
    "keller-3d": Keller3DRegistration,
    "rotationaxis-3d": RotationAxis3DRegistration,
    "translation-3d": TranslationFFT3DRegistration,
}
REGISTRATION_METHODS = REGISTRATION_METHODS_2D | REGISTRATION_METHODS_3D


def show_debug_images(result: RegistrationResult, **kwargs: Any) -> None:
    if debug_images_2d := result.get_debug_images(dim=2, **kwargs):
        Image2D.show_all(*(Image2D(im.data, im.name) for im in debug_images_2d))

    if debug_images_3d := result.get_debug_images(dim=3, **kwargs):
        Image3D.show_all(*(Image3D(im.data, im.name) for im in debug_images_3d))


def save_result_images(images: Sequence[Image], output_dir: Path) -> None:
    extension = "png" if images[0].dim == 2 else "tif"
    images[0].save_all(*images, extension=extension, directory=output_dir / "results")


def save_debug_images(
    result: RegistrationResult, output_dir: Path, **kwargs: Any
) -> None:
    # TODO: Fix saving images that are not in range [-1, 1] (clip?).
    if debug_images_2d := result.get_debug_images(dim=2, **kwargs):
        Image2D.save_all(
            *(Image2D(im.data, im.name) for im in debug_images_2d),
            extension="png",
            directory=output_dir,
        )

    if debug_images_3d := result.get_debug_images(dim=3, **kwargs):
        Image3D.save_all(
            *(Image3D(im.data, im.name) for im in debug_images_3d),
            extension="tif",
            directory=output_dir,
        )


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", level))


def setup_cpu_fft_backend(backend: CpuFftBackend) -> None:
    """TODO."""
    logger.debug(f"Setting '{backend}' as CPU FFT backend")
    fft.set_global_backend(get_fft_backend(backend))


def setup_gpu(*, device_id: int) -> None:
    """TODO."""
    try:
        import cupy as cp
    except ImportError:
        print("Error: GPU support ist not available as CuPy could not be imported.")
        sys.exit(1)

    try:
        available_devices = {
            i: cp.cuda.runtime.getDeviceProperties(i)["name"].decode("utf-8")
            for i in range(cp.cuda.runtime.getDeviceCount())
        }
        print(f"Available CUDA devices: {available_devices}")

        print(f"Selecting CUDA device {device_id} ({available_devices[device_id]})...")
        cp.cuda.runtime.setDevice(device_id)

    except cp.cuda.runtime.CUDARuntimeError:
        print(f"Error: CUDA device ID {device_id} is not available.")
        sys.exit(1)


def get_pre_processors(
    *,
    dim: Dimension,
    rescale: bool,
    zoom: Zoom,
    bandpass: BandpassFilter,
    window: WindowFilter,
) -> list[DataProcessor]:
    from ndimreg.processor import (
        GaussianBandPassFilter,
        RescaleProcessor,
        WindowFilter,
        Zoomer,
    )

    pre_processors: list[DataProcessor] = []
    if zoom is not None:
        pre_processors.append(Zoomer(zoom, dim=dim))
    if bandpass:
        pre_processors.append(GaussianBandPassFilter(*bandpass))
    if window:
        pre_processors.append(WindowFilter(window))
    if rescale:
        pre_processors.append(RescaleProcessor(group=True))

    return pre_processors
