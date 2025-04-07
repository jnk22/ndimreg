"""CLI register module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from cyclopts import App
from cyclopts.types import Json, ResolvedDirectory, ResolvedExistingFile  # noqa: TC002

from ._types import (  # noqa: TC001
    BandpassFilter,
    ContextType,
    CpuFftBackend,
    Device,
    InterpolationOrder,
    MaxPad,
    RegistrationDebugStep,
    RegistrationMethod2D,
    RegistrationMethod3D,
    Resize,
    RotationAxis3D,
    SafePad,
    Spacing2D,
    Spacing3D,
    TransformationOrder,
    WindowFilter,
    Zoom,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndimreg.image import Image, ImageLoader
    from ndimreg.registration import Registration

EMPTY_OPTIONS: Final[dict] = {}

app = App(name="register")

# TODO: Combine debug output with result images (show mode).
# TODO: Allow setting debug output steps (preprocessing, registration, ...).


@app.command(name="2d")
def register_2d(  # noqa: PLR0913
    fixed_image_path: ResolvedExistingFile,
    moving_image_paths: list[ResolvedExistingFile],
    *,
    method: RegistrationMethod2D = "keller-adf-2d",
    options: Json | dict = EMPTY_OPTIONS,
    interpolation_order: InterpolationOrder = 3,
    transformation_order: TransformationOrder = "trs",
    normalize: bool = True,
    resize: Resize = None,
    spacing: Spacing2D | None = None,
    max_pad: MaxPad = True,
    safe_pad: SafePad = False,
    bandpass: BandpassFilter = None,
    window: WindowFilter = None,
    zoom: Zoom = None,
    rescale: bool = False,
    device: Device = "cpu",
    device_id: int = 0,
    debug_save: bool = False,
    debug_show: bool = False,
    debug_depth: int | None = None,
    debug_steps: list[RegistrationDebugStep] | None = None,
    debug_context: ContextType = "notebook",
    result_show: bool = True,
    result_save: bool = False,
    output_dir: ResolvedDirectory = Path("output"),
    fft_backend: CpuFftBackend | None = None,
) -> None:
    """Register 2D input images."""
    from ndimreg.image import Image2D, Paths2DImageLoader
    from ndimreg.utils import get_available_fft_backends
    from ndimreg.utils.image import prepare_benchmark_image
    from ndimreg.utils.plot_images import set_matplotlib_context

    from ._common import (
        REGISTRATION_METHODS_2D,
        get_pre_processors,
        setup_cpu_fft_backend,
        setup_gpu,
        setup_logging,
    )

    chosen_fft_backend = fft_backend or next(get_available_fft_backends())

    setup_logging()
    set_matplotlib_context(debug_context)
    setup_cpu_fft_backend(chosen_fft_backend)
    print(f"Using CPU FFT backend: {chosen_fft_backend}")

    if device == "gpu":
        setup_gpu(device_id=device_id)

    print("Loading images from input paths...")
    fixed_image = Image2D.from_path(fixed_image_path)
    moving_images = [*Paths2DImageLoader(moving_image_paths)]

    for image in (fixed_image, *moving_images):
        prepare_benchmark_image(
            image,
            normalize=normalize,
            resize=resize,
            spacing=spacing,
            max_pad=max_pad,
            safe_pad=safe_pad,
        )

    pre_processors = get_pre_processors(
        zoom=zoom, rescale=rescale, bandpass=bandpass, window=window, dim=2
    )

    registration = REGISTRATION_METHODS_2D[method](
        processors=pre_processors, debug=debug_show or debug_save, **options
    )

    __register_nd(
        fixed_image,
        moving_images,
        registration,
        interpolation_order=interpolation_order,
        transformation_order=transformation_order,
        debug_save=debug_save,
        debug_show=debug_show,
        debug_depth=debug_depth,
        debug_steps=debug_steps,
        result_save=result_save,
        result_show=result_show,
        output_dir=output_dir,
        device=device,
    )


@app.command(name="3d")
def register_3d(  # noqa: PLR0913
    fixed_image_path: ResolvedExistingFile,
    moving_image_paths: list[ResolvedExistingFile],  # TODO: Require at least one.
    *,
    method: RegistrationMethod3D = "keller-3d",
    options: Json | dict = EMPTY_OPTIONS,
    interpolation_order: InterpolationOrder = 3,
    transformation_order: TransformationOrder = "trs",
    rotation_axis: RotationAxis3D = "z",
    normalize: bool = True,
    resize: Resize = None,
    spacing: Spacing3D | None = None,
    max_pad: MaxPad = False,
    safe_pad: SafePad = False,
    bandpass: BandpassFilter = None,
    window: WindowFilter = None,
    zoom: Zoom = None,
    rescale: bool = False,
    device: Device = "cpu",
    device_id: int = 0,
    debug_save: bool = False,
    debug_show: bool = False,
    debug_depth: int | None = None,
    debug_steps: list[RegistrationDebugStep] | None = None,
    debug_context: ContextType = "notebook",
    result_show: bool = True,
    result_save: bool = False,
    output_dir: ResolvedDirectory = Path("output"),
    fft_backend: CpuFftBackend | None = None,
) -> None:
    """Register 3D input images."""
    from ndimreg.image import Image3D, Paths3DImageLoader
    from ndimreg.utils import get_available_fft_backends
    from ndimreg.utils.image import prepare_benchmark_image
    from ndimreg.utils.plot_images import set_matplotlib_context

    from ._common import (
        REGISTRATION_METHODS_3D,
        get_pre_processors,
        setup_cpu_fft_backend,
        setup_gpu,
        setup_logging,
    )

    chosen_fft_backend = fft_backend or next(get_available_fft_backends())

    setup_logging()
    set_matplotlib_context(debug_context)
    setup_cpu_fft_backend(chosen_fft_backend)
    print(f"Using CPU FFT backend: {chosen_fft_backend}")

    if device == "gpu":
        setup_gpu(device_id=device_id)

    fixed_image = Image3D.from_path(fixed_image_path)
    moving_images = [*Paths3DImageLoader(moving_image_paths)]

    for image in (fixed_image, *moving_images):
        prepare_benchmark_image(
            image,
            normalize=normalize,
            resize=resize,
            spacing=spacing,
            max_pad=max_pad,
            safe_pad=safe_pad,
        )

    pre_processors = get_pre_processors(
        zoom=zoom, rescale=rescale, bandpass=bandpass, window=window, dim=3
    )

    registration = REGISTRATION_METHODS_3D[method](
        axis=rotation_axis,
        processors=pre_processors,
        debug=debug_show or debug_save,
        **options,
    )

    __register_nd(
        fixed_image,
        moving_images,
        registration,
        interpolation_order=interpolation_order,
        transformation_order=transformation_order,
        debug_save=debug_save,
        debug_show=debug_show,
        debug_depth=debug_depth,
        debug_steps=debug_steps,
        result_save=result_save,
        result_show=result_show,
        output_dir=output_dir,
        device=device,
    )


def __register_nd(  # noqa: PLR0913
    fixed_image: Image,
    moving_images: Sequence[Image] | ImageLoader,
    registration: Registration,
    *,
    interpolation_order: InterpolationOrder = 3,
    transformation_order: TransformationOrder = "trs",
    debug_save: bool = False,
    debug_show: bool = False,
    debug_depth: int | None,
    debug_steps: list[RegistrationDebugStep] | None,
    result_save: bool,
    result_show: bool,
    output_dir: Path,
    device: Device,
) -> None:
    import sys

    from ndimreg.fusion import MergeFusion
    from ndimreg.image import Image2D, Image3D
    from ndimreg.utils import format_time

    from ._common import save_debug_images, save_result_images, show_debug_images

    results = fixed_image.register(registration, *moving_images, device=device)
    for result in results:
        print(f"Recovered: {result.transformation}")
        print(f"Duration: {format_time(result.total_duration)}")

    recovered_images = [
        im.copy(name=f"{im.name}-recovered").transform(
            result.transformation,
            inverse=True,
            interpolation_order=interpolation_order,
            transformation_order=transformation_order,
        )
        for im, result in zip(moving_images, results, strict=True)
    ]

    if not any((result_save, result_show, debug_save, debug_show)):
        # If we do not show or save any output, any further processing
        # is redundant.
        sys.exit(0)

    if debug_show:
        for result in results:
            show_debug_images(result, step=debug_steps, depth=debug_depth)
    if debug_save:
        for result, moving_image in zip(results, moving_images, strict=True):
            debug_output_dir = output_dir / f"debug_{moving_image.name}"
            save_debug_images(
                result, debug_output_dir, step=debug_steps, depth=debug_depth
            )

    if not (result_show or result_save):
        # If we do not show or save any result images, any further
        # processing is redundant.
        sys.exit(0)

    fused_image = fixed_image.fuse(MergeFusion(), *recovered_images, device=device)
    fused_image.to_device("cpu")

    result_images = (fixed_image, *moving_images, *recovered_images, fused_image)

    if result_show:
        fixed_image.show_all(*result_images)
    if result_save:
        save_result_images(result_images, output_dir / fixed_image.name)

        if screenshots_3d := [
            Image2D(im.get_screenshot_2d(), name=f"{im.name}-2d")
            for im in result_images
            if isinstance(im, Image3D)
        ]:
            save_result_images(screenshots_3d, output_dir / fixed_image.name)
