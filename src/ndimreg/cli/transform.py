"""CLI transform module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    RotationAngle,
    RotationAxis3D,
    RotationEulerXYZ,
    SafePad,
    Scale,
    Spacing2D,
    Spacing3D,
    TransformationOrder,
    Translation2D,
    Translation3D,
    TranslationsRelativeBool,
    WindowFilter,
    Zoom,
)

if TYPE_CHECKING:
    from ndimreg.image import Image
    from ndimreg.registration import Registration
    from ndimreg.transform import Transformation

app = App(name="transform")

# TODO: Combine debug output with result images (show mode).
# TODO: Allow quaternion rotation inputs for 3D.
# TODO: Allow to choose radians/degrees for input angles (2D + 3D Euler angles).
# TODO: Allow setting debug output steps (preprocessing, registration, ...).


@app.command(name="2d")
def transform_2d(  # noqa: PLR0913
    image_paths: list[ResolvedExistingFile] | None = None,
    *,
    image_datasets: list[str] | None = None,
    method: RegistrationMethod2D = "keller-adf-2d",
    translation: Translation2D | None = None,
    translation_relative: TranslationsRelativeBool = True,
    rotation: RotationAngle | None = None,
    scale: Scale | None = None,
    interpolation_order: InterpolationOrder = 3,
    transformation_order: TransformationOrder = "trs",
    options: Json = "{}",
    normalize: bool = True,
    resize: Resize = None,
    spacing: Spacing2D | None = None,
    max_pad: MaxPad = False,
    safe_pad: SafePad = False,
    bandpass: BandpassFilter = None,
    window: WindowFilter = None,
    zoom: Zoom = None,
    rescale: bool = False,
    noise: float | None = None,
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
    """Transform 2D image and register it with the original image."""
    import sys

    import numpy as np

    from ndimreg.image import Paths2DImageLoader, Scikit2DImageLoader
    from ndimreg.transform import Transformation2D
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

    images = []
    if image_paths:
        print("Loading images from input paths...")
        images.extend(Paths2DImageLoader(image_paths))
    if image_datasets:
        print("Loading images from scikit-image samples...")
        images.extend(Scikit2DImageLoader(image_datasets))
    if not images:
        print("No images to process")
        sys.exit(1)

    pre_processors = get_pre_processors(
        zoom=zoom, rescale=rescale, bandpass=bandpass, window=window, dim=3
    )

    # Registration options must be coerced into actual dictionary.
    # If empty JSON string ('{}') has been provided, it will not
    # automatically be a proper dict which would result in an error.
    registration_options = options if isinstance(options, dict) else {}
    registration = REGISTRATION_METHODS_2D[method](
        processors=pre_processors,
        debug=debug_show or debug_save,
        **registration_options,
    )

    transformation_params = {
        "transformation_order": transformation_order,
        "interpolation_order": interpolation_order,
    }

    for image in images:
        prepare_benchmark_image(
            image,
            normalize=normalize,
            spacing=spacing,
            resize=resize,
            max_pad=max_pad,
            safe_pad=safe_pad,
        )

        if translation is not None:
            translation_absolute = (
                tuple(np.array(translation) * image.resolution / 100)
                if translation_relative
                else translation
            )
        else:
            translation_absolute = None

        transformation_input = Transformation2D(
            translation=translation_absolute, rotation=rotation, scale=scale
        )

        __transform_nd(
            image,
            registration,
            tform_input=transformation_input,
            tform_params=transformation_params,
            noise=noise,
            device=device,
            debug_save=debug_save,
            debug_show=debug_show,
            debug_depth=debug_depth,
            debug_steps=debug_steps,
            result_save=result_save,
            result_show=result_show,
            output_dir=output_dir,
        )


@app.command(name="3d")
def transform_3d(  # noqa: PLR0913
    image_paths: list[ResolvedExistingFile] | None = None,
    *,
    image_datasets: list[str] | None = None,
    method: RegistrationMethod3D = "keller-3d",
    translation: Translation3D | None = None,
    translation_relative: TranslationsRelativeBool = True,
    rotation: RotationEulerXYZ | None = None,
    scale: Scale | None = None,
    interpolation_order: InterpolationOrder = 3,
    transformation_order: TransformationOrder = "trs",
    rotation_axis: RotationAxis3D = "z",
    options: Json = "{}",
    normalize: bool = True,
    resize: Resize = None,
    spacing: Spacing3D | None = None,
    max_pad: MaxPad = False,
    safe_pad: SafePad = False,
    bandpass: BandpassFilter = None,
    window: WindowFilter = None,
    zoom: Zoom = None,
    rescale: bool = False,
    noise: float | None = None,
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
    """Transform 3D image and register it with the original image."""
    import sys

    import numpy as np

    from ndimreg.image import Paths3DImageLoader, Scikit3DImageLoader
    from ndimreg.transform import Transformation3D
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

    images = []
    if image_paths:
        images.extend(Paths3DImageLoader(image_paths))
    if image_datasets:
        images.extend(Scikit3DImageLoader(image_datasets))
    if not images:
        print("No images to process")
        sys.exit(1)

    pre_processors = get_pre_processors(
        zoom=zoom, rescale=rescale, bandpass=bandpass, window=window, dim=3
    )

    # Registration options must be coerced into actual dictionary.
    # If empty JSON string ('{}') has been provided, it will not
    # automatically be a proper dict which would result in an error.
    registration_options = options if isinstance(options, dict) else {}
    registration = REGISTRATION_METHODS_3D[method](
        axis=rotation_axis,
        processors=pre_processors,
        debug=debug_show or debug_save,
        **registration_options,
    )

    transformation_params = {
        "transformation_order": transformation_order,
        "interpolation_order": interpolation_order,
    }

    for image in images:
        prepare_benchmark_image(
            image,
            normalize=normalize,
            spacing=spacing,
            resize=resize,
            max_pad=max_pad,
            safe_pad=safe_pad,
        )

        if translation is not None:
            translation_absolute = (
                tuple(np.array(translation) * np.roll(image.resolution, 1) / 100)
                if translation_relative
                else translation
            )
        else:
            translation_absolute = None

        # TODO: Add rotation based on single angle + axis.
        transformation_input = Transformation3D(
            translation=translation_absolute, rotation=rotation, scale=scale
        )

        __transform_nd(
            image,
            registration,
            tform_input=transformation_input,
            tform_params=transformation_params,
            noise=noise,
            device=device,
            debug_save=debug_save,
            debug_show=debug_show,
            debug_depth=debug_depth,
            debug_steps=debug_steps,
            result_save=result_save,
            result_show=result_show,
            output_dir=output_dir,
        )


def __transform_nd(  # noqa: PLR0913
    image: Image,
    registration: Registration,
    *,
    tform_params: dict[str, str | int],
    tform_input: Transformation,
    noise: float | None = None,
    debug_save: bool,
    debug_show: bool,
    debug_depth: int | None,
    debug_steps: list[RegistrationDebugStep] | None,
    result_save: bool,
    result_show: bool,
    device: Device,
    output_dir: Path,
) -> None:
    import sys

    from skimage.util import random_noise

    from ndimreg.fusion import MergeFusion
    from ndimreg.image import Image2D, Image3D
    from ndimreg.utils import format_time

    from ._common import save_debug_images, save_result_images, show_debug_images

    image_basename = image.name
    image.name = f"{image_basename}-original"

    image_transformed = image.copy(name=f"{image_basename}-transformed")
    image_transformed.transform(tform_input, **tform_params)

    if noise is not None:
        image_transformed.data = random_noise(image_transformed.data, var=noise)

    result = image.register(registration, image_transformed, device=device)[0]
    print(f"Expected: {tform_input}")
    print(f"Recovered: {result.transformation}")
    print(f"Duration: {format_time(result.total_duration)}")

    if not any((result_save, result_show, debug_save, debug_show)):
        # If we do not show or save any output, any further processing
        # is redundant.
        sys.exit(0)

    if debug_show:
        show_debug_images(result, step=debug_steps, depth=debug_depth)
    if debug_save:
        save_debug_images(
            result,
            output_dir / image.name / "debug",
            step=debug_steps,
            depth=debug_depth,
        )

    if not (result_show or result_save):
        # If we do not show or save any result images, any further
        # processing is redundant.
        sys.exit(0)

    image_inverted = image_transformed.copy(name=f"{image_basename}-inverted")
    image_inverted.transform(tform_input, inverse=True, **tform_params)

    image_recovered = image_transformed.copy(name=f"{image_basename}-recovered")
    image_recovered.transform(result.transformation, inverse=True, **tform_params)

    image_fused = image.fuse(
        MergeFusion(), image_recovered, name=f"{image_basename}-fused", device=device
    )
    image_fused.to_device("cpu")

    result_images = (
        image,
        image_transformed,
        image_inverted,
        image_recovered,
        image_fused,
    )

    if result_show:
        image.show_all(*result_images)
    if result_save:
        save_result_images(result_images, output_dir / image.name)

        if screenshots_3d := [
            Image2D(im.get_screenshot_2d(), name=f"{im.name}-2d")
            for im in result_images
            if isinstance(im, Image3D)
        ]:
            save_result_images(screenshots_3d, output_dir / image.name)
