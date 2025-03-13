"""CLI main module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from cyclopts import App
from cyclopts.types import (  # noqa: TC002
    Directory,
    File,
    Json,
    PositiveInt,
    ResolvedExistingFile,
)

from ._types import (  # noqa: TC001
    BenchmarkParallelBool,
    CpuFftBackend,
    Device,
    GenerationStrategy,
    MaxPad,
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
    Translation2D,
    Translation3D,
    TranslationsRelativeBool,
)

if TYPE_CHECKING:
    from ndimreg.image import Image

app = App(name="benchmark")

# TODO: Allow setting interpolation order 0-5 for transformed test images.
# TODO: Allow setting for all rotation angles/differences to be radians or degrees.
# TODO: Add optional warmup run(s) for sequential/performance mode.
# TODO: Allow using random data during performance measurement runs (RGB + grayscale).
# TODO: Add 'noop' basic input for empty translation, rotation, and scale.
# TODO: Save meta/config before running benchmark.
# TODO: Add aliases for rotations, translations, image sizes etc. for singular input.
# TODO: Add parameters: noise, filters/pre-processors, ...
# TODO: Fix order/conversion/... when using safe and max paddings.
# TODO: Add append mode for output files.
# TODO: Implement to/from configuration file (auto-save current config).
# TODO: Implement checkpoints for writing results to disk.
# TODO: Sanitize/validate input (e.g., registration options, datasets, ...).
# TODO: Implement lazy-loading for images.
# TODO: Run benchmarks per image (i.e., only one image at the time).
# TODO: Output of all angles as radians and/or degrees, or only one consistently.
# TODO: Allow setting various FFT backends for performance benchmarking.
# TODO: Allow selecting multiple device IDs (e.g., for performance benchmarks).
# TODO: Add run duration (in addition to actual registration duration).
# TODO: Add seed input for generating same rotations again.
# TODO: Allow random translation per rotation (i.e., non-matrix runs).
# FIX: If something fails, transformation output and diff is not present (error).

DEFAULT_BENCHMARK_SAFE_PADS: Final = [True]
DEFAULT_BENCHMARK_MAX_PADS: Final = [True]
DEFAULT_BENCHMARK_NORMALIZE_INPUT: Final = [True]
DEFAULT_BENCHMARK_FFT_BACKENDS: Final[list[CpuFftBackend]] = ["scipy"]
DEFAULT_BENCHMARK_DEVICES: Final[list[Device]] = ["cpu"]
DEFAULT_BENCHMARK_ROTATION_AXIS_AXES: Final[list[RotationAxis3D]] = ["x", "y", "z"]


@app.command(name="2d")
def benchmark_2d(  # noqa: PLR0913
    image_paths: list[ResolvedExistingFile] | None = None,
    *,
    image_datasets: list[str] | None = None,
    image_sizes: list[Resize],
    registration_methods: list[RegistrationMethod2D],
    registration_options: Json = "{}",
    translations: list[Translation2D] | None = None,
    translations_strategy: GenerationStrategy = "uniform",
    translations_strategy_low: Translation2D = (0.0, 0.0),
    translations_strategy_high: Translation2D = (10.0, 10.0),
    translations_strategy_amount: PositiveInt | None = None,
    translations_relative: TranslationsRelativeBool = True,
    rotations: list[RotationAngle] | None = None,
    rotations_strategy: GenerationStrategy = "uniform",
    rotations_strategy_low: float = 0.0,
    rotations_strategy_high: float = 360.0,
    rotations_strategy_amount: PositiveInt | None = None,
    scales: list[Scale] | None = None,
    scales_strategy: GenerationStrategy = "uniform",
    scales_strategy_low: float = 0.5,
    scales_strategy_high: float = 2.0,
    scales_strategy_amount: PositiveInt | None = None,
    random_seed: int | None = None,
    safe_pads: list[SafePad] = DEFAULT_BENCHMARK_SAFE_PADS,
    max_pads: list[MaxPad] = DEFAULT_BENCHMARK_MAX_PADS,
    normalize_inputs: list[bool] = DEFAULT_BENCHMARK_NORMALIZE_INPUT,
    spacings: list[Spacing2D] | None = None,
    spacings_limit_mode: Literal["max", "normalize"] | None = "max",
    devices: list[Device] = DEFAULT_BENCHMARK_DEVICES,
    device_id: int = 0,
    fft_backends: list[CpuFftBackend] = DEFAULT_BENCHMARK_FFT_BACKENDS,
    parallel: BenchmarkParallelBool = True,
    parallel_workers: PositiveInt | None = None,
    benchmark_note: str | None = None,
    progress_bar: bool = True,
    output_dir: Directory = Path("results"),
    output_file_json: File | None = None,
    output_file_csv: File | None = None,
    output_file_meta: File | None = None,
    output_file_config: File | None = None,
    output_precision: int | None = None,
) -> None:
    """Run registration benchmarks for 2D images.

    Parameters
    ----------
    registration_options:
        Additional options for registration methods as JSON formatted
        string. Parameter names must match the registration method's
        keyword arguments. All values must be provided as a list of
        values. Note that duplicate registration methods with
        unsupported **will not** be filtered!
    translations_relative:
        Interpret input translations as relative or absolute. Defaults
        to relative.
    parallel:
        Run benchmarks in parallel. Defaults to true. **Caution:** When
        benchmarking for registration duration, parallel mode should be
        disabled.
    parallel_workers:
        Amount of workers for parallel mode. Uses maximum available if
        none provided.
    """
    import sys

    import numpy as np

    from ndimreg.benchmark import BenchmarkParameters, BenchmarkRunner
    from ndimreg.image import Paths2DImageLoader, Scikit2DImageLoader
    from ndimreg.utils.transformation_generators import (
        generate_random_rotations_2d,
        generate_random_scales,
        generate_random_translations_2d,
        generate_uniform_rotations_2d,
        generate_uniform_scales,
        generate_uniform_translations_2d,
    )

    from ._common import setup_gpu, setup_logging

    setup_logging()
    if "gpu" in devices:
        setup_gpu(device_id=device_id)

    images = []
    if image_paths:
        images.extend(Paths2DImageLoader(image_paths))
    if image_datasets:
        images.extend(Scikit2DImageLoader(image_datasets))

    if not images:
        print("No input images provided")
        sys.exit(1)

    if len({(im.name, im.shape) for im in images}) < len(images):
        # This is due to the caching method in the runner that caches
        # pre-processing operations on the same image using the image's
        # name and shape as keys.
        # TODO: Improve this behavior to not require this check if possible.
        print("All images must have distinct names")
        sys.exit(1)

    if spacings is not None:
        print("Applying spacing to images...")
        _apply_spacing(
            images, spacings, limit_mode=spacings_limit_mode, sizes=image_sizes
        )

    rng = np.random.default_rng(random_seed)

    translations = translations or []
    if translations_strategy_amount:
        match translations_strategy:
            case "random":
                generator_func = generate_random_translations_2d
            case "uniform":
                generator_func = generate_uniform_translations_2d

        translations.extend(
            generator_func(
                translations_strategy_amount,
                low=translations_strategy_low,
                high=translations_strategy_high,
                rng=rng,
            )
        )

    rotations = rotations or []
    if rotations_strategy_amount:
        match rotations_strategy:
            case "random":
                generator_func = generate_random_rotations_2d
            case "uniform":
                generator_func = generate_uniform_rotations_2d

        rotations.extend(
            generator_func(
                rotations_strategy_amount,
                low=rotations_strategy_low,
                high=rotations_strategy_high,
                rng=rng,
            )
        )

    scales = scales or []
    if scales_strategy_amount:
        match scales_strategy:
            case "random":
                generator_func = generate_random_scales
            case "uniform":
                generator_func = generate_uniform_scales

        scales.extend(
            generator_func(
                scales_strategy_amount,
                low=scales_strategy_low,
                high=scales_strategy_high,
                rng=rng,
            )
        )

    # Registration options must be coerced into actual dictionary.
    # If empty JSON string ('{}') has been provided, it will not
    # automatically be a proper dict which would result in an error.
    registration_options = (
        registration_options if isinstance(registration_options, dict) else {}
    )

    print("Building benchmark parameters...")
    parameters = BenchmarkParameters(
        images=images,
        sizes=image_sizes,
        registration_methods=registration_methods,
        registration_options=registration_options,
        translations=translations,
        translations_relative=translations_relative,
        rotations=[(None, rotation, None) for rotation in rotations],
        scales=scales,
        safe_pads=safe_pads,
        max_pads=max_pads,
        normalize_inputs=normalize_inputs,
        devices=devices,
        fft_backends=fft_backends,
    )

    if not len(parameters):
        print("Benchmark test matrix based on test parameters is empty")
        sys.exit(1)

    print(f"Benchmark parameters consist of {len(parameters)} combinations.")

    print("Creating benchmark runner...")
    benchmark = BenchmarkRunner(parameters, note=benchmark_note)

    print("Starting benchmark...")
    on_gpu = "gpu" in devices
    results = benchmark.run(
        on_gpu=on_gpu,
        parallel=parallel,
        workers=parallel_workers,
        progress_bar=progress_bar,
    )

    timestamp = results[0].benchmark.datetime.strftime("%Y%m%d-%H%M%S")
    output_file_config = (
        output_file_config or output_dir / f"result_{timestamp}-config.json"
    )
    output_file_meta = output_file_meta or output_dir / f"result_{timestamp}-meta.json"
    output_file_json = output_file_json or output_dir / f"result_{timestamp}.json"
    output_file_csv = output_file_csv or output_dir / f"result_{timestamp}.csv"

    print(f"Saving benchmark configuration to '{output_file_config}'...")
    parameters.write_config(output_file_config)

    print(f"Saving benchmark metadata to '{output_file_meta}'...")
    results.write_meta(output_file_meta)

    print(f"Saving JSON results to '{output_file_json}'...")
    results.write_json(output_file_json, precision=output_precision)

    print(f"Saving CSV results to '{output_file_csv}'...")
    results.write_csv(output_file_csv, precision=output_precision)


@app.command(name="3d")
def benchmark_3d(  # noqa: PLR0913
    image_paths: list[ResolvedExistingFile] | None = None,
    *,
    image_datasets: list[str] | None = None,
    image_sizes: list[Resize],
    registration_methods: list[RegistrationMethod3D],
    registration_options: Json = "{}",
    translations: list[Translation3D] | None = None,
    translations_strategy: GenerationStrategy = "uniform",
    translations_strategy_low: Translation3D = (0.0, 0.0, 0.0),
    translations_strategy_high: Translation3D = (10.0, 10.0, 10.0),
    translations_strategy_amount: PositiveInt | None = None,
    translations_relative: TranslationsRelativeBool = True,
    rotations: list[RotationEulerXYZ] | None = None,
    rotations_strategy: GenerationStrategy = "uniform",
    rotations_strategy_low: RotationEulerXYZ = (-180.0, -90.0, -180.0),
    rotations_strategy_high: RotationEulerXYZ = (180.0, 90.0, 180.0),
    rotations_strategy_amount: PositiveInt | None = None,
    rotations_axis: list[RotationAngle] | None = None,
    rotations_axis_axes: list[RotationAxis3D] = DEFAULT_BENCHMARK_ROTATION_AXIS_AXES,
    rotations_axis_strategy: GenerationStrategy = "uniform",
    rotations_axis_strategy_low: RotationAngle = 0.0,
    rotations_axis_strategy_high: RotationAngle = 360.0,
    rotations_axis_strategy_amount: PositiveInt | None = None,
    scales: list[Scale] | None = None,
    scales_strategy: GenerationStrategy = "uniform",
    scales_strategy_low: float = 0.5,
    scales_strategy_high: float = 2.0,
    scales_strategy_amount: PositiveInt | None = None,
    random_seed: int | None = None,
    safe_pads: list[SafePad] = DEFAULT_BENCHMARK_SAFE_PADS,
    max_pads: list[MaxPad] = DEFAULT_BENCHMARK_MAX_PADS,
    normalize_inputs: list[bool] = DEFAULT_BENCHMARK_NORMALIZE_INPUT,
    spacings: list[Spacing3D] | None = None,
    spacings_limit_mode: Literal["max", "normalize"] | None = "max",
    devices: list[Device] = DEFAULT_BENCHMARK_DEVICES,
    device_id: int = 0,
    fft_backends: list[CpuFftBackend] = DEFAULT_BENCHMARK_FFT_BACKENDS,
    parallel: BenchmarkParallelBool = True,
    parallel_workers: PositiveInt | None = None,
    benchmark_note: str | None = None,
    progress_bar: bool = True,
    output_dir: Directory = Path("results"),
    output_file_json: File | None = None,
    output_file_csv: File | None = None,
    output_file_meta: File | None = None,
    output_file_config: File | None = None,
    output_precision: int | None = None,
) -> None:
    """Run registration benchmarks for 3D images.

    Parameters
    ----------
    registration_options:
        Additional options for registration methods as JSON formatted
        string. Parameter names must match the registration method's
        keyword arguments. All values must be provided as a list of
        values. Note that duplicate registration methods with
        unsupported **will not** be filtered!
    translations_relative:
        Interpret input translations as relative or absolute. Defaults
        to relative.
    parallel:
        Run benchmarks in parallel. Defaults to true. **Caution:** When
        benchmarking for registration duration, parallel mode should be
        disabled.
    parallel_workers:
        Amount of workers for parallel mode. Uses maximum available if
        none provided.
    """
    import sys

    import numpy as np

    from ndimreg.benchmark import BenchmarkParameters, BenchmarkRunner
    from ndimreg.image import Paths3DImageLoader, Scikit3DImageLoader
    from ndimreg.utils.transformation_generators import (
        generate_random_rotations_2d,
        generate_random_rotations_3d,
        generate_random_scales,
        generate_random_translations_3d,
        generate_uniform_rotations_2d,
        generate_uniform_rotations_3d,
        generate_uniform_scales,
        generate_uniform_translations_3d,
    )

    from ._common import setup_gpu, setup_logging

    setup_logging()

    if "gpu" in devices:
        setup_gpu(device_id=device_id)

    images = []
    if image_paths:
        images.extend(Paths3DImageLoader(image_paths))
    if image_datasets:
        images.extend(Scikit3DImageLoader(image_datasets))

    if not images:
        print("No input images provided")
        sys.exit(1)

    if len({(im.name, im.shape) for im in images}) < len(images):
        # This is due to the caching method in the runner that caches
        # pre-processing operations on the same image using the image's
        # name and shape as keys.
        # TODO: Improve this behavior to not require this check if possible.
        print("All images must have distinct names")
        sys.exit(1)

    if spacings is not None:
        print("Applying spacing to images...")
        _apply_spacing(
            images, spacings, limit_mode=spacings_limit_mode, sizes=image_sizes
        )

    rng = np.random.default_rng(random_seed)

    translations = translations or []
    if translations_strategy_amount:
        match translations_strategy:
            case "random":
                generator_func = generate_random_translations_3d
            case "uniform":
                generator_func = generate_uniform_translations_3d

        translations.extend(
            generator_func(
                translations_strategy_amount,
                low=translations_strategy_low,
                high=translations_strategy_high,
                rng=rng,
            )
        )

    rotations = rotations or []
    if rotations_strategy_amount:
        match rotations_strategy:
            case "random":
                generator_func = generate_random_rotations_3d
            case "uniform":
                generator_func = generate_uniform_rotations_3d

        rotations.extend(
            generator_func(
                rotations_strategy_amount,
                low=rotations_strategy_low,
                high=rotations_strategy_high,
                rng=rng,
            )
        )

    rotations_axis = rotations_axis or []
    if rotations_axis_strategy_amount:
        match rotations_axis_strategy:
            case "random":
                generator_func = generate_random_rotations_2d
            case "uniform":
                generator_func = generate_uniform_rotations_2d

        rotations_axis.extend(
            generator_func(
                rotations_axis_strategy_amount,
                low=rotations_axis_strategy_low,
                high=rotations_axis_strategy_high,
                rng=rng,
            )
        )

    scales = scales or []
    if scales_strategy_amount:
        match scales_strategy:
            case "random":
                generator_func = generate_random_scales
            case "uniform":
                generator_func = generate_uniform_scales

        scales.extend(
            generator_func(
                scales_strategy_amount,
                low=scales_strategy_low,
                high=scales_strategy_high,
            )
        )

    euler_rotations = ((None, None, rot) for rot in rotations)

    single_axis_rotations = []
    for axis in rotations_axis_axes:
        single_axis_rotations.extend((axis, rot, None) for rot in rotations_axis)

    # Registration options must be coerced into actual dictionary.
    # If empty JSON string ('{}') has been provided, it will not
    # automatically be a proper dict which would result in an error.
    registration_options = (
        registration_options if isinstance(registration_options, dict) else {}
    )

    print("Building benchmark parameters...")
    parameters = BenchmarkParameters(
        images=images,
        sizes=image_sizes,
        registration_methods=registration_methods,
        registration_options=registration_options or {},
        translations=translations,
        translations_relative=translations_relative,
        rotations=[*euler_rotations, *single_axis_rotations],
        scales=scales,
        safe_pads=safe_pads,
        max_pads=max_pads,
        normalize_inputs=normalize_inputs,
        devices=devices,
        fft_backends=fft_backends,
    )

    if not len(parameters):
        print("Benchmark test matrix based on test parameters is empty")
        sys.exit(1)

    print(f"Benchmark parameters consist of {len(parameters)} combinations.")

    print("Creating benchmark runner...")
    benchmark = BenchmarkRunner(parameters, note=benchmark_note)

    print("Starting benchmark...")
    on_gpu = "gpu" in devices
    results = benchmark.run(
        on_gpu=on_gpu,
        parallel=parallel,
        workers=parallel_workers,
        progress_bar=progress_bar,
    )

    timestamp = results[0].benchmark.datetime.strftime("%Y%m%d-%H%M%S")
    output_file_config = (
        output_file_config or output_dir / f"result_{timestamp}-config.json"
    )
    output_file_meta = output_file_meta or output_dir / f"result_{timestamp}-meta.json"
    output_file_json = output_file_json or output_dir / f"result_{timestamp}.json"
    output_file_csv = output_file_csv or output_dir / f"result_{timestamp}.csv"

    print(f"Saving benchmark configuration to '{output_file_config}'...")
    parameters.write_config(output_file_config)

    print(f"Saving benchmark metadata to '{output_file_meta}'...")
    results.write_meta(output_file_meta)

    print(f"Saving JSON results to '{output_file_json}'...")
    results.write_json(output_file_json, precision=output_precision)

    print(f"Saving CSV results to '{output_file_csv}'...")
    results.write_csv(output_file_csv, precision=output_precision)


def _apply_spacing(
    images: list[Image],
    spacings: list[Spacing3D] | list[Spacing2D],
    *,
    sizes: list[Resize],
    limit_mode: Literal["max", "normalize"] | None,
) -> None:
    import numpy as np

    normalize_spacing = limit_mode == "normalize"
    max_size = (
        (None if None in sizes else max(s for s in sizes if s is not None))
        if limit_mode == "max"
        else None
    )

    if len(spacings) == 1:
        sp = spacings[0]
        sp = np.array(sp) / np.max(sp) if normalize_spacing else sp

        for im in images:
            im.apply_spacing(sp, max_size=max_size)

    else:
        if len(spacings) != len(images):
            msg = "The amount of provided spacings must match the amount of images if multiple are provided"
            raise ValueError(msg)

        for im, sp in zip(images, spacings, strict=True):
            spx = np.array(sp) / np.max(sp) if normalize_spacing else sp
            im.apply_spacing(spx, max_size=max_size)
