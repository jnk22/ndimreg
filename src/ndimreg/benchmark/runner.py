"""TODO."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytransform3d.rotations as pr
from cachetools import cached
from cachetools.keys import hashkey
from loguru import logger
from mpire.context import DEFAULT_START_METHOD
from mpire.pool import WorkerPool
from scipy import fft
from tqdm import tqdm

from ndimreg.transform import axis_rotation_matrix
from ndimreg.utils import Timer
from ndimreg.utils.arrays import to_device_arrays
from ndimreg.utils.fft import get_fft_backend
from ndimreg.utils.image import prepare_benchmark_image

from .result import (
    AppInfo,
    BenchmarkInfo,
    BenchmarkResult,
    BenchmarkResults,
    DeviceInfo,
    ImageInfo,
    RegistrationInfo,
    SystemInfo,
)
from .transformation import BenchmarkTransformation

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ndimreg.image import Image

    from .parameters import (
        BenchmarkImagePreprocessing,
        BenchmarkParameter,
        BenchmarkParameters,
    )


class BenchmarkRunner:
    """TODO."""

    def __init__(
        self, parameters: BenchmarkParameters, *, note: str | None = None
    ) -> None:
        """TODO."""
        self._parameters: BenchmarkParameters = parameters
        self._app_info: AppInfo = AppInfo.from_metadata()
        self._system_info: SystemInfo = SystemInfo.from_platform()
        self._benchmark_info: BenchmarkInfo = BenchmarkInfo(note=note)

    def run(
        self,
        *,
        on_gpu: bool,
        parallel: bool = True,
        workers: int | None = None,
        progress_bar: bool = False,
    ) -> BenchmarkResults:
        """TODO."""
        with Timer() as timer:
            if parallel and (workers is None or workers >= 1):
                start_method = "spawn" if on_gpu else DEFAULT_START_METHOD
                user_output = f"workers: {workers}, start method: {start_method}"
                print(f"Running benchmark in parallel mode ({user_output})")

                with WorkerPool(n_jobs=workers, start_method=start_method) as pool:
                    results = pool.map(
                        self._run, self._parameters, progress_bar=progress_bar
                    )

            else:
                print("Running benchmark in sequential mode")
                progress = tqdm if progress_bar else iter
                results = [self._run(p) for p in progress(self._parameters)]

        return BenchmarkResults(
            results, duration=timer.total_duration, parallel=parallel, workers=workers
        )

    # TODO: Wrapping inside dataclass parameter might be too slow. Cache it?
    # TODO: Absolute translation is also dependent on padding modes!
    def _run(self, param: BenchmarkParameter) -> BenchmarkResult:
        # Remove logger to keep progress bar clean.
        logger.remove()

        input_image = param.image
        image_fixed = _prepare_image(
            input_image.copy(name=input_image.name),
            resize=param.size,
            preprocessing=param.preprocessing,
        )

        image_size = image_fixed.resolution
        if param.translation_relative:
            translation_rel = param.translation
            translation_abs = _translation_rel2abs(translation_rel, image_size)
        else:
            translation_abs = param.translation
            translation_rel = _translation_abs2rel(translation_abs, image_size)

        # TODO: Implement translation/rotation as dataclass wrapper.
        rotation_axis = param.rotation[0]
        rotation_angle_in = param.rotation[1]
        rotation_euler_in = param.rotation[2]

        dim = input_image.dim
        if dim == 2 and rotation_angle_in is not None:
            rotation = rotation_angle_in
            axis_kwargs = {}

        elif dim == 3 and rotation_axis is not None and rotation_angle_in is not None:
            # This is the case where we only rotate around a single 3D
            # axis. We therefore convert the single axis rotation to
            # Euler for debug/benchmark outputs and set the 'axis'
            # parameter for 3D registration methods.
            rotation = axis_rotation_matrix(
                rotation_angle_in, axis=rotation_axis, dim=dim
            )
            rotation_euler_in = tuple(
                np.rad2deg(pr.euler_from_matrix(rotation, 0, 1, 2, extrinsic=False))
            )
            axis_kwargs = {"axis": rotation_axis}

        elif rotation_euler_in is not None:
            rotation = rotation_euler_in
            axis_kwargs = {}

        else:
            msg = "Rotation must be either single axis rotation (2D/3D) or Euler angles"
            raise ValueError(msg)

        image_moving = image_fixed.copy().transform(
            translation=translation_abs, rotation=rotation, scale=param.scale
        )

        device_info = DeviceInfo.from_device(param.device)
        image_info = ImageInfo(
            name=image_fixed.name,
            resolution=image_fixed.resolution,
            resolution_original=input_image.resolution,
            multichannel=image_fixed.multichannel,
        )
        registration_options_str = json.dumps(param.registration_options)
        fft_backend_name = param.fft_backend if param.device == "cpu" else "cupy"

        tform_input = BenchmarkTransformation(
            translation_abs=translation_abs,
            translation_rel=translation_rel,
            rotation_angle=np.deg2rad(rotation_angle_in)
            if rotation_angle_in is not None
            else None,
            rotation_axis=rotation_axis,
            rotation_euler=tuple(np.deg2rad(rotation_euler_in))
            if rotation_euler_in is not None
            else None,
            scale=param.scale,
        )

        try:
            registration = param.registration_method[0](
                **(param.registration_options | axis_kwargs)
            )

        except Exception as e:  # noqa: BLE001 # Benchmark runner shall never break.
            registration_info = RegistrationInfo(
                dimension=dim,
                id=param.registration_method[1],
                options=registration_options_str,
                fft_backend=fft_backend_name,
            )
            return BenchmarkResult(
                app=self._app_info,
                benchmark=self._benchmark_info,
                system=self._system_info,
                image=image_info,
                preprocessing=param.preprocessing,
                device=device_info,
                registration=registration_info,
                fail_reason=str(e),
                tform_input=tform_input,
            )

        registration_info = RegistrationInfo.from_registration(
            registration,
            id=param.registration_method[1],
            options=registration_options_str,
            fft_backend=fft_backend_name,
        )

        with warnings.catch_warnings():
            # Registrations might throw warnings (e.g., for equal
            # images), but this is not the place to check for that.
            warnings.simplefilter("ignore")

            # TODO: Log timing for registration + data transfer.
            fixed_data, moving_data = to_device_arrays(
                image_fixed.data, image_moving.data, device=param.device
            )

            try:
                # TODO: Improve this.
                if param.device == "cpu":
                    with fft.set_backend(get_fft_backend(param.fft_backend)):
                        result = registration.register(fixed_data, moving_data)

                else:
                    result = registration.register(fixed_data, moving_data)

            except Exception as e:  # noqa: BLE001 # Benchmark runner shall never break.
                return BenchmarkResult(
                    app=self._app_info,
                    benchmark=self._benchmark_info,
                    system=self._system_info,
                    image=image_info,
                    preprocessing=param.preprocessing,
                    device=device_info,
                    registration=registration_info,
                    fail_reason=str(e),
                    tform_input=tform_input,
                )

        result_rotation = result.transformation.rotation
        if result_rotation is not None and dim == 2:
            rotation_angle_out = np.deg2rad(result_rotation)
            rotation_euler_out = None

        elif result_rotation is not None and dim == 3:
            rotation_euler_out = np.deg2rad(result_rotation)

            if rotation_axis is not None:
                # This is the case for 3D registrations that only recover
                # rotation around a single axis.
                # TODO: Move this to result class instead.
                axis_angle = pr.axis_angle_from_matrix(
                    pr.matrix_from_euler(rotation_euler_out, 0, 1, 2, extrinsic=False)
                )
                axis, angle = axis_angle[:3], axis_angle[3]
                # TODO: Description.
                rotation_angle_out = -angle if sum(axis) > 0 else angle
            else:
                rotation_angle_out = None
        else:
            rotation_angle_out = None
            rotation_euler_out = None

        result_translation_abs = result.transformation.translation
        result_translation_rel = (
            _translation_abs2rel(result_translation_abs, image_size)
            if result_translation_abs
            else None
        )

        tform_output = BenchmarkTransformation(
            translation_abs=result_translation_abs,
            translation_rel=result_translation_rel,
            rotation_angle=rotation_angle_out,
            rotation_axis=rotation_axis,
            rotation_euler=rotation_euler_out,
            scale=result.transformation.scale,
        )

        return BenchmarkResult(
            app=self._app_info,
            benchmark=self._benchmark_info,
            system=self._system_info,
            image=image_info,
            preprocessing=param.preprocessing,
            device=device_info,
            registration=registration_info,
            duration_total=result.total_duration,
            duration=result.get_durations(depth=0)[0][1],
            tform_input=tform_input,
            tform_output=tform_output,
        )


def _prepare_image_key_maker(image: Image, **kwargs: Any) -> tuple[Hashable, ...]:
    return hashkey(image.name, image.shape, **kwargs)


@cached(cache={}, key=_prepare_image_key_maker)
def _prepare_image(
    image: Image, *, preprocessing: BenchmarkImagePreprocessing, resize: int | None
) -> Image:
    return prepare_benchmark_image(
        image=image,
        normalize=preprocessing.normalize,
        max_pad=preprocessing.max_pad,
        safe_pad=preprocessing.safe_pad,
        resize=resize,
    )


def _translation_abs2rel(
    translation: tuple[float, ...], image_size: tuple[int, ...]
) -> tuple[float, ...]:
    return tuple(np.array(translation) * 100 / image_size)


def _translation_rel2abs(
    translation: tuple[float, ...], image_size: tuple[int, ...]
) -> tuple[float, ...]:
    return tuple(np.array(image_size) * translation / 100)
