"""TODO."""

from __future__ import annotations

import platform
import sys
from dataclasses import KW_ONLY, asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import cpuinfo
import polars as pl
import tzlocal

try:
    from cupy.cuda import runtime as cuda_runtime
except ImportError:
    cuda_runtime = None

from ndimreg.utils.dataframes import explode_nested_column

from .transformation import BenchmarkTransformationDiff

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from ndimreg.benchmark.parameters import BenchmarkImagePreprocessing
    from ndimreg.image.image import Device
    from ndimreg.registration import Registration
    from ndimreg.registration.result import RegistrationDuration
    from ndimreg.utils.fft import FftBackend

    from .transformation import BenchmarkTransformation


XYZ_FIELDS: Final = ("x", "y", "z")
QUAT_FIELDS: Final = ("w", "x", "y", "z")

NESTED_FIELDS: Final = (
    ("image.resolution", XYZ_FIELDS),
    ("image.resolution_original", XYZ_FIELDS),
    ("tform_input.translation_abs", XYZ_FIELDS),
    ("tform_input.translation_rel", XYZ_FIELDS),
    ("tform_input.rotation_euler", XYZ_FIELDS),
    ("tform_input.rotation_quaternion", QUAT_FIELDS),
    ("tform_output.translation_abs", XYZ_FIELDS),
    ("tform_output.translation_rel", XYZ_FIELDS),
    ("tform_output.rotation_euler", XYZ_FIELDS),
    ("tform_output.rotation_quaternion", QUAT_FIELDS),
    ("tform_diff.translation_abs", XYZ_FIELDS),
    ("tform_diff.translation_rel", XYZ_FIELDS),
    ("tform_diff.rotation_euler", XYZ_FIELDS),
)
NESTED_COLUMN_DROPS = ("duration", "tform_diff", "tform_output")


@dataclass(frozen=True, slots=True)
class BenchmarkResults:
    """TODO."""

    results: list[BenchmarkResult]

    _: KW_ONLY
    duration: float
    parallel: bool
    workers: int | None
    command: str = field(default=" ".join(sys.argv))

    def __len__(self) -> int:
        """TODO."""
        return len(self.results)

    def __iter__(self) -> Iterator[BenchmarkResult]:
        """TODO."""
        yield from self.results

    def __getitem__(self, idx: int) -> BenchmarkResult:
        """TODO."""
        return self.results[idx]

    def write_meta(self, file: Path | str) -> None:
        """TODO."""
        if self.results:
            app = (res := self.results[0]).app
            benchmark = res.benchmark
            system = res.system
        else:
            app, benchmark, system = (None,) * 3

        meta_data = {
            "results": len(self.results),
            "duration": self.duration,
            "app": app,
            "benchmark": benchmark,
            "system": system,
            "command": self.command,
        }

        pl.DataFrame(meta_data).write_json(file)

    def write_json(self, file: Path | str, precision: int | None = None) -> None:
        """TODO."""
        results_df = pl.DataFrame(self.__data, infer_schema_length=None)

        if precision:
            results_df = results_df.with_columns(
                pl.col(column).round(precision)
                for column in results_df.columns
                if results_df.schema[column] == pl.Float64
            )

        Path(file).parent.mkdir(exist_ok=True, parents=True)
        results_df.write_json(file)

    def write_csv(
        self, file: Path | str, precision: int | None = None, **kwargs: Any
    ) -> None:
        """TODO."""
        results_df = pl.json_normalize(self.__data, infer_schema_length=None)

        # FIX: If a nested field is 'None', replace its exploded fields with NaN values.
        column_aliases = (
            expr
            for column, fields in NESTED_FIELDS
            for expr in explode_nested_column(results_df, column, fields)
        )
        column_drops = (name for name, _ in NESTED_FIELDS)

        results_df = results_df.with_columns(column_aliases)
        results_df = results_df.drop(column_drops, strict=False)
        results_df = results_df.select(*sorted(results_df.columns))

        Path(file).parent.mkdir(exist_ok=True, parents=True)
        results_df.write_csv(file, float_precision=precision, **kwargs)

    @property
    def __data(self) -> list[dict[str, Any]]:
        return [asdict(result) for result in self.results]


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkResult:
    """TODO."""

    # Static information that is not changing for different registrations.
    app: AppInfo
    benchmark: BenchmarkInfo
    system: SystemInfo

    # Registration dependent data.
    image: ImageInfo
    preprocessing: BenchmarkImagePreprocessing
    registration: RegistrationInfo
    device: DeviceInfo
    tform_input: BenchmarkTransformation
    tform_output: BenchmarkTransformation | None = None
    tform_diff: BenchmarkTransformationDiff | None = None
    duration: RegistrationDuration | None = None
    duration_total: float | None = None
    fail_reason: str | None = None

    def __post_init__(self) -> None:
        """TODO."""
        if (tf_output := self.tform_output) is not None:
            tform_diff = BenchmarkTransformationDiff.build(self.tform_input, tf_output)
            object.__setattr__(self, "tform_diff", tform_diff)


@dataclass(frozen=True, slots=True, kw_only=True)
class AppInfo:
    """TODO."""

    version: str

    @classmethod
    def from_metadata(cls) -> Self:
        """TODO."""
        version = "unknown" if __package__ is None else metadata.version("ndimreg")
        return cls(version=version)


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkInfo:
    """TODO."""

    note: str | None = None
    datetime: datetime = field(default=datetime.now(tzlocal.get_localzone()))


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInfo:
    """TODO."""

    name: str
    resolution: tuple[int, ...]
    resolution_original: tuple[int, ...]
    multichannel: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceInfo:
    """TODO."""

    type: Device
    name: str

    @classmethod
    def from_device(cls, device: Device) -> Self:
        """TODO."""
        if device == "cpu":
            name = cpuinfo.get_cpu_info()["brand_raw"]

        elif device == "gpu" and cuda_runtime:
            cuda_device = cuda_runtime.getDevice()
            name = cuda_runtime.getDeviceProperties(cuda_device)["name"].decode("utf-8")

        else:
            name = "unknown"

        return cls(type=device, name=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemInfo:
    """TODO."""

    # TODO: Add GPU/CPU specs.

    system: str
    node_name: str
    release: str
    version: str
    machine: str
    processor: str
    python_version: str
    python_implementation: str

    @classmethod
    def from_platform(cls) -> Self:
        """TODO."""
        return cls(
            system=platform.system(),
            node_name=platform.node(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            python_implementation=platform.python_implementation(),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RegistrationInfo:
    """TODO."""

    id: str
    name: str | None = None
    dimension: Literal[2, 3]
    options: str
    fft_backend: FftBackend

    @classmethod
    def from_registration(
        cls, registration: Registration, fft_backend: FftBackend, id: str, options: str
    ) -> Self:
        """TODO."""
        return cls(
            id=id,
            name=registration.name,
            dimension=registration.dim,
            options=options,
            fft_backend=fft_backend,
        )
