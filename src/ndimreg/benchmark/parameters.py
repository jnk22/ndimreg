"""TODO."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ndimreg.cli._common import REGISTRATION_METHODS

if TYPE_CHECKING:
    from collections.abc import Collection, Generator, Iterator, Sequence

    # TODO: Do not import from CLI types.
    from ndimreg.cli._types import (
        RegistrationMethod,
        RegistrationMethod2D,
        RegistrationMethod3D,
    )
    from ndimreg.image import Device, Image, ImageLoader
    from ndimreg.registration import Registration
    from ndimreg.transform import RotationAxis
    from ndimreg.utils.fft import CpuFftBackend


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkParameters:
    """Actual benchmark parameters such as images and their transformations.

    This class contains all attributes that will be used for generating
    a benchmarking matrix. Images, registrations, and transformations
    will be generated as cartesian product.
    """

    images: Sequence[Image] | ImageLoader
    sizes: Collection[int | None]
    devices: Collection[Device]
    fft_backends: Collection[CpuFftBackend]
    registration_methods: list[RegistrationMethod2D] | list[RegistrationMethod3D]
    registration_options: dict[str, Any]
    translations: Collection[tuple[float, ...]]
    translations_relative: bool
    rotations: Collection[
        tuple[None, float, None]
        | tuple[RotationAxis, float, None]
        | tuple[None, None, tuple[float, float, float]]
    ]
    scales: Collection[float]
    safe_pads: Collection[bool]
    max_pads: Collection[bool]
    normalize_inputs: Collection[bool]

    def __len__(self) -> int:
        """TODO."""
        return len(list(self._params))

    def __iter__(self) -> Iterator:
        """TODO."""
        yield from self._params

    def write_config(self, file: Path | str) -> None:
        """TODO."""
        # TODO: Add missing images.
        # TODO: Do not use set here, but filter duplicates on initialization.
        # TODO: Compare with parameter generation from below.
        file = Path(file)
        config = {
            "sizes": list(set(self.sizes)),
            "devices": list(set(self.devices)),
            "fft_backends": list(set(self.fft_backends)),
            "translations": list(set(self.translations)),
            "translations_relative": self.translations_relative,
            "rotations": list(set(self.rotations)),
            "scales": list(set(self.scales)),
            "safe_pads": list(set(self.safe_pads)),
            "max_pads": list(set(self.max_pads)),
            "normalize_input": list(set(self.normalize_inputs)),
            "registration_methods": list(set(self.registration_methods)),
            "registration_options": list(self.registration_options.items()),
        }

        file.parent.mkdir(exist_ok=True, parents=True)
        file.write_text(json.dumps(config))

    @property
    def _params(self) -> Generator[BenchmarkParameter]:
        """TODO."""
        # TODO: Modify translations/rotations etc. here?
        # TODO: Use transformation wrappers here (translation, rotation, scale).
        # TODO: Compare speed with a) use set on each attribute + b) without wrapper.
        # TODO: Build different registrations based on options here already.
        # TODO: Filter duplicate registration instances.
        # TODO: Use caching where appropriate.
        # TODO: Filter duplicates (rotations, translations etc.) by normalization.
        # TODO: Only test multiple FFT backends per CPU, ignore for GPU.

        registration_methods: list[tuple[type[Registration], RegistrationMethod]] = [
            (REGISTRATION_METHODS[reg_id], reg_id)
            for reg_id in self.registration_methods
        ]

        # Sort each option arguments for better output in results file.
        keys = self.registration_options.keys()
        values = self.registration_options.values()
        registration_options = [
            dict(sorted(dict(zip(keys, opts, strict=True)).items(), key=lambda x: x[0]))
            for opts in set(itertools.product(*values))
        ]

        preprocessings = (
            BenchmarkImagePreprocessing(*options)
            for options in itertools.product(
                set(self.safe_pads), set(self.max_pads), set(self.normalize_inputs)
            )
        )

        yield from (
            BenchmarkParameter(*combination)
            for combination in itertools.product(
                *(
                    self.images,
                    set(self.sizes),
                    set(self.devices),
                    set(self.fft_backends),
                    registration_methods,
                    registration_options,
                    set(self.translations),
                    (self.translations_relative,),
                    set(self.rotations),
                    set(self.scales),
                    set(preprocessings),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class BenchmarkImagePreprocessing:
    """TODO."""

    max_pad: bool
    safe_pad: bool
    normalize: bool


@dataclass(frozen=True, slots=True)
class BenchmarkParameter:
    """TODO."""

    image: Image
    size: int | None
    device: Device | Any
    fft_backend: CpuFftBackend
    registration_method: tuple[type[Registration], RegistrationMethod]
    registration_options: dict[str, Any]
    translation: tuple[float, ...]
    translation_relative: bool
    rotation: (
        tuple[None, float, None]
        | tuple[RotationAxis, float, None]
        | tuple[None, None, tuple[float, float, float]]
    )
    scale: float
    preprocessing: BenchmarkImagePreprocessing
