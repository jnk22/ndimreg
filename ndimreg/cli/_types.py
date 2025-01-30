"""TODO."""

from __future__ import annotations

from typing import Annotated, Literal

from cyclopts import Parameter
from cyclopts.types import NonNegativeFloat, PositiveFloat, PositiveInt

# NOTE: This is a workaround. Instead of importing these from the actual
# modules, types are re-defined for faster CLI startup time.
TransformationOrder = Literal["trs", "tsr", "rts", "rst", "str", "srt"]
TransformationMode = Literal["crop", "resize", "extend"]
InterpolationOrder = Literal[0, 1, 2, 3, 4, 5]
RotationAxis3DName = Literal["x", "y", "z"]
RotationAxis3DIndex = Literal[0, 1, 2]
RotationAxis3D = RotationAxis3DName | RotationAxis3DIndex
RegistrationDebugStep = Literal["input", "preprocessing", "registration"]
Dimension = Literal[2, 3]
Device = Literal["cpu", "gpu"]

# Registration methods.
RegistrationMethod2D = Literal[
    "keller-adf-2d", "scikit-2d", "imregdft-2d", "translation-2d"
]
RegistrationMethod3D = Literal["keller-3d", "rotationaxis-3d", "translation-3d"]
RegistrationMethod = RegistrationMethod2D | RegistrationMethod3D

# Transformation.
Translation2D = tuple[float, float]
Translation3D = tuple[float, float, float]
RotationAngle = float
RotationEulerXYZ = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Scale = PositiveFloat

# Padding, Spacing + Size.
SafePad = bool
MaxPad = bool
Resize = PositiveInt | None
Spacing2D = tuple[PositiveFloat, PositiveFloat]
Spacing3D = tuple[PositiveFloat, PositiveFloat, PositiveFloat]
Spacing = Spacing2D | Spacing3D

# Preprocesser functions.
BandpassFilter = tuple[NonNegativeFloat, PositiveFloat] | None
WindowFilter = Literal["hann"] | None
Normalize = bool
Zoom = NonNegativeFloat | None

# Benchmarks.
GenerationStrategy = Literal["uniform", "random"]
TranslationsRelativeBool = Annotated[
    bool, Parameter(negative="--translations-absolute", show_default=False)
]
BenchmarkParallelBool = Annotated[
    bool, Parameter(negative="--sequential", show_default=False)
]

# matplotlib configuration.
ContextType = Literal["paper", "notebook", "talk", "poster"]

# FFT backend.
CpuFftBackend = Literal["scipy", "pyfftw", "mkl"]
