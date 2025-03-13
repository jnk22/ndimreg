"""TODO."""

from __future__ import annotations

from typing import Final, Literal

TransformationOrder = Literal["trs", "tsr", "rts", "rst", "str", "srt"]
InterpolationOrder = Literal[0, 1, 2, 3, 4, 5]
TransformationMode = Literal["crop", "resize", "extend"]
Dimension = Literal[2, 3]

RotationAxis3DName = Literal["x", "y", "z"]
RotationAxis3DIndex = Literal[0, 1, 2]
RotationAxis3D = RotationAxis3DName | RotationAxis3DIndex
RotationAxis2DName = Literal["x"]
RotationAxis2DIndex = Literal[0]
RotationAxis2D = RotationAxis2DName | RotationAxis2DIndex
RotationAxis = RotationAxis2D | RotationAxis3D

# TODO: Replace with something better if possible.
AXIS_MAPPING: Final[
    dict[RotationAxis3D | int, tuple[RotationAxis3DName, RotationAxis3DIndex]]
] = {0: ("z", 0), "z": ("z", 0), 1: ("x", 1), "x": ("x", 1), 2: ("y", 2), "y": ("y", 2)}
AXIS_FLIP_MAPPING: Final = {"z": "x", "y": "y", "x": "z"}
