"""TODO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pytransform3d.rotations import norm_angle, norm_euler, quaternion_from_euler
from scipy.spatial.distance import euclidean

from ndimreg.utils.diffs import (
    angle_diff,
    euler_angles_diff,
    quaternion_dist,
    scale_diff_abs,
    scale_diff_rel,
    translation_diff,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from ndimreg.transform import RotationAxis


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkTransformation:
    """TODO."""

    translation_abs: tuple[float, ...] | None
    translation_rel: tuple[float, ...] | None
    rotation_angle: float | None
    rotation_axis: RotationAxis | None
    rotation_euler: tuple[float, float, float] | None
    rotation_quaternion: tuple[float, float, float, float] | None = None
    scale: float | None

    def __post_init__(self) -> None:
        """TODO."""
        if self.rotation_euler is not None:
            normed_euler = norm_euler(self.rotation_euler, 0, 1, 2)
            quaternion = quaternion_from_euler(normed_euler, 0, 1, 2, extrinsic=False)
            object.__setattr__(self, "rotation_euler", tuple(normed_euler))
            object.__setattr__(self, "rotation_quaternion", tuple(quaternion))

        if self.rotation_angle is not None:
            object.__setattr__(self, "rotation_angle", norm_angle(self.rotation_angle))


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkTransformationDiff:
    """TODO."""

    # TODO: Add 'TRE' metric for transformation difference.

    translation_abs: tuple[float, ...] | None = None
    translation_abs_total: float | None = None
    translation_abs_euclidean: float | None = None
    translation_rel: tuple[float, ...] | None = None
    translation_rel_total: float | None = None
    translation_rel_euclidean: float | None = None
    rotation_angle: float | None = None
    rotation_euler: tuple[float, float, float] | None = None
    rotation_euler_total: float | None = None
    rotation_quaternion_dist: float | None = None
    scale_abs: float | None = None
    scale_rel: float | None = None

    @classmethod
    def build(
        cls, tform1: BenchmarkTransformation, tform2: BenchmarkTransformation
    ) -> Self:
        """TODO."""
        translation_abs, translation_abs_total, translation_abs_euclidean = (None,) * 3
        if tform1.translation_abs is not None and tform2.translation_abs is not None:
            translations = tform1.translation_abs, tform2.translation_abs
            translation_abs = tuple(translation_diff(*translations))
            translation_abs_total = sum(translation_abs)
            translation_abs_euclidean = euclidean(*translations)

        translation_rel, translation_rel_total, translation_rel_euclidean = (None,) * 3
        if tform1.translation_rel is not None and tform2.translation_rel is not None:
            translations = tform1.translation_rel, tform2.translation_rel
            translation_rel = tuple(translation_diff(*translations))
            translation_rel_total = sum(translation_rel)
            translation_rel_euclidean = euclidean(*translations)

        rotation_angle = None
        if tform1.rotation_angle is not None and tform2.rotation_angle is not None:
            rotation_angle = angle_diff(
                tform1.rotation_angle, tform2.rotation_angle, degrees=False
            )

        rotation_euler, rotation_euler_total = None, None
        if tform2.rotation_euler is not None and tform1.rotation_euler is not None:
            rotation_euler = tuple(
                euler_angles_diff(
                    tform2.rotation_euler, tform1.rotation_euler, degrees=False
                )
            )
            rotation_euler_total = sum(rotation_euler)

        rotation_quaternion_dist = None
        if (
            tform2.rotation_quaternion is not None
            and tform1.rotation_quaternion is not None
        ):
            rotation_quaternion_dist = quaternion_dist(
                tform2.rotation_quaternion, tform1.rotation_quaternion
            )

        scale_abs, scale_rel = None, None
        if tform2.scale is not None and tform1.scale is not None:
            scale_abs = scale_diff_abs(tform2.scale, tform1.scale)
            scale_rel = scale_diff_rel(tform2.scale, tform1.scale)

        return cls(
            translation_abs=translation_abs,
            translation_abs_total=translation_abs_total,
            translation_abs_euclidean=translation_abs_euclidean,
            translation_rel=translation_rel,
            translation_rel_total=translation_rel_total,
            translation_rel_euclidean=translation_rel_euclidean,
            rotation_angle=rotation_angle,
            rotation_euler=rotation_euler,
            rotation_euler_total=rotation_euler_total,
            rotation_quaternion_dist=rotation_quaternion_dist,
            scale_abs=scale_abs,
            scale_rel=scale_rel,
        )
