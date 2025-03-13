"""2D and 3D image transformations."""

from __future__ import annotations

import functools
from operator import matmul
from typing import TYPE_CHECKING, Any

import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import get_namespace
from loguru import logger

from ndimreg.utils import array_to_shape, log_time
from ndimreg.utils.image_operations import affine_transform
from ndimreg.utils.misc import calculate_cval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ndimreg.transform import TransformationMode
    from ndimreg.utils.misc import CvalMode

    from .types import Dimension, InterpolationOrder, RotationAxis, TransformationOrder

# TODO: Allow device input to provide explicit GPU support.


@log_time(print_func=logger.info)
def transform(  # noqa: PLR0913
    data: NDArray,
    *,
    dim: Dimension,
    matrix: NDArray | None = None,
    translation: Sequence[float] | NDArray | None = None,
    rotation: float | Sequence[float] | NDArray | None = None,
    scale: float | Sequence[float] | NDArray | None = None,
    origin: Sequence[float] | NDArray | None = None,
    degrees: bool = True,
    transformation_order: TransformationOrder = "trs",
    interpolation_order: InterpolationOrder = 3,
    mode: TransformationMode = "crop",
    inverse: bool = False,
    output_shape: tuple[int, ...] | None = None,
    cval: float | CvalMode = 0.0,
    clip: bool = True,
) -> NDArray:
    # TODO: Validate/Sanitize input.
    # TODO: Define behavior on present 'output_shape' input.
    origin = (get_center(data.shape, dim=dim) if origin is None else origin)[::-1]
    if matrix is not None:
        transformation = matrix
    else:
        transformation = transform_matrix(
            dim=dim,
            translation=translation,
            rotation=rotation,
            scale=scale,
            origin=origin,
            degrees=degrees,
            transformation_order=transformation_order,
        )

    # FIX: Modes 'resize' and 'extend' do not work for non-isometric scaling!
    match mode:
        case "crop":
            out_shape_mode = data.shape[:dim]

        case "extend":
            input_shape = data.shape[:dim]
            transformed_shape = __transform_shape(input_shape, transformation, dim=dim)
            out_shape_mode = transformed_shape

            origin_tform = __build_translation_matrix(origin, dim=dim)
            transformed_origin_tform = __build_translation_matrix(
                get_center(tuple(np.roll(transformed_shape, -1)), dim=dim), dim=dim
            )

            transformation @= np.linalg.inv(origin_tform) @ transformed_origin_tform

        case "resize":
            input_shape = data.shape[:dim]
            out_shape_mode = input_shape

            origin_tform = __build_translation_matrix(origin, dim=dim)
            transformed_shape = __transform_shape(input_shape, transformation, dim=dim)
            downscale_factor = (np.array(input_shape) / transformed_shape).min()
            downscale_tform = __build_scale_matrix(downscale_factor, dim=dim)

            transformation @= (
                np.linalg.inv(origin_tform) @ downscale_tform @ (origin_tform)
            )

    logger.debug(f"Mode '{mode}' target shape: {out_shape_mode}'")

    kwargs = {
        "order": interpolation_order,
        "output_shape": output_shape or out_shape_mode,
        "cval": calculate_cval(data, cval),
    }

    # NOTE: 2D transformations might need to be inverted.
    xp = get_namespace(data)
    tform = xp.asarray(np.linalg.inv(transformation) if inverse else transformation)
    transformed_data = (
        xp.stack(
            [affine_transform(data[..., i], tform, **kwargs) for i in range(sh[dim])],
            axis=dim,
        )
        if len(sh := data.shape) > dim
        else affine_transform(data, tform, **kwargs)
    )

    return xp.clip(transformed_data, 0.0, 1.0) if clip else transformed_data


def rotate_axis(
    data: NDArray,
    angle: float,
    *,
    dim: Dimension,
    axis: RotationAxis,
    degrees: bool = True,
    **kwargs: Any,
) -> NDArray:
    """Rotate image around an axis.

    This rotates an image around a single axis, either X, Y, or Z.

    Parameters
    ----------
    data
        3D input image to be rotated.
    angle
        Rotation angle. Defaults to degrees if not specified otherwise.
    axis
        Rotation axis, either one of (1, 2, 3) or (x, y, z).
        Will be ignored for 2D images as only XY-axis exists.
    degrees
        Interpret angles as degrees, radians otherwise.
    **kwargs
        Passthrough parameters to `transform`.
    """
    rotation_matrix = axis_rotation_matrix(angle, dim=dim, axis=axis, degrees=degrees)

    return transform(data, dim=dim, rotation=rotation_matrix, **kwargs)


def resize(data: NDArray, factor: float, *, dim: Dimension, **kwargs: Any) -> NDArray:
    """Scale whole image including data.

    This method changes the output shape.

    Parameters
    ----------
    data
        2D or 3D image data.
    factor
        Zoom factor. Values > 1.0 zoom in, values < 1.0 zoom out.
    dim
        Image dimension, 2 or 3.
    **kwargs
        Passthrough parameters to `transform`.
    """
    # TODO: Allow multiple factors for different dimension scalings.
    # TODO: Calculate missing translation/offset here.
    # FIX: Resizing does not properly update the output shape.
    output_shape = array_to_shape(np.array(data.shape[:dim]) * factor)

    return transform(data, scale=factor, dim=dim, output_shape=output_shape, **kwargs)


def transform_matrix(  # noqa: PLR0913
    *,
    dim: Dimension,
    translation: Sequence[float] | NDArray | None = None,
    rotation: float | Sequence[float] | NDArray | None = None,
    scale: float | Sequence[float] | NDArray | None = None,
    origin: Sequence[float] | NDArray | None = None,
    degrees: bool = True,
    transformation_order: TransformationOrder = "trs",
) -> NDArray:
    """Transform an image with interpolation in a single step.

    Parameters
    ----------
    dim
        Image dimension, either 2 or 3.
    translation
        Translation in x, y[, z] direction.
    rotation
        Rotation in degrees.
    scale
        Scaling factor.
    origin
        Origin of rotation and scaling as (x,y[,z]) coordinates.
        If None, the center of the image is used.
    degrees
        If True, `rotation` is interpreted as degrees (default).
    transformation_order
        This defines the order of transformations applied to the image.

        Possible orders are:
        - 'trs': Translate, rotate, scale. (<-- Default)
        - 'tsr': Translate, scale, rotate.
        - 'rts': Rotate, translate, scale.
        - 'rst': Rotate, scale, translate.
        - 'str': Scale, translate, rotate.
        - 'srt': Scale, rotate, translate.
    """
    # FIX: When input os on GPU (e.g., cp.ndarray), internal checks fail.
    # TODO: With 2D usage of affine_transform, all operations must be verified again.
    origin_op = __build_translation_matrix(origin, dim=dim)
    ops = {
        "t": __build_translation_matrix(translation, dim=dim),
        "r": __build_rotation_matrix(rotation, dim=dim, degrees=degrees),
        "s": __build_scale_matrix(scale, dim=dim),
    }
    logger.debug(f"Generated transformation operations: {ops}")
    logger.debug(f"Generated center operation: {origin_op}")

    enabled_ops = (ops[op] for op in transformation_order if op in ops)

    return functools.reduce(matmul, (*enabled_ops, origin_op), np.linalg.inv(origin_op))


def axis_rotation_matrix(
    angle: float, *, dim: Dimension, axis: RotationAxis, degrees: bool = True
) -> NDArray:
    """Create a rotation matrix to rotate around a single axis.

    Parameters
    ----------
    angle
        Rotation angle. Defaults to degrees if not specified otherwise.
    dim
        Dimension of rotation matrix, either 2 or 3.
    axis
        Rotation axis, either one of (1, 2, 3) or (x, y, z).
        This parameter vill be ignored for 2D images as only XY-axis exists.
    degrees
        Interpret angles as degrees, radians otherwise.
    """
    theta = np.deg2rad(angle) if degrees else angle
    if dim == 2:
        return np.array(((ct := np.cos(theta), -np.sin(theta)), (np.sin(theta), ct)))

    match axis:
        case "x" | 1:
            basis = 2
        case "y" | 2:
            basis = 1
        case "z" | 0:
            basis = 0

    return pr.active_matrix_from_angle(basis, -theta)


@functools.lru_cache
def get_center(shape: tuple[int, ...], *, dim: int) -> tuple[float, ...]:
    """Return the center of the image as origin."""
    # TODO: Check whether this is correct (e.g., for 1x1 images).
    return tuple(np.array(shape[:dim]) / 2 - 0.5)


def __transform_shape(
    image_shape: tuple[int, ...], transformation_matrix: NDArray, *, dim: Dimension
) -> tuple[int, ...]:
    match dim:
        case 2:
            x, y = image_shape[:2]
            corners = np.array([[0, 0, 1], [x, 0, 1], [0, y, 1], [x, y, 1]])
        case 3:
            x, y, z = image_shape[:3]
            corners = np.array(
                [
                    [0, 0, 0, 1],
                    [x, 0, 0, 1],
                    [0, y, 0, 1],
                    [x, y, 0, 1],
                    [0, 0, z, 1],
                    [x, 0, z, 1],
                    [0, y, z, 1],
                    [x, y, z, 1],
                ]
            )

    transformed_corners = transformation_matrix @ corners.T
    transformed_points = transformed_corners.T[:, :-1]

    min_coords = np.min(transformed_points, axis=0)
    max_coords = np.max(transformed_points, axis=0)

    return array_to_shape(max_coords - min_coords)


def __build_translation_matrix(
    translation: Sequence[float] | NDArray | None, *, dim: Dimension
) -> NDArray:
    matrix = np.eye(dim + 1)
    if translation is None:
        return matrix

    matrix[:dim, dim] = -np.roll(translation, 1)
    return matrix


def __build_rotation_matrix(
    rotation: float | Sequence[float] | NDArray | None,
    *,
    degrees: bool = True,
    dim: Dimension,
) -> NDArray:
    # TODO: Verify that rotation direction is correct for 2D (matrix vs. angle).
    if rotation is None:
        matrix = np.eye(dim)

    elif isinstance(rotation, np.ndarray) and rotation.shape == (dim,) * 2:
        matrix = rotation

    elif isinstance(rotation, np.ndarray) and rotation.ndim != 1:
        msg = "Unsupported rotation (2D: angle/matrix, 3D: matrix/XYZ-Euler/quaternion)"
        raise ValueError(msg)

    elif isinstance(rotation, float | int):
        rotation = rotation if dim == 3 else -rotation

        # We must norm the angle, otherwise 'affine_transform' would
        # crop some data on the image edges due to incorrect rotation.
        theta = pr.norm_angle(np.deg2rad(rotation) if degrees else rotation)
        matrix = np.array(((ct := np.cos(theta), -np.sin(theta)), (np.sin(theta), ct)))

    elif len(rotation) == 3:
        # We also norm the Euler angles just to be sure, this might not
        # be absolutely necessary though.
        euler = pr.norm_euler(np.deg2rad(rotation) if degrees else rotation, 0, 1, 2)
        matrix = pr.matrix_from_euler(euler, 0, 1, 2, extrinsic=False)

    elif len(rotation) == 4:
        matrix = pr.matrix_from_quaternion(rotation)

    else:
        msg = "Unsupported rotation (2D: angle/matrix, 3D: matrix/XYZ-Euler/quaternion)"
        raise ValueError(msg)

    tform = np.eye(dim + 1)
    tform[:dim, :dim] = matrix
    return tform


def __build_scale_matrix(
    scale: float | Sequence[float] | NDArray | None = None, *, dim: Dimension
) -> NDArray:
    # FIX: Handle division by zero or use minimum fallback value!
    # TODO: Verify that scale is correct for 2D scaling.
    if scale is None:
        return np.diag((*(1,) * dim, 1))

    if isinstance(scale, float | int):
        return np.diag((*(1 / scale,) * dim, 1))

    if isinstance(scale, np.ndarray) and scale.shape == (dim + 1,) * 2:
        return scale

    if len(scale) == dim:
        return np.diag((*(1 / np.array(scale)), 1))

    msg = "Unsupported scale (matrix or single value)"
    raise ValueError(msg)
