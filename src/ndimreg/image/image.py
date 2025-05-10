"""Base class for image representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import skimage
import skimage.io as skio
from array_api_compat import array_namespace
from loguru import logger

from ndimreg.transform import get_center, resize, rotate_axis, transform
from ndimreg.utils import array_to_shape, to_device_array, to_device_arrays
from ndimreg.utils.image_operations import (
    img_as_float,
    img_as_ubyte,
    rescale_intensity,
    resize_local_mean,
    rgb2gray,
)
from ndimreg.utils.misc import calculate_cval

from .utils import chainable, log_shape

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray
    from typing_extensions import Self

    from ndimreg.fusion import Fusion
    from ndimreg.registration import Registration, RegistrationResult
    from ndimreg.transform import RotationAxis, Transformation
    from ndimreg.utils.misc import CvalMode

Device = Literal["cpu", "gpu"]
Dimension = Literal[2, 3]
Spacing = Sequence[float]

# TODO: Add function 'crop' (similar to 'pad').


class Image(ABC):
    """Base class for images."""

    def __init__(
        self,
        data: NDArray,
        name: str | None = None,
        *,
        copy: bool = True,
        device: Device | None = None,
        **_kwargs: Any,
    ) -> None:
        """Initialize image.

        Parameters
        ----------
        data
            Image data.
        name
            Image name.
        copy
            If True, copy image data to prevent side effects on original
            data.
        """
        self.name: str = name or f"image{self.dim}d"
        self.data: NDArray = img_as_float(
            to_device_array(data, device=device, copy=copy)
        )

        logger.debug(f"Image '{self.name}' with shape {self.data.shape} created")

    def __str__(self) -> str:
        return f"{self.name} ({'x'.join(str(i) for i in self.resolution)})"

    @classmethod
    def from_path(
        cls, path: Path | str, *, name: str | None = None, device: Device = "cpu"
    ) -> Self:
        """Load image from path.

        Parameters
        ----------
        path
            Path to image.
        name
            Name of image.
        """
        # TODO: Check whether device (GPU) is available before loading data.
        # TODO: Handle case when image is not found.
        if not isinstance(path, Path):
            path = Path(path)

        logger.debug(f"Loading image from '{path}'")
        data = skio.imread(path)

        return cls(data, name or path.stem, copy=False, device=device)

    @classmethod
    def from_skimage(
        cls, dataset_name: str, *, name: str | None = None, device: Device = "cpu"
    ) -> Self:
        """Load scikit image.

        Parameters
        ----------
        dataset_name
            Name of dataset from `skimage.data`.
        name
            Name of image.
        """
        # TODO: Handle case when dataset does not exist.
        # TODO: Check whether device (GPU) is available before loading data.
        logger.debug(f"Loading scikit image '{dataset_name}'")
        data = getattr(skimage.data, dataset_name)()

        return cls(data, name or dataset_name, copy=False, device=device)

    @classmethod
    @abstractmethod
    def show_all(cls, image: Self, *additional_images: Self, **kwargs: Any) -> None:
        """Show multiple images with subclass specific viewer."""

    @classmethod
    def save_all(
        cls,
        image: Self,
        *additional_images: Self,
        extension: str,
        directory: Path | str,
    ) -> None:
        """Save multiple images."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for im in (image, *additional_images):
            im.save(directory / f"{im.name}.{extension}")

    def __eq__(self, other: object) -> bool:
        """Check if images are equal.

        Images are considered equal if their data is equal, regardless
        of their names.
        """
        if not isinstance(other, Image):
            return NotImplemented

        if type(self) is not type(other):
            # Subclasses such as Image2D and Image3D cannot be equal.
            return False

        # TODO: Ensure that data is on the same device/namespace.
        return all((self.name == other.name, np.array_equal(self.data, other.data)))

    @property
    def multichannel(self) -> bool:
        """Return True if image has multiple channels."""
        return len(self.shape) > self.dim

    @property
    def channel_axis(self) -> int | None:
        """Return channel axis if channels are present, else None."""
        return self.dim if self.multichannel else None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return image shape."""
        return self.data.shape

    @property
    def resolution(self) -> tuple[int, ...]:
        """Return image shape excluding channel axis."""
        return self.data.shape[: self.dim]

    @property
    def device(self) -> Device:
        """Return image shape excluding channel axis."""
        # TODO: Return actual device information (type, name, id, ...).
        return "cpu" if self.__namespace.device(self.data) == "cpu" else "gpu"

    @property
    @abstractmethod
    def dim(self) -> Dimension:
        """Return dimension of underlying data.

        This returns the number of dimensions of the underlying data
        array. This is not necessarily the same as the number of
        dimensions of the image, which is one more if the image is
        multichannel.
        """

    @property
    def center(self) -> tuple[float, ...]:
        """Return center of image."""
        return get_center(self.resolution, dim=self.dim)

    @chainable
    def to_device(self, device: Device) -> None:
        """Transfer image data to device."""
        self.data = to_device_array(self.data, device=device)

    @chainable
    @log_shape(print_func=logger.debug)
    def transform(
        self, transformation: Transformation | None = None, **kwargs: Any
    ) -> None:
        """Transform an image with interpolation in a single step.

        Parameters
        ----------
        transformation
            Wrapper for similarity transformations (translation,
            rotation, and scale).
        kwargs
            Passthrough parameters to `transform` method.
        """
        if t := transformation:
            # Setting the default values this way ensures that an
            # explicit transformation value within 'kwargs' overrides
            # the bundled parameter from 'transformation'.
            kwargs.setdefault("translation", t.translation)
            kwargs.setdefault("rotation", t.rotation)
            kwargs.setdefault("scale", t.scale)

        self.data = transform(self.data, dim=self.dim, **kwargs)

    @chainable
    @log_shape(print_func=logger.debug)
    def resize(self, factor: float, **kwargs: Any) -> None:
        """Resize image dimensions.

        Parameters
        ----------
        factor
            Scaling factor for all axes.
        kwargs
            Passthrough parameters to `resize` method.
        """
        # TODO: Allow resizing factors for specific axes.
        # TODO: Check whether this should be removed (see Image.transform()).
        self.data = resize(self.data, factor, dim=self.dim, **kwargs)

    @chainable
    @log_shape(print_func=logger.debug)
    def apply_spacing(
        self, spacing: Spacing, *, max_size: int | None = None, **kwargs: Any
    ) -> None:
        """TODO."""
        if len(spacing) != self.dim:
            msg = f"Spacing length {len(spacing)} must match image dimension {self.dim}"
            raise ValueError(msg)

        # TODO: Does numpy namespace work for images on GPU?
        target_shape = np.array(self.resolution) * spacing
        if max_size is not None and (max_side := target_shape.max()) > max_size:
            target_shape /= max_side / max_size

        return self.resize_to_shape(array_to_shape(target_shape), **kwargs)

    @chainable
    @log_shape(print_func=logger.debug)
    def rotate_axis(self, angle: float, *, axis: RotationAxis, **kwargs: Any) -> None:
        """Rotate image around an axis.

        This rotates a 3D image around a single axis, either X, Y, or Z.
        For 2D images, this is a just the default rotation.

        Parameters
        ----------
        angle
            Rotation angle. Defaults to degrees if not specified
            otherwise.
        axis
            Rotation axis.
        kwargs
            Passthrough parameters to `rotate_axis` method.
        """
        self.data = rotate_axis(self.data, angle, axis=axis, dim=self.dim, **kwargs)

    @abstractmethod
    def show(self, **kwargs: Any) -> None:
        """Show an image with a viewer."""

    @chainable
    def normalize(self, low: float = 0.0, high: float = 1.0) -> None:
        """Normalize an image to specified range."""
        # TODO: Make output range configurable.
        logger.info(f"Normalizing image '{self.name}'")

        self.data = rescale_intensity(self.data, out_range=(low, high))

    @chainable
    def clip(self) -> None:
        """Clip image content to expected range of [0.0, 1.0]."""
        logger.info(f"Clipping image '{self.name}'")
        self.data = self.__namespace.clip(self.data, 0.0, 1.0)

    @chainable
    def cut(
        self,
        low: float | None = None,
        high: float | None = None,
        cval: float | CvalMode = 0.0,
    ) -> None:
        """Cut image data below/above threshold."""
        logger.info(f"Cutting image data below/above thresholds for '{self.name}'")

        low = low or 0
        high = high or 1

        self.data[(self.data < low) | (self.data > high)] = self.__calculate_cval(cval)

    @chainable
    @log_shape(print_func=logger.debug)
    def grayscale(self) -> None:
        """Convert an RGB image to grayscale."""
        logger.info(f"Converting image '{self.name}' to grayscale")

        if self.multichannel:
            self.data = rgb2gray(self.data)

    @chainable
    @log_shape(print_func=logger.debug)
    def resize_to_shape(self, shape: tuple[int, ...] | int, **kwargs: Any) -> None:
        """Resize image dimensions to target shape.

        Parameters
        ----------
        shape
            Target shape. The target shape must either match the image
            resolution or can be a single value that will automatically
            be converted the the image resolution with equal sides.
        **kwargs
            Additional arguments passed to `skimage.transform.resize_local_mean`.
        """
        target_shape = (shape,) * self.dim if isinstance(shape, int) else shape

        # TODO: Compare with 'scipy.zoom'.
        # TODO: Replace with 'transformations.resize' if possible.
        logger.info(f"Scaling image '{self.name}' to shape {target_shape}")
        self.data = resize_local_mean(self.data, target_shape, **kwargs)

    @chainable
    @log_shape(print_func=logger.debug)
    def pad_safe_rotation(
        self, *, cval: float | CvalMode = 0.0, keep_shape: bool = False
    ) -> None:
        """Pad an image to a size that does not crop any rotation.

        This adds ``fill_value`` around the image content so that the
        maximum rotation of 45° (both for 2D and 3D) does not crop any
        of the actual content.


        Parameters
        ----------
        fill_value
            Value to put as padding content.
        keep_shape
            Resize to original shape before padding, defaults to False.
        """
        ceil = max(self.resolution) * np.hypot(1, 1) - self.resolution
        paddings = [((x := int(np.ceil(y / 2))), x) for y in ceil]
        if self.multichannel:
            paddings.append((0, 0))

        original_shape = self.resolution
        _cval = self.__calculate_cval(cval)

        self.data = self.__namespace.pad(self.data, paddings, constant_values=_cval)

        if keep_shape:
            self.resize_to_shape(original_shape)

    @chainable
    @log_shape(print_func=logger.debug)
    def pad_to_size(self, target_size: int, *, cval: float | CvalMode = 0.0) -> None:
        """Pad all axes of an image a specific size."""
        logger.info(
            f"Padding image '{self.name}' to target size of {(target_size,) * self.dim}"
        )

        paddings = []
        for size in self.resolution:
            diff = target_size - size
            left = diff // 2
            right = left if 2 * left == diff else left + 1
            paddings.append((left, right))

        if self.multichannel:
            paddings.append((0, 0))

        _cval = self.__calculate_cval(cval)
        self.data = self.__namespace.pad(self.data, paddings, constant_values=_cval)

    @chainable
    @log_shape(print_func=logger.debug)
    def pad_equal_sides(self, *, cval: float | CvalMode = 0.0) -> None:
        """Pad an image to the size of its maximums axis."""
        self.pad_to_size(max(self.resolution), cval=cval)

    def register(
        self,
        registration: Registration,
        image: Self,
        *additional_images: Self,
        device: Device | None = None,
        **kwargs: Any,
    ) -> list[RegistrationResult]:
        """TODO."""
        images = [image, *additional_images]
        msg_imagenames = ", ".join(f"'{im.name}'" for im in images)
        logger.info(f"Registering image '{self.name}' with images: {msg_imagenames}")

        # Prevent multiple data transfers by transfering fixed image once.
        fixed = to_device_array(self.data, device=device)

        results = []
        for im in images:
            # NOTE: Replace with single call once multi-image registrations are allowed.
            moving = to_device_array(im.data, device=device)
            results.append(registration.register(fixed, moving, **kwargs))

        return results

    def fuse(
        self,
        fusion: Fusion,
        image: Self,
        *additional_images: Self,
        name: str | None = None,
        device: Device | None = None,
        **kwargs: Any,
    ) -> Self:
        """Fuse with other image(s).

        Parameters
        ----------
        image
            Image(s) to be fused with the image.
        fusion
            Fusion algorithm to fuse input images into single image.
        alpha
            Alpha value for merging/blending.
        name
            Name of the new image.
        """
        other_images = [image, *additional_images]
        image_names = ", ".join(f"'{im.name}'" for im in other_images)
        logger.info(f"Fusing image '{self.name}' with images: {image_names}")

        all_images = to_device_arrays(
            *(im.data for im in (self, *other_images)), device=device
        )
        fused_data = fusion.fuse(*all_images, **kwargs)

        return self.__class__(fused_data, name=name or f"{self.name}-fused")

    def copy(self, *, name: str | None = None, device: Device | None = None) -> Self:
        """Create a copy of the image.

        Returns
        -------
        Image
            Copy of the image.
        """
        new_name = name if name is not None else f"{self.name}-copy"
        data = to_device_array(self.data, device=device)

        return self.__class__(data.copy(), new_name)

    def save(self, path: Path | str) -> None:
        """Store image on disk.

        Parameters
        ----------
        path
            Path to store image at.
        """
        if isinstance(path, str):
            path = Path(path)

        logger.debug(f"Saving image '{self.name}' at '{path}'")
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            skio.imsave(path, img_as_ubyte(self.data))
        except ValueError as e:
            logger.error(f"Failed to save image '{self.name}' at '{path}': {e}")

    def __calculate_cval(self, cval: float | CvalMode) -> float:
        return calculate_cval(self.data, cval)

    @property
    def __namespace(self) -> ModuleType:
        return array_namespace(self.data)
