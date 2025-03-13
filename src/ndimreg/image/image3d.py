"""3D image representation."""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, Any, Final, Literal

import napari
from loguru import logger
from typing_extensions import override

from ndimreg.transform.types import AXIS_MAPPING
from ndimreg.utils import to_numpy_array

from .image import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    from ndimreg.transform.types import RotationAxis3D

    from .image import Dimension

# TODO: Add all supported napari colormaps.
COLOR_MAPS: Final = ("green", "blue", "yellow", "red", "gray")


class Image3D(Image):
    """3D image class.

    A 3D image is a representation for a 3D array of pixels. It can be
    used to represent a single 3D image or a stack of 2D images.

    The coordinates of the pixels are defined as follows:
    - The first axis is the z-axis (or planes).
    - The second axis is the x-axis (or rows).
    - The third axis is the y-axis (or columns).
    - The fourth axis (optional) is the channel axis (or colors).

    This follows the convention of scikit-image:
    https://scikit-image.org/docs/stable/user_guide/numpy_images.html#coordinate-conventions
    """

    @classmethod
    @override
    def show_all(
        cls,
        image: Self,
        *additional_images: Self,
        title: str | None = None,
        colors: bool = False,
        display_mode: Dimension = 3,
        **kwargs: Any,
    ) -> None:
        """Show multiple 3D images with napari.

        Parameters
        ----------
        images
            Images to show.
        title
            Title of the napari viewer window.
        display_mode
            Display mode of the viewer. Possible values are 2 and 3,
            representing 2D and 3D display mode, respectively.
        colors
            If True, each image is shown with a different colormap.
        **kwargs
            Additional arguments passed to `napari.Viewer.add_image`.
        """
        images = [image, *additional_images]
        title = title or f"napari: {', '.join(image.name for image in images)}"

        viewer = napari.Viewer(title=title, ndisplay=display_mode)
        viewer.grid.enabled = True

        with contextlib.suppress(KeyError):
            # If available, add the manual transforms plugin.
            viewer.window.add_plugin_dock_widget("napari-manual-transforms")

        colormaps = itertools.cycle(COLOR_MAPS) if colors else itertools.repeat("gray")

        for im, cmap in zip(images, colormaps, strict=False):
            viewer.add_image(
                to_numpy_array(im.data), name=im.name, colormap=cmap, **kwargs
            )

        viewer.show()
        napari.run()

    def get_slice_2d(self, axis: RotationAxis3D) -> NDArray:
        """Return the middle slice of the data as array."""
        slicer: list = [slice(None)] * 3
        axis_id = AXIS_MAPPING[axis][1]
        slicer[axis_id] = self.resolution[axis_id] // 2

        return self.data[slicer]

    def get_screenshot_2d(self) -> NDArray:
        """Return a screenshot of the 3D object using napari."""
        # TODO: Allow axis input, angle, etc.
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(self.data)
        screenshot = viewer.screenshot()
        viewer.close()

        return screenshot

    @override
    def show(self, **kwargs: Any) -> None:
        """Show image with napari.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to `napari.view_image`.
        """
        logger.debug(f"Showing image '{self.name}'")
        napari.view_image(to_numpy_array(self.data), **kwargs)
        napari.run()

    @property
    @override
    def dim(self) -> Literal[3]:
        return 3
