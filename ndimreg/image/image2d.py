"""2D image representation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger
from typing_extensions import override

from ndimreg.utils import to_numpy_array

from .image import Image

if TYPE_CHECKING:
    from typing_extensions import Self


class Image2D(Image):
    """2D image class.

    An image is a representation for a 2D array of pixels. It can be
    used to represent a single 2D image.

    The coordinates of the pixels are defined as follows:
    - The first axis is the x-axis (or rows).
    - The second axis is the y-axis (or columns).
    - The third axis (optional) is the channel axis (or colors).

        Y-
        ^ * (0,0)
        |
        |
        |
     X- |----------> X+
        Y+

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
        layout: Literal["row", "col", "grid"] = "grid",
        **kwargs: Any,
    ) -> None:
        """Show multiple images with matplotlib.

        Parameters
        ----------
        images
            Images to show.
        title
            Title of the figure. If None, no title is shown.
        layout
            How to arrange the images. Possible values are:
            - 'row': Arrange images in a row.
            - 'col': Arrange images in a column.
            - 'grid': Arrange images in a grid.
        **kwargs
            Additional arguments passed to `matplotlib.pyplot.imshow`.
        """
        # Backend might be disabled during debugging, therefore we have
        # to re-enable it.
        mpl.use("qtagg")
        images = [image, *additional_images]

        match layout:
            case "row":
                plot_layout = (1, len(images))
            case "col":
                plot_layout = (len(images), 1)
            case "grid" | _:
                ncols = math.ceil(math.sqrt(num_images := len(images)))
                nrows = math.ceil(num_images / ncols)
                plot_layout = ncols, nrows

        # Set squeeze to False that we can use 'flat' on it afterwards.
        fig, axes = plt.subplots(*plot_layout, figsize=(8, 8), squeeze=False)

        for ax, im in zip(axes.flat, images, strict=False):
            ax.set_title(im.name)
            ax.imshow(to_numpy_array(im.data), **kwargs)

        # Hide any remaining axes if there are fewer images than axes.
        for ax in axes.ravel()[len(images) :]:
            ax.axis("off")

        if title is not None:
            fig.suptitle(title)

        plt.show()

    @override
    def show(self, title: str | None = None, **kwargs: Any) -> None:
        """Show image with matplotlib.

        Parameters
        ----------
        title
            Title of the figure. If None, the name of the image is used.
        **kwargs
            Additional arguments passed to `matplotlib.pyplot.imshow`.
        """
        logger.debug(f"Showing image '{self.name}'")

        # Backend might be disabled during debugging, therefore we have
        # to re-enable it.
        mpl.use("qtagg")
        plt.imshow(to_numpy_array(self.data), **kwargs)
        plt.title(self.name if title is None else title)
        plt.show()

    @property
    @override
    def dim(self) -> Literal[2]:
        return 2
