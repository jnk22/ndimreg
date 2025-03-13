"""CLI debug module."""

from __future__ import annotations

from cyclopts import App
from cyclopts.types import PositiveInt, ResolvedExistingFile  # noqa: TC002

from ._types import RegistrationMethod2D  # noqa: TC001

app = App(name="debug", show=False)


@app.command(name="2d")
def debug_2d(
    image_path: ResolvedExistingFile | None,
    *,
    image_dataset: str | None = None,
    method: RegistrationMethod2D = "keller-adf-2d",
    resize: PositiveInt | None = None,
    debug: bool = False,
) -> None:
    """Apply transformation to 2D image and register it with the original image.

    Sources:
    - <https://matplotlib.org/3.9.2/gallery/widgets/slider_demo.html>
    - <https://www.tutorialspoint.com/how-to-update-matplotlib-s-imshow-window-interactively>
    """
    # TODO: Implement napari alternative for 3D visualization.
    # TODO: Allow multiple input images.
    # TODO: Implement checkboxes to enable/disable debug output images.
    # TODO: Implement caching for already rotated images.
    # TODO: Allow option for setting max cache values.
    # TODO: Allow 3D input images for matplotlib using `get_2d_image()`.
    # TODO: Show tables: input/result/errors/time.
    # TODO: Convert to napari plugin.
    # TODO: Allow 'safe padding' for non-destructive rotation/scale/...
    # TODO: Add reset on individual sliders.
    # TODO: Add option to select which debug image types to show.
    # TODO: Add mode to compare: registration methods, images, options, ...

    import sys
    from functools import lru_cache
    from typing import TYPE_CHECKING, Any

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Button, Slider

    from ndimreg.image import Image2D
    from ndimreg.transform import Transformation2D

    from ._common import REGISTRATION_METHODS_2D, setup_logging

    if TYPE_CHECKING:
        from ndimreg.registration import RegistrationDebugImage

    setup_logging()

    if image_path:
        image = Image2D.from_path(image_path)
    elif image_dataset:
        image = Image2D.from_skimage(image_dataset)
    else:
        print("No images to process")
        sys.exit(1)

    if resize is not None:
        image.resize_to_shape(resize)

    image_transformed = image.copy(name=f"{image.name}-transformed")
    registration = REGISTRATION_METHODS_2D[method](debug=debug)
    result = registration.register(image.data, image_transformed.data)
    image_recovered = image_transformed.copy(name=f"{image.name}-recovered")
    image_recovered.transform(transformation=result.transformation, inverse=True)

    # We have at least one original, transformed, and recovered image.
    debug_images = result.get_debug_images(dim=2, step="registration")
    cols = max(3, len(debug_images))

    fig, axs = plt.subplots(nrows=2, ncols=cols)
    axs[0][0].set_xlabel("original")
    axs[0][1].set_xlabel("transformed")
    axs[0][2].set_xlabel("recovered")
    axs[0][0].imshow(image.data)
    axs[0][1].imshow(image_transformed.data)
    axs[0][2].imshow(image_recovered.data)

    for ax in axs[0][3:]:
        ax.axis("off")
    for ax in axs[1]:
        ax.axis("off")

    for ax, debug_image in zip(axs[1], debug_images, strict=False):
        ax.imshow(debug_image.data)
        ax.set_xlabel(debug_image.name)
        ax.axis("on")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Adjust the main plot to make room for the sliders.
    fig.subplots_adjust(bottom=0.35)

    ax_rotation = fig.add_axes((0.25, 0.25, 0.65, 0.03))
    ax_scale = fig.add_axes((0.25, 0.2, 0.65, 0.03))
    ax_shift_x = fig.add_axes((0.25, 0.15, 0.65, 0.03))
    ax_shift_y = fig.add_axes((0.25, 0.1, 0.65, 0.03))

    slider_rotation = Slider(
        ax=ax_rotation, label="Angle [°]", valmin=0, valmax=360, valinit=0, valstep=1
    )
    slider_scale = Slider(
        ax=ax_scale, label="Scale", valmin=0.01, valmax=3, valinit=1, valstep=0.01
    )
    slider_shift_x = Slider(
        ax=ax_shift_x,
        label="X-Shift [%]",
        valmin=-100,
        valmax=100,
        valinit=0,
        valstep=1,
    )
    slider_shift_y = Slider(
        ax=ax_shift_y,
        label="Y-Shift [%]",
        valmin=-100,
        valmax=100,
        valinit=0,
        valstep=1,
    )

    @lru_cache
    def generate_images(
        transformation: Transformation2D,
    ) -> tuple[Image2D, Image2D, list[RegistrationDebugImage]]:
        image_transformed = image.copy().transform(transformation=transformation)
        result = registration.register(image.data, image_transformed.data)

        image_recovered = image_transformed.copy(name=f"{image.name}-recovered")
        image_recovered.transform(transformation=result.transformation, inverse=True)

        return (
            image_transformed,
            image_recovered,
            result.get_debug_images(dim=2, step="registration"),
        )

    def get_slider_values() -> Transformation2D:
        translation_values = np.array((slider_shift_x.val, slider_shift_y.val))

        return Transformation2D(
            translation=tuple((translation_values) * image.resolution / 100),
            rotation=slider_rotation.val,
            scale=slider_scale.val,
        )

    def update(_: Any) -> None:
        tform = get_slider_values()
        image_transformed, image_recovered, debug_images = generate_images(tform)

        axs[0][1].imshow(image_transformed.data)
        axs[0][2].imshow(image_recovered.data)

        for ax, debug_image in zip(axs[1], debug_images, strict=False):
            ax.imshow(debug_image.data)
            ax.set_xlabel(debug_image.name)

        fig.canvas.draw_idle()

    def reset(*sliders: Slider) -> None:
        for slider in sliders:
            slider.reset()

    def reset_all(_: Any) -> None:
        reset(*all_sliders)

    all_sliders = (slider_rotation, slider_shift_x, slider_shift_y, slider_scale)
    for slider in all_sliders:
        slider.on_changed(update)

    resetax = fig.add_axes((0.8, 0.025, 0.1, 0.04))
    button = Button(resetax, "Reset all", hovercolor="0.975")
    button.on_clicked(reset_all)

    # FIX: Generating debug images closes the current canvas.
    plt.show()
