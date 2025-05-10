from __future__ import annotations

from ndimreg.fusion import MergeFusion
from ndimreg.image import Image2D
from ndimreg.registration import Keller2DRegistration
from ndimreg.transform import Transformation2D
from ndimreg.utils import format_time

# Load a test 2D image from 'scikit-image' and resize it to 256x256.
original = Image2D.from_skimage("astronaut").resize_to_shape(256)

# Copy the original image and transform it.
transformation = Transformation2D(translation=(12.8, 25.6), rotation=22)
transformed = original.copy().transform(transformation=transformation)

# Use the registration method with 'shift_upsample_factor=10' for a translation
# precision of 0.1 to register the original image with the transformed image.
registration = Keller2DRegistration(shift_upsample_factor=10)
result = original.register(registration, transformed)[0]

# We now transform the previously modified image with the recovered
# transformation output and fuse it with the original image.
recovered = transformed.copy().transform(
    transformation=result.transformation, inverse=True
)
fused = original.fuse(MergeFusion(), recovered)

print(f"Expected: {transformation}")
print(f"Recovered: {result.transformation}")
print(f"Duration: {format_time(result.total_duration)}")

# Save all images.
original.name = "astronaut-original"
transformed.name = "astronaut-transformed"
recovered.name = "astronaut-recovered"
fused.name = "astronaut-fused"

Image2D.save_all(
    original,
    transformed,
    recovered,
    fused,
    extension="png",
    directory="examples_output/keller2d",
)
