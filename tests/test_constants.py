"""TODO."""

from __future__ import annotations

import os

# Shift.
TEST_SHIFTS_2D = [0, 5, 10, 15]
TEST_SHIFTS_3D = [0, 5, 10]
TEST_SHIFTS_2D_EXTENDED = [0, 5, 10, 20, 30]
TEST_SHIFTS_3D_EXTENDED = [0, 5, 10, 20, 30]

# Rotation.
TEST_ROT_POS_45 = list(range(46))
TEST_ROT_POS_90 = list(range(91))
TEST_ROT_POS_180 = list(range(181))
TEST_ROT_NEG_45 = list(range(-45, 1))
TEST_ROT_NEG_90 = list(range(-90, 1))
TEST_ROT_NEG_180 = list(range(-180, 1))
TEST_ROT_FULL_45 = sorted({*TEST_ROT_NEG_45[1:], *TEST_ROT_POS_45})
TEST_ROT_FULL_90 = sorted({*TEST_ROT_NEG_90[1:], *TEST_ROT_POS_90})
TEST_ROT_FULL_180 = sorted({*TEST_ROT_NEG_180[1:], *TEST_ROT_POS_180})

TEST_ROT_POS_45_5 = list(range(0, 46, 5))
TEST_ROT_POS_90_5 = list(range(0, 91, 5))
TEST_ROT_POS_180_5 = list(range(0, 181, 5))
TEST_ROT_NEG_45_5 = list(range(-45, 1, 5))
TEST_ROT_NEG_90_5 = list(range(-90, 1, 5))
TEST_ROT_NEG_180_5 = list(range(-180, 1, 5))
TEST_ROT_FULL_45_5 = sorted({*TEST_ROT_NEG_45_5[1:], *TEST_ROT_POS_45_5})
TEST_ROT_FULL_90_5 = sorted({*TEST_ROT_NEG_90_5[1:], *TEST_ROT_POS_90_5})
TEST_ROT_FULL_180_5 = sorted({*TEST_ROT_NEG_180_5[1:], *TEST_ROT_POS_180_5})

TEST_ROT_POS_45_10 = list(range(0, 46, 10))
TEST_ROT_POS_90_10 = list(range(0, 91, 10))
TEST_ROT_POS_180_10 = list(range(0, 181, 10))
TEST_ROT_NEG_45_10 = list(range(-45, 1, 10))
TEST_ROT_NEG_90_10 = list(range(-90, 1, 10))
TEST_ROT_NEG_180_10 = list(range(-180, 1, 10))
TEST_ROT_FULL_45_10 = sorted({*TEST_ROT_NEG_45_10[1:], *TEST_ROT_POS_45_10})
TEST_ROT_FULL_90_10 = sorted({*TEST_ROT_NEG_90_10[1:], *TEST_ROT_POS_90_10})
TEST_ROT_FULL_180_10 = sorted({*TEST_ROT_NEG_180_10[1:], *TEST_ROT_POS_180_10})

# Rotation axis.
TEST_ROT_AXES_VAR = list(os.getenv("AXES", "xyz"))
TEST_ROTATION_AXIS_3D = os.getenv("AXIS", "x").strip().lower()

# Scale.
TEST_SCALES_2D = [0.9, 1.0, 1.1]

# Image size.
TEST_IMAGE_SIZE_2D = [4, 8, 16, 32, 64]
TEST_IMAGE_SIZE_3D = [4, 8, 16, 32, 64]
TEST_IMAGE_SIZE_VAR_32 = int(os.getenv("SIZE", "32"))
TEST_IMAGE_SIZE_VAR_64 = int(os.getenv("SIZE", "64"))

# Images.
TEST_IMAGES_3D_DATA = ["illumination_image_3d", "haase_image_3d"]
TEST_IMAGES_3D_ARTIFICIAL = [
    "gradient_image_3d",
    "empty_image_3d",
    "full_image_3d",
    "random_image_3d",
]
TEST_IMAGES_3D = TEST_IMAGES_3D_DATA + TEST_IMAGES_3D_ARTIFICIAL

TEST_IMAGES_2D_DATA = ["f16_image_2d", "astronaut_image_2d"]
TEST_IMAGES_2D_ARTIFICIAL = [
    "gradient_image_2d",
    "empty_image_2d",
    "full_image_2d",
    "random_image_2d",
]
TEST_IMAGES_2D = TEST_IMAGES_2D_DATA + TEST_IMAGES_2D_ARTIFICIAL
