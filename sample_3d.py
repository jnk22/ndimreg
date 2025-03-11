"""Sample script for benchmarking with 'py-spy'.

Run 'py-spy record scripts/benchmark_registration.py' to create flamegraph.
"""

from pathlib import Path

from ndimreg.image import Image3D
from ndimreg.registration import Keller3DRegistration
from ndimreg.transform import Transformation3D

RUNS = 2
SIZE = 64
DEVICE = "gpu"

fixed = Image3D.from_path(Path("data/3d/haase_mrt.tif"))
fixed.pad_safe_rotation().resize_to_shape(SIZE).to_device(DEVICE)

for _ in range(RUNS):
    tform_input = Transformation3D(translation=(10, 5, 2), rotation=(10, 20, 30))
    moving = fixed.copy().transform(transformation=tform_input)

    result = Keller3DRegistration(
        rotation_angle_vectorized=True, rotation_axis_vectorized=True
    ).register(fixed.data, moving.data)
