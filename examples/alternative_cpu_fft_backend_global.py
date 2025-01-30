import mkl_fft._scipy_fft_backend as mkl_fft
from scipy import fft

from ndimreg.image import Image2D
from ndimreg.registration import Keller2DRegistration
from ndimreg.utils import format_time

# Load and prepare sample image as fixed and moving.
sample_image = Image2D.from_skimage("astronaut").resize_to_shape(256)
fixed_image = sample_image.copy().data
moving_image = sample_image.copy().transform(translation=(12.8, 25.6), rotation=22).data

# Setup a registration method.
registration = Keller2DRegistration()

# Set the global FFT backend to `mkl_fft` which will then be used
# during registration. You can use any backend that supports SciPy FFT.
fft.set_global_backend(mkl_fft)

result = registration.register(fixed_image, moving_image)

print(f"Duration: {format_time(result.total_duration)}")
