import numpy as np
import pytest

from ndimreg.registration import (
    BaseRegistration,
    ImregDft2DRegistration,
    Keller2DRegistration,
    Keller3DRegistration,
    RotationAxis3DRegistration,
    Scikit2DRegistration,
    TranslationFFT2DRegistration,
    TranslationFFT3DRegistration,
)

REGISTRATION_METHODS_2D = [
    Keller2DRegistration,
    TranslationFFT2DRegistration,
    Scikit2DRegistration,
    ImregDft2DRegistration,
]

REGISTRATION_METHODS_3D = [
    Keller3DRegistration,
    TranslationFFT3DRegistration,
    RotationAxis3DRegistration,
]


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("registration_method", REGISTRATION_METHODS_2D)
def test_registration_2d_contains_debug_images_if_enabled(
    registration_method: type[BaseRegistration], *, debug: bool
) -> None:
    """TODO."""
    rng = np.random.default_rng()
    registration = registration_method(debug=debug)

    image_size = (8,) * 2
    result = registration.register(rng.random(image_size), rng.random(image_size))

    assert bool(len(result.get_debug_images(dim=2))) == debug


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("registration_method", REGISTRATION_METHODS_3D)
def test_registration_3d_contains_debug_images_if_enabled(
    registration_method: type[BaseRegistration], *, debug: bool
) -> None:
    """TODO."""
    rng = np.random.default_rng()
    registration = registration_method(debug=debug)

    image_size = (8,) * 3
    result = registration.register(rng.random(image_size), rng.random(image_size))

    assert bool(len(result.get_debug_images(dim=3))) == debug
