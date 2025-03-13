"""TODO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

from array_api_compat import get_namespace, is_cupy_namespace
from scipy import fft

try:
    import cupyx.scipy.fft as cufft
except ImportError:
    cufft = None

try:
    import mkl_fft._scipy_fft_backend as mkl_fft
except ImportError:
    mkl_fft = None

try:
    import pyfftw
    import pyfftw.interfaces.scipy_fft as fftw

    # If pyFFTW is available, we also enable the cache as suggested in
    # https://pyfftw.readthedocs.io/en/latest/source/tutorial.html.
    pyfftw.interfaces.cache.enable()
except ImportError:
    fftw = None

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType, TracebackType

    from numpy.typing import NDArray
    from typing_extensions import Self

CpuFftBackend = Literal["scipy", "pyfftw", "mkl"]
FftBackend = CpuFftBackend | Literal["cupy"]

CPU_BACKENDS: Final[dict[CpuFftBackend, ModuleType | str | None]] = {
    "mkl": mkl_fft,
    "pyfftw": fftw,
    "scipy": "scipy",
}
"""Mapping of known FFT backends in order of priority."""


def get_available_fft_backends() -> Generator[CpuFftBackend]:
    """TODO."""
    return (name for name, backend in CPU_BACKENDS.items() if backend is not None)


def get_fft_backend(backend: CpuFftBackend) -> ModuleType | str:
    """TODO."""
    if backend in CPU_BACKENDS and (module := CPU_BACKENDS[backend]) is not None:
        return module

    msg = f"Selected backend '{backend}' is not available"
    raise ValueError(msg)


class AutoScipyFftBackend:
    """TODO."""

    def __init__(self, namespace: ModuleType) -> None:
        """TODO."""
        self.namespace: ModuleType = namespace
        self.scipy_backend_context: Any | None = None

    @classmethod
    def from_array(cls, *arrays: NDArray) -> Self:
        """TODO."""
        return cls(get_namespace(*arrays))

    def __enter__(self) -> Any:
        """TODO."""
        if self.namespace and is_cupy_namespace(self.namespace):
            self.scipy_backend_context = fft.set_backend(cufft)
            return self.scipy_backend_context.__enter__()

        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """TODO."""
        if self.scipy_backend_context is not None:
            self.scipy_backend_context.__exit__(exc_type, exc_val, exc_tb)
