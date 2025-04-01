"""3D image registration using various approaches based on literature."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import get_namespace
from matplotlib.patches import Circle
from numpy.linalg import inv
from ppftpy import ppft3, rppft3
from typing_extensions import override

from ndimreg.processor import GrayscaleProcessor3D
from ndimreg.transform import Transformation3D
from ndimreg.utils import fig_to_array, to_numpy_arrays
from ndimreg.utils.arrays import to_numpy_array
from ndimreg.utils.fft import AutoScipyFftBackend

from .base import BaseRegistration
from .result import RegistrationDebugImage, ResultInternal3D
from .rotation_axis_3d import RotationAxis3DRegistration
from .translation_fft_3d import TranslationFFT3DRegistration

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from numpy.typing import NDArray

    from ndimreg.transform.types import RotationAxis3D

DEFAULT_DEBUG_CMAP: Final = "rainbow"

_U1_AXIS_MAP: dict[RotationAxis3D, tuple[float, float, float]] = {
    "z": (1.0, 0.0, 0.0),
    0: (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    1: (0.0, 1.0, 0.0),
    "x": (0.0, 0.0, 1.0),
    2: (0.0, 0.0, 1.0),
}

# PERF: Parallelize V1-tilde and V2-tilde transformations.
# PERF: Parallelize magnitude calculation if not on GPU.
# PERF: Parallelize debug plot generation (debug only).
# TODO: Rename class with better name.
# TODO: Implement 'rotation optimization' shift as in Keller2D.
# TODO: Test for flipped images by checking 180 degrees flips on all axes.


class Keller3DRegistration(BaseRegistration):
    """3D image registration using pseudo log-polar and FFT fourier transformation.

    This is an implementation of [1].

    Notes
    -----
    [1] references an algorithm for sub-pixel shift estimation,
    however we use the `phase_cross_correlation` from `scikit-image`
    instead, which uses another approach. Sub-pixel accuracy can be set
    by the `shift_upsample_factor` parameter.

    Capabilities
    ------------
    - Dimension: 3D
    - Translation: Yes
    - Rotation: Yes
    - Scale: No
    - Shear: No

    Limitations
    ------------
    - Images must be of same shape, i.e., NxNxN.
    - N must be even.

    References
    ----------
    .. [1] Keller, Y., Shkolnisky, Y., Averbuch, A.,
           "Volume Registration Using the 3-D Pseudopolar Fourier Transform,"
           IEEE Transactions on Signal Processing,
           Vol. 54, No. 11, pp. 4323-4331, 2006. :DOI:`10.1109/tsp.2006.881217`
    """

    def __init__(
        self,
        *,
        rotation_axis_normalization: bool = False,  # NOTE: WIP.
        rotation_axis_optimization: bool = False,  # NOTE: Not yet implemented.
        rotation_axis_vectorized: bool = False,
        rotation_angle_axis: RotationAxis3D = "z",
        rotation_angle_normalization: bool = True,
        rotation_angle_optimization: bool = True,
        rotation_angle_vectorized: bool = False,
        rotation_angle_shift_normalization: bool = True,
        rotation_angle_shift_disambiguate: bool = False,  # WARNING: True does not work.
        rotation_angle_shift_upsample_factor: int = 1,
        shift_normalization: bool = True,
        shift_disambiguate: bool = False,
        shift_upsample_factor: int = 1,
        highpass_filter: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the 3D Keller registration.

        Parameters
        ----------
        rotation_normalization
            Whether to normalize the rotation, by default True.
            In general, this should improve the accuracy of the rotation
            and only slightly increase the computation time.
        shift_normalization
            Whether to normalize the shift, by default True.
            In general, this should improvde the accuracy of the shift.
        shift_upsample_factor
            Upsample factor for the shift, by default 1.
            The upsample factor is used to increase the accuracy of the
            shift. The higher the factor, the more accurate the shift.
            However, it also increases the computation time.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self._processors.insert(0, GrayscaleProcessor3D())

        self.__rotation_axis_normalization: bool = rotation_axis_normalization
        self.__rotation_angle_axis: RotationAxis3D = rotation_angle_axis
        self.__rotation_axis_optimization: bool = rotation_axis_optimization
        self.__rotation_axis_vectorized: bool = rotation_axis_vectorized
        self.__highpass_filter: bool = highpass_filter

        # TODO: Test parameters 'disambiguate' and 'normalization'.
        self.__rotation_angle_registration = RotationAxis3DRegistration(
            axis=rotation_angle_axis,
            shift_disambiguate=rotation_angle_shift_disambiguate,
            shift_normalization=rotation_angle_shift_normalization,
            shift_upsample_factor=rotation_angle_shift_upsample_factor,
            rotation_optimization=rotation_angle_optimization,
            rotation_normalization=rotation_angle_normalization,
            rotation_vectorized=rotation_angle_vectorized,
            debug=self.debug,
        )

        self.__shift_registration = TranslationFFT3DRegistration(
            disambiguate=shift_disambiguate,
            normalization=shift_normalization,
            upsample_factor=shift_upsample_factor,
            debug=self.debug,
        )

    @property
    @override
    def dim(self) -> Literal[3]:
        return 3

    @override
    def _register(
        self, fixed: NDArray, moving: NDArray, **_kwargs: Any
    ) -> ResultInternal3D:
        images = (fixed, moving)
        xp = get_namespace(*images)

        n = len(fixed)
        mask = _generate_mask(n, xp=xp) if self.__highpass_filter else False
        is_complex = any(xp.iscomplexobj(im) for im in images)
        ppft, idx = (ppft3, n) if is_complex else (rppft3, 0)
        ppft_kwargs = {"vectorized": self.__rotation_axis_vectorized, "scipy_fft": True}

        magnitudes = (
            xp.where(mask, xp.nan, xp.abs(ppft(im, **ppft_kwargs)[:, :, idx:]))
            for im in images
        )

        delta_v_func = (
            _delta_v_normalized
            if self.__rotation_axis_normalization
            else _delta_v_default
        )

        with AutoScipyFftBackend(xp):
            if self.debug:
                # Convert generator into re-usable tuple to keep for debug.
                magnitudes = tuple(magnitudes)

            delta_v = delta_v_func(*magnitudes, xp=xp)

        # We build the rotation matrix for Z-axis alignment as defined
        # in section '4: Planar rotation'.
        u1 = _U1_AXIS_MAP[self.__rotation_angle_axis]
        u2 = to_numpy_array(_build_u2(delta_v, xp=xp))

        axis_angle = pr.axis_angle_from_two_directions(u1, u2)
        axis_rotation_matrix = pr.matrix_from_axis_angle(axis_angle)

        # We align both volumes onto the Z-axis.
        axis_aligned_images = (
            self._transform(im, rotation=axis_rotation_matrix) for im in images
        )

        if self.debug:
            # Convert generator into re-usable tuple to keep for debug.
            axis_aligned_images = tuple(axis_aligned_images)

        angle_result = self.__rotation_angle_registration.register(*axis_aligned_images)
        angle_rotation_matrix = pr.matrix_from_euler(
            np.deg2rad(angle_result.transformation.rotation), 0, 1, 2, extrinsic=False
        )

        inv_axis_rotation = inv(axis_rotation_matrix)
        matrix = axis_rotation_matrix @ inv(angle_rotation_matrix) @ inv_axis_rotation

        angles = np.rad2deg(pr.euler_from_matrix(inv(matrix), 0, 1, 2, extrinsic=False))

        moving_rotated = self._transform(moving, rotation=matrix)
        shift_result = self.__shift_registration.register(fixed, moving_rotated)
        shifts = shift_result.transformation.translation

        if self.debug:
            debug_data = (*axis_aligned_images, moving_rotated)
            debug_names = ("v1-tilde", "v2-tilde", "moving-rerotated")
            debug_images = [
                *_create_magnitude_debug_images(tuple(magnitudes)),
                *self._build_debug_images(debug_data, debug_names),
            ]
        else:
            debug_images = None

        tform = Transformation3D(rotation=tuple(angles), translation=shifts)
        return ResultInternal3D(
            tform, sub_results=[angle_result, shift_result], debug_images=debug_images
        )


@functools.lru_cache
def _generate_mask(n: int, *, xp: ModuleType) -> NDArray:
    radial_limit = (3 * n + 1) / 2
    rsi = __generate_radial_sampling_intervals(n, xp=xp)
    distances = xp.arange(radial_limit)[:, None, None] * rsi

    return (distances > radial_limit)[None, ...]


@functools.lru_cache
def __generate_radial_sampling_intervals(n: int, *, xp: ModuleType) -> NDArray:
    coords = xp.linspace(-1, 1, n + 1) ** 2 + 0.5

    return xp.hypot(coords[:, None], coords)


def _build_u2(delta_v: NDArray, *, xp: ModuleType) -> NDArray:
    n = len(delta_v[1]) - 1
    sector, *min_index = xp.unravel_index(xp.argmin(delta_v), delta_v.shape)
    coords = -2 * (xp.array(min_index) - n // 2) / n

    return xp.concatenate((coords[:sector], xp.array([1.0]), coords[sector:]))


def _delta_v_default(m1: NDArray, m2: NDArray, *, xp: ModuleType) -> NDArray:
    rsi = __generate_radial_sampling_intervals(m1.shape[2] - 1, xp=xp)

    return xp.nansum(xp.abs(m1 - m2) * rsi, axis=1)


def _delta_v_normalized(m1: NDArray, m2: NDArray, *, xp: ModuleType) -> NDArray:
    centered_1 = m1 - xp.nanmean(m1, axis=1, keepdims=True)
    centered_2 = m2 - xp.nanmean(m2, axis=1, keepdims=True)

    std_1 = xp.sqrt(xp.nanvar(m1, axis=1, ddof=0))
    std_2 = xp.sqrt(xp.nanvar(m2, axis=1, ddof=0))

    return xp.nansum((centered_1 - centered_2) ** 2, axis=1) / (std_1 * std_2)


def _create_magnitude_debug_images(
    magnitudes: Sequence[NDArray],
) -> list[RegistrationDebugImage]:
    # TODO: Add angles to x/y.
    # TODO: Implement circle version for output.
    # TODO: Implement -log(...) representation as the paper does (Fig. 3).
    # TODO: Add spherical output as in Figure 3.
    # TODO: Add spherical output using 3D rotation vectors (pytransform3d?).

    magnitudes = tuple(to_numpy_arrays(*magnitudes))
    delta_v_norm = _delta_v_normalized(*magnitudes, xp=np)
    delta_v_default = _delta_v_default(*magnitudes, xp=np)

    mpl.rc("font", size=8)
    # TODO: Check whether this is really faster.
    mpl.use("Agg")

    kwargs = {"dim": 2, "copy": False}
    return [
        RegistrationDebugImage(
            __combined_fig(delta_v_default), "ppft3-combined-default", **kwargs
        ),
        RegistrationDebugImage(
            __combined_fig(delta_v_norm), "ppft3-combined-normalized", **kwargs
        ),
        RegistrationDebugImage(
            __sectors_fig(delta_v_default), "ppft3-sectors-default", **kwargs
        ),
        RegistrationDebugImage(
            __sectors_fig(delta_v_norm), "ppft3-sectors-normalized", **kwargs
        ),
    ]


def __sectors_fig(delta_v: NDArray) -> NDArray:
    """TODO."""
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplots(nrows=1, ncols=3, sharey=True, sharex=True)

    # Source: https://stackoverflow.com/a/68553479/24321379
    im = None
    for ax, sec in zip(axs.flat, delta_v, strict=True):
        im = ax.imshow(
            sec, vmin=delta_v.min(), vmax=delta_v.max(), cmap=DEFAULT_DEBUG_CMAP
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if im is not None:
        fig.colorbar(im, ax=axs, location="bottom")

    # Source: https://stackoverflow.com/a/70928130
    min_index = np.unravel_index(delta_v.argmin(), delta_v.shape)
    min_mark = Circle(tuple((np.array(min_index[1:]) + 0.5)[::-1]), 0.5, color="red")
    min_ax = axs[min_index[0]]
    min_ax.add_patch(min_mark)
    min_ax.patch.set_linewidth(5)
    min_ax.patch.set_edgecolor("red")

    fig.suptitle(__build_title(delta_v))

    return fig_to_array(fig)


def __combined_fig(delta_v: NDArray) -> NDArray:
    """TODO."""
    fig = plt.figure(constrained_layout=True)
    ax = fig.subplots()

    data = __combine_sectors(delta_v)
    im = ax.imshow(data, cmap=DEFAULT_DEBUG_CMAP)

    # Source: https://stackoverflow.com/a/70928130
    min_index = np.unravel_index(data.argmin(), data.shape)
    min_mark = Circle(tuple((np.array(min_index) + 0.5)[::-1]), 0.5, color="red")
    ax.add_patch(min_mark)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle(__build_title(delta_v))
    fig.colorbar(im, ax=ax, location="bottom")

    return fig_to_array(fig)


def __combine_sectors(sectors: NDArray) -> NDArray:
    k = len(sectors[0]) // 2

    mid = sectors[2]
    top = np.rot90(sectors[0, ::, ::-1])[k:0:-1]
    bot = np.rot90(sectors[0, ::-1])[1 : k + 1]
    left = sectors[1, ::, ::-1][:, k:-1]
    right = sectors[1, ::-1, ::-1][:, 1 : k + 1]
    empty = np.ma.masked_array(np.empty((k, k)), mask=True)

    row_1 = np.ma.hstack([empty, top, empty])
    row_2 = np.ma.hstack([left, mid, right])
    row_3 = np.ma.hstack([empty, bot, empty])

    return np.ma.vstack((row_1, row_2, row_3))


def __build_title(delta_v: NDArray) -> str:
    min_index = np.unravel_index(delta_v.argmin(), delta_v.shape)

    n_sector = f"N: {len(delta_v[1]) - 1}, Sector: {min_index[0] + 1}"
    index = f"Index: {tuple(int(x) for x in min_index[1:])}"
    value = f"Value: {delta_v.min():.3f}"

    theta, phi = np.rad2deg(_build_u2(delta_v, xp=np)[1:])
    degrees = rf"$\phi$: {phi:.2f}°, $\theta$: {theta:.2f}°"

    return f"{n_sector}, {index}, {value}\n{degrees}"
