"""3D image registration using various approaches based on literature."""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.rotations as pr
from array_api_compat import get_namespace
from loguru import logger
from matplotlib.patches import Circle
from numpy.linalg import inv
from numpy.typing import NDArray
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


Sector = int
MinimumIndex = NDArray

DEFAULT_DEBUG_CMAP: Final = "rainbow"
U1: Final = (0.0, 0.0, 1.0)
OPTIMIZATION_DIRECTIONS: Final = np.array(
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
)

# PERF: Parallelize V1-tilde and V2-tilde transformations.
# PERF: Parallelize magnitude calculation if not on GPU.
# PERF: Parallelize debug plot generation (debug only).
# TODO: Rename class with better name.
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
        rotation_axis_optimization: bool = True,
        rotation_axis_vectorized: bool = False,
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
        self.__rotation_axis_optimization: bool = rotation_axis_optimization
        self.__rotation_axis_vectorized: bool = rotation_axis_vectorized
        self.__highpass_filter: bool = highpass_filter

        # TODO: Test parameters 'disambiguate' and 'normalization'.
        self.__rotation_angle_registration = RotationAxis3DRegistration(
            axis="x",
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

        normalize = self.__rotation_axis_normalization
        delta_v_func = _delta_v_normalized if normalize else _delta_v_default

        with AutoScipyFftBackend(xp):
            if self.debug:
                # Convert generator into re-usable tuple to keep for debug.
                magnitudes = tuple(magnitudes)

            delta_v = delta_v_func(*magnitudes, xp=xp)

        # We build the roation matrix for Z-axis alignment as defined
        # in section '4: Planar rotation'.
        index_func = (
            _index_optimized if self.__rotation_axis_optimization else _index_default
        )
        sector, min_index = index_func(delta_v)

        pseudopolar_coords = xp.array(min_index) - n // 2
        indices = to_numpy_array(-2 * pseudopolar_coords / n)
        u2 = np.insert(indices, sector, 1)

        axis_angle = pr.axis_angle_from_two_directions(U1, u2)
        rot_mat_r_tilde = pr.matrix_from_axis_angle(axis_angle)

        # We align both volumes onto the Z-axis.
        tilde_images = (self._transform(im, rotation=rot_mat_r_tilde) for im in images)

        if self.debug:
            # Convert generator into re-usable tuple to keep for debug.
            tilde_images = tuple(tilde_images)

        angle_result = self.__rotation_angle_registration.register(*tilde_images)
        z_rot = angle_result.transformation.rotation
        rot_mat_z_axis = pr.matrix_from_euler(
            np.deg2rad(z_rot), 0, 1, 2, extrinsic=False
        )
        matrix = rot_mat_r_tilde @ inv(rot_mat_z_axis) @ inv(rot_mat_r_tilde)
        angles = np.rad2deg(pr.euler_from_matrix(inv(matrix), 0, 1, 2, extrinsic=False))
        logger.debug(f"Recovered angles: [{', '.join(f'{x:.2f}' for x in angles)}]")

        moving_rotated = self._transform(moving, rotation=matrix)
        shift_result = self.__shift_registration.register(fixed, moving_rotated)
        shifts = shift_result.transformation.translation
        logger.debug(f"Recovered shifts: [{', '.join(f'{x:.2f}' for x in shifts)}]")

        if self.debug:
            debug_data = (*tilde_images, moving_rotated)
            debug_names = ("v1-tilde", "v2-tilde", "moving-rerotated")
            debug_images = [
                # Magnitudes of PPFT3D output have previously converted
                # to a tuple.
                *_debug_images(magnitudes),  # type: ignore[reportArgumentType]
                *(
                    RegistrationDebugImage(data, name, dim=3, copy=False)
                    for name, data in zip(debug_names, debug_data, strict=True)
                ),
            ]
        else:
            debug_images = None

        tform = Transformation3D(rotation=tuple(angles), translation=shifts)
        return ResultInternal3D(
            tform, sub_results=[angle_result, shift_result], debug_images=debug_images
        )


def _index_default(delta_v: NDArray) -> tuple[Sector, MinimumIndex]:
    xp = get_namespace(delta_v)
    sector, *min_index = xp.unravel_index(xp.argmin(delta_v), delta_v.shape)

    return sector.item(), to_numpy_array(xp.array(min_index))


def _index_optimized(delta_v: NDArray) -> tuple[Sector, MinimumIndex]:
    # FIX: Handle zero-valued neighbors.
    # TODO: Implement debug output.
    # TODO: Handle (e.g., exclude) 'NaN' values for corner indices.
    # PERF: Access edge/corner neighbors without combining all sectors.
    # The current method builds a full matrix that consists of all
    # values to allow for a wrap-around if the minimum index (i,j) is
    # on the edge (e.g., i=0) or at a corner (e.g., i=j=0). However,
    # this requires more memory and O(n) time. Direct access reduces
    # the asymptotic runtime complexity to O(1).
    sector, min_index = _index_default(delta_v)
    combined_sectors = _combine_sectors(to_numpy_array(delta_v), sector=sector)
    row, col = min_index + len(delta_v[0]) // 2

    center_matrix = combined_sectors[row - 1 : row + 2, col - 1 : col + 2]
    center_value = center_matrix[1, 1]

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    neighbors = np.ma.array(center_matrix, mask=mask).compressed()

    if np.max(neighbors) <= 0:
        logger.warning("Neighbor values are <= 0, cannot optimize minimum index")
        return sector, min_index

    weights = (center_value / neighbors) * (1 / 3)
    directed_weights = weights[:, None] * OPTIMIZATION_DIRECTIONS

    refinement = np.sum(directed_weights, axis=0)
    optimized_index = min_index + refinement

    logger.debug(f"Refined peak position from index {min_index} to {optimized_index}")

    return sector, optimized_index


@functools.lru_cache
def _generate_mask(n: int, *, xp: ModuleType) -> NDArray:
    radial_limit = (3 * n + 1) / 2
    rsi = __generate_radial_sampling_intervals(n, xp=xp)
    distances = rsi * xp.arange(radial_limit)[:, None, None]

    return (distances > radial_limit)[None, :]


@functools.lru_cache
def __generate_radial_sampling_intervals(n: int, *, xp: ModuleType) -> NDArray:
    target_shape = (n + 1, n + 1)
    x = (-2 * (xp.array(tuple(xp.ndindex(target_shape))) - n // 2) / n) ** 2 + 0.5

    return xp.sqrt(xp.sum(x, axis=1)).reshape(target_shape)


def _delta_v_normalized(m1: NDArray, m2: NDArray, *, xp: ModuleType) -> NDArray:
    # This is the implementation of the 'normalized correlation',
    # equation 3.9. This does not seem to produce anything useful
    # yet in comparison to the non-normalized version.
    # Source: https://de.mathworks.com/help/images/ref/normxcorr2.html
    # TODO: Compare with the solution as presented in the paper:
    # 'Normalized correlation for pattern recognition' [24].
    x1 = m1 - xp.nanmean(m1, 1, keepdims=True)
    x2 = m2 - xp.nanmean(m2, 1, keepdims=True)
    denominator = xp.sqrt(xp.nansum(x1**2, axis=1) * xp.nansum(x2**2, axis=1))

    return -(xp.nansum(x1 * x2, axis=1) / denominator)


def _delta_v_default(m1: NDArray, m2: NDArray, *, xp: ModuleType) -> NDArray:
    rsi = __generate_radial_sampling_intervals(m1.shape[2] - 1, xp=xp)
    return xp.nansum(xp.abs(m1 - m2) * rsi, axis=1)


def _combine_sectors(delta_v: NDArray, sector: int) -> NDArray:
    k = len(delta_v[0]) // 2

    if sector == 0:
        mid = delta_v[0]
        top = delta_v[1, ::-1][k:-1:]
        bot = delta_v[1, ::-1, ::-1][1 : k + 1]
        left = np.rot90(delta_v[2, ::-1, ::-1])[:, k:-1]
        right = np.rot90(delta_v[2, ::-1])[:, 1 : k + 1]

    elif sector == 1:
        mid = delta_v[1]
        left = delta_v[2, ::, ::-1][:, k:-1]
        right = delta_v[2, ::-1, ::-1][:, 1 : k + 1]
        top = delta_v[0, ::-1][k:-1:]
        bot = delta_v[0, ::-1, ::-1][1 : k + 1]

    else:
        mid = delta_v[2]
        top = np.rot90(delta_v[0, ::, ::-1])[k:0:-1]
        bot = np.rot90(delta_v[0, ::-1])[1 : k + 1]
        left = delta_v[1, ::, ::-1][:, k:-1]
        right = delta_v[1, ::-1, ::-1][:, 1 : k + 1]

    empty = np.ma.masked_array(np.empty((k, k)), mask=True)
    row_1 = np.ma.hstack([empty, top, empty])
    row_2 = np.ma.hstack([left, mid, right])
    row_3 = np.ma.hstack([empty, bot, empty])

    return np.ma.vstack((row_1, row_2, row_3))


def _debug_images(magnitudes: Sequence[NDArray]) -> list[RegistrationDebugImage]:
    # TODO: Add angles to x/y.
    # TODO: Implement circle version for output.
    # TODO: Implement -log(...) representation as the paper does (Fig. 3).
    # TODO: Add spherical output as in Figure 3.
    # TODO: Add spherical output using 3D rotation vectors (pytransform3d?).
    # TODO: Add titles again (sector, minimum index, angles, ...).

    magnitudes = tuple(to_numpy_arrays(*magnitudes))

    mpl.rc("font", size=8)
    # TODO: Check whether this is really faster.
    mpl.use("Agg")

    debug_image_kwargs = {"dim": 2, "copy": False}

    debug_images = []
    for normalize in (True, False):
        delta_func = _delta_v_normalized if normalize else _delta_v_default
        delta_v = delta_func(*magnitudes, xp=np)

        debug_images.append(
            RegistrationDebugImage(
                __build_rotation_axis_optimization_plots(delta_v),
                f"optimization-norm={normalize}",
                **debug_image_kwargs,
            )
        )

        for optimize in (True, False):
            index_func = _index_optimized if optimize else _index_default
            sector_min_index = index_func(delta_v)

            suffix = f"norm={normalize}-optimize={optimize}"
            debug_images.extend(
                (
                    RegistrationDebugImage(
                        __combined_fig(delta_v, *sector_min_index),
                        f"combined-{suffix}",
                        **debug_image_kwargs,
                    ),
                    RegistrationDebugImage(
                        __sectors_fig(delta_v, *sector_min_index),
                        f"sectors-{suffix}",
                        **debug_image_kwargs,
                    ),
                )
            )

    return debug_images


def __sectors_fig(delta_v: NDArray, sector: Sector, min_index: MinimumIndex) -> NDArray:
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplots(nrows=1, ncols=3)

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
    min_mark = Circle((min_index[1], min_index[0]), 0.5, color="red")
    min_ax = axs[sector]
    min_ax.add_patch(min_mark)
    min_ax.patch.set_linewidth(5)
    min_ax.patch.set_edgecolor("red")

    fig.suptitle("TODO")

    return fig_to_array(fig)


def __combined_fig(delta_v: NDArray, sector: int, min_index: MinimumIndex) -> NDArray:
    n = len(delta_v[0]) - 1
    fig = plt.figure(constrained_layout=True)
    ax = fig.subplots()

    data = _combine_sectors(delta_v, sector=sector)
    im = ax.imshow(data, cmap=DEFAULT_DEBUG_CMAP)

    min_index_shifted = tuple(np.array(min_index[::-1]) + n // 2)
    # Source: https://stackoverflow.com/a/70928130
    min_mark = Circle(min_index_shifted, 0.5, color="red")
    ax.add_patch(min_mark)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle("TODO")
    fig.colorbar(im, ax=ax, location="bottom")

    return fig_to_array(fig)


def __build_rotation_axis_optimization_plots(delta_v: NDArray) -> NDArray:
    # TODO: If possible, merge with actual optimization code.
    sector, min_index = _index_default(delta_v)
    combined_sectors = _combine_sectors(to_numpy_array(delta_v), sector=sector)
    row, col = np.array(min_index) + len(delta_v[0]) // 2

    center_matrix = combined_sectors[row - 1 : row + 2, col - 1 : col + 2]
    center_value = center_matrix[1, 1]

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    neighbors = np.ma.array(center_matrix, mask=mask).compressed()

    if np.max(neighbors) <= 0:
        msg = "Neighbor values are <= 0, cannot optimize minimum index"
        logger.warning(msg)

        plt.text(0.5, 0.5, msg, color="red", ha="center", va="center")
        plt.axis("off")
        return fig_to_array()

    weights = (center_value / neighbors) * (1 / 3)
    directed_weights = weights[:, None] * OPTIMIZATION_DIRECTIONS

    refinement = np.sum(directed_weights, axis=0)

    matrix = np.array(center_matrix)
    fig, ax = plt.subplots()
    plt.title("3x3 Matrix with Center as Minimum")
    plt.imshow(matrix, cmap="coolwarm")
    plt.colorbar(label="Value")

    for i, j in itertools.product(range(3), repeat=2):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black")

    indices = ["-1", "0", "1"]
    ax.set_xticks(range(3), labels=indices)
    ax.set_yticks(range(3), labels=indices)

    plt.title("3x3 Matrix with Center as Minimum + Weighted Vectors and Refinement")

    refinement_text = f"{np.array2string(refinement, precision=2, floatmode='fixed')}"
    ax.text(0.52, 0.52, refinement_text, color="red", ha="left", va="top")

    for weight in directed_weights:
        plt.arrow(
            1, 1, *weight[::-1], head_width=0.05, head_length=0.05, fc="gray", ec="gray"
        )

    plt.arrow(
        1, 1, *refinement[::-1], head_width=0.1, head_length=0.1, fc="red", ec="red"
    )

    return fig_to_array(fig)
