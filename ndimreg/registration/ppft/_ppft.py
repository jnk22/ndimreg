"""TODO."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from scipy import fft

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray


@functools.cache
def _calculate_pq_pz(n: int, dim: int, xp: ModuleType) -> tuple[NDArray, NDArray]:
    # Pre-calculate values that are only dependent on n.
    np = n + 1
    nh = n // 2
    nhx = nh + 1
    m = n * dim + 1
    x = m // 2

    px = (
        2j
        * xp.pi
        / (n * m)
        * xp.arange(-x, x + 1)[:, None]
        * xp.arange(-np, np + 1) ** 2
    )
    pq = xp.exp(-px[:, nhx:-nhx])
    pz = fft.fft(xp.pad(xp.exp(px), ((0, 0), (nh, nh))))

    return pq, pz


@functools.cache
def _calculate_rpq_rpz(n: int, dim: int, xp: ModuleType) -> tuple[NDArray, NDArray]:
    pq, pz = _calculate_pq_pz(n, dim, xp)
    x = (dim * n) // 2

    return pq[x:], pz[x:]


@functools.cache
def _calculate_nh_idx(n: int) -> tuple[int, int, slice]:
    np = n + 1
    md = np + np // 2

    return n // 2, np * 3, slice(md, md + np)
