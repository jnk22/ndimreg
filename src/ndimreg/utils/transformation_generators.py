import itertools
from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import pytransform3d.rotations as pr

# TODO: Normalize rotation angle inputs (or restrict to range).
# TODO: Validate input (amount in [0, -x])
# TODO: Create test suite.


def generate_random_translations_2d(
    amount: int,
    *,
    low: tuple[float, float],
    high: tuple[float, float],
    rng: np.random.Generator | None = None,
) -> Generator[tuple[float, float]]:
    rng = rng or np.random.default_rng()

    yield from (tuple(x) for x in rng.uniform(low, high, (amount, 2)))


def generate_random_translations_3d(
    amount: int,
    *,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    rng: np.random.Generator | None = None,
) -> Generator[tuple[float, float, float]]:
    rng = rng or np.random.default_rng()

    yield from (tuple(x) for x in rng.uniform(low, high, (amount, 3)))


def generate_random_rotations_2d(
    amount: int, *, low: float, high: float, rng: np.random.Generator | None = None
) -> Generator[float]:
    rng = rng or np.random.default_rng()

    yield from rng.uniform(low, high, amount)


def generate_random_rotations_3d(
    amount: int,
    *,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    rng: np.random.Generator | None = None,
) -> Generator[tuple[float, float, float]]:
    rng = rng or np.random.default_rng()

    quaternions = []
    while len(quaternions) < amount:
        quat = pr.random_quaternion(rng)
        euler = np.rad2deg(pr.euler_from_quaternion(quat, 0, 1, 2, extrinsic=False))
        if np.all(euler >= low) and np.all(euler <= high):
            quaternions.append(euler)

    yield from (tuple(x) for x in quaternions)


def generate_random_scales(
    amount: int, *, low: float, high: float, rng: np.random.Generator | None = None
) -> Generator[float]:
    rng = rng or np.random.default_rng()

    yield from rng.uniform(low, high, amount)


def generate_uniform_translations_2d(
    amount: int, *, low: tuple[float, float], high: tuple[float, float], **kwargs: Any
) -> Generator[tuple[float, float]]:
    values_generator = __uniform_values(amount, low=low, high=high)

    yield from ((x, y) for x, y in itertools.product(*values_generator))


def generate_uniform_translations_3d(
    amount: int,
    *,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    **kwargs: Any,
) -> Generator[tuple[float, float, float]]:
    values_generator = __uniform_values(amount, low=low, high=high)

    yield from ((x, y, z) for x, y, z in itertools.product(*values_generator))


def generate_uniform_rotations_2d(
    amount: int, *, low: float, high: float, **kwargs: Any
) -> Generator[float]:
    yield from np.linspace(low, high, num=amount)


def generate_uniform_rotations_3d(
    amount: int,
    *,
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    **kwargs: Any,
) -> Generator[tuple[float, float, float]]:
    values_generator = __uniform_values(amount, low=low, high=high)

    yield from ((x, y, z) for x, y, z in itertools.product(*values_generator))


def generate_uniform_scales(
    amount: int, *, low: float, high: float, **kwargs: Any
) -> Generator[float]:
    yield from np.linspace(low, high, num=amount)


def __uniform_values(
    amount: int, *, low: Sequence[float], high: Sequence[float]
) -> Generator:
    ranges = zip(low, high, strict=True)

    yield from (np.linspace(_low, _high, num=amount) for _low, _high in ranges)
