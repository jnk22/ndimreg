"""TODO."""

from __future__ import annotations

import functools
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np
import polars as pl
from typing_extensions import override

from ndimreg.utils.dataframes import explode_nested_column

if TYPE_CHECKING:
    from collections.abc import Collection

PPFT2DSector = Literal[1, 2]
PPFT3DSector = Literal[1, 2, 3]
PolarCoordinates = tuple[float, float]
CartesianCoordinates2D = tuple[float, float]
PseudoPolarCoordinates2D = tuple[int, int]
SphericalCoordinates = tuple[float, float, float]
CartesianCoordinates3D = tuple[float, float, float]
PseudoPolarCoordinates3D = tuple[int, int, int]

SECTOR_IDS_2D: list[PPFT2DSector] = [1, 2]
SECTOR_IDS_3D: list[PPFT3DSector] = [1, 2, 3]


def pseudopolar_to_cartesian_2d(
    coordinates: PseudoPolarCoordinates2D, sector: PPFT2DSector, n: int
) -> CartesianCoordinates2D:
    """TODO."""
    k, i = coordinates

    match sector:
        case 1:
            return -(2 * i / n) * k, k
        case 2:
            return k, -(2 * i / n) * k


def pseudopolar_to_cartesian_3d(
    coordinates: PseudoPolarCoordinates3D, sector: PPFT3DSector, n: int
) -> CartesianCoordinates3D:
    """Convert pseudo-polar coordinates to cartesian coordinates.

    Returns
    -------
    CartesianCoordinates3D
        Cartesian coordinates on 3D grid for pseudo-polar coordinates.
    """
    k, i, j = coordinates

    match sector:
        case 1:
            return k, -2 * i * k / n, -2 * j * k / n
        case 2:
            return -2 * i * k / n, k, -2 * j * k / n
        case 3:
            return -2 * i * k / n, -2 * j * k / n, k


def pseudopolar_to_polar(
    coordinates: PseudoPolarCoordinates2D, sector: PPFT2DSector, n: int
) -> PolarCoordinates:
    """TODO."""
    k, i = coordinates

    r = k * np.sqrt(4 * (i / n) ** 2 + 1)
    angle = np.arctan(2 * i / n)

    match sector:
        case 1:
            return r, np.pi / 2 - angle
        case 2:
            return r, angle


def pseudopolar_to_spherical(
    coordinates: PseudoPolarCoordinates3D, sector: PPFT3DSector, n: int
) -> SphericalCoordinates:
    """Convert pseudo-polar coordinates to spherical coordinates.

    Returns
    -------
    SphericalCoordinates
        Spherical coordinates for pseudo-polar coordinates as radians.
    """
    x, y, z = pseudopolar_to_cartesian_3d(coordinates, sector, n)

    r = np.sqrt(x**2 + y**2 + z**2)

    phi = np.arccos(z / r) if r != 0 else 0
    theta = np.arctan2(y, x)

    return r, phi, theta


class PolarGridValue(ABC):
    """Base class for representation of a nD polar grid value."""

    n: int
    sector: Any

    def __lt__(self, other: object) -> bool:
        """TODO."""
        if not isinstance(other, type(self)):
            return False

        return (
            self.sector < other.sector
            or self.pseudopolar_coordinates < other.pseudopolar_coordinates
        )

    def __eq__(self, other: object) -> bool:
        """TODO."""
        if not isinstance(other, type(self)):
            return False

        return (
            self.sector == other.sector
            and self.pseudopolar_coordinates == other.pseudopolar_coordinates
        )

    @property
    @abstractmethod
    def m(self) -> int:
        """Return the radial resoution `M`."""
        ...

    @property
    @abstractmethod
    def pseudopolar_coordinates(self) -> tuple[float, ...]:
        """Pseudo-polar point coordinates."""

    @property
    @abstractmethod
    def cartesian_coordinates(self) -> tuple[float, ...]:
        """Cartesian coordinates."""

    @property
    @abstractmethod
    def spatial_coordinates(self) -> tuple[float, ...]:
        """Spatial coordinates."""

    @property
    @abstractmethod
    def radial_sampling_interval(self) -> float:
        """Return the radial sampling interval of the polar grid value."""

    @property
    @abstractmethod
    def in_range(self) -> bool:
        """Return whether pseudo-polar point is in the circle.

        Based on the paper, a point **outside** the circle is in the
        _high-frequency_ range.
        """

    @property
    @abstractmethod
    def radius(self) -> float:
        """Return the radius of the polar grid value."""


@functools.total_ordering
@dataclass(frozen=True, slots=True)
class PolarGridValue2D(PolarGridValue):
    """Representation of a 2D polar grid value."""

    n: int
    """Total input size of polar grid as NxN."""

    sector: PPFT2DSector
    """Pseudopolar sector."""

    k: int
    """Pseudoradius."""

    i: int
    """Pseudoangle."""

    @property
    @override
    def m(self) -> int:
        return self.n * 2 + 1

    @property
    @override
    def pseudopolar_coordinates(self) -> PseudoPolarCoordinates2D:
        return self.k, self.i

    @property
    @override
    def cartesian_coordinates(self) -> CartesianCoordinates2D:
        return pseudopolar_to_cartesian_2d(
            self.pseudopolar_coordinates, self.sector, self.n
        )

    @property
    @override
    def spatial_coordinates(self) -> PolarCoordinates:
        """Polar coordinates as defined in equations 3.10 and 3.11."""
        return pseudopolar_to_polar(self.pseudopolar_coordinates, self.sector, self.n)

    @property
    @override
    def radial_sampling_interval(self) -> float:
        return pseudopolar_to_polar((1, self.i), 1, self.n)[0]

    @property
    @override
    def in_range(self) -> bool:
        return self.radius <= self.n

    @property
    @override
    def radius(self) -> float:
        return abs(self.spatial_coordinates[0])

    @property
    def theta(self) -> float:
        """Return the angle in radians from the x-axis to the point."""
        return self.spatial_coordinates[1]


@functools.total_ordering
@dataclass(frozen=True, slots=True)
class PolarGridValue3D(PolarGridValue):
    """Representation of a 3D polar grid value."""

    n: int
    """Total input size of polar grid as NxNxN."""

    sector: PPFT3DSector
    """Pseudopolar sector."""

    k: int
    """Radial resolution M = 3xN+1."""

    i: int
    """Angular resolution N+1."""

    j: int
    """Angular resolution N+1."""

    @property
    @override
    def m(self) -> int:
        return self.n * 3 + 1

    @property
    @override
    def pseudopolar_coordinates(self) -> PseudoPolarCoordinates3D:
        return self.k, self.i, self.j

    @property
    @override
    def cartesian_coordinates(self) -> CartesianCoordinates3D:
        return pseudopolar_to_cartesian_3d(
            self.pseudopolar_coordinates, self.sector, self.n
        )

    @property
    @override
    def spatial_coordinates(self) -> SphericalCoordinates:
        """Spherical coordinates as defined in equations 3.10 and 3.11."""
        return pseudopolar_to_spherical(
            self.pseudopolar_coordinates, self.sector, self.n
        )

    @property
    @override
    def radial_sampling_interval(self) -> float:
        return pseudopolar_to_spherical((1, self.i, self.j), 1, self.n)[0]

    @property
    @override
    def in_range(self) -> bool:
        return self.radius <= self.m / 2

    @property
    @override
    def radius(self) -> float:
        return abs(self.spatial_coordinates[0])

    @property
    def theta(self) -> float:
        """TODO."""
        return self.spatial_coordinates[1]

    @property
    def phi(self) -> float:
        """TODO."""
        return self.spatial_coordinates[2]


T = TypeVar("T", bound=PolarGridValue)


class PolarGrid(ABC, Generic[T]):
    """TODO."""

    @abstractmethod
    def generate_values(self, sectors: Collection | None) -> set[T]:
        """TODO."""


@dataclass(frozen=True, slots=True)
class PolarGrid2D(PolarGrid):
    """Representation of a polar grid."""

    n: int = 4

    @override
    def generate_values(
        self, sectors: Collection[PPFT2DSector] | None = None
    ) -> set[PolarGridValue2D]:
        """Generate all possible values for the polar grid.

        Parameters
        ----------
        sectors
            The sectors to generate values for, by default None.
            Allowed values are 1 and 2.

        Returns
        -------
        set[PolarGridValue2D]
            A set of polar grid values.
        """
        sectors_range = SECTOR_IDS_2D if sectors is None else set(sectors) & {1, 2}
        k_range = range(-self.n, self.n + 1)
        i_range = range(-self.n // 2, self.n // 2 + 1)

        ranges = itertools.product(sectors_range, k_range, i_range)

        return {PolarGridValue2D(self.n, *values) for values in ranges}


@dataclass(frozen=True, slots=True)
class PolarGrid3D(PolarGrid):
    """Representation of a polar grid."""

    n: int = 4

    @override
    def generate_values(
        self, sectors: Collection[PPFT3DSector] | None = None
    ) -> set[PolarGridValue3D]:
        """Generate all possible values for the polar grid.

        Parameters
        ----------
        sectors
            The sectors to generate values for, by default None.
            Allowed values are 1, 2, and 3.

        Returns
        -------
        set[PolarGridValue3D]
            A set of polar grid values.
        """
        sectors_range = SECTOR_IDS_3D if sectors is None else set(sectors) & {1, 2, 3}
        k_range = range(-3 * self.n // 2, 3 * self.n // 2 + 1)
        i_range = range(-self.n // 2, self.n // 2 + 1)
        j_range = range(-self.n // 2, self.n // 2 + 1)

        ranges = itertools.product(sectors_range, k_range, i_range, j_range)

        return {PolarGridValue3D(self.n, *values) for values in ranges}


class Table:
    """TODO."""

    def __len__(self) -> int:
        """Return table rows."""
        return len(self.rows)

    @property
    @abstractmethod
    def headers(self) -> tuple:
        """TODO."""

    @property
    @abstractmethod
    def rows(self) -> list[tuple]:
        """TODO."""

    @property
    def as_dict(self) -> dict[str, list]:
        """Return pseudo-polar coordinate table as dictionary."""
        return dict(
            zip(
                self.headers, list(map(list, zip(*self.rows, strict=True))), strict=True
            )
        )

    @property
    def as_dataframe(self) -> pl.DataFrame:
        """Return pseudo-polar coordinate table as dictionary."""
        return pl.DataFrame(self.as_dict, strict=False)

    @property
    @abstractmethod
    def _nested_fields(self) -> list[tuple[str, tuple[str, ...]]]:
        """TODO."""

    def write_json(self, file: Path | str, precision: int | None = None) -> None:
        """TODO."""
        results_df = self.as_dataframe

        if precision:
            results_df = results_df.with_columns(
                pl.col(column).round(precision)
                for column in results_df.columns
                if results_df.schema[column] == pl.Float64
            )

        Path(file).parent.mkdir(exist_ok=True, parents=True)
        results_df.write_json(file)

    def write_csv(
        self, file: Path | str, precision: int | None = None, **kwargs: Any
    ) -> None:
        """TODO."""
        results_df = self.as_dataframe

        nested_fields = self._nested_fields
        column_aliases = (
            expr
            for col, fields in nested_fields
            for expr in explode_nested_column(results_df, col, fields)
        )
        column_drops = (name for name, _ in nested_fields)

        results_df = results_df.with_columns(column_aliases).drop(column_drops)
        results_df = results_df.select(*sorted(results_df.columns))

        Path(file).parent.mkdir(exist_ok=True, parents=True)
        results_df.write_csv(file, float_precision=precision, **kwargs)


@dataclass(frozen=True, slots=True)
class Table2D(Table):
    """TODO."""

    n: int
    limit_k: int | None
    sectors: list[PPFT2DSector] | None
    ignore_mirror: bool

    @property
    @override
    def headers(self) -> tuple:
        return (
            "sector",
            "pseudopolar",
            "cartesian",
            "polar",
            "θ (rad)",
            "θ (deg)",
            "radius",
            "Δr_i",
            "◯",
        )

    @property
    @override
    def rows(self) -> list[tuple]:
        return [
            (
                point.sector,
                point.pseudopolar_coordinates,
                point.cartesian_coordinates,
                point.spatial_coordinates,
                point.theta,
                np.rad2deg(point.theta),
                point.radius,
                point.radial_sampling_interval,
                point.in_range,
            )
            for point in (PolarGrid2D(self.n).generate_values(sectors=self.sectors))
            if (self.limit_k is None or point.k == self.limit_k)
            and (self.ignore_mirror and point.k >= 0)
        ]

    @property
    @override
    def _nested_fields(self) -> list[tuple[str, tuple[str, str]]]:
        """TODO."""
        return [
            ("pseudopolar", ("k", "i")),
            ("cartesian", ("x", "y")),
            ("polar", ("r", "theta")),
        ]


@dataclass(frozen=True, slots=True)
class Table3D(Table):
    """TODO."""

    n: int
    limit_k: int | None
    sectors: list[PPFT3DSector] | None
    ignore_mirror: bool

    @property
    @override
    def headers(self) -> tuple:
        return (
            "sector",
            "pseudopolar",
            "cartesian",
            "spherical",
            "θ (rad)",
            "θ (deg)",
            "Φ (rad)",
            "Φ (deg)",
            "radius",
            "Δr_ij",
            "◯",
        )

    @property
    @override
    def rows(self) -> list[tuple]:
        return [
            (
                point.sector,
                point.pseudopolar_coordinates,
                point.cartesian_coordinates,
                point.spatial_coordinates,
                point.theta,
                np.rad2deg(point.theta),
                point.phi,
                np.rad2deg(point.phi),
                point.radius,
                point.radial_sampling_interval,
                point.in_range,
            )
            for point in PolarGrid3D(self.n).generate_values(sectors=self.sectors)
            if (self.limit_k is None or point.k == self.limit_k)
            and (self.ignore_mirror and point.k >= 0)
        ]

    @property
    @override
    def _nested_fields(self) -> list[tuple[str, tuple[str, str, str]]]:
        """TODO."""
        return [
            ("pseudopolar", ("k", "i", "j")),
            ("cartesian", ("x", "y", "z")),
            ("spherical", ("r", "phi", "theta")),
        ]
