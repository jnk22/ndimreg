"""CLI to calculate PPFT2D and PPFT3D output.

This generates angles for both 2D and 3D pseudo-polar representations
in cartesian and polar/spherical coordinates.
"""

from __future__ import annotations

from typing import Literal

from cyclopts import App
from cyclopts.types import NonNegativeInt, PositiveInt, ResolvedFile  # noqa: TC002

from ._types import Dimension  # noqa: TC001

app = App(name="tools", show=False)


@app.command
def pseudo_polar_coordinates(  # noqa: PLR0913
    dimension: Dimension,
    output_file: ResolvedFile,
    *,
    n: PositiveInt = 4,
    sectors: Literal[1, 2, 3] | None = None,
    k: NonNegativeInt | None = None,
    ignore_mirror: bool = True,
    precision: PositiveInt | None = None,
) -> None:
    """Generate pseudo-polar coordinates.

    Parameters
    ----------
    dimension:
        Pseudo-polar coordinates dimension.
    n:
        Image size to generate coordinates on.
    sectors:
        Generate output only for specific sector. Defaults to all.
    k:
        Generate output only for specific pseudo-polar radius.
    ignore_mirror:
        Ignore mirrored/duplicate data that exists due to Fourier
        Transform.
    precision:
        Formatting precision for float output.
    output_file:
        File to store dataframe to, can be either CSV or JSON.
    """
    import sys

    from ndimreg.tools.pseudo_polar_coordinates import Table2D, Table3D

    table_cls = Table2D if dimension == 2 else Table3D  # noqa: PLR2004
    table = table_cls(n, k, sectors, ignore_mirror=ignore_mirror)  # type: ignore[reportArgumentType]

    if not len(table):
        print("No data points for pseudo-polar coordinates match the input criteria")
        sys.exit(1)

    match output_file.suffix.lower():
        case ".json":
            table.write_json(output_file, precision=precision)
        case ".csv":
            table.write_csv(output_file, precision=precision)
        case _:
            print(f"Unsupported file format '{output_file.suffix}'")
            sys.exit(1)
