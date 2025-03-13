"""Utilities for DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


def explode_nested_column(
    df: pl.DataFrame, column: str, fields: Sequence[str]
) -> Generator[pl.Expr]:
    """TODO."""
    # PERF: Use generic expression instead of explicit null check.
    func = _alias_empty if df.schema[column] == pl.Null else _alias_numbers
    yield from func(column, fields)


def _alias_empty(col_name: str, fields: Sequence[str]) -> Generator[pl.Expr]:
    yield from (pl.lit(None).alias(f"{col_name}.{field}") for field in fields)


def _alias_numbers(col_name: str, fields: Sequence[str]) -> Generator[pl.Expr]:
    yield from (
        pl.col(col_name).list.get(i, null_on_oob=True).alias(f"{col_name}.{field}")
        for i, field in enumerate(fields)
    )
