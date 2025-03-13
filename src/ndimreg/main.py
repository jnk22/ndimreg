"""CLI main module."""

from __future__ import annotations

from cyclopts import App

from ._version import version
from .cli import benchmark, debug, register, tools, transform

# TODO: Add debug/results show difference (default true for results, false for debug).
# TODO: Properly configure logger (e.g., via CLI callback).
# TODO: Implement user input known parameters: rotation, scale, shifts.
# TODO: Implement multi-image registration.
# TODO: Make pre-processing order configurable.
# TODO: Implement 3D compatible pre-processing functions.

app = App(version=version)

app.command(register.app)
app.command(transform.app)
app.command(benchmark.app)
app.command(debug.app)
app.command(tools.app)
