"""TODO."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from array_api_compat import is_array_api_obj

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


# PERF: Only run when explicitly enabled (e.g., via environment variable).
def log_memory(
    func: Callable | None = None,
    *,
    print_func: Callable = print,
    positions: Sequence[int] | None = None,
) -> Callable:
    """TODO."""
    # TODO: Allow aggregation (e.g., sum and mean).
    # TODO: Improve logged return value positions: all/auto/explicit/...
    # TODO: Add backend device where data resides.
    # TODO: Allow precision format + float precision (auto/user-specific).
    # TODO: Print object name if possible.
    # TODO: Extend support for non-array objects.

    def __print_memory_usage(obj: NDArray) -> None:
        if is_array_api_obj(obj):
            mem = obj.nbytes
            print_func(
                f"Array memory usage: {mem} bytes ({mem / 1024:.2f} KB, {mem / (1024**2):.2f} MB)"
            )
        else:
            print_func("Array memory usage: Returned value is not an array object")

    def actual_decorator(f: Callable) -> Callable:
        @wraps(f)
        def log_memory_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = f(*args, **kwargs)

            if not positions:
                __print_memory_usage(result)
                return result

            for i in positions:
                try:
                    __print_memory_usage(result[i])
                except IndexError:
                    print_func(
                        f"Array memory usage: Result object at position {i} does not exist"
                    )

            return result

        return log_memory_wrapper

    return actual_decorator(func) if func else actual_decorator
