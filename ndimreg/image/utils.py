"""Utility functions for images."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .image import Image


def chainable(func: Callable) -> Callable:
    """Return `self` to make function calls chainable.

    Parameters
    ----------
    func
        Class method that should return `self` attribute after its call.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        func(self, *args, **kwargs)
        return self

    return wrapper


def log_shape(
    func: Callable | None = None, *, print_func: Callable = print
) -> Callable:
    """Log image shape before and after function call.

    Parameters
    ----------
    func
        Class method that provides a `shape` attribute.
    print_func
        Callable that logs the output string.
    """

    def actual_decorator(func: Callable) -> Callable:
        @wraps(func)
        def log_shape_wrapper(self: Image, *args: Any, **kwargs: Any) -> None:
            before = self.shape
            result = func(self, *args, **kwargs)
            after = self.shape

            if before != after:
                change_msg = "changed" if before != after else "unchanged"
                change_details = f"{before} --> {after}"
            else:
                change_msg = "unchanged"
                change_details = str(before)

            func_name = f"{self.__class__.__name__}.{func.__name__}"
            log_msg = f"Image shape {change_msg} during '{func_name}': {change_details}"
            print_func(log_msg)

            return result

        return log_shape_wrapper

    return actual_decorator(func) if func else actual_decorator
