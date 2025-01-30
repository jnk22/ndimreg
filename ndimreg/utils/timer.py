"""TODO."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

MILLI_SECONDS_DIFF = 1e-0
MICRO_SECONDS_DIFF = 1e-3
NANO_SECONDS_DIFF = 1e-6


def format_time(seconds: float, /, *, precision: int = 2) -> str:
    """Format time into human-readable output."""
    if seconds < NANO_SECONDS_DIFF:
        return f"{seconds * 1e9:.{precision}f} ns"
    if seconds < MICRO_SECONDS_DIFF:
        return f"{seconds * 1e6:.{precision}f} μs"
    if seconds < MILLI_SECONDS_DIFF:
        return f"{seconds * 1e3:.{precision}f} ms"
    return f"{seconds:.{precision}f} s"


@dataclass(frozen=True, slots=True)
class TimerInterval:
    """TODO."""

    name: str | None
    start: float
    duration: float


class Timer:
    """A utility class for timing code execution.

    Can be used as a context manager or as a standalone timer and
    includes support for intervals.
    """

    def __init__(self, name: str | None = None, interval_name: str | None = None) -> None:
        """Initialize the timer.

        Arguments:
        ---------
        name:
            Optional label for the timer.
        interval_name:
            Optional label for the first interval.
        """
        self.name: str | None = name

        self.__start_time: float | None = None
        self.__end_time: float | None = None
        self.__first_interval_name: str | None = interval_name
        self.__current_interval_name: str | None = interval_name
        self.__elapsed: float | None = None
        self.__intervals: list[TimerInterval] = []

    @property
    def start_time(self) -> float | None:
        """Return the start time."""
        return self.__start_time

    @property
    def end_time(self) -> float | None:
        """Return the end time."""
        return self.__end_time

    @property
    def total_duration(self) -> float:
        """TODO."""
        if self.__start_time is None:
            msg = "Timer must be started before total duration is available"
            raise RuntimeError(msg)

        if self.__elapsed is None:
            msg = (
                "Timer must be stopped with .stop() before total duration is available"
            )
            raise RuntimeError(msg)

        return self.__elapsed

    @property
    def elapsed(self) -> float:
        """TODO."""
        if self.__start_time is None:
            msg = "Timer must be started with .start() before elapsed time is available"
            raise RuntimeError(msg)

        return self.__elapsed or time.perf_counter() - self.__start_time

    @property
    def intervals(self) -> list[TimerInterval]:
        """Return saved intervals."""
        return self.__intervals.copy()

    def start(self) -> None:
        """Start the timer."""
        if self.__start_time is not None:
            msg = "Timer must be reset with .reset() before calling .start() again"
            raise RuntimeError(msg)

        self.__start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and calculate total elapsed time.

        Returns
        -------
        float: The elapsed time in seconds.
        """
        if self.__start_time is None:
            msg = "Timer must be started with .start() before calling .stop()"
            raise RuntimeError(msg)

        self.__elapsed = time.perf_counter() - self.__start_time
        self.__finish_interval()

        return self.__elapsed

    def reset(self, interval_name: str | None = None) -> None:
        """Reset the timer to an empty state."""
        self.__start_time = None
        self.__end_time = None
        self.__current_interval_name = interval_name or self.__first_interval_name
        self.__elapsed = None
        self.__intervals = []

    def start_interval(self, interval_name: str | None = None) -> None:
        """Finish an interval and start a new one.

        Arguments:
        ---------
        interval_name:
            A name or description for the next interval.
        """
        if self.__start_time is None:
            msg = "Timer must be started with .start() before starting a new interval"
            raise RuntimeError(msg)

        self.__finish_interval()
        self.__current_interval_name = interval_name

    def __finish_interval(self) -> None:
        """Finish an interval and start a new one.

        Arguments:
        ---------
        interval_name:
            A name or description for the next interval.
        """
        if self.__start_time is None:
            msg = "Timer must be started with .start() before starting a new interval"
            raise RuntimeError(msg)

        start = self.intervals[-1].start if self.__intervals else self.__start_time
        now = time.perf_counter()
        duration = now - start
        interval = TimerInterval(self.__current_interval_name, now, duration)
        self.__intervals.append(interval)

    def __enter__(self) -> Self:
        """Enter the runtime context related to this object."""
        self.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context related to this object."""
        self.stop()


# PERF: Only run when explicitly enabled (e.g., via environment variable).
def log_time(func: Callable | None = None, *, print_func: Callable = print) -> Callable:
    """TODO."""
    # TODO: Allow precision format + float precision (auto/user-specific).

    def actual_decorator(f: Callable) -> Callable:
        @wraps(f)
        def log_time_wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer() as timer:
                result = f(*args, **kwargs)

            print_func(
                f"{format_time(timer.total_duration)} during function '{f.__name__}'"
            )
            return result

        return log_time_wrapper

    return actual_decorator(func) if func else actual_decorator
