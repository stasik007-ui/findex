import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

timing_logger = logging.getLogger("findex.timing")


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log wall-clock execution time of a function call in milliseconds."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timing_logger.info("%s took %.2f ms", fn.__qualname__, elapsed_ms)

    return wrapper
