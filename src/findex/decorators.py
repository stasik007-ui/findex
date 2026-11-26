from collections.abc import Callable
import functools
import logging
import time

log = logging.getLogger(__name__)


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log function execution duration in milliseconds at DEBUG level."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.debug("%s took %.2f ms", fn.__qualname__, elapsed_ms)

    return wrapper