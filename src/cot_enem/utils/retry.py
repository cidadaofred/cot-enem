"""Small retry primitive with exponential backoff and injectable sleep."""

from collections.abc import Callable
import time
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except retry_on:
            if attempt == max_attempts:
                raise
            sleep(delay)
            delay *= multiplier
    raise AssertionError("unreachable")
