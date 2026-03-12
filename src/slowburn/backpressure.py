"""
Backpressure logging for SlowBurn.

When acquire() blocks because the budget is exhausted, SlowBurn logs a
warning so the user knows the system is slowing down (not hanging).

Enabled by default. Disable with ``slowburn.set_backpressure_warnings(False)``.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

logger = logging.getLogger("slowburn.backpressure")

_warnings_enabled: bool = True

BACKPRESSURE_THRESHOLD_SECONDS: float = 0.5


def set_backpressure_warnings(enabled: bool) -> None:
    """Enable or disable backpressure warning messages.

    When enabled (default), SlowBurn logs a WARNING whenever an acquire()
    call blocks for more than 0.5 seconds due to budget exhaustion.
    """
    global _warnings_enabled
    _warnings_enabled = enabled


def backpressure_warnings_enabled() -> bool:
    return _warnings_enabled


@contextmanager
def timed_acquire(limit_set: Any, requested: Dict[str, int], context: str = ""):
    """Context manager that wraps limit_set.acquire() with backpressure timing.

    If acquire() blocks for longer than BACKPRESSURE_THRESHOLD_SECONDS,
    logs a warning with the wait duration and context.

    Usage::

        with timed_acquire(limit_set, requested, context="step 5") as acq:
            acq.update(usage={...})

    Yields the LimitSetAcquisition object.
    """
    start = time.monotonic()
    acq = limit_set.acquire(requested=requested)
    elapsed = time.monotonic() - start

    if _warnings_enabled and elapsed > BACKPRESSURE_THRESHOLD_SECONDS:
        cost_key = next(
            (k for k in requested if "cost" in k.lower()),
            next(iter(requested), "?"),
        )
        requested_amount = requested[cost_key]
        if isinstance(requested_amount, (int, float)) and requested_amount > 0:
            dollar_amount = requested_amount / 1_000_000
            amount_str = f"~${dollar_amount:.4f}"
        else:
            amount_str = str(requested_amount)
        logger.warning(
            f"SlowBurn BACKPRESSURE: blocked for {elapsed:.1f}s waiting for "
            f"budget. Requested {amount_str}. {context}"
        )
        if not logger.handlers and not logging.getLogger().handlers:
            import sys
            print(
                f"  ** SlowBurn: backpressure active — waited {elapsed:.1f}s "
                f"for budget (requested {amount_str}, {context}) **",
                file=sys.stderr,
            )

    try:
        yield acq
    except BaseException:
        raise
