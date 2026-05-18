"""
CostLimit: A dollar-denominated rate limit for LLM budget control.

CostLimit extends Concurry's RateLimit so that "capacity" means "dollars"
and "usage" means "dollars spent per call." Internally, all values are stored
as microdollars (1 USD = 1,000,000 microdollars) because Concurry's RateLimit
uses integer arithmetic.

When the budget is exhausted, acquire() blocks (backpressure) until the time
window rolls over. The agent slows down rather than crashing.
"""

import math
import sys
from typing import Any, Optional, Union

from concurry import RateLimit, RateWindow

from .config import slowburn_config

MICRODOLLARS_PER_DOLLAR: int = 1_000_000

DEFAULT_COST_LIMIT_KEY: str = "cost_microdollars"

_MAX_MICRODOLLARS: int = sys.maxsize


def dollars_to_microdollars(usd: float) -> int:
    """Convert a dollar amount to microdollars (integer).

    Minimum return value is 1 microdollar to avoid zero-capacity limits.
    ``float('inf')`` maps to ``sys.maxsize`` (effectively unlimited).
    """
    if math.isinf(usd) and usd > 0:
        return _MAX_MICRODOLLARS
    return max(int(usd * MICRODOLLARS_PER_DOLLAR), 1)


def microdollars_to_dollars(micro: int) -> float:
    """Convert microdollars back to a dollar amount (float)."""
    return micro / MICRODOLLARS_PER_DOLLAR


class CostLimit(RateLimit):
    """Dollar-denominated rate limit that blocks when budget is exhausted.

    This is a RateLimit where "capacity" is a dollar budget expressed in
    microdollars and "usage" is the dollar cost of each LLM call. The
    TokenBucket algorithm (default) allows bursty spending while enforcing
    an average spend rate over the time window.

    Args:
        budget_usd: Maximum dollar spend allowed per window.
        window: Length of the budget window. Accepts a
            :class:`RateWindow` member, a string alias (``"daily"``,
            ``"hourly"``, ``"weekly"``, etc.), or a positive number of seconds.
            Defaults to ``slowburn_config.defaults.budget_usd_window``
            (``RateWindow.Daily``).
        key: Limit key used in acquire/update dicts. Defaults to
            ``"cost_microdollars"``.
        **kwargs: Additional arguments passed to RateLimit (e.g., algorithm).

    Example::

        from slowburn.limits import CostLimit
        from concurry import LimitSet, RateWindow

        limit_set = LimitSet(
            limits=[CostLimit(budget_usd=5.0, window=RateWindow.Daily)],
            mode="Asyncio",
            shared=True,
        )
    """

    def __init__(
        self,
        budget_usd: float,
        window: Optional[Union[RateWindow, str, int, float]] = None,
        window_seconds: Optional[float] = None,
        key: str = DEFAULT_COST_LIMIT_KEY,
        **kwargs: Any,
    ):
        # Backwards-compat: callers passing the deprecated ``window_seconds=``
        # kwarg get the value forwarded as ``window`` (which accepts numeric
        # seconds verbatim). Concurry's RateLimit emits the DeprecationWarning.
        if window is None and window_seconds is not None:
            window = window_seconds
        elif window is not None and window_seconds is not None:
            raise ValueError("Pass either `window` (preferred) or `window_seconds`, not both.")
        if window is None:
            window = slowburn_config.defaults.budget_usd_window
        capacity_microdollars = dollars_to_microdollars(budget_usd)
        super().__init__(
            key=key,
            capacity=capacity_microdollars,
            window=window,
            **kwargs,
        )
        self._budget_usd = budget_usd

    @property
    def budget_usd(self) -> float:
        """The original dollar budget for this limit."""
        return self._budget_usd

    @property
    def budget_microdollars(self) -> int:
        """The budget in microdollars (same as capacity)."""
        return self.capacity
