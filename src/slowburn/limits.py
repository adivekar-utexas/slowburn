"""
CostLimit: A dollar-denominated rate limit for LLM budget control.

CostLimit extends Concurry's RateLimit so that "capacity" means "dollars"
and "usage" means "dollars spent per call." Internally, all values are stored
as microdollars (1 USD = 1,000,000 microdollars) because Concurry's RateLimit
uses integer arithmetic.

When the budget is exhausted, acquire() blocks (backpressure) until the time
window rolls over. The agent slows down rather than crashing.
"""

from typing import Optional

from concurry import RateLimit

from .config import _NO_ARG, is_no_arg, slowburn_config

MICRODOLLARS_PER_DOLLAR: int = 1_000_000

DEFAULT_COST_LIMIT_KEY: str = "cost_microdollars"


def dollars_to_microdollars(usd: float) -> int:
    """Convert a dollar amount to microdollars (integer).

    Minimum return value is 1 microdollar to avoid zero-capacity limits.
    """
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
        window_seconds: Length of the budget window in seconds.
            Common values: 3600 (hourly), 86400 (daily).
        key: Limit key used in acquire/update dicts.
            Defaults to "cost_usd".
        **kwargs: Additional arguments passed to RateLimit (e.g., algorithm).

    Example::

        from slowburn.limits import CostLimit
        from concurry import LimitSet

        limit_set = LimitSet(
            limits=[CostLimit(budget_usd=5.0, window_seconds=86400)],
            mode="asyncio",
            shared=True,
        )

        # Inside an async worker method:
        with self.limits.acquire(requested={"cost_usd": estimated_microdollars}) as acq:
            response = await litellm.acompletion(...)
            acq.update(usage={"cost_usd": actual_microdollars})
    """

    def __init__(
        self,
        budget_usd: float,
        window_seconds: Optional[float] = None,
        key: str = DEFAULT_COST_LIMIT_KEY,
        **kwargs,
    ):
        if window_seconds is None:
            window_seconds = slowburn_config.defaults.default_window_seconds
        capacity_microdollars = dollars_to_microdollars(budget_usd)
        super().__init__(
            key=key,
            capacity=capacity_microdollars,
            window_seconds=window_seconds,
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
