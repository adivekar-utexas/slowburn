"""
CostLimit: A dollar-denominated rate limit for LLM budget control.

CostLimit extends Concurry's ``RateLimit`` so that ``capacity`` means
"dollars" and ``usage`` means "dollars spent per call." With Concurry's
``confloat(gt=0)`` capacity field, dollars are stored directly as
``float`` — no boundary conversion needed.

When the budget is exhausted, ``acquire()`` blocks (backpressure) until
the time window rolls over. The agent slows down rather than crashing.
"""

from typing import Any, Union

from concurry import RateLimit, RateLimitAlgorithm, RateWindow

DEFAULT_COST_LIMIT_KEY: str = "cost_usd"


class CostLimit(RateLimit):
    """Dollar-denominated rate limit that blocks when budget is exhausted.

    This is a ``RateLimit`` whose ``capacity`` is a dollar budget stored
    as a positive ``float``. ``usage`` is the dollar cost of each LLM
    call. The TokenBucket / GCRA algorithms (the only ones that admit
    fractional acquires) are the only valid choices; CostLimit defaults
    to GCRA so dollar amounts can be acquired without rounding.

    Args:
        budget_usd: Maximum dollar spend allowed per window. Must be
            positive; ``float('inf')`` is admitted to mean "unlimited"
            (the worker side recognizes infinite budgets and skips cost
            enforcement entirely).
        window: Length of the budget window. Accepts a
            :class:`RateWindow` member, a string alias (``"daily"``,
            ``"hourly"``, ``"weekly"``, etc.), or a positive number of
            seconds. Required — there is no library default; the user
            must always specify the window explicitly. Use shorthand
            kwargs like ``budget_per_day=5.0`` on
            :class:`slowburn.SlowBurnLimits` for a more readable API.
        key: Limit key used in acquire/update dicts. Defaults to
            ``"cost_usd"``.
        algorithm: Rate-limiter algorithm. Defaults to GCRA, which is the
            most precise of the two algorithms that admit fractional
            (sub-cent) acquires. Pass ``RateLimitAlgorithm.TokenBucket``
            for bursty workloads. Other algorithms (SlidingWindow,
            FixedWindow, LeakyBucket) reject fractional cost values.
        **kwargs: Additional arguments passed to RateLimit.

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
        window: Union[RateWindow, str, int, float],
        key: str = DEFAULT_COST_LIMIT_KEY,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.GCRA,
        **kwargs: Any,
    ):
        super().__init__(
            key=key,
            capacity=budget_usd,
            window=window,
            algorithm=algorithm,
            **kwargs,
        )

    @property
    def budget_usd(self) -> float:
        """The dollar budget for this limit (alias for ``capacity``)."""
        return float(self.capacity)
