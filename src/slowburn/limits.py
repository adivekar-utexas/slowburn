"""
CostLimit: A dollar-denominated rate limit for LLM budget control.

CostLimit extends Concurry's ``RateLimit`` so that ``capacity`` means
"dollars" and ``usage`` means "dollars spent per call." With Concurry's
``confloat(gt=0)`` capacity field, dollars are stored directly as
``float`` — no boundary conversion needed.

When the budget is exhausted, ``acquire()`` blocks (backpressure) until
the time window rolls over. The agent slows down rather than crashing.
"""

from typing import Any, Dict

from concurry import RateLimit, RateLimitAlgorithm
from concurry.utils import _NO_ARG

DEFAULT_COST_LIMIT_KEY: str = "cost_usd"


class CostLimit(RateLimit):
    """Dollar-denominated rate limit that blocks when budget is exhausted.

    This is a ``RateLimit`` whose ``capacity`` is a dollar budget stored
    as a positive ``float``. ``usage`` is the dollar cost of each LLM
    call. The TokenBucket / GCRA algorithms (the only ones that admit
    fractional acquires) are the only valid choices; CostLimit defaults
    to GCRA so dollar amounts can be acquired without rounding.

    Construction kwargs (all keyword-only):

        budget_usd: Maximum dollar spend allowed per window. Must be
            positive; ``float('inf')`` is admitted to mean "unlimited"
            (the worker side recognizes infinite budgets and skips cost
            enforcement entirely). Stored internally as ``capacity``.
        window: Length of the budget window. Accepts a
            :class:`RateWindow` member, a string alias (``"daily"``,
            ``"hourly"``, ``"weekly"``, etc.), or a positive number of
            seconds. Required — there is no library default.
        key: Limit key used in acquire/update dicts. Defaults to
            ``"cost_usd"``.
        algorithm: Rate-limiter algorithm. Defaults to GCRA, the most
            precise of the two algorithms that admit fractional
            (sub-cent) acquires. Pass ``RateLimitAlgorithm.TokenBucket``
            for bursty workloads. Other algorithms (SlidingWindow,
            FixedWindow, LeakyBucket) reject fractional cost values.

    Like every other ``Typed`` subclass in concurry / slowburn, this
    class does **not** override ``__init__``. The ``budget_usd`` →
    ``capacity`` translation, the ``key`` default, and the GCRA default
    all happen in :meth:`pre_initialize` so they apply uniformly to
    every construction path: direct kwargs, ``model_validate``,
    ``model_copy``, etc. Use ``capacity=`` and ``budget_usd=``
    interchangeably (both names are accepted as input; ``capacity`` is
    the canonical stored field).

    Example::

        from slowburn.limits import CostLimit
        from concurry import LimitSet, RateWindow

        limit_set = LimitSet(
            limits=[CostLimit(budget_usd=5.0, window=RateWindow.Daily)],
            mode="Asyncio",
            shared=True,
        )
    """

    @classmethod
    def pre_initialize(cls, data: Dict[str, Any]) -> None:
        """Normalize ``CostLimit``-specific kwargs before pydantic validation.

        - Map ``budget_usd=<float>`` to ``capacity=<float>`` (the canonical
          field on the parent ``RateLimit``). Passing both is a contradiction
          and raises ``ValueError``.
        - Default ``key`` to ``"cost_usd"`` if not specified.
        - Default ``algorithm`` to ``RateLimitAlgorithm.GCRA`` if the user
          did not pick one. Concurry's ``RateLimit`` field-default for
          ``algorithm`` is the ``_NO_ARG`` sentinel (which ``post_initialize``
          would otherwise replace with ``slowburn_config.defaults`` →
          SlidingWindow). Fractional dollar acquires are only well-defined
          for GCRA / TokenBucket, so for ``CostLimit`` we override that
          fallback to GCRA.

        ``RateLimit.pre_initialize`` (window / per / window_seconds
        normalization) runs automatically before this hook because
        ``morphic.Typed`` chains ``pre_initialize`` in MRO order from base
        to derived. So no ``super().pre_initialize(data)`` call is needed
        or correct.
        """
        if not isinstance(data, dict):
            return

        # Translate budget_usd -> capacity (the stored field on RateLimit).
        if "budget_usd" in data:
            budget_usd = data.pop("budget_usd")
            if "capacity" in data and data["capacity"] is not None and budget_usd is not None:
                raise ValueError(
                    "CostLimit: pass either `budget_usd=` or `capacity=`, not "
                    "both. They are aliases for the same dollar-denominated "
                    "capacity field."
                )
            if budget_usd is not None:
                data["capacity"] = budget_usd

        # Default the key. ``setdefault`` is correct here because ``key`` has
        # no field-level default on ``RateLimit`` (it's a required ``str``);
        # if it's missing from ``data``, it really is missing.
        data.setdefault("key", DEFAULT_COST_LIMIT_KEY)

        # Default the algorithm. We can't use ``setdefault`` because pydantic
        # already filled ``algorithm`` with the ``_NO_ARG`` sentinel from the
        # parent class's field default before this hook runs. Replace it only
        # when the user didn't pick a real algorithm.
        if data.get("algorithm", _NO_ARG) is _NO_ARG:
            data["algorithm"] = RateLimitAlgorithm.GCRA

    @property
    def budget_usd(self) -> float:
        """The dollar budget for this limit (alias for ``capacity``)."""
        return float(self.capacity)
