"""Tests for CostLimit's dollar-native float-capacity behavior.

``CostLimit.capacity`` *is* dollars stored as a positive ``float``.
``budget_usd`` is a property that returns ``float(self.capacity)``.

These tests cover:

- Construction (capacity / window / budget_usd).
- Acquire/update inside a ``LimitSet``.
- Combined with other rate limits.
"""

import math

import pytest
from concurry import LimitSet, RateLimit

from slowburn.limits import DEFAULT_COST_LIMIT_KEY, CostLimit


class TestCostLimitCreation:
    """Test CostLimit instantiation and properties."""

    def test_basic_creation(self) -> None:
        cl = CostLimit(budget_usd=5.0, window=86400)
        assert cl.key == DEFAULT_COST_LIMIT_KEY
        # capacity is dollars-as-float.
        assert cl.capacity == 5.0
        assert cl.window == 86400
        assert cl.budget_usd == 5.0

    def test_custom_key(self) -> None:
        cl = CostLimit(budget_usd=1.0, window=86400, key="my_budget")
        assert cl.key == "my_budget"

    def test_hourly_window(self) -> None:
        cl = CostLimit(budget_usd=2.0, window=3600)
        assert cl.window == 3600

    def test_subcent_budget(self) -> None:
        """Sub-cent budgets are preserved as-is (no rounding to int)."""
        cl = CostLimit(budget_usd=0.000001, window=86400)
        assert cl.capacity == 0.000001
        assert cl.budget_usd == 0.000001

    def test_is_rate_limit_subclass(self) -> None:
        cl = CostLimit(budget_usd=5.0, window=86400)
        assert isinstance(cl, RateLimit)

    def test_infinite_budget(self) -> None:
        """``float('inf')`` budget produces a CostLimit with infinite
        capacity (the SlowBurn default for "no enforcement")."""
        cl = CostLimit(budget_usd=float("inf"), window=86400)
        assert math.isinf(cl.budget_usd)
        assert math.isinf(cl.capacity)


class TestCostLimitWithLimitSet:
    """Test that CostLimit works correctly inside a Concurry LimitSet."""

    def test_acquire_and_update(self) -> None:
        """Basic acquire/update cycle with CostLimit inside a LimitSet.

        Steps:
        1. Create a CostLimit with $5 budget.
        2. Wrap in a thread-mode LimitSet.
        3. Acquire $0.10 (a typical mid-size LLM call).
        4. Update with actual usage of $0.05.
        5. Verify the acquisition succeeds (no exception).
        """
        cl = CostLimit(budget_usd=5.0, window=3600)
        ls = LimitSet(limits=[cl], mode="Threads", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 0.10}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 0.05})

    def test_multiple_acquisitions(self) -> None:
        """Multiple sequential acquisitions should all succeed within budget.

        Steps:
        1. Create a CostLimit with $1 budget.
        2. Make 10 acquisitions of $0.05 each (= $0.50 total).
        3. Each should succeed without blocking.
        """
        cl = CostLimit(budget_usd=1.0, window=3600)
        ls = LimitSet(limits=[cl], mode="Threads", shared=True)

        for _ in range(10):
            with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 0.05}) as acq:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: 0.05})

    def test_try_acquire_fails_over_budget(self) -> None:
        """``try_acquire`` should fail when requesting more than available
        capacity.

        Steps:
        1. Create a CostLimit with a very small budget ($0.01).
        2. Acquire the full budget.
        3. A subsequent ``try_acquire`` for more should fail.
        """
        cl = CostLimit(budget_usd=0.01, window=3600)
        ls = LimitSet(limits=[cl], mode="Threads", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 0.01}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 0.01})

        result = ls.try_acquire(requested={DEFAULT_COST_LIMIT_KEY: 0.01})
        assert not result.successful

    def test_works_with_asyncio_mode(self) -> None:
        """CostLimit should also work with asyncio-mode LimitSet."""
        cl = CostLimit(budget_usd=5.0, window=3600)
        ls = LimitSet(limits=[cl], mode="Asyncio", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 0.001}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 0.0005})

    def test_combined_with_other_limits(self) -> None:
        """CostLimit works alongside RateLimit in the same LimitSet.

        Steps:
        1. Create a LimitSet with both CostLimit and a token RateLimit.
        2. Acquire both in one call.
        3. Update both with actual usage.
        """
        cl = CostLimit(budget_usd=5.0, window=3600)
        tl = RateLimit(key="tokens", window=60, capacity=10_000)
        ls = LimitSet(limits=[cl, tl], mode="Threads", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 0.001, "tokens": 500}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 0.0008, "tokens": 400})
