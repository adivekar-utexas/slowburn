"""Tests for CostLimit and microdollar conversion functions."""

import pytest
from concurry import LimitSet, RateLimit

from slowburn.limits import (
    DEFAULT_COST_LIMIT_KEY,
    CostLimit,
    dollars_to_microdollars,
    microdollars_to_dollars,
)


class TestDollarsToMicrodollars:
    """Test the dollars_to_microdollars conversion function."""

    def test_one_dollar(self) -> None:
        assert dollars_to_microdollars(1.0) == 1_000_000

    def test_five_dollars(self) -> None:
        assert dollars_to_microdollars(5.0) == 5_000_000

    def test_fractional_cents(self) -> None:
        assert dollars_to_microdollars(0.05) == 50_000

    def test_sub_cent(self) -> None:
        assert dollars_to_microdollars(0.000001) == 1

    def test_minimum_clamp(self) -> None:
        """Amounts below 1 microdollar are clamped to 1."""
        assert dollars_to_microdollars(0.0) == 1
        assert dollars_to_microdollars(0.0000001) == 1

    def test_large_budget(self) -> None:
        assert dollars_to_microdollars(50.0) == 50_000_000

    def test_infinity_maps_to_max(self) -> None:
        """float('inf') should produce a very large capacity, not crash."""
        import sys
        assert dollars_to_microdollars(float('inf')) == sys.maxsize

    def test_negative_returns_one(self) -> None:
        """Negative amounts are clamped to the minimum of 1."""
        assert dollars_to_microdollars(-1.0) == 1


class TestMicrodollarsToDollars:
    """Test the microdollars_to_dollars conversion function."""

    def test_one_million(self) -> None:
        assert microdollars_to_dollars(1_000_000) == 1.0

    def test_fifty_thousand(self) -> None:
        assert microdollars_to_dollars(50_000) == 0.05

    def test_one_microdollar(self) -> None:
        assert microdollars_to_dollars(1) == pytest.approx(0.000001)

    def test_zero(self) -> None:
        assert microdollars_to_dollars(0) == 0.0

    def test_roundtrip(self) -> None:
        """Converting to microdollars and back should be approximately equal."""
        for usd in [0.01, 0.50, 1.0, 5.0, 100.0]:
            micro = dollars_to_microdollars(usd)
            back = microdollars_to_dollars(micro)
            assert back == pytest.approx(usd, abs=1e-6)


class TestCostLimitCreation:
    """Test CostLimit instantiation and properties."""

    def test_basic_creation(self) -> None:
        cl = CostLimit(budget_usd=5.0, window_seconds=86400)
        assert cl.key == DEFAULT_COST_LIMIT_KEY
        assert cl.capacity == 5_000_000
        assert cl.window_seconds == 86400
        assert cl.budget_usd == 5.0
        assert cl.budget_microdollars == 5_000_000

    def test_custom_key(self) -> None:
        cl = CostLimit(budget_usd=1.0, key="my_budget")
        assert cl.key == "my_budget"

    def test_hourly_window(self) -> None:
        cl = CostLimit(budget_usd=2.0, window_seconds=3600)
        assert cl.window_seconds == 3600

    def test_small_budget(self) -> None:
        """Very small budgets should still have capacity >= 1."""
        cl = CostLimit(budget_usd=0.000001)
        assert cl.capacity >= 1

    def test_is_rate_limit_subclass(self) -> None:
        cl = CostLimit(budget_usd=5.0)
        assert isinstance(cl, RateLimit)

    def test_infinite_budget(self) -> None:
        """float('inf') budget should create a CostLimit with very large capacity."""
        import sys
        cl = CostLimit(budget_usd=float('inf'))
        assert cl.budget_usd == float('inf')
        assert cl.capacity == sys.maxsize


class TestCostLimitWithLimitSet:
    """Test that CostLimit works correctly inside a Concurry LimitSet."""

    def test_acquire_and_update(self) -> None:
        """Basic acquire/update cycle with CostLimit inside a LimitSet.

        Steps:
        1. Create a CostLimit with $5 budget.
        2. Wrap in a thread-mode LimitSet.
        3. Acquire 100,000 microdollars (= $0.10).
        4. Update with actual usage of 50,000 microdollars (= $0.05).
        5. Verify the acquisition succeeds (no exception).
        """
        cl = CostLimit(budget_usd=5.0, window_seconds=3600)
        ls = LimitSet(limits=[cl], mode="thread", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 100_000}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 50_000})

    def test_multiple_acquisitions(self) -> None:
        """Multiple sequential acquisitions should all succeed within budget.

        Steps:
        1. Create a CostLimit with $1 budget (1,000,000 microdollars).
        2. Make 10 acquisitions of 50,000 microdollars each (= $0.50 total).
        3. Each should succeed without blocking.
        """
        cl = CostLimit(budget_usd=1.0, window_seconds=3600)
        ls = LimitSet(limits=[cl], mode="thread", shared=True)

        for _ in range(10):
            with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 50_000}) as acq:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: 50_000})

    def test_try_acquire_fails_over_budget(self) -> None:
        """try_acquire should fail when requesting more than available capacity.

        Steps:
        1. Create a CostLimit with a very small budget ($0.01 = 10,000 microdollars).
        2. Acquire the full budget.
        3. A subsequent try_acquire for more should fail (not successful).
        """
        cl = CostLimit(budget_usd=0.01, window_seconds=3600)
        ls = LimitSet(limits=[cl], mode="thread", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 10_000}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 10_000})

        result = ls.try_acquire(requested={DEFAULT_COST_LIMIT_KEY: 10_000})
        assert not result.successful

    def test_works_with_asyncio_mode(self) -> None:
        """CostLimit should also work with asyncio-mode LimitSet."""
        cl = CostLimit(budget_usd=5.0, window_seconds=3600)
        ls = LimitSet(limits=[cl], mode="asyncio", shared=True)

        with ls.acquire(requested={DEFAULT_COST_LIMIT_KEY: 1_000}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 500})

    def test_combined_with_other_limits(self) -> None:
        """CostLimit works alongside RateLimit in the same LimitSet.

        Steps:
        1. Create a LimitSet with both CostLimit and a token RateLimit.
        2. Acquire both in one call.
        3. Update both with actual usage.
        """
        cl = CostLimit(budget_usd=5.0, window_seconds=3600)
        tl = RateLimit(key="tokens", window_seconds=60, capacity=10_000)
        ls = LimitSet(limits=[cl, tl], mode="thread", shared=True)

        with ls.acquire(
            requested={DEFAULT_COST_LIMIT_KEY: 1_000, "tokens": 500}
        ) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 800, "tokens": 400})
