"""Edge-case tests for SlowBurn's float-dollar ``CostLimit``.

``CostLimit`` stores dollars natively as a ``float``. These tests
red-team that:

- Sub-cent budgets and sub-cent usage round-trip without loss of precision.
- ``inf`` budget admits any positive usage and signals "no enforcement"
  via the ``budget_usd`` property.
- Warning messages and reporter logs use natural dollar formatting.
- The ``cost_usd`` key is what callers must use in ``acquire`` / ``update``.
- Construction-time validation rejects zero / negative / NaN.
"""

import math

import pytest
from concurry import LimitSet, RateLimitAlgorithm, RateWindow

from slowburn import CostLimit, DEFAULT_COST_LIMIT_KEY


class TestCostLimitConstruction:
    def test_default_key_is_cost_usd(self) -> None:
        cl = CostLimit(budget_usd=5.0, window=RateWindow.Daily)
        assert cl.key == "cost_usd"
        assert DEFAULT_COST_LIMIT_KEY == "cost_usd"

    def test_capacity_is_budget_usd(self) -> None:
        cl = CostLimit(budget_usd=0.05, window=RateWindow.Hourly)
        assert cl.capacity == 0.05
        assert cl.budget_usd == 0.05

    def test_subcent_budget_preserved(self) -> None:
        """A budget like $0.0001 must be stored as 0.0001, not rounded to 0."""
        cl = CostLimit(budget_usd=0.0001, window=RateWindow.Hourly)
        assert cl.budget_usd == 0.0001
        # Sanity: capacity is float-valued.
        assert isinstance(cl.capacity, float)

    def test_infinite_budget(self) -> None:
        """``budget_usd=inf`` must be admitted and signals "unlimited"."""
        cl = CostLimit(budget_usd=float("inf"), window=RateWindow.Daily)
        assert math.isinf(cl.budget_usd)

    def test_rejects_zero_budget(self) -> None:
        """Zero budget would deadlock acquire forever."""
        with pytest.raises((ValueError, Exception)):
            CostLimit(budget_usd=0.0, window=RateWindow.Daily)

    def test_rejects_negative_budget(self) -> None:
        with pytest.raises((ValueError, Exception)):
            CostLimit(budget_usd=-1.0, window=RateWindow.Daily)

    def test_rejects_nan_budget(self) -> None:
        with pytest.raises((ValueError, Exception)):
            CostLimit(budget_usd=float("nan"), window=RateWindow.Daily)

    def test_rejects_negative_infinity(self) -> None:
        with pytest.raises((ValueError, Exception)):
            CostLimit(budget_usd=float("-inf"), window=RateWindow.Daily)

    def test_window_required(self) -> None:
        """``window`` is positional/required; no library default."""
        with pytest.raises(TypeError):
            CostLimit(budget_usd=5.0)  # type: ignore[call-arg]

    def test_custom_key(self) -> None:
        cl = CostLimit(budget_usd=1.0, window=RateWindow.Daily, key="my_budget")
        assert cl.key == "my_budget"


class TestCostLimitAcquireAndUpdate:
    """Acquire/update cycle with float dollar amounts."""

    def test_acquire_typical_cost(self) -> None:
        """A real LLM call costs around $0.000084 — must acquire fine."""
        cl = CostLimit(budget_usd=1.0, window=3600, algorithm=RateLimitAlgorithm.GCRA)
        ls = LimitSet(limits=[cl])
        with ls.acquire(requested={"cost_usd": 0.000084}) as acq:
            acq.update(usage={"cost_usd": 0.000084})

    def test_acquire_zero_cost_call(self) -> None:
        """A free model (cost=0) must still acquire cleanly."""
        cl = CostLimit(budget_usd=1.0, window=3600, algorithm=RateLimitAlgorithm.GCRA)
        ls = LimitSet(limits=[cl])
        with ls.acquire(requested={"cost_usd": 0.0}) as acq:
            acq.update(usage={"cost_usd": 0.0})

    def test_acquire_above_capacity_raises(self) -> None:
        """Estimated cost above capacity is unfulfillable; raise rather
        than blocking forever."""
        cl = CostLimit(budget_usd=0.0001, window=3600, algorithm=RateLimitAlgorithm.GCRA)
        ls = LimitSet(limits=[cl])
        with pytest.raises(ValueError, match="exceeds capacity"):
            with ls.acquire(requested={"cost_usd": 0.001}) as acq:
                acq.update(usage={"cost_usd": 0.001})

    def test_validate_usage_warns_in_dollars(self, caplog) -> None:
        """When ``used > requested``, the warning must show dollar amounts
        directly."""
        import logging

        cl = CostLimit(budget_usd=1.0, window=3600, algorithm=RateLimitAlgorithm.GCRA)
        with caplog.at_level(logging.WARNING):
            cl.validate_usage(requested=0.000050, used=0.000100)
        msg = caplog.text
        # Both values appear in the warning text (formatted as float).
        # Python may render 0.00005 as ``5e-05``; both are acceptable.
        assert "0.0001" in msg or "1e-04" in msg
        assert "5e-05" in msg or "0.000050" in msg or "0.00005" in msg
        assert "cost_usd" in msg


class TestCostLimitInfiniteBudget:
    """``budget_usd=inf`` is the SlowBurn default for "no enforcement"."""

    def test_inf_admits_any_acquire(self) -> None:
        cl = CostLimit(budget_usd=float("inf"), window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        ls = LimitSet(limits=[cl])
        # Even a giant acquire must succeed against inf capacity.
        with ls.acquire(requested={"cost_usd": 1e9}) as acq:
            acq.update(usage={"cost_usd": 1e9})

    def test_inf_budget_property(self) -> None:
        cl = CostLimit(budget_usd=float("inf"), window=RateWindow.Daily)
        assert math.isinf(cl.budget_usd)


class TestCostLimitParamsSignature:
    """Two CostLimits with identical (budget, window) must produce the
    same ``params_signature``. SlowBurn's pool-builder relies on this for
    shared-limit caching across endpoints."""

    def test_identical_params_match(self) -> None:
        a = CostLimit(budget_usd=5.0, window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        b = CostLimit(budget_usd=5.0, window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        assert a.params_signature() == b.params_signature()

    def test_different_budgets_differ(self) -> None:
        a = CostLimit(budget_usd=5.0, window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        b = CostLimit(budget_usd=10.0, window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        assert a.params_signature() != b.params_signature()

    def test_inf_budget_signature_stable(self) -> None:
        """Two infinite-budget CostLimits should share a signature so the
        SlowBurn pool builder dedupes them across endpoints."""
        a = CostLimit(budget_usd=float("inf"), window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        b = CostLimit(budget_usd=float("inf"), window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        assert a.params_signature() == b.params_signature()
        assert "inf" in a.params_signature()

    def test_subcent_budget_signature_distinct(self) -> None:
        """A $0.0001 budget produces a signature distinct from $0.001 — i.e.,
        precision is preserved, not rounded to int."""
        a = CostLimit(budget_usd=0.0001, window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        b = CostLimit(budget_usd=0.001, window=RateWindow.Daily, algorithm=RateLimitAlgorithm.GCRA)
        assert a.params_signature() != b.params_signature()
