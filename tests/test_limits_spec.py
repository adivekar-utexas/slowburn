"""Unit tests for ``slowburn.limits_spec.SlowBurnLimits``.

These tests cover the shorthand-kwarg parser, slot conflict detection, the
``max_`` prefix, and the library-default factory. They are pure unit tests
and do not require any LLM or network access.
"""

from __future__ import annotations

from typing import List

import pytest
from concurry import RateLimit, RateWindow

from slowburn import CostLimit
from slowburn.limits_spec import (
    SLOT_TO_LIMIT_KEY,
    SlowBurnLimits,
    default_slowburn_limits,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _windows(limits: List) -> List[float]:
    """Extract the resolved-seconds window from each limit, preserving order."""
    return [float(limit.window) for limit in limits]


def _capacities(limits: List) -> List[int]:
    return [int(limit.capacity) for limit in limits]


# ----------------------------------------------------------------------------
# Empty / inheritance
# ----------------------------------------------------------------------------


class TestEmpty:
    def test_empty_yields_all_none(self) -> None:
        sl = SlowBurnLimits()
        assert sl.requests is None
        assert sl.input_tokens is None
        assert sl.output_tokens is None
        assert sl.budget is None
        assert sl.concurrency is None

    def test_explicit_none_equivalent_to_empty(self) -> None:
        sl = SlowBurnLimits(requests=None, budget=None, concurrency=None)
        assert sl.requests is None
        assert sl.budget is None
        assert sl.concurrency is None


# ----------------------------------------------------------------------------
# Compact-letter shorthands (rpm/rps/rph/rpd/rpw)
# ----------------------------------------------------------------------------


class TestCompactRequestShorthands:
    @pytest.mark.parametrize(
        "kwarg,expected_seconds",
        [
            ("rps", 1.0),
            ("rpm", 60.0),
            ("rph", 3600.0),
            ("rpd", 86400.0),
            ("rpw", 604800.0),
        ],
    )
    def test_each_compact_form(self, kwarg: str, expected_seconds: float) -> None:
        sl = SlowBurnLimits(**{kwarg: 100})
        assert sl.requests is not None
        assert len(sl.requests) == 1
        assert float(sl.requests[0].window) == expected_seconds
        assert sl.requests[0].capacity == 100
        assert sl.requests[0].key == SLOT_TO_LIMIT_KEY["requests"]

    def test_max_prefix_stripped(self) -> None:
        sl = SlowBurnLimits(max_rpm=300)
        assert sl.requests is not None
        assert _windows(sl.requests) == [60.0]
        assert _capacities(sl.requests) == [300]

    def test_multiple_compact_forms_merge(self) -> None:
        sl = SlowBurnLimits(rpm=300, rpd=10_000)
        assert sl.requests is not None
        # Order is insertion order (rpm before rpd).
        assert _windows(sl.requests) == [60.0, 86400.0]
        assert _capacities(sl.requests) == [300, 10_000]


# ----------------------------------------------------------------------------
# Verbose request shorthands (requests_per_*)
# ----------------------------------------------------------------------------


class TestVerboseRequestShorthands:
    @pytest.mark.parametrize(
        "kwarg,expected_seconds",
        [
            ("requests_per_second", 1.0),
            ("requests_per_minute", 60.0),
            ("requests_per_hour", 3600.0),
            ("requests_per_day", 86400.0),
            ("requests_per_week", 604800.0),
        ],
    )
    def test_each_verbose_form(self, kwarg: str, expected_seconds: float) -> None:
        sl = SlowBurnLimits(**{kwarg: 100})
        assert sl.requests is not None
        assert _windows(sl.requests) == [expected_seconds]


# ----------------------------------------------------------------------------
# Input/output token shorthands
# ----------------------------------------------------------------------------


class TestTokenShorthands:
    @pytest.mark.parametrize(
        "kwarg,expected_slot,expected_seconds",
        [
            # input_tokens
            ("itps", "input_tokens", 1.0),
            ("itpm", "input_tokens", 60.0),
            ("itph", "input_tokens", 3600.0),
            ("itpd", "input_tokens", 86400.0),
            ("itpw", "input_tokens", 604800.0),
            ("input_tps", "input_tokens", 1.0),
            ("input_tpm", "input_tokens", 60.0),
            ("input_tph", "input_tokens", 3600.0),
            ("input_tpd", "input_tokens", 86400.0),
            ("input_tpw", "input_tokens", 604800.0),
            ("input_tokens_per_second", "input_tokens", 1.0),
            ("input_tokens_per_minute", "input_tokens", 60.0),
            ("input_tokens_per_hour", "input_tokens", 3600.0),
            ("input_tokens_per_day", "input_tokens", 86400.0),
            ("input_tokens_per_week", "input_tokens", 604800.0),
            # output_tokens
            ("otps", "output_tokens", 1.0),
            ("otpm", "output_tokens", 60.0),
            ("otph", "output_tokens", 3600.0),
            ("otpd", "output_tokens", 86400.0),
            ("otpw", "output_tokens", 604800.0),
            ("output_tps", "output_tokens", 1.0),
            ("output_tpm", "output_tokens", 60.0),
            ("output_tph", "output_tokens", 3600.0),
            ("output_tpd", "output_tokens", 86400.0),
            ("output_tpw", "output_tokens", 604800.0),
            ("output_tokens_per_second", "output_tokens", 1.0),
            ("output_tokens_per_minute", "output_tokens", 60.0),
            ("output_tokens_per_hour", "output_tokens", 3600.0),
            ("output_tokens_per_day", "output_tokens", 86400.0),
            ("output_tokens_per_week", "output_tokens", 604800.0),
        ],
    )
    def test_token_shorthand_routes_to_correct_slot(
        self, kwarg: str, expected_slot: str, expected_seconds: float
    ) -> None:
        sl = SlowBurnLimits(**{kwarg: 50_000})
        slot_value = getattr(sl, expected_slot)
        assert slot_value is not None
        assert _windows(slot_value) == [expected_seconds]
        assert slot_value[0].key == SLOT_TO_LIMIT_KEY[expected_slot]
        # Other slots remain None.
        for other in ("requests", "input_tokens", "output_tokens"):
            if other != expected_slot:
                assert getattr(sl, other) is None


# ----------------------------------------------------------------------------
# Budget shorthand
# ----------------------------------------------------------------------------


class TestBudgetShorthand:
    @pytest.mark.parametrize(
        "kwarg,expected_seconds",
        [
            ("budget_per_second", 1.0),
            ("budget_per_minute", 60.0),
            ("budget_per_hour", 3600.0),
            ("budget_per_day", 86400.0),
            ("budget_per_week", 604800.0),
        ],
    )
    def test_budget_per_window_creates_costlimit(self, kwarg: str, expected_seconds: float) -> None:
        sl = SlowBurnLimits(**{kwarg: 5.0})
        assert sl.budget is not None
        assert len(sl.budget) == 1
        assert isinstance(sl.budget[0], CostLimit)
        assert float(sl.budget[0].window) == expected_seconds

    def test_max_prefix_works_for_budget_too(self) -> None:
        sl = SlowBurnLimits(max_budget_per_day=10.0)
        assert sl.budget is not None
        assert _windows(sl.budget) == [86400.0]


# ----------------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrency_int(self) -> None:
        sl = SlowBurnLimits(concurrency=10)
        assert sl.concurrency == 10

    def test_max_concurrency(self) -> None:
        sl = SlowBurnLimits(max_concurrency=10)
        assert sl.concurrency == 10

    def test_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            SlowBurnLimits(concurrency=0)
        with pytest.raises(ValueError):
            SlowBurnLimits(concurrency=-1)


# ----------------------------------------------------------------------------
# Conflict detection
# ----------------------------------------------------------------------------


class TestConflicts:
    def test_two_shorthands_same_window_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            SlowBurnLimits(rpm=300, requests_per_minute=500)

    def test_canonical_and_shorthand_same_window_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            SlowBurnLimits(
                rpm=300,
                requests=[RateLimit(key="requests", capacity=500, window="minute")],
            )

    def test_compact_and_verbose_same_window_raises(self) -> None:
        # itpm vs input_tokens_per_minute → both → 60s on input_tokens
        with pytest.raises(ValueError, match="duplicate"):
            SlowBurnLimits(itpm=100, input_tokens_per_minute=200)

    def test_different_windows_on_same_slot_merge(self) -> None:
        # No conflict: 60s vs 86400s
        sl = SlowBurnLimits(rpm=300, rpd=10_000)
        assert _windows(sl.requests) == [60.0, 86400.0]


# ----------------------------------------------------------------------------
# Canonical fields
# ----------------------------------------------------------------------------


class TestCanonicalFields:
    def test_canonical_requests_single_ratelimit(self) -> None:
        rl = RateLimit(key="requests", capacity=100, window="minute")
        sl = SlowBurnLimits(requests=rl)
        assert sl.requests == [rl]

    def test_canonical_requests_list(self) -> None:
        rl1 = RateLimit(key="requests", capacity=100, window="minute")
        rl2 = RateLimit(key="requests", capacity=10_000, window="day")
        sl = SlowBurnLimits(requests=[rl1, rl2])
        assert sl.requests == [rl1, rl2]

    def test_canonical_budget_single_costlimit(self) -> None:
        cl = CostLimit(budget_usd=5.0, window="day")
        sl = SlowBurnLimits(budget=cl)
        assert sl.budget == [cl]

    def test_canonical_budget_list(self) -> None:
        cl1 = CostLimit(budget_usd=5.0, window="day")
        cl2 = CostLimit(budget_usd=100.0, window="week")
        sl = SlowBurnLimits(budget=[cl1, cl2])
        assert sl.budget == [cl1, cl2]

    def test_canonical_requests_rejects_int(self) -> None:
        # Bare int is forbidden — would need a window.
        with pytest.raises(ValueError, match="RateLimit"):
            SlowBurnLimits(requests=300)

    def test_canonical_budget_rejects_float(self) -> None:
        # Bare float is forbidden — would need a window.
        with pytest.raises(ValueError, match="CostLimit"):
            SlowBurnLimits(budget=5.0)


# ----------------------------------------------------------------------------
# Mixed canonical + shorthand
# ----------------------------------------------------------------------------


class TestMixed:
    def test_canonical_and_shorthand_different_windows(self) -> None:
        sl = SlowBurnLimits(
            rpm=300,
            requests=[RateLimit(key="requests", capacity=10_000, window="day")],
        )
        assert sl.requests is not None
        # Canonical entries are processed first; shorthand appends after.
        assert _windows(sl.requests) == [86400.0, 60.0]

    def test_full_realistic_call(self) -> None:
        sl = SlowBurnLimits(
            rpm=10,
            max_input_tokens_per_minute=100_000_000,
            otph=10_000_000,
            budget_per_hour=10.0,
            concurrency=3,
        )
        assert _windows(sl.requests) == [60.0]
        assert _windows(sl.input_tokens) == [60.0]
        assert _windows(sl.output_tokens) == [3600.0]
        assert _windows(sl.budget) == [3600.0]
        assert sl.concurrency == 3


# ----------------------------------------------------------------------------
# Unknown kwargs
# ----------------------------------------------------------------------------


class TestUnknownKwargs:
    def test_unknown_kwarg_raises(self) -> None:
        with pytest.raises(ValueError):
            SlowBurnLimits(notathing=5)

    def test_typo_in_shorthand_raises(self) -> None:
        # "rqp" is not a known pattern.
        with pytest.raises(ValueError):
            SlowBurnLimits(rqpm=5)


# ----------------------------------------------------------------------------
# Library defaults
# ----------------------------------------------------------------------------


class TestLibraryDefaults:
    def test_default_factory_populates_every_slot(self) -> None:
        d = default_slowburn_limits()
        assert d.requests is not None
        assert d.input_tokens is not None
        assert d.output_tokens is not None
        assert d.budget is not None
        assert d.concurrency is not None

    def test_default_budget_is_inf(self) -> None:
        d = default_slowburn_limits()
        assert d.budget[0].budget_usd == float("inf")
        assert float(d.budget[0].window) == 86400.0  # daily

    def test_default_concurrency_is_unbounded(self) -> None:
        d = default_slowburn_limits()
        assert d.concurrency >= 1_000_000

    def test_default_keys_match_slot_keys(self) -> None:
        d = default_slowburn_limits()
        assert d.requests[0].key == SLOT_TO_LIMIT_KEY["requests"]
        assert d.input_tokens[0].key == SLOT_TO_LIMIT_KEY["input_tokens"]
        assert d.output_tokens[0].key == SLOT_TO_LIMIT_KEY["output_tokens"]
