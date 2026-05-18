"""Tests for infinite budget behavior: ``inf`` budget → no cost enforcement,
no pricing DB required.

In the new ``SlowBurnLimits`` design every slot is always populated, so a
worker with the library default budget *does* have a CostLimit — but with
``budget_usd=float("inf")``. The worker recognizes this and skips cost
enforcement (and pricing-database lookups) the same way it used to skip
when the slot was absent.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from slowburn import SlowBurnNonRetryableError, create_llm
from slowburn.limits import DEFAULT_COST_LIMIT_KEY

from .conftest import MOCK_MODEL_NAME

UNKNOWN_MODEL = "together_ai/some-org/unknown-model-xyz"


def _set_mock_response(mock_acompletion, cost=None):
    """Configure a mock acompletion to return a standard response."""
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    message = SimpleNamespace(content="ok", tool_calls=None)
    choice = SimpleNamespace(message=message)
    hidden = {"response_cost": cost} if cost is not None else {}
    mock_acompletion.return_value = SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=UNKNOWN_MODEL,
        _hidden_params=hidden,
    )


def _get_limit_keys(llm) -> set:
    """Extract all limit keys from a SlowBurnLLM's LimitPool."""
    keys = set()
    for limit_set in llm.limits.limit_sets:
        for lim in limit_set.limits:
            key = getattr(lim, "key", None)
            if key is not None:
                keys.add(key)
    return keys


def _get_cost_limit_budget(llm) -> float:
    """Return the budget_usd of the (single) CostLimit in the first LimitSet."""
    for limit_set in llm.limits.limit_sets:
        for lim in limit_set.limits:
            if getattr(lim, "key", None) == DEFAULT_COST_LIMIT_KEY:
                return lim.budget_usd
    raise AssertionError("no CostLimit on the pool")


class TestInfiniteBudgetCreateLLM:
    """``create_llm`` with default (inf) budget should produce a CostLimit
    with ``budget_usd=inf`` (the library default), and the worker should treat
    that as 'no real cost enforcement' for unknown models."""

    def test_default_budget_has_inf_cost_limit(self) -> None:
        """Default budget should produce a CostLimit with ``budget_usd=inf``."""
        llm = create_llm(model=MOCK_MODEL_NAME)
        try:
            assert DEFAULT_COST_LIMIT_KEY in _get_limit_keys(llm)
            assert _get_cost_limit_budget(llm) == float("inf")
        finally:
            llm.stop()

    def test_explicit_budget_has_finite_cost_limit(self) -> None:
        """Explicit ``budget_per_day=5.0`` should include a finite CostLimit."""
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_day=5.0))
        try:
            assert DEFAULT_COST_LIMIT_KEY in _get_limit_keys(llm)
            assert _get_cost_limit_budget(llm) == 5.0
        finally:
            llm.stop()

    def test_explicit_inf_budget_keeps_inf_cost_limit(self) -> None:
        """Passing ``budget_per_day=float('inf')`` explicitly is equivalent to default."""
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_day=float("inf")))
        try:
            assert _get_cost_limit_budget(llm) == float("inf")
        finally:
            llm.stop()

    def test_token_and_call_limits_still_present(self) -> None:
        """Even with inf budget, token and call rate limits should exist."""
        llm = create_llm(model=MOCK_MODEL_NAME)
        try:
            keys = _get_limit_keys(llm)
            assert "input_tokens" in keys
            assert "output_tokens" in keys
            assert "requests" in keys
        finally:
            llm.stop()


class TestInfiniteBudgetCallLLM:
    """call_llm with infinite budget should work without the model being in the pricing DB."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_unknown_model_works_with_inf_budget(self, mock_acompletion) -> None:
        """An unknown model should NOT raise ModelNotFoundError when budget is inf.

        Steps:
        1. Create worker with an unknown model and default (inf) budget.
        2. Mock acompletion to return a valid response (without _hidden_params cost).
        3. Call call_llm.
        4. Verify it succeeds (no ModelNotFoundError).
        """
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        message = SimpleNamespace(content="test output", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=UNKNOWN_MODEL,
            _hidden_params={},
        )
        llm = create_llm(model=UNKNOWN_MODEL)
        try:
            result = llm.call_llm(prompt="Hello").result(timeout=10.0)
            assert result == "test output"
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_unknown_model_reporter_tracks_calls_and_tokens(self, mock_acompletion) -> None:
        """Reporter should still track call count and tokens even without cost data."""
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
        message = SimpleNamespace(content="response", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=UNKNOWN_MODEL,
            _hidden_params={},
        )
        llm = create_llm(model=UNKNOWN_MODEL)
        try:
            llm.call_llm(prompt="Hi").result(timeout=10.0)
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            summary = reporter.summary()
            assert UNKNOWN_MODEL in summary
            assert summary[UNKNOWN_MODEL]["input_tokens"] == 50
            assert summary[UNKNOWN_MODEL]["output_tokens"] == 20
            assert summary[UNKNOWN_MODEL]["total_tokens"] == 70
            assert reporter.total_cost() == 0.0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_known_model_with_inf_budget_still_tracks_cost(self, mock_acompletion) -> None:
        """A known model with inf budget should still report actual cost."""
        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        message = SimpleNamespace(content="output", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=MOCK_MODEL_NAME,
            _hidden_params={"response_cost": 0.0005},
        )
        llm = create_llm(model=MOCK_MODEL_NAME)
        try:
            llm.call_llm(prompt="Hi").result(timeout=10.0)
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_multiple_calls_unknown_model_accumulate(self, mock_acompletion) -> None:
        """Multiple calls with unknown model should accumulate call count."""
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=UNKNOWN_MODEL,
            _hidden_params={},
        )
        llm = create_llm(model=UNKNOWN_MODEL)
        try:
            for _ in range(3):
                llm.call_llm(prompt="test").result(timeout=10.0)
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 3
            assert reporter.total_cost() == 0.0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_budget_unknown_model_raises_by_default(self, mock_acompletion) -> None:
        """An explicit (finite) budget with unknown pricing raises a non-retryable error by default."""
        _set_mock_response(mock_acompletion)
        llm = create_llm(model=UNKNOWN_MODEL, limits=dict(budget_per_day=1.0))
        try:
            with pytest.raises(SlowBurnNonRetryableError, match="not in the pricing database"):
                llm.call_llm(prompt="test").result(timeout=10.0)
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_budget_unknown_model_warn_succeeds(self, mock_acompletion) -> None:
        """on_pricing_unavailable='warn' should log a warning and proceed."""
        _set_mock_response(mock_acompletion)
        llm = create_llm(
            model=UNKNOWN_MODEL,
            limits=dict(budget_per_day=1.0),
            on_pricing_unavailable="warn",
        )
        try:
            result = llm.call_llm(prompt="test").result(timeout=10.0)
            assert result == "ok"
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() == 0.0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_budget_unknown_model_ignore_succeeds(self, mock_acompletion) -> None:
        """on_pricing_unavailable='ignore' should silently skip cost tracking."""
        _set_mock_response(mock_acompletion)
        llm = create_llm(
            model=UNKNOWN_MODEL,
            limits=dict(budget_per_day=1.0),
            on_pricing_unavailable="ignore",
        )
        try:
            for _ in range(3):
                llm.call_llm(prompt="test").result(timeout=10.0)
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 3
            assert reporter.total_cost() == 0.0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_known_model_with_budget_still_tracks_cost(self, mock_acompletion) -> None:
        """A known model with explicit budget should track cost normally."""
        _set_mock_response(mock_acompletion, cost=0.001)
        llm = create_llm(
            model=MOCK_MODEL_NAME,
            limits=dict(budget_per_day=5.0),
            on_pricing_unavailable="warn",
        )
        try:
            llm.call_llm(prompt="test").result(timeout=10.0)
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_validator_works_with_inf_budget_unknown_model(self, mock_acompletion) -> None:
        """Validators should work with inf budget and unknown model."""
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        message = SimpleNamespace(content="42", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=UNKNOWN_MODEL,
            _hidden_params={},
        )
        llm = create_llm(model=UNKNOWN_MODEL)
        try:
            result = llm.call_llm(
                prompt="What is 6*7?",
                validator=lambda text: int(text.strip()),
            ).result(timeout=10.0)
            assert result == 42
        finally:
            llm.stop()
