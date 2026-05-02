"""Tests for infinite budget behavior: no CostLimit, no pricing DB required."""

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


class TestInfiniteBudgetCreateLLM:
    """create_llm with default budget (inf) should not create a CostLimit."""

    def test_default_budget_has_no_cost_limit(self) -> None:
        """Default budget_usd=inf should produce a LimitSet without CostLimit.

        Steps:
        1. Create worker with default budget (no budget_usd arg).
        2. Inspect the LimitSet's limits.
        3. Verify no limit has the cost_microdollars key.
        """
        llm = create_llm(model=MOCK_MODEL_NAME)
        try:
            assert DEFAULT_COST_LIMIT_KEY not in _get_limit_keys(llm)
        finally:
            llm.stop()

    def test_explicit_budget_has_cost_limit(self) -> None:
        """Explicit budget_usd=5.0 should include a CostLimit."""
        llm = create_llm(model=MOCK_MODEL_NAME, budget_usd=5.0)
        try:
            assert DEFAULT_COST_LIMIT_KEY in _get_limit_keys(llm)
        finally:
            llm.stop()

    def test_explicit_inf_budget_has_no_cost_limit(self) -> None:
        """Passing budget_usd=float('inf') explicitly should also skip CostLimit."""
        llm = create_llm(model=MOCK_MODEL_NAME, budget_usd=float("inf"))
        try:
            assert DEFAULT_COST_LIMIT_KEY not in _get_limit_keys(llm)
        finally:
            llm.stop()

    def test_token_and_call_limits_still_present(self) -> None:
        """Even with inf budget, token and call rate limits should exist."""
        llm = create_llm(model=MOCK_MODEL_NAME)
        try:
            keys = _get_limit_keys(llm)
            assert "input_tokens" in keys
            assert "output_tokens" in keys
            assert "call_count" in keys
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
        """Reporter should still track call count and tokens even without cost data.

        Steps:
        1. Create worker with unknown model, inf budget.
        2. Make a call.
        3. Verify reporter has 1 call and correct token counts.
        4. Verify cost is 0 (unknown model, no pricing).
        """
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
        """A known model with inf budget should still report actual cost.

        Steps:
        1. Create worker with known model (gpt-4o-mini), inf budget.
        2. Mock acompletion with _hidden_params response_cost.
        3. Verify reporter logs the cost from the response.
        """
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
        """Multiple calls with unknown model should accumulate call count.

        Steps:
        1. Create worker with unknown model, inf budget.
        2. Make 3 calls.
        3. Verify reporter has 3 calls, 0 cost.
        """
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
        """An explicit budget with unknown pricing raises a non-retryable error by default.

        Steps:
        1. Create worker with unknown model and budget_usd=1.0.
        2. Call call_llm.
        3. Verify it raises SlowBurnNonRetryableError.
        """
        _set_mock_response(mock_acompletion)
        llm = create_llm(model=UNKNOWN_MODEL, budget_usd=1.0)
        try:
            with pytest.raises(SlowBurnNonRetryableError, match="not in the pricing database"):
                llm.call_llm(prompt="test").result(timeout=10.0)
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_budget_unknown_model_warn_succeeds(self, mock_acompletion) -> None:
        """on_pricing_unavailable='warn' should log a warning and proceed.

        Steps:
        1. Create worker with unknown model, budget_usd=1.0, on_pricing_unavailable='warn'.
        2. Call call_llm.
        3. Verify it succeeds (no exception).
        4. Verify reporter shows 0 cost (pricing unavailable).
        """
        _set_mock_response(mock_acompletion)
        llm = create_llm(model=UNKNOWN_MODEL, budget_usd=1.0, on_pricing_unavailable="warn")
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
        """on_pricing_unavailable='ignore' should silently skip cost tracking.

        Steps:
        1. Create worker with unknown model, budget_usd=1.0, on_pricing_unavailable='ignore'.
        2. Make multiple calls.
        3. Verify all succeed, reporter tracks calls but not cost.
        """
        _set_mock_response(mock_acompletion)
        llm = create_llm(model=UNKNOWN_MODEL, budget_usd=1.0, on_pricing_unavailable="ignore")
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
        """A known model with explicit budget should track cost normally
        regardless of on_pricing_unavailable setting.
        """
        _set_mock_response(mock_acompletion, cost=0.001)
        llm = create_llm(model=MOCK_MODEL_NAME, budget_usd=5.0, on_pricing_unavailable="warn")
        try:
            llm.call_llm(prompt="test").result(timeout=10.0)
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_validator_works_with_inf_budget_unknown_model(self, mock_acompletion) -> None:
        """Validators should work with inf budget and unknown model.

        Steps:
        1. Create worker with unknown model, inf budget.
        2. Call with a validator that parses int.
        3. Verify parsed result is returned.
        """
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
