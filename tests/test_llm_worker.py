"""Tests for SlowBurnLLM worker with mocked litellm calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from concurry import CallLimit, LimitSet, RateLimit

from slowburn.limits import CostLimit
from slowburn.llm_worker import SlowBurnLLM, _estimate_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_acompletion_response(
    content: str = "Hello from the LLM",
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    model: str = "gpt-4o-mini",
    cost: float = 0.001,
):
    """Build a mock litellm acompletion response."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=model,
        _hidden_params={"response_cost": cost},
    )


def _build_worker(budget_usd: float = 10.0) -> SlowBurnLLM:
    """Create a SlowBurnLLM worker with a reasonable limit set."""
    limit_set = LimitSet(
        limits=[
            CostLimit(budget_usd=budget_usd, window_seconds=3600),
            RateLimit(key="input_tokens", window_seconds=60, capacity=1_000_000),
            RateLimit(key="output_tokens", window_seconds=60, capacity=200_000),
            CallLimit(window_seconds=60, capacity=500),
        ],
        mode="asyncio",
        shared=True,
    )
    return SlowBurnLLM.options(
        mode="asyncio",
        limits=limit_set,
        num_retries={"call_llm": 0, "*": 0},
    ).init(
        name="test-llm",
        model_name="gpt-4o-mini",
        api_key="test-key",
        temperature=0.5,
        max_tokens=100,
        timeout=10.0,
    )


# ===========================================================================
# Tests: _estimate_tokens helper
# ===========================================================================

class TestEstimateTokens:
    def test_basic_estimate(self) -> None:
        assert _estimate_tokens("hello world") >= 1

    def test_empty_string(self) -> None:
        """Empty string should return at least 1."""
        assert _estimate_tokens("") >= 1

    def test_scales_with_length(self) -> None:
        short = _estimate_tokens("hi")
        long = _estimate_tokens("a" * 3000)
        assert long > short


# ===========================================================================
# Tests: SlowBurnLLM call_llm
# ===========================================================================

class TestSlowBurnLLMCallLLM:
    """Test the core call_llm method with mocked litellm.acompletion."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_basic_call(self, mock_acompletion) -> None:
        """A basic call should return the response text and log to the reporter.

        Steps:
        1. Mock litellm.acompletion to return a known response.
        2. Call call_llm with a simple prompt.
        3. Verify the returned text matches.
        4. Verify the reporter logged one call with correct cost.
        """
        mock_acompletion.return_value = _make_acompletion_response(
            content="Test response", cost=0.0005,
        )
        w = _build_worker()
        try:
            result = w.call_llm(prompt="Say hello").result(timeout=10.0)
            assert result == "Test response"

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() == pytest.approx(0.0005, abs=1e-6)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_call_with_system_prompt(self, mock_acompletion) -> None:
        """System prompt should be included in the messages sent to litellm.

        Steps:
        1. Call call_llm with both prompt and system_prompt.
        2. Verify litellm.acompletion was called with 2 messages.
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            w.call_llm(prompt="Hello", system_prompt="Be helpful").result(timeout=10.0)
            call_args = mock_acompletion.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_call_with_validator(self, mock_acompletion) -> None:
        """A validator should parse the response text before returning.

        Steps:
        1. Return "42" from the mock.
        2. Pass a validator that converts to int.
        3. Verify the result is int(42).
        """
        mock_acompletion.return_value = _make_acompletion_response(content="42")
        w = _build_worker()
        try:
            result = w.call_llm(
                prompt="What is 6*7?",
                validator=lambda text: int(text.strip()),
            ).result(timeout=10.0)
            assert result == 42
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_validator_failure_raises(self, mock_acompletion) -> None:
        """A failing validator should propagate the ValueError.

        Steps:
        1. Return "not a number" from the mock.
        2. Pass a validator that does int(text).
        3. Verify ValueError is raised.
        """
        mock_acompletion.return_value = _make_acompletion_response(content="not a number")
        w = _build_worker()
        try:
            with pytest.raises(ValueError):
                w.call_llm(
                    prompt="bad",
                    validator=lambda text: int(text),
                ).result(timeout=10.0)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_multiple_calls_accumulate_cost(self, mock_acompletion) -> None:
        """Multiple calls should accumulate cost in the reporter.

        Steps:
        1. Make 5 calls, each costing $0.001.
        2. Verify reporter shows 5 calls and ~$0.005 total.
        """
        mock_acompletion.return_value = _make_acompletion_response(cost=0.001)
        w = _build_worker()
        try:
            for _ in range(5):
                w.call_llm(prompt="test").result(timeout=10.0)

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 5
            assert reporter.total_cost() == pytest.approx(0.005, abs=1e-5)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_api_error_propagates(self, mock_acompletion) -> None:
        """API errors from litellm should propagate as exceptions.

        Steps:
        1. Make acompletion raise a RuntimeError.
        2. Verify the error propagates through .result().
        """
        mock_acompletion.side_effect = RuntimeError("API connection failed")
        w = _build_worker()
        try:
            with pytest.raises(RuntimeError, match="API connection failed"):
                w.call_llm(prompt="fail").result(timeout=10.0)
        finally:
            w.stop()


# ===========================================================================
# Tests: SlowBurnLLM call_llm_batch
# ===========================================================================

class TestSlowBurnLLMBatch:
    """Test batch execution."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_returns_all_results(self, mock_acompletion) -> None:
        """Batch call should return one result per prompt.

        Steps:
        1. Submit 3 prompts.
        2. Verify 3 results returned.
        3. Verify reporter logged 3 calls.
        """
        mock_acompletion.return_value = _make_acompletion_response(
            content="batch result", cost=0.0002,
        )
        w = _build_worker()
        try:
            results = w.call_llm_batch(
                prompts=["p1", "p2", "p3"],
            ).result(timeout=15.0)
            assert len(results) == 3
            assert all(r == "batch result" for r in results)

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 3
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_empty_batch(self, mock_acompletion) -> None:
        """Empty prompt list should return empty results."""
        w = _build_worker()
        try:
            results = w.call_llm_batch(prompts=[]).result(timeout=10.0)
            assert results == []
        finally:
            w.stop()


# ===========================================================================
# Tests: SlowBurnLLM get_reporter
# ===========================================================================

class TestSlowBurnLLMReporter:
    """Test reporter access pattern."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_get_reporter_returns_same_instance(self, mock_acompletion) -> None:
        """get_reporter() should return the same CostReporter each time."""
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            r1 = w.get_reporter().result(timeout=5.0)
            w.call_llm(prompt="test").result(timeout=10.0)
            r2 = w.get_reporter().result(timeout=5.0)
            assert r1 is r2
            assert r2.num_calls == 1
        finally:
            w.stop()
