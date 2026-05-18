"""Tests for SlowBurnCallbackHandler (LangChain integration) with mocked objects."""

import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from .conftest import MOCK_MODEL_NAME

from slowburn.integrations.langchain import SlowBurnCallbackHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_serialized(
    model_name: str = MOCK_MODEL_NAME,
    max_tokens: int = 500,
) -> dict:
    """Fake LangChain serialized LLM config dict."""
    return {
        "kwargs": {
            "model_name": model_name,
            "max_tokens": max_tokens,
        },
        "id": ["langchain_openai", "ChatOpenAI"],
    }


def _make_llm_result(
    text: str = "Generated text",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> SimpleNamespace:
    """Fake LangChain LLMResult with token_usage."""
    gen = SimpleNamespace(text=text)
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        },
    )


def _make_llm_result_no_usage(text: str = "Generated text") -> SimpleNamespace:
    """Fake LLMResult without token_usage (some providers don't return it)."""
    gen = SimpleNamespace(text=text)
    return SimpleNamespace(
        generations=[[gen]],
        llm_output=None,
    )


# ===========================================================================
# Tests: __init__
# ===========================================================================


class TestSlowBurnCallbackHandlerInit:
    def test_creates_limit_set_and_reporter(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0, window=3600)
        assert cb.limit_set is not None
        assert cb.reporter is not None
        assert cb.reporter.num_calls == 0

    def test_accepts_external_reporter(self) -> None:
        from slowburn.reporter import CostReporter

        r = CostReporter()
        cb = SlowBurnCallbackHandler(budget_usd=5.0, reporter=r)
        assert cb.reporter is r

    def test_raise_error_is_true(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        assert cb.raise_error is True


# ===========================================================================
# Tests: _extract_model_name
# ===========================================================================


class TestExtractModelName:
    def test_from_kwargs_model_name(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {"model_name": MOCK_MODEL_NAME}}
        assert cb._extract_model_name(serialized) == MOCK_MODEL_NAME

    def test_from_kwargs_model(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {"model": "claude-3-haiku"}}
        assert cb._extract_model_name(serialized) == "claude-3-haiku"

    def test_from_id_fallback(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {}, "id": ["langchain_openai", "ChatOpenAI"]}
        assert cb._extract_model_name(serialized) == "ChatOpenAI"

    def test_raises_when_no_name(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {}}
        with pytest.raises(RuntimeError, match="Could not determine model name"):
            cb._extract_model_name(serialized)

    def test_raises_on_empty_string(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {"model_name": ""}}
        with pytest.raises(RuntimeError, match="Could not determine model name"):
            cb._extract_model_name(serialized)


# ===========================================================================
# Tests: _extract_max_tokens
# ===========================================================================


class TestExtractMaxTokens:
    def test_from_kwargs_max_tokens(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {"max_tokens": 512}}
        assert cb._extract_max_tokens(serialized) == 512

    def test_from_kwargs_max_output_tokens(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {"max_output_tokens": 1024}}
        assert cb._extract_max_tokens(serialized) == 1024

    def test_raises_when_missing(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=5.0)
        serialized = {"kwargs": {"model_name": MOCK_MODEL_NAME}}
        with pytest.raises(RuntimeError, match="Could not determine max_tokens"):
            cb._extract_max_tokens(serialized)


# ===========================================================================
# Tests: on_llm_start + on_llm_end (full cycle)
# ===========================================================================


class TestCallbackFullCycle:
    def test_start_end_logs_cost(self) -> None:
        """Full on_llm_start -> on_llm_end cycle should log one call.

        Steps:
        1. Call on_llm_start with serialized config and prompt.
        2. Call on_llm_end with a response that has token_usage.
        3. Verify reporter has 1 call with correct tokens.
        """
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        run_id = uuid4()
        serialized = _make_serialized()

        cb.on_llm_start(serialized, ["What is 2+2?"], run_id=run_id)
        cb.on_llm_end(_make_llm_result(prompt_tokens=15, completion_tokens=5), run_id=run_id)

        assert cb.reporter.num_calls == 1
        assert cb.reporter.total_cost() > 0
        summary = cb.reporter.summary()
        assert summary[MOCK_MODEL_NAME]["input_tokens"] == 15
        assert summary[MOCK_MODEL_NAME]["output_tokens"] == 5

    def test_multiple_cycles(self) -> None:
        """Multiple start/end cycles should accumulate correctly."""
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        serialized = _make_serialized()

        for _ in range(3):
            run_id = uuid4()
            cb.on_llm_start(serialized, ["Hello"], run_id=run_id)
            cb.on_llm_end(_make_llm_result(), run_id=run_id)

        assert cb.reporter.num_calls == 3

    def test_end_without_start_raises(self) -> None:
        """on_llm_end without on_llm_start should raise RuntimeError."""
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        with pytest.raises(RuntimeError, match="No pending acquisition"):
            cb.on_llm_end(_make_llm_result(), run_id=uuid4())

    def test_no_token_usage_falls_back_to_text_length(self) -> None:
        """If token_usage is missing, cost should be estimated from text length."""
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        run_id = uuid4()
        serialized = _make_serialized()

        cb.on_llm_start(serialized, ["Hello"], run_id=run_id)
        cb.on_llm_end(_make_llm_result_no_usage(text="A" * 300), run_id=run_id)

        assert cb.reporter.num_calls == 1
        assert cb.reporter.total_cost() > 0

    def test_uses_run_id_to_match_start_and_end(self) -> None:
        """Two concurrent calls with different run_ids should be tracked separately.

        Steps:
        1. Start two calls with different run_ids.
        2. End them in reverse order.
        3. Verify both are logged correctly (2 calls total).
        """
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        serialized = _make_serialized()
        rid_a = uuid4()
        rid_b = uuid4()

        cb.on_llm_start(serialized, ["Query A"], run_id=rid_a)
        cb.on_llm_start(serialized, ["Query B"], run_id=rid_b)

        cb.on_llm_end(_make_llm_result(text="Answer B"), run_id=rid_b)
        cb.on_llm_end(_make_llm_result(text="Answer A"), run_id=rid_a)

        assert cb.reporter.num_calls == 2


# ===========================================================================
# Tests: on_llm_error
# ===========================================================================


class TestCallbackOnError:
    def test_error_charges_full_estimated_cost(self) -> None:
        """on_llm_error should release the acquisition with full estimated cost.

        Steps:
        1. Call on_llm_start.
        2. Call on_llm_error (simulating API failure).
        3. Verify the pending acquisition is cleaned up.
        4. Verify no call is logged in reporter (error, not success).
        """
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        run_id = uuid4()
        serialized = _make_serialized()

        cb.on_llm_start(serialized, ["Hello"], run_id=run_id)
        cb.on_llm_error(RuntimeError("API timeout"), run_id=run_id)

        assert cb.reporter.num_calls == 0
        assert len(cb._pending) == 0

    def test_error_without_start_is_safe(self) -> None:
        """on_llm_error for an unknown run_id should not raise."""
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        cb.on_llm_error(RuntimeError("oops"), run_id=uuid4())

    def test_error_then_end_raises(self) -> None:
        """If on_llm_error already cleaned up, on_llm_end should raise."""
        cb = SlowBurnCallbackHandler(budget_usd=10.0, window=3600)
        run_id = uuid4()
        serialized = _make_serialized()

        cb.on_llm_start(serialized, ["Hello"], run_id=run_id)
        cb.on_llm_error(RuntimeError("fail"), run_id=run_id)

        with pytest.raises(RuntimeError, match="No pending acquisition"):
            cb.on_llm_end(_make_llm_result(), run_id=run_id)


# ===========================================================================
# Tests: Thread safety
# ===========================================================================


class TestCallbackThreadSafety:
    def test_concurrent_start_end_from_multiple_threads(self) -> None:
        """Concurrent on_llm_start/on_llm_end from 10 threads should all be tracked.

        Steps:
        1. Spawn 10 threads, each doing start+end with unique run_id.
        2. Wait for all threads.
        3. Verify reporter logged exactly 10 calls.
        """
        cb = SlowBurnCallbackHandler(budget_usd=100.0, window=3600)
        serialized = _make_serialized()
        num_threads = 10
        errors = []

        def do_call():
            try:
                rid = uuid4()
                cb.on_llm_start(serialized, ["Thread query"], run_id=rid)
                cb.on_llm_end(_make_llm_result(), run_id=rid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_call) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in threads: {errors}"
        assert cb.reporter.num_calls == num_threads


# ===========================================================================
# Tests: on_llm_start validation
# ===========================================================================


class TestCallbackStartValidation:
    def test_raises_if_no_model_name(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=10.0)
        serialized = {"kwargs": {"max_tokens": 500}}
        with pytest.raises(RuntimeError, match="Could not determine model name"):
            cb.on_llm_start(serialized, ["Hello"], run_id=uuid4())

    def test_raises_if_no_max_tokens(self) -> None:
        cb = SlowBurnCallbackHandler(budget_usd=10.0)
        serialized = {"kwargs": {"model_name": MOCK_MODEL_NAME}}
        with pytest.raises(RuntimeError, match="Could not determine max_tokens"):
            cb.on_llm_start(serialized, ["Hello"], run_id=uuid4())
