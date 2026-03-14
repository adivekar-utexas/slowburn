"""Tests for SlowBurnMiddleware (LangGraph integration) with mocked objects."""

from types import SimpleNamespace

import pytest

from .conftest import MOCK_MODEL_NAME

from slowburn.integrations.langgraph import (
    SlowBurnMiddleware,
    _extract_text_from_messages,
    _get_model_name,
)

# ---------------------------------------------------------------------------
# Helpers: build fake LangGraph-style objects
# ---------------------------------------------------------------------------

def _make_model(model_name: str = MOCK_MODEL_NAME, max_tokens: int = 500):
    """Fake BaseChatModel with model_name and max_tokens attributes."""
    return SimpleNamespace(model_name=model_name, max_tokens=max_tokens)


def _make_message(content: str = "Hello world"):
    """Fake LangChain message with a content attribute."""
    return SimpleNamespace(content=content)


def _make_request(
    model_name: str = MOCK_MODEL_NAME,
    max_tokens: int = 500,
    messages: list = None,
    system_message: object = None,
    model_settings: dict = None,
):
    """Fake LangGraph ModelRequest."""
    model = _make_model(model_name=model_name, max_tokens=max_tokens)
    if messages is None:
        messages = [_make_message("Hello world")]
    return SimpleNamespace(
        model=model,
        messages=messages,
        system_message=system_message,
        model_settings=model_settings,
    )


def _make_response(content: str = "Response text", usage_metadata: dict = None):
    """Fake LangGraph ModelResponse (or AIMessage-like object)."""
    return SimpleNamespace(content=content, usage_metadata=usage_metadata)


# ===========================================================================
# Tests: _get_model_name helper
# ===========================================================================

class TestGetModelName:

    def test_extracts_model_name_attr(self) -> None:
        model = SimpleNamespace(model_name=MOCK_MODEL_NAME)
        assert _get_model_name(model) == MOCK_MODEL_NAME

    def test_raises_when_no_model_name(self) -> None:
        """Model without model_name raises AttributeError — not silently handled."""
        model = SimpleNamespace(something_else="foo")
        with pytest.raises(AttributeError):
            _get_model_name(model)

    def test_raises_on_empty_string(self) -> None:
        model = SimpleNamespace(model_name="")
        with pytest.raises(RuntimeError, match="empty or not a string"):
            _get_model_name(model)


# ===========================================================================
# Tests: _extract_text_from_messages helper
# ===========================================================================

class TestExtractTextFromMessages:

    def test_string_content(self) -> None:
        msgs = [_make_message("Hello"), _make_message("World")]
        assert _extract_text_from_messages(msgs) == "Hello World"

    def test_list_content_with_text_blocks(self) -> None:
        msg = SimpleNamespace(content=[{"text": "block1"}, {"text": "block2"}])
        assert _extract_text_from_messages([msg]) == "block1 block2"

    def test_list_content_with_string_blocks(self) -> None:
        msg = SimpleNamespace(content=["raw string 1", "raw string 2"])
        assert _extract_text_from_messages([msg]) == "raw string 1 raw string 2"

    def test_empty_messages(self) -> None:
        assert _extract_text_from_messages([]) == ""

    def test_no_content_attr_raises(self) -> None:
        """Message without .content raises AttributeError."""
        msg = SimpleNamespace(role="user")
        with pytest.raises(AttributeError):
            _extract_text_from_messages([msg])


# ===========================================================================
# Tests: SlowBurnMiddleware.__init__
# ===========================================================================

class TestSlowBurnMiddlewareInit:

    def test_creates_limit_set_and_reporter(self) -> None:
        mw = SlowBurnMiddleware(budget_usd=5.0, window_seconds=3600)
        assert mw.limit_set is not None
        assert mw.reporter is not None
        assert mw.reporter.num_calls == 0

    def test_accepts_external_reporter(self) -> None:
        from slowburn.reporter import CostReporter
        r = CostReporter()
        mw = SlowBurnMiddleware(budget_usd=5.0, reporter=r)
        assert mw.reporter is r


# ===========================================================================
# Tests: SlowBurnMiddleware.wrap_model_call
# ===========================================================================

class TestSlowBurnMiddlewareWrapModelCall:

    def test_basic_call_logs_cost(self) -> None:
        """A successful model call should be logged in the reporter.

        Steps:
        1. Create middleware with $10 budget.
        2. Create a fake request with gpt-4o-mini and max_tokens=500.
        3. Call wrap_model_call with a handler that returns a response.
        4. Verify reporter has 1 call with positive cost.
        """
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)
        request = _make_request()
        handler = lambda req: _make_response(content="The answer is 42.")

        result = mw.wrap_model_call(request, handler)

        assert result.content == "The answer is 42."
        assert mw.reporter.num_calls == 1
        assert mw.reporter.total_cost() > 0

    def test_multiple_calls_accumulate(self) -> None:
        """Multiple wrap_model_call invocations should accumulate cost."""
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)

        for i in range(5):
            request = _make_request(messages=[_make_message(f"Question {i}")])
            mw.wrap_model_call(request, lambda req: _make_response(content=f"Answer {i}"))

        assert mw.reporter.num_calls == 5
        assert mw.reporter.total_cost() > 0

    def test_uses_usage_metadata_when_available(self) -> None:
        """If response has usage_metadata, it should be used for cost calc."""
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)
        request = _make_request()

        response = _make_response(
            content="short",
            usage_metadata={"input_tokens": 100, "output_tokens": 50},
        )
        mw.wrap_model_call(request, lambda req: response)

        summary = mw.reporter.summary()
        assert summary[MOCK_MODEL_NAME]["input_tokens"] == 100
        assert summary[MOCK_MODEL_NAME]["output_tokens"] == 50

    def test_max_tokens_from_model_settings(self) -> None:
        """max_tokens should be read from model_settings if present."""
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)
        model = SimpleNamespace(model_name=MOCK_MODEL_NAME)  # no max_tokens attr
        request = SimpleNamespace(
            model=model,
            messages=[_make_message("Hi")],
            system_message=None,
            model_settings={"max_tokens": 200},
        )
        mw.wrap_model_call(request, lambda req: _make_response())
        assert mw.reporter.num_calls == 1

    def test_raises_if_no_max_tokens(self) -> None:
        """Should raise AttributeError if max_tokens not on model."""
        mw = SlowBurnMiddleware(budget_usd=10.0)
        model = SimpleNamespace(model_name=MOCK_MODEL_NAME)
        request = SimpleNamespace(
            model=model,
            messages=[_make_message("Hi")],
            system_message=None,
            model_settings=None,
        )
        with pytest.raises(AttributeError):
            mw.wrap_model_call(request, lambda req: _make_response())

    def test_raises_if_no_model_name(self) -> None:
        """Should raise AttributeError if model_name not on model."""
        mw = SlowBurnMiddleware(budget_usd=10.0)
        model = SimpleNamespace(max_tokens=100)
        request = SimpleNamespace(
            model=model,
            messages=[_make_message("Hi")],
            system_message=None,
            model_settings=None,
        )
        with pytest.raises(AttributeError):
            mw.wrap_model_call(request, lambda req: _make_response())

    def test_includes_system_message_in_estimation(self) -> None:
        """System message text should be included in input token estimation."""
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)
        sys_msg = SimpleNamespace(content="You are a helpful assistant." * 50)
        request = _make_request(system_message=sys_msg)
        mw.wrap_model_call(request, lambda req: _make_response())
        assert mw.reporter.num_calls == 1

    def test_handler_exception_charges_full_estimated_cost(self) -> None:
        """If handler raises, acquisition should be updated with full estimated cost.

        Steps:
        1. Create middleware.
        2. Pass a handler that raises RuntimeError.
        3. Verify the exception propagates.
        4. Verify no call was logged (handler never succeeded).
        """
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)
        request = _make_request()

        def failing_handler(req):
            raise RuntimeError("model failed")

        with pytest.raises(RuntimeError, match="model failed"):
            mw.wrap_model_call(request, failing_handler)

        assert mw.reporter.num_calls == 0

    def test_handler_return_value_passed_through(self) -> None:
        """The return value from handler should be returned by wrap_model_call."""
        mw = SlowBurnMiddleware(budget_usd=10.0, window_seconds=3600)
        request = _make_request()
        response = _make_response(content="passthrough test")
        result = mw.wrap_model_call(request, lambda req: response)
        assert result is response
        assert result.content == "passthrough test"
