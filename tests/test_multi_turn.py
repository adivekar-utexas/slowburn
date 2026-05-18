"""Tests for multi-turn conversation API: history, return_messages, tools, build_messages."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, patch

import pytest
from concurry import CallLimit, LimitSet, RateLimit

from slowburn.exceptions import SlowBurnNonRetryableError
from slowburn.limits import CostLimit
from slowburn.llm_worker import SlowBurnLLM

from .conftest import MOCK_MODEL_NAME


class _MockLiteLLMMessage(SimpleNamespace):
    """Minimal LiteLLM message mock with the model_dump() API used by SlowBurn."""

    content: Optional[str]
    tool_calls: Optional[List[SimpleNamespace]]

    def model_dump(self, *, exclude_none: bool = False) -> Dict[str, Any]:
        tool_calls: Optional[List[Dict[str, Any]]] = None
        if self.tool_calls is not None:
            tool_calls = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]

        message_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": tool_calls,
        }
        if exclude_none:
            message_dict = {key: value for key, value in message_dict.items() if value is not None}
        return message_dict


def _make_response(
    content: Optional[str] = "Hello",
    tool_calls: Optional[List[SimpleNamespace]] = None,
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    cost: float = 0.001,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = _MockLiteLLMMessage(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=MOCK_MODEL_NAME,
        _hidden_params={"response_cost": cost},
    )


def _make_tool_call(
    tool_call_id: str = "tc_1",
    name: str = "get_weather",
    arguments: str = '{"city": "SF"}',
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _build_worker(**init_kwargs: Any) -> SlowBurnLLM:
    limit_set = LimitSet(
        limits=[
            CostLimit(budget_usd=10.0, window=3600),
            RateLimit(key="input_tokens", window=60, capacity=1_000_000),
            RateLimit(key="output_tokens", window=60, capacity=200_000),
            CallLimit(window=60, capacity=500),
        ],
        mode="Asyncio",
        shared=True,
    )
    worker_defaults: Dict[str, Any] = dict(
        name="test-llm",
        model_name=MOCK_MODEL_NAME,
        api_key="test-key",
        temperature=0.5,
        max_tokens=100,
        timeout=10.0,
    )
    worker_defaults.update(init_kwargs)
    return SlowBurnLLM.options(
        mode="Asyncio",
        limits=limit_set,
        num_retries={"call_llm": 0, "*": 0},
    ).init(**worker_defaults)


# ===========================================================================
# Tests: build_messages
# ===========================================================================


class TestBuildMessages:
    """Test the build_messages method (sync, no LLM call)."""

    def test_simple_prompt(self) -> None:
        """String prompt with no history produces [user] message."""
        worker = _build_worker()
        try:
            messages = worker.build_messages(prompt="Hello").result(timeout=5.0)
            assert len(messages) == 1
            assert messages[0] == {"role": "user", "content": "Hello"}
        finally:
            worker.stop()

    def test_prompt_with_system(self) -> None:
        """String prompt + system_prompt produces [system, user]."""
        worker = _build_worker()
        try:
            messages = worker.build_messages(
                prompt="Hello",
                system_prompt="Be helpful",
            ).result(timeout=5.0)
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
        finally:
            worker.stop()

    def test_pre_built_messages_passthrough(self) -> None:
        """List-of-dicts prompt is returned as-is (copied)."""
        worker = _build_worker()
        try:
            original = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]
            messages = worker.build_messages(prompt=original).result(timeout=5.0)
            assert messages == original
            assert messages is not original
        finally:
            worker.stop()

    def test_history_appends_user_message(self) -> None:
        """With history, new user message is appended to a copy."""
        worker = _build_worker()
        try:
            history = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "resp 1"},
            ]
            messages = worker.build_messages(
                prompt="turn 2",
                history=history,
            ).result(timeout=5.0)
            assert len(messages) == 4
            assert messages[-1] == {"role": "user", "content": "turn 2"}
            assert len(history) == 3
        finally:
            worker.stop()

    def test_history_empty_prompt_no_append(self) -> None:
        """With history and empty prompt, no new user message appended."""
        worker = _build_worker()
        try:
            history = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            ]
            messages = worker.build_messages(
                prompt="",
                history=history,
            ).result(timeout=5.0)
            assert len(messages) == 3
            assert messages[-1]["role"] == "tool"
        finally:
            worker.stop()

    def test_history_injects_system_if_missing(self) -> None:
        """System prompt is prepended when history has no system message."""
        worker = _build_worker()
        try:
            history = [
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "resp 1"},
            ]
            messages = worker.build_messages(
                prompt="turn 2",
                system_prompt="Be helpful",
                history=history,
            ).result(timeout=5.0)
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be helpful"
            assert len(messages) == 4
        finally:
            worker.stop()

    def test_history_skips_system_if_present(self) -> None:
        """System prompt is NOT prepended when history already has one."""
        worker = _build_worker()
        try:
            history = [
                {"role": "system", "content": "existing"},
                {"role": "user", "content": "turn 1"},
            ]
            messages = worker.build_messages(
                prompt="turn 2",
                system_prompt="new system",
                history=history,
            ).result(timeout=5.0)
            assert messages[0]["content"] == "existing"
            assert len(messages) == 3
        finally:
            worker.stop()


# ===========================================================================
# Tests: return_messages auto-detection
# ===========================================================================


class TestReturnMessagesAutoDetect:
    """Test that return_messages auto-detects based on input type."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_string_prompt_returns_string(self, mock_acompletion: AsyncMock) -> None:
        """Default: string prompt, no history -> returns string."""
        mock_acompletion.return_value = _make_response(content="world")
        worker = _build_worker()
        try:
            result = worker.call_llm(prompt="hello").result(timeout=10.0)
            assert isinstance(result, str)
            assert result == "world"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_history_returns_messages(self, mock_acompletion: AsyncMock) -> None:
        """With history provided -> returns messages list."""
        mock_acompletion.return_value = _make_response(content="resp")
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt="turn 2",
                history=[
                    {"role": "user", "content": "turn 1"},
                    {"role": "assistant", "content": "resp 1"},
                ],
            ).result(timeout=10.0)
            assert isinstance(result, list)
            assert result[-1]["role"] == "assistant"
            assert result[-1]["content"] == "resp"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_list_prompt_returns_messages(self, mock_acompletion: AsyncMock) -> None:
        """Pre-built messages list prompt -> returns messages list."""
        mock_acompletion.return_value = _make_response(content="output")
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
            ).result(timeout=10.0)
            assert isinstance(result, list)
            assert result[-1]["role"] == "assistant"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_return_messages_true(self, mock_acompletion: AsyncMock) -> None:
        """return_messages=True forces messages output even for string prompt."""
        mock_acompletion.return_value = _make_response(content="forced")
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt="hello",
                return_messages=True,
            ).result(timeout=10.0)
            assert isinstance(result, list)
            assert result[-1]["content"] == "forced"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_return_messages_false(self, mock_acompletion: AsyncMock) -> None:
        """return_messages=False forces string output even with history."""
        mock_acompletion.return_value = _make_response(content="forced str")
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt="turn 2",
                history=[{"role": "user", "content": "turn 1"}],
                return_messages=False,
            ).result(timeout=10.0)
            assert isinstance(result, str)
            assert result == "forced str"
        finally:
            worker.stop()


# ===========================================================================
# Tests: tool_calls in messages return
# ===========================================================================


class TestToolCallsInMessages:
    """Test that tool_calls are properly included in returned messages."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_tool_calls_in_assistant_message(self, mock_acompletion: AsyncMock) -> None:
        """When LLM returns tool_calls, they appear in the assistant message."""
        tool_call = _make_tool_call(tool_call_id="tc_42", name="search", arguments='{"q": "test"}')
        mock_acompletion.return_value = _make_response(content=None, tool_calls=[tool_call])
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt="search for something",
                history=[],
                system_prompt="agent",
            ).result(timeout=10.0)
            assert isinstance(result, list)
            assistant = result[-1]
            assert assistant["role"] == "assistant"
            assert assistant["content"] is None
            assert len(assistant["tool_calls"]) == 1
            assert assistant["tool_calls"][0]["id"] == "tc_42"
            assert assistant["tool_calls"][0]["function"]["name"] == "search"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_tool_results_resubmit(self, mock_acompletion: AsyncMock) -> None:
        """After appending tool results, re-submit with empty prompt."""
        tool_call = _make_tool_call(tool_call_id="tc_1", name="calc", arguments='{"x": 1}')
        mock_acompletion.side_effect = [
            _make_response(content=None, tool_calls=[tool_call]),
            _make_response(content="final answer"),
        ]
        worker = _build_worker()
        try:
            messages = worker.call_llm(
                prompt="compute",
                history=[],
                system_prompt="agent",
            ).result(timeout=10.0)

            assert messages[-1].get("tool_calls") is not None

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "tc_1",
                    "content": "42",
                }
            )

            messages = worker.call_llm(
                prompt="",
                history=messages,
            ).result(timeout=10.0)

            assert messages[-1]["role"] == "assistant"
            assert messages[-1]["content"] == "final answer"
            assert messages[-1].get("tool_calls") is None
        finally:
            worker.stop()


# ===========================================================================
# Tests: tools parameter resolution
# ===========================================================================


class TestToolsResolution:
    """Test worker-level and per-call tools/tool_choice resolution."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_worker_level_tools(self, mock_acompletion: AsyncMock) -> None:
        """Worker-level tools are passed to litellm when messages are requested."""
        mock_acompletion.return_value = _make_response()
        tool_schema = [{"type": "function", "function": {"name": "foo"}}]
        worker = _build_worker(tools=tool_schema, tool_choice="auto")
        try:
            worker.call_llm(prompt="hello", return_messages=True).result(timeout=10.0)
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["tools"] == tool_schema
            assert call_kwargs["tool_choice"] == "auto"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_tools_require_return_messages(self, mock_acompletion: AsyncMock) -> None:
        """Tool calls require messages mode because tool calls are structured assistant data."""
        tool_schema = [{"type": "function", "function": {"name": "foo"}}]
        worker = _build_worker(tools=tool_schema, tool_choice="auto")
        try:
            with pytest.raises(SlowBurnNonRetryableError, match="return_messages=True"):
                worker.call_llm(prompt="hello", return_messages=False).result(timeout=10.0)
            assert mock_acompletion.call_count == 0
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_per_call_tools_override(self, mock_acompletion: AsyncMock) -> None:
        """Per-call tools override worker-level tools."""
        mock_acompletion.return_value = _make_response()
        worker_tools = [{"type": "function", "function": {"name": "foo"}}]
        call_tools = [{"type": "function", "function": {"name": "bar"}}]
        worker = _build_worker(tools=worker_tools, tool_choice="auto")
        try:
            worker.call_llm(
                prompt="hello",
                tools=call_tools,
                tool_choice="required",
                return_messages=True,
            ).result(timeout=10.0)
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["tools"] == call_tools
            assert call_kwargs["tool_choice"] == "required"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_per_call_tools_none_disables(self, mock_acompletion: AsyncMock) -> None:
        """Per-call tools=None disables worker-level tools."""
        mock_acompletion.return_value = _make_response()
        worker_tools = [{"type": "function", "function": {"name": "foo"}}]
        worker = _build_worker(tools=worker_tools, tool_choice="auto")
        try:
            worker.call_llm(
                prompt="hello",
                tools=None,
                tool_choice=None,
            ).result(timeout=10.0)
            call_kwargs = mock_acompletion.call_args.kwargs
            assert "tools" not in call_kwargs
            assert "tool_choice" not in call_kwargs
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_no_tools_by_default(self, mock_acompletion: AsyncMock) -> None:
        """Without tools on worker or call, no tools in litellm params."""
        mock_acompletion.return_value = _make_response()
        worker = _build_worker()
        try:
            worker.call_llm(prompt="hello").result(timeout=10.0)
            call_kwargs = mock_acompletion.call_args.kwargs
            assert "tools" not in call_kwargs
            assert "tool_choice" not in call_kwargs
        finally:
            worker.stop()


# ===========================================================================
# Tests: batch with history_per_prompt
# ===========================================================================


class TestBatchMultiTurn:
    """Test call_llm_batch with the new multi-turn params."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_with_history(self, mock_acompletion: AsyncMock) -> None:
        """Batch with history_per_prompt passes history to each call."""
        mock_acompletion.return_value = _make_response(content="batch resp")
        worker = _build_worker()
        try:
            history_1 = [
                {"role": "user", "content": "t1"},
                {"role": "assistant", "content": "r1"},
            ]
            results = worker.call_llm_batch(
                prompts=["follow-up 1", "standalone"],
                history_per_prompt=[history_1, None],
            ).result(timeout=15.0)
            assert len(results) == 2
            assert isinstance(results[0], list)
            assert isinstance(results[1], str)
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_history_length_mismatch(self, mock_acompletion: AsyncMock) -> None:
        """Mismatched history_per_prompt length raises a non-retryable error."""
        worker = _build_worker()
        try:
            with pytest.raises(SlowBurnNonRetryableError, match="history_per_prompt length"):
                worker.call_llm_batch(
                    prompts=["a", "b", "c"],
                    history_per_prompt=[None, None],
                ).result(timeout=10.0)
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_with_tools(self, mock_acompletion: AsyncMock) -> None:
        """Batch tools param is forwarded to each call."""
        mock_acompletion.return_value = _make_response()
        tool_schema = [{"type": "function", "function": {"name": "foo"}}]
        worker = _build_worker()
        try:
            worker.call_llm_batch(
                prompts=["a", "b"],
                tools=tool_schema,
                tool_choice="auto",
                return_messages=True,
            ).result(timeout=15.0)
            for call_arguments in mock_acompletion.call_args_list:
                assert call_arguments.kwargs["tools"] == tool_schema
                assert call_arguments.kwargs["tool_choice"] == "auto"
        finally:
            worker.stop()


# ===========================================================================
# Tests: backward compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Verify existing call patterns still work unchanged."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_simple_string_call_unchanged(self, mock_acompletion: AsyncMock) -> None:
        """call_llm(prompt="...") still returns a string."""
        mock_acompletion.return_value = _make_response(content="hi")
        worker = _build_worker()
        try:
            result = worker.call_llm(prompt="hello").result(timeout=10.0)
            assert result == "hi"
            assert isinstance(result, str)
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_string_call_with_system_prompt(self, mock_acompletion: AsyncMock) -> None:
        """call_llm(prompt="...", system_prompt="...") still returns string."""
        mock_acompletion.return_value = _make_response(content="resp")
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt="hello",
                system_prompt="be nice",
            ).result(timeout=10.0)
            assert isinstance(result, str)
            sent_messages = mock_acompletion.call_args.kwargs["messages"]
            assert len(sent_messages) == 2
            assert sent_messages[0]["role"] == "system"
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_validator_still_works(self, mock_acompletion: AsyncMock) -> None:
        """Validator on string-return path still works."""
        mock_acompletion.return_value = _make_response(content="42")
        worker = _build_worker()
        try:
            result = worker.call_llm(
                prompt="what is 6*7?",
                validator=lambda text: int(text.strip()),
            ).result(timeout=10.0)
            assert result == 42
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_string_prompts_unchanged(self, mock_acompletion: AsyncMock) -> None:
        """call_llm_batch(prompts=["...", "..."]) returns List[str]."""
        mock_acompletion.return_value = _make_response(content="r")
        worker = _build_worker()
        try:
            results = worker.call_llm_batch(prompts=["a", "b"]).result(timeout=15.0)
            assert len(results) == 2
            assert all(isinstance(result, str) for result in results)
        finally:
            worker.stop()
