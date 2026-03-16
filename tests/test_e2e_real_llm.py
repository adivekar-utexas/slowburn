"""
End-to-end integration tests with real LLM calls.

API keys are loaded from .env by conftest.py at session startup.
Tests are skipped if no key is available.
"""

import re
import time

import pytest
from concurry import LimitSet

from slowburn import CostLimit, CostReporter, create_llm
from slowburn.integrations.autogen import SlowBurnModelClient

from .conftest import skip_no_api_key


@skip_no_api_key
class TestCreateLLMRealCalls:
    """Real LLM calls via create_llm() convenience API."""

    @pytest.fixture(autouse=True)
    def _setup_llm(self, llm_model_and_key):
        model, key = llm_model_and_key
        self.llm = create_llm(
            model=model,
            budget_usd=1.0,
            window="hourly",
            api_key=key,
            max_tokens=150,
            temperature=0.3,
        )
        yield
        self.llm.stop()

    def test_single_call(self) -> None:
        """A basic call should return text and log cost to the reporter."""
        start = time.time()
        result = self.llm.call_llm(
            prompt="What is 2+2? Answer in one word.",
        ).result(timeout=30.0)
        elapsed = time.time() - start

        reporter = self.llm.get_reporter().result(timeout=5.0)
        print(f"Response: {result!r}  ({elapsed:.2f}s)")
        print(f"Calls: {reporter.num_calls}, Cost: ${reporter.total_cost():.6f}")

        assert len(result) > 0
        assert reporter.num_calls == 1
        assert reporter.total_cost() > 0

    def test_multiple_sequential_calls(self) -> None:
        """Multiple calls should accumulate cost."""
        prompts = ["Name one planet.", "Name one color.", "Name one animal."]
        for p in prompts:
            r = self.llm.call_llm(prompt=p).result(timeout=30.0)
            print(f"  Q: {p} -> A: {r.strip()!r}")

        reporter = self.llm.get_reporter().result(timeout=5.0)
        print(f"Total calls: {reporter.num_calls}, Cost: ${reporter.total_cost():.6f}")
        assert reporter.num_calls == 3
        assert reporter.total_cost() > 0

    def test_batch_concurrent(self) -> None:
        """Batch call should return one result per prompt."""
        start = time.time()
        results = self.llm.call_llm_batch(
            prompts=["Capital of France?", "Capital of Japan?", "Capital of Brazil?"],
        ).result(timeout=30.0)
        elapsed = time.time() - start

        print(f"Batch ({elapsed:.2f}s):")
        for i, r in enumerate(results):
            print(f"  [{i}] {r.strip()!r}")

        assert len(results) == 3
        for r in results:
            assert len(r) > 0


@skip_no_api_key
class TestValidatorRealCall:
    """Validator parsing with a real LLM call."""

    def test_parse_integer(self, llm_model_and_key) -> None:
        """Validator should parse the response into an int."""
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            budget_usd=0.50,
            window="hourly",
            api_key=key,
            max_tokens=50,
            temperature=0.0,
            num_retries=2,
        )
        try:
            def extract_number(text: str) -> int:
                match = re.search(r"\d+", text)
                if match is None:
                    raise ValueError(f"No number found in: {text!r}")
                return int(match.group())

            result = llm.call_llm(
                prompt="What is 17 * 3? Reply with just the number.",
                validator=extract_number,
            ).result(timeout=30.0)

            print(f"Result: {result} (type: {type(result).__name__})")
            assert isinstance(result, int)
            assert result == 51
        finally:
            llm.stop()


@skip_no_api_key
class TestAutoGenModelClient:
    """SlowBurnModelClient (AG2 protocol) with a real LLM call."""

    def test_create_and_retrieve(self, llm_model_and_key) -> None:
        """create() should return a response, log cost, and expose usage."""
        model, key = llm_model_and_key

        limit_set = LimitSet(
            limits=[CostLimit(budget_usd=0.50, window_seconds=3600)],
            mode="Threads",
            shared=True,
        )
        ag_reporter = CostReporter()

        client = SlowBurnModelClient(
            config={"model": model},
            limit_set=limit_set,
            reporter=ag_reporter,
        )

        response = client.create({
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            "model": model,
            "max_tokens": 50,
            "temperature": 0.0,
            "timeout": 120.0,
        })

        content = response.choices[0].message.content
        print(f"Response: {content!r}")
        print(f"Cost: ${client.cost(response):.6f}")
        print(f"Usage: {client.get_usage(response)}")

        msgs = client.message_retrieval(response)
        print(f"Messages: {msgs}")

        assert ag_reporter.num_calls == 1
        assert ag_reporter.total_cost() > 0
        assert len(msgs) > 0


@skip_no_api_key
class TestCostReporterFormats:
    """Verify reporter output formats with real accumulated data."""

    def test_markdown_and_latex(self, llm_model_and_key) -> None:
        """Make a call, then verify markdown/latex output is non-empty."""
        model, key = llm_model_and_key
        llm = create_llm(
            model=model, budget_usd=0.50, window="hourly",
            api_key=key, max_tokens=50, temperature=0.3,
        )
        try:
            llm.call_llm(prompt="Say hi").result(timeout=30.0)
            reporter = llm.get_reporter().result(timeout=5.0)

            md = reporter.to_markdown()
            print(f"\n--- Markdown ---\n{md}")
            assert "Total" in md

            tex = reporter.to_latex()
            print(f"\n--- LaTeX ---\n{tex}")
            assert "Total" in tex
        finally:
            llm.stop()


@skip_no_api_key
class TestMultiTurnRealCalls:
    """Multi-turn conversation API with real LLM calls."""

    def test_history_preserves_context(self, llm_model_and_key) -> None:
        """Two-turn conversation where the second turn references the first.

        Steps:
        1. First turn: tell the LLM a fact ("My name is Zephyr").
        2. Second turn: ask the LLM to recall it ("What is my name?").
        3. Verify the response contains "Zephyr" (proves history was sent).
        """
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            budget_usd=0.50,
            window="hourly",
            api_key=key,
            max_tokens=100,
            temperature=0.0,
        )
        try:
            messages = llm.call_llm(
                prompt="My name is Zephyr. Remember it.",
                system_prompt="You are a helpful assistant with perfect memory.",
                history=[],
            ).result(timeout=30.0)

            assert isinstance(messages, list)
            assert len(messages) >= 2
            assert messages[-1]["role"] == "assistant"
            print(f"Turn 1 response: {messages[-1]['content']!r}")

            messages = llm.call_llm(
                prompt="What is my name?",
                history=messages,
            ).result(timeout=30.0)

            assert isinstance(messages, list)
            final_response = messages[-1]["content"]
            print(f"Turn 2 response: {final_response!r}")
            assert "Zephyr" in final_response

            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 2
            print(f"Cost: ${reporter.total_cost():.6f}")
        finally:
            llm.stop()

    def test_tool_call_round_trip(self, llm_model_and_key) -> None:
        """LLM returns a tool_call, we append the result, LLM produces final text.

        Steps:
        1. Send a prompt with a tool schema asking for the weather.
        2. Verify the assistant message contains tool_calls.
        3. Append a fake tool result.
        4. Re-submit; verify the assistant produces a text response using the tool result.
        """
        model, key = llm_model_and_key
        tool_schemas = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }]
        llm = create_llm(
            model=model,
            budget_usd=0.50,
            window="hourly",
            api_key=key,
            max_tokens=150,
            temperature=0.0,
            tools=tool_schemas,
            tool_choice="required",
        )
        try:
            messages = llm.call_llm(
                prompt="What is the weather in Paris right now?",
                system_prompt="Use the get_weather tool to answer weather questions.",
                history=[],
            ).result(timeout=30.0)

            assert isinstance(messages, list)
            assistant_message = messages[-1]
            assert assistant_message["role"] == "assistant"
            assert assistant_message.get("tool_calls") is not None
            assert len(assistant_message["tool_calls"]) >= 1
            print(f"Tool calls: {assistant_message['tool_calls']}")

            tool_call = assistant_message["tool_calls"][0]
            assert tool_call["function"]["name"] == "get_weather"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": '{"temperature": "18°C", "condition": "partly cloudy"}',
            })

            messages = llm.call_llm(
                prompt="",
                history=messages,
                tool_choice=None,
            ).result(timeout=30.0)

            assert isinstance(messages, list)
            final_response = messages[-1]["content"]
            print(f"Final response: {final_response!r}")
            assert final_response is not None
            assert len(final_response) > 0

            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 2
            print(f"Cost: ${reporter.total_cost():.6f}")
        finally:
            llm.stop()
