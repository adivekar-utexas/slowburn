"""
End-to-end integration tests with real LLM calls.

API keys are loaded from .env by conftest.py at session startup.
Tests are skipped if no key is available.
"""

import re
import time

import litellm
import pytest
from concurry import LimitSet

from slowburn import CostLimit, CostReporter, _DEFAULT_RETRY_ON, create_llm
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
            limits=dict(budget_per_day=1.0),
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
            limits=dict(budget_per_day=0.50),
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
            limits=[CostLimit(budget_usd=0.50, window=3600)],
            mode="Threads",
            shared=True,
        )
        ag_reporter = CostReporter()

        client = SlowBurnModelClient(
            config={"model": model},
            limit_set=limit_set,
            reporter=ag_reporter,
        )

        response = client.create(
            {
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "model": model,
                "max_tokens": 50,
                "temperature": 0.0,
                "timeout": 120.0,
            }
        )

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
            model=model,
            limits=dict(budget_per_day=0.50),
            window="hourly",
            api_key=key,
            max_tokens=50,
            temperature=0.3,
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
class TestRetryConfigRealCalls:
    """E2E tests for create_llm() retry parameters with real LLM calls.

    These tests verify that:
    - All four retry params (retry_on, retry_wait, retry_algorithm, retry_jitter)
      are accepted and the worker functions correctly with a real LLM.
    - The default retry_on list (which now includes litellm error types) does not
      interfere with normal successful calls.
    - Transient ValueError failures from a validator are retried and recovered.
    """

    def test_explicit_retry_params_with_real_call(self, llm_model_and_key) -> None:
        """create_llm() with all four retry params makes a real call successfully.

        Steps:
        1. Create worker with explicit retry_on, retry_wait, retry_algorithm,
           retry_jitter.
        2. Make a real call.
        3. Verify success and cost recorded.
        """
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            limits=dict(budget_per_day=0.50),
            window="hourly",
            api_key=key,
            max_tokens=50,
            temperature=0.0,
            num_retries=3,
            retry_on=_DEFAULT_RETRY_ON,
            retry_wait=1.0,
            retry_algorithm="Exponential",
            retry_jitter=0.1,
        )
        try:
            result = llm.call_llm(prompt="What is 3 + 3? Reply with just the number.").result(timeout=30.0)
            print(f"Response: {result!r}")

            assert len(result) > 0
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()

    def test_linear_backoff_with_real_call(self, llm_model_and_key) -> None:
        """retry_algorithm='Linear' works correctly with a real LLM call.

        Steps:
        1. Create worker with retry_algorithm='Linear'.
        2. Make a real call.
        3. Verify success.
        """
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            limits=dict(budget_per_day=0.50),
            window="hourly",
            api_key=key,
            max_tokens=50,
            temperature=0.0,
            retry_algorithm="Linear",
            retry_wait=1.0,
        )
        try:
            result = llm.call_llm(prompt="Name one continent. Reply with just its name.").result(timeout=30.0)
            print(f"Response: {result!r}")
            assert len(result) > 0
        finally:
            llm.stop()

    def test_validator_retry_recovers_after_transient_failure(self, llm_model_and_key) -> None:
        """A validator that fails on first call triggers a retry and eventually succeeds.

        This is the closest real-LLM analogue to the APIError retry test — we
        inject a controlled ValueError from the validator on the first attempt,
        verifying that the retry machinery replays the LLM call and the validator
        passes on the second attempt.

        Steps:
        1. Define a stateful validator that raises ValueError on attempt 1 and
           parses normally on attempt 2+.
        2. Call call_llm() with num_retries=2.
        3. Verify the final result is correct (the validator passed on retry).
        4. Verify the LLM was called twice (one failure + one success).
        """
        model, key = llm_model_and_key

        call_count = {"n": 0}

        def flaky_number_validator(text: str) -> int:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValueError(f"Simulated transient parse failure on attempt 1 (text={text!r})")
            match = re.search(r"\d+", text)
            if match is None:
                raise ValueError(f"No number found in: {text!r}")
            return int(match.group())

        llm = create_llm(
            model=model,
            limits=dict(budget_per_day=0.50),
            window="hourly",
            api_key=key,
            max_tokens=50,
            temperature=0.0,
            num_retries=2,
            retry_on=_DEFAULT_RETRY_ON,
            retry_wait=1.0,
            retry_algorithm="Exponential",
        )
        try:
            result = llm.call_llm(
                prompt="What is 6 * 7? Reply with just the number.",
                validator=flaky_number_validator,
            ).result(timeout=60.0)

            print(f"Result: {result} (validator called {call_count['n']} times)")
            assert isinstance(result, int)
            assert result == 42
            assert call_count["n"] == 2
        finally:
            llm.stop()

    def test_default_retry_on_does_not_break_normal_calls(self, llm_model_and_key) -> None:
        """Default retry_on (comprehensive litellm error list) doesn't affect success path.

        Steps:
        1. Create worker using all defaults (including _DEFAULT_RETRY_ON).
        2. Make a batch of 3 real calls.
        3. Verify all succeed — confirms the new error list doesn't introduce
           false retries on successful responses.
        """
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            limits=dict(budget_per_day=0.50),
            window="hourly",
            api_key=key,
            max_tokens=30,
            temperature=0.0,
        )
        try:
            results = llm.call_llm_batch(
                prompts=["Name one ocean.", "Name one mountain.", "Name one river."],
            ).result(timeout=60.0)

            print("Batch results:")
            for i, r in enumerate(results):
                print(f"  [{i}] {r.strip()!r}")

            assert len(results) == 3
            for r in results:
                assert len(r) > 0

            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 3
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
            limits=dict(budget_per_day=0.50),
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
        tool_schemas = [
            {
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
            }
        ]
        llm = create_llm(
            model=model,
            limits=dict(budget_per_day=0.50),
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

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": '{"temperature": "18°C", "condition": "partly cloudy"}',
                }
            )

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
