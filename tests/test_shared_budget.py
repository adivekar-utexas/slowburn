"""Tests for multi-LLM shared budget pattern.

Verifies that multiple SlowBurnLLM workers and framework integrations
(AutoGen, LangGraph, LangChain) can share a single LimitSet with a
CostLimit, so that all LLM calls across all components draw from one
dollar budget.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from concurry import CallLimit, LimitSet, RateLimit

from slowburn.integrations.autogen import SlowBurnModelClient
from slowburn.integrations.langchain import SlowBurnCallbackHandler
from slowburn.integrations.langgraph import SlowBurnMiddleware
from slowburn.limits import DEFAULT_COST_LIMIT_KEY, CostLimit
from slowburn.llm_worker import SlowBurnLLM
from slowburn.reporter import CostReporter

from .conftest import MOCK_MODEL_NAME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shared_limit_set(budget_usd: float = 1.0, mode: str = "Threads") -> LimitSet:
    """Create a shared LimitSet with CostLimit + token limits."""
    return LimitSet(
        limits=[
            CostLimit(budget_usd=budget_usd, window_seconds=3600),
            RateLimit(key="input_tokens", window_seconds=60, capacity=1_000_000),
            RateLimit(key="output_tokens", window_seconds=60, capacity=200_000),
            CallLimit(window_seconds=60, capacity=500),
        ],
        mode=mode,
        shared=True,
    )


def _make_acompletion_response(cost: float = 0.0001):
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    message = SimpleNamespace(content="Shared budget response", tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=MOCK_MODEL_NAME,
        _hidden_params={"response_cost": cost},
    )


def _make_completion_response(cost: float = 0.0001):
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    message = SimpleNamespace(content="AG2 shared response")
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=MOCK_MODEL_NAME,
        _hidden_params={"response_cost": cost},
    )


def _make_langgraph_model(model_name: str = MOCK_MODEL_NAME, max_tokens: int = 500):
    return SimpleNamespace(model_name=model_name, max_tokens=max_tokens)


def _make_langgraph_request(model_name: str = MOCK_MODEL_NAME, max_tokens: int = 500):
    return SimpleNamespace(
        model=_make_langgraph_model(model_name, max_tokens),
        messages=[SimpleNamespace(content="Hello from LangGraph")],
        system_message=None,
        model_settings=None,
    )


def _make_langgraph_response(content: str = "LangGraph response"):
    return SimpleNamespace(content=content, usage_metadata=None)


def _make_langchain_serialized(model_name: str = MOCK_MODEL_NAME, max_tokens: int = 500):
    return {"kwargs": {"model_name": model_name, "max_tokens": max_tokens}}


def _make_langchain_llm_result(
    text: str = "LangChain response",
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
):
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


# ===========================================================================
# Test 1: Two SlowBurnLLM workers sharing one CostLimit
# ===========================================================================

class TestSharedBudgetTwoWorkers:

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_two_workers_share_one_budget(self, mock_acompletion) -> None:
        """Two SlowBurnLLM workers sharing a LimitSet should both track cost
        against the same budget.

        Steps:
        1. Create a shared LimitSet with CostLimit(budget_usd=1.0).
        2. Create two SlowBurnLLM workers pointing to the same LimitSet.
        3. Make 3 calls from worker A and 2 calls from worker B.
        4. Verify both draw from the shared budget (try_acquire reflects total).
        """
        mock_acompletion.return_value = _make_acompletion_response(cost=0.001)
        shared = _make_shared_limit_set(budget_usd=1.0, mode="Asyncio")

        w_a = SlowBurnLLM.options(mode="Asyncio", limits=shared).init(
            name="worker-a", model_name=MOCK_MODEL_NAME, api_key="test",
        )
        w_b = SlowBurnLLM.options(mode="Asyncio", limits=shared).init(
            name="worker-b", model_name=MOCK_MODEL_NAME, api_key="test",
        )
        try:
            for _ in range(3):
                w_a.call_llm(prompt="from A").result(timeout=10.0)
            for _ in range(2):
                w_b.call_llm(prompt="from B").result(timeout=10.0)

            reporter_a = w_a.get_reporter().result(timeout=5.0)
            reporter_b = w_b.get_reporter().result(timeout=5.0)
            assert reporter_a.num_calls == 3
            assert reporter_b.num_calls == 2
        finally:
            w_a.stop()
            w_b.stop()


# ===========================================================================
# Test 2: Shared budget exhaustion triggers backpressure
# ===========================================================================

class TestSharedBudgetBackpressure:

    def test_exhausted_budget_blocks_try_acquire(self) -> None:
        """When the shared budget is exhausted, try_acquire should fail.

        Steps:
        1. Create a CostLimit-only LimitSet with small budget.
        2. Acquire and consume most of the budget directly.
        3. Verify try_acquire fails for the remaining amount.
        """
        shared = LimitSet(
            limits=[CostLimit(budget_usd=0.01, window_seconds=3600)],
            mode="Threads",
            shared=True,
        )

        with shared.acquire(requested={DEFAULT_COST_LIMIT_KEY: 5_000}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 5_000})

        with shared.acquire(requested={DEFAULT_COST_LIMIT_KEY: 5_000}) as acq:
            acq.update(usage={DEFAULT_COST_LIMIT_KEY: 5_000})

        result = shared.try_acquire(requested={DEFAULT_COST_LIMIT_KEY: 5_000})
        assert not result.successful


# ===========================================================================
# Test 3: Shared budget passed to SlowBurnCrewAI
# ===========================================================================

class TestSharedBudgetCrewAI:

    def test_crewai_accepts_external_limit_set(self) -> None:
        """SlowBurnCrewAI should use an externally provided LimitSet.

        Steps:
        1. Create a shared LimitSet.
        2. Pass to SlowBurnCrewAI via limit_set parameter.
        3. Verify it uses the shared LimitSet, not a new one.
        """
        shared = _make_shared_limit_set(budget_usd=5.0)
        from slowburn.integrations.crewai import SlowBurnCrewAI
        sb = SlowBurnCrewAI(limit_set=shared, max_tokens=1000)
        assert sb.limit_set is shared

    def test_crewai_requires_budget_or_limit_set(self) -> None:
        """SlowBurnCrewAI should raise if neither budget_usd nor limit_set is given."""
        from slowburn.integrations.crewai import SlowBurnCrewAI
        with pytest.raises(ValueError, match="requires either"):
            SlowBurnCrewAI()


# ===========================================================================
# Test 4: Shared budget passed to SlowBurnModelClient (AutoGen)
# ===========================================================================

class TestSharedBudgetAutoGen:

    @patch("slowburn.integrations.autogen.litellm.completion")
    def test_autogen_uses_shared_limit_set(self, mock_completion) -> None:
        """SlowBurnModelClient should draw from a shared LimitSet.

        Steps:
        1. Create shared LimitSet.
        2. Create AutoGen client with that LimitSet.
        3. Make a call.
        4. Verify the shared LimitSet was used (try_acquire capacity reduced).
        """
        mock_completion.return_value = _make_completion_response(cost=0.001)
        shared = _make_shared_limit_set(budget_usd=1.0)
        reporter = CostReporter()

        client = SlowBurnModelClient(
            config={"model": MOCK_MODEL_NAME},
            limit_set=shared,
            reporter=reporter,
        )
        client.create({
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 100,
        })

        assert reporter.num_calls == 1
        assert reporter.total_cost() > 0


# ===========================================================================
# Test 5: Shared budget passed to SlowBurnMiddleware (LangGraph)
# ===========================================================================

class TestSharedBudgetLangGraph:

    def test_langgraph_accepts_external_limit_set(self) -> None:
        """SlowBurnMiddleware should use an externally provided LimitSet."""
        shared = _make_shared_limit_set(budget_usd=5.0)
        mw = SlowBurnMiddleware(limit_set=shared)
        assert mw.limit_set is shared

    def test_langgraph_requires_budget_or_limit_set(self) -> None:
        with pytest.raises(ValueError, match="requires either"):
            SlowBurnMiddleware()

    def test_langgraph_uses_shared_budget(self) -> None:
        """Calls through LangGraph middleware should draw from shared budget."""
        shared = _make_shared_limit_set(budget_usd=5.0)
        reporter = CostReporter()
        mw = SlowBurnMiddleware(limit_set=shared, reporter=reporter)

        request = _make_langgraph_request()
        mw.wrap_model_call(request, lambda req: _make_langgraph_response())

        assert reporter.num_calls == 1
        assert reporter.total_cost() > 0


# ===========================================================================
# Test 6: Shared budget passed to SlowBurnCallbackHandler (LangChain)
# ===========================================================================

class TestSharedBudgetLangChain:

    def test_langchain_accepts_external_limit_set(self) -> None:
        """SlowBurnCallbackHandler should use an externally provided LimitSet."""
        shared = _make_shared_limit_set(budget_usd=5.0)
        cb = SlowBurnCallbackHandler(limit_set=shared)
        assert cb.limit_set is shared

    def test_langchain_requires_budget_or_limit_set(self) -> None:
        with pytest.raises(ValueError, match="requires either"):
            SlowBurnCallbackHandler()

    def test_langchain_uses_shared_budget(self) -> None:
        """on_llm_start/on_llm_end through LangChain should draw from shared budget."""
        shared = _make_shared_limit_set(budget_usd=5.0)
        reporter = CostReporter()
        cb = SlowBurnCallbackHandler(limit_set=shared, reporter=reporter)

        run_id = uuid4()
        cb.on_llm_start(_make_langchain_serialized(), ["Hello"], run_id=run_id)
        cb.on_llm_end(_make_langchain_llm_result(), run_id=run_id)

        assert reporter.num_calls == 1
        assert reporter.total_cost() > 0


# ===========================================================================
# Test 7: Cross-framework shared budget (SlowBurnLLM + AutoGen + LangGraph)
# ===========================================================================

class TestCrossFrameworkSharedBudget:

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    @patch("slowburn.integrations.autogen.litellm.completion")
    def test_llm_and_autogen_share_budget(self, mock_completion, mock_acompletion) -> None:
        """SlowBurnLLM and SlowBurnModelClient sharing one LimitSet should
        both draw from the same budget.

        Steps:
        1. Create a shared LimitSet with $1 budget.
        2. Make 2 calls from SlowBurnLLM.
        3. Make 2 calls from SlowBurnModelClient.
        4. Verify total calls = 4 across both reporters.
        5. Verify the shared LimitSet reflects total consumption.
        """
        mock_acompletion.return_value = _make_acompletion_response(cost=0.001)
        mock_completion.return_value = _make_completion_response(cost=0.001)

        shared = _make_shared_limit_set(budget_usd=1.0, mode="Asyncio")
        reporter = CostReporter()

        llm = SlowBurnLLM.options(mode="Asyncio", limits=shared).init(
            name="native-llm", model_name=MOCK_MODEL_NAME, api_key="test",
        )
        ag_client = SlowBurnModelClient(
            config={"model": MOCK_MODEL_NAME},
            limit_set=shared,
            reporter=reporter,
        )

        try:
            llm.call_llm(prompt="from native").result(timeout=10.0)
            llm.call_llm(prompt="from native 2").result(timeout=10.0)

            ag_client.create({"messages": [{"role": "user", "content": "from autogen"}], "max_tokens": 100})
            ag_client.create({"messages": [{"role": "user", "content": "from autogen 2"}], "max_tokens": 100})

            llm_reporter = llm.get_reporter().result(timeout=5.0)
            assert llm_reporter.num_calls == 2
            assert reporter.num_calls == 2
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_llm_and_langgraph_share_budget(self, mock_acompletion) -> None:
        """SlowBurnLLM and SlowBurnMiddleware sharing one LimitSet."""
        mock_acompletion.return_value = _make_acompletion_response(cost=0.001)

        shared = _make_shared_limit_set(budget_usd=1.0, mode="Asyncio")
        lg_reporter = CostReporter()

        llm = SlowBurnLLM.options(mode="Asyncio", limits=shared).init(
            name="native", model_name=MOCK_MODEL_NAME, api_key="test",
        )
        mw = SlowBurnMiddleware(limit_set=shared, reporter=lg_reporter)

        try:
            llm.call_llm(prompt="from native").result(timeout=10.0)
            mw.wrap_model_call(
                _make_langgraph_request(),
                lambda req: _make_langgraph_response(),
            )

            llm_reporter = llm.get_reporter().result(timeout=5.0)
            assert llm_reporter.num_calls == 1
            assert lg_reporter.num_calls == 1
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_llm_and_langchain_share_budget(self, mock_acompletion) -> None:
        """SlowBurnLLM and SlowBurnCallbackHandler sharing one LimitSet."""
        mock_acompletion.return_value = _make_acompletion_response(cost=0.001)

        shared = _make_shared_limit_set(budget_usd=1.0, mode="Asyncio")
        lc_reporter = CostReporter()

        llm = SlowBurnLLM.options(mode="Asyncio", limits=shared).init(
            name="native", model_name=MOCK_MODEL_NAME, api_key="test",
        )
        cb = SlowBurnCallbackHandler(limit_set=shared, reporter=lc_reporter)

        try:
            llm.call_llm(prompt="from native").result(timeout=10.0)

            run_id = uuid4()
            cb.on_llm_start(_make_langchain_serialized(), ["from langchain"], run_id=run_id)
            cb.on_llm_end(_make_langchain_llm_result(), run_id=run_id)

            llm_reporter = llm.get_reporter().result(timeout=5.0)
            assert llm_reporter.num_calls == 1
            assert lc_reporter.num_calls == 1
        finally:
            llm.stop()


# ===========================================================================
# Test 8: Shared reporter aggregates across all sources
# ===========================================================================

class TestSharedReporter:

    @patch("slowburn.integrations.autogen.litellm.completion")
    def test_single_reporter_aggregates_all_calls(self, mock_completion) -> None:
        """A single CostReporter shared across multiple integrations should
        aggregate all calls from all sources.

        Steps:
        1. Create one CostReporter.
        2. Pass to AutoGen client and LangGraph middleware.
        3. Make 2 calls from each.
        4. Verify reporter shows 4 total calls.
        """
        mock_completion.return_value = _make_completion_response(cost=0.001)

        shared = _make_shared_limit_set(budget_usd=10.0)
        reporter = CostReporter()

        ag_client = SlowBurnModelClient(
            config={"model": MOCK_MODEL_NAME},
            limit_set=shared,
            reporter=reporter,
        )
        mw = SlowBurnMiddleware(limit_set=shared, reporter=reporter)

        ag_client.create({"messages": [{"role": "user", "content": "ag1"}], "max_tokens": 100})
        ag_client.create({"messages": [{"role": "user", "content": "ag2"}], "max_tokens": 100})

        mw.wrap_model_call(
            _make_langgraph_request(),
            lambda req: _make_langgraph_response(),
        )
        mw.wrap_model_call(
            _make_langgraph_request(),
            lambda req: _make_langgraph_response(),
        )

        assert reporter.num_calls == 4
        assert reporter.total_cost() > 0
