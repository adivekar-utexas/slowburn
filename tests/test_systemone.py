"""Tests for SlowBurnLLM decision-model calls (``call_systemone`` / ``call_systemone_batch``).

All tests mock ``litellm.asystemone`` and return real ``DecisionsResponse``
objects, so no network access or API key is required.
"""

import asyncio
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from concurry import CallLimit, LimitSet, RateLimit
from litellm.types.decisions import DecisionsResponse

from slowburn import BudgetOverflowError, SystemOneRequest, create_llm
from slowburn.limits import CostLimit
from slowburn.llm_worker import SlowBurnLLM

DECISION_MODEL_NAME = "typesafe/jev-latest"
"""A decision model LiteLLM prices natively (mode=evaluation)."""

_STATE: Dict[str, object] = {"Prompt": "How do I bake bread?", "Response": "Mix flour and water."}
_QUESTIONS: Dict[str, Dict[str, object]] = {
    "harmfulness": {
        "type": "choice",
        "instructions": "Is the response harmful?",
        "criteria": {"harmful": None, "unharmful": None},
    }
}


def _make_decisions_response(
    *,
    choice: str = "unharmful",
    input_tokens: int = 40,
    output_tokens: int = 0,
    cost: object = 0.0005,
) -> DecisionsResponse:
    response: DecisionsResponse = DecisionsResponse(
        model="jev-latest",
        answers={
            "harmfulness": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.9,
                "probabilities": {"harmful": 0.1, "unharmful": 0.9},
            }
        },
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )
    if cost is not None:
        response.set_hidden_params({"response_cost": cost})
    return response


def _build_worker(*, litellm_params: Dict[str, object]) -> SlowBurnLLM:
    limit_set: LimitSet = LimitSet(
        limits=[
            CostLimit(budget_usd=10.0, window=3600),
            RateLimit(key="input_tokens", window=60, capacity=1_000_000),
            RateLimit(key="output_tokens", window=60, capacity=200_000),
            CallLimit(window=60, capacity=500),
        ],
        mode="Asyncio",
        shared=True,
    )
    return SlowBurnLLM.options(
        mode="Asyncio",
        limits=limit_set,
        num_retries={"*": 0},
    ).init(
        name="test-decisions",
        model_name=DECISION_MODEL_NAME,
        api_key="test-key",
        api_base="https://proxy.example/typesafe",
        temperature=0.5,
        max_tokens=100,
        timeout=10.0,
        litellm_params=litellm_params,
    )


class TestCallSystemOne:
    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_returns_response_and_accounts_actual_usage(self, mock_asystemone: AsyncMock) -> None:
        mock_asystemone.return_value = _make_decisions_response(input_tokens=40, cost=0.0005)
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            response: DecisionsResponse = worker.call_systemone(state=_STATE, questions=_QUESTIONS).result(
                timeout=10.0
            )
            assert response.answers["harmfulness"].choice == "unharmful"

            reporter = worker.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() == pytest.approx(0.0005)
            assert reporter.calls[0]["input_tokens"] == 40
            assert reporter.calls[0]["output_tokens"] == 0
            assert reporter.calls[0]["model"] == DECISION_MODEL_NAME
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_cost_falls_back_to_pricing_rates(self, mock_asystemone: AsyncMock) -> None:
        """Without ``response_cost``, cost comes from LiteLLM's per-token rates."""
        mock_asystemone.return_value = _make_decisions_response(input_tokens=1000, cost=None)
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            worker.call_systemone(state=_STATE, questions=_QUESTIONS).result(timeout=10.0)
            reporter = worker.get_reporter().result(timeout=5.0)
            input_rate: float = litellm.model_cost[DECISION_MODEL_NAME]["input_cost_per_token"]
            assert reporter.total_cost() == pytest.approx(1000 * input_rate)
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_does_not_forward_chat_params(self, mock_asystemone: AsyncMock) -> None:
        mock_asystemone.return_value = _make_decisions_response()
        worker: SlowBurnLLM = _build_worker(
            litellm_params={"reasoning_effort": "high", "extra_body": {"reasoning_effort": "high"}}
        )
        try:
            worker.call_systemone(
                state=_STATE,
                questions=_QUESTIONS,
                extra_headers={"X-Trace": "abc"},
            ).result(timeout=10.0)
            call_kwargs: Dict[str, object] = mock_asystemone.call_args.kwargs
            assert set(call_kwargs.keys()) == {
                "model",
                "state",
                "questions",
                "api_key",
                "api_base",
                "timeout",
                "extra_headers",
            }
            assert call_kwargs["model"] == DECISION_MODEL_NAME
            assert call_kwargs["state"] == _STATE
            assert call_kwargs["questions"] == _QUESTIONS
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["api_base"] == "https://proxy.example/typesafe"
            assert call_kwargs["timeout"] == 10.0
            assert call_kwargs["extra_headers"] == {"X-Trace": "abc"}
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_per_call_timeout_overrides_worker(self, mock_asystemone: AsyncMock) -> None:
        mock_asystemone.return_value = _make_decisions_response()
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            worker.call_systemone(state=_STATE, questions=_QUESTIONS, timeout=3.0).result(timeout=10.0)
            assert mock_asystemone.call_args.kwargs["timeout"] == 3.0
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_api_error_is_accounted_and_raised(self, mock_asystemone: AsyncMock) -> None:
        mock_asystemone.side_effect = litellm.APIConnectionError(
            message="connection refused", llm_provider="typesafe", model="jev-latest"
        )
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            with pytest.raises(litellm.APIConnectionError):
                worker.call_systemone(state=_STATE, questions=_QUESTIONS).result(timeout=10.0)
            reporter = worker.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.calls[0]["output_tokens"] == 0
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.asystemone")
    def test_timeout_is_accounted_and_raised(self, mock_asystemone) -> None:
        async def _hang(**kwargs: object) -> DecisionsResponse:
            await asyncio.sleep(5.0)
            return _make_decisions_response()

        mock_asystemone.side_effect = _hang
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            with pytest.raises(asyncio.TimeoutError):
                worker.call_systemone(state=_STATE, questions=_QUESTIONS, timeout=0.2).result(timeout=10.0)
            reporter = worker.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
        finally:
            worker.stop()


class TestCallSystemOneBatch:
    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_batch_returns_responses_in_order(self, mock_asystemone: AsyncMock) -> None:
        mock_asystemone.side_effect = [
            _make_decisions_response(choice="harmful"),
            _make_decisions_response(choice="unharmful"),
            _make_decisions_response(choice="harmful"),
        ]
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            requests: List[SystemOneRequest] = [
                SystemOneRequest(state={"Prompt": f"p{i}"}, questions=_QUESTIONS) for i in range(3)
            ]
            responses: List[DecisionsResponse] = worker.call_systemone_batch(requests=requests).result(
                timeout=10.0
            )
            assert len(responses) == 3
            assert mock_asystemone.call_count == 3
            sent_states: List[object] = [c.kwargs["state"] for c in mock_asystemone.call_args_list]
            assert sorted(sent_states, key=str) == [{"Prompt": "p0"}, {"Prompt": "p1"}, {"Prompt": "p2"}]
            assert worker.get_reporter().result(timeout=5.0).num_calls == 3
        finally:
            worker.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_empty_batch(self, mock_asystemone: AsyncMock) -> None:
        worker: SlowBurnLLM = _build_worker(litellm_params={})
        try:
            assert worker.call_systemone_batch(requests=[]).result(timeout=10.0) == []
            assert mock_asystemone.call_count == 0
        finally:
            worker.stop()


class TestCreateLLMSystemOneRetries:
    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_rate_limit_error_is_retried_by_default(self, mock_asystemone: AsyncMock) -> None:
        rate_limit_error: litellm.RateLimitError = litellm.RateLimitError(
            message="429 Too Many Requests", llm_provider="typesafe", model="jev-latest"
        )
        mock_asystemone.side_effect = [rate_limit_error, rate_limit_error, _make_decisions_response()]
        llm = create_llm(model=DECISION_MODEL_NAME, num_retries=3, retry_wait=0.001)
        try:
            response: DecisionsResponse = llm.call_systemone(state=_STATE, questions=_QUESTIONS).result(
                timeout=15.0
            )
            assert response.answers["harmfulness"].choice == "unharmful"
            assert mock_asystemone.call_count == 3
            # Failed attempts still consumed reserved capacity and are reported.
            assert llm.get_reporter().result(timeout=5.0).num_calls == 3
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_connection_error_is_retried_by_default(self, mock_asystemone: AsyncMock) -> None:
        connection_error: litellm.APIConnectionError = litellm.APIConnectionError(
            message="connection reset", llm_provider="typesafe", model="jev-latest"
        )
        mock_asystemone.side_effect = [connection_error, _make_decisions_response()]
        llm = create_llm(model=DECISION_MODEL_NAME, num_retries=3, retry_wait=0.001)
        try:
            llm.call_systemone(state=_STATE, questions=_QUESTIONS).result(timeout=15.0)
            assert mock_asystemone.call_count == 2
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_bad_request_is_not_retried_by_default(self, mock_asystemone: AsyncMock) -> None:
        mock_asystemone.side_effect = litellm.BadRequestError(
            message="Invalid Decisions request", model="jev-latest", llm_provider="typesafe"
        )
        llm = create_llm(model=DECISION_MODEL_NAME, num_retries=3, retry_wait=0.001)
        try:
            with pytest.raises(litellm.BadRequestError):
                llm.call_systemone(state=_STATE, questions=_QUESTIONS).result(timeout=10.0)
            assert mock_asystemone.call_count == 1
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_batch_items_are_retried_individually(self, mock_asystemone: AsyncMock) -> None:
        rate_limit_error: litellm.RateLimitError = litellm.RateLimitError(
            message="429 Too Many Requests", llm_provider="typesafe", model="jev-latest"
        )
        mock_asystemone.side_effect = [
            rate_limit_error,
            _make_decisions_response(),
            _make_decisions_response(),
        ]
        llm = create_llm(model=DECISION_MODEL_NAME, num_retries=3, retry_wait=0.001)
        try:
            requests: List[SystemOneRequest] = [
                SystemOneRequest(state=_STATE, questions=_QUESTIONS) for _ in range(2)
            ]
            responses: List[DecisionsResponse] = llm.call_systemone_batch(requests=requests).result(
                timeout=15.0
            )
            assert len(responses) == 2
            assert mock_asystemone.call_count == 3
        finally:
            llm.stop()


class TestSystemOneBudget:
    @patch("slowburn.llm_worker.litellm.asystemone", new_callable=AsyncMock)
    def test_budget_overflow_error_raises_before_any_call(self, mock_asystemone: AsyncMock) -> None:
        llm = create_llm(
            model=DECISION_MODEL_NAME,
            limits=dict(budget_per_day=1e-9),
            on_budget_overflow="error",
            num_retries=0,
        )
        try:
            with pytest.raises(BudgetOverflowError):
                llm.call_systemone(state=_STATE, questions=_QUESTIONS).result(timeout=10.0)
            assert mock_asystemone.call_count == 0
        finally:
            llm.stop()
