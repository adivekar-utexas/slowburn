"""Tests for SlowBurnModelClient (AutoGen/AG2 integration) with mocked litellm."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from concurry import LimitSet

from .conftest import MOCK_MODEL_NAME

from slowburn.integrations.autogen import SlowBurnModelClient
from slowburn.limits import CostLimit
from slowburn.reporter import CostReporter


def _make_completion_response(
    content: str = "AG2 response",
    prompt_tokens: int = 80,
    completion_tokens: int = 40,
    model: str = MOCK_MODEL_NAME,
    cost: float = 0.002,
):
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


def _make_client(
    budget_usd: float = 10.0,
    model: str = f"slowburn/{MOCK_MODEL_NAME}",
) -> tuple:
    limit_set = LimitSet(
        limits=[CostLimit(budget_usd=budget_usd, window_seconds=3600)],
        mode="thread",
        shared=True,
    )
    reporter = CostReporter()
    config = {"model": model}
    client = SlowBurnModelClient(config=config, limit_set=limit_set, reporter=reporter)
    return client, limit_set, reporter


class TestSlowBurnModelClientInit:

    def test_strips_slowburn_prefix(self) -> None:
        client, _, _ = _make_client(model=f"slowburn/{MOCK_MODEL_NAME}")
        assert client.litellm_model == MOCK_MODEL_NAME

    def test_no_prefix(self) -> None:
        client, _, _ = _make_client(model=MOCK_MODEL_NAME)
        assert client.litellm_model == MOCK_MODEL_NAME

    def test_missing_model_raises(self) -> None:
        """Config without 'model' key should raise ValueError."""
        limit_set = LimitSet(
            limits=[CostLimit(budget_usd=1.0)],
            mode="thread", shared=True,
        )
        with pytest.raises(ValueError, match="requires 'model' in config"):
            SlowBurnModelClient(config={}, limit_set=limit_set)

    def test_creates_own_reporter_if_none(self) -> None:
        limit_set = LimitSet(
            limits=[CostLimit(budget_usd=1.0)],
            mode="thread", shared=True,
        )
        client = SlowBurnModelClient(config={"model": MOCK_MODEL_NAME}, limit_set=limit_set, reporter=None)
        assert client.reporter is not None


class TestSlowBurnModelClientCreate:

    @patch("slowburn.integrations.autogen.litellm.completion")
    def test_basic_create(self, mock_completion) -> None:
        """create() should return the response and log to reporter."""
        mock_completion.return_value = _make_completion_response(cost=0.002)
        client, _, reporter = _make_client()

        response = client.create({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": MOCK_MODEL_NAME,
            "max_tokens": 100,
        })

        assert response.choices[0].message.content == "AG2 response"
        assert reporter.num_calls == 1
        assert reporter.total_cost() == pytest.approx(0.002, abs=1e-5)

    @patch("slowburn.integrations.autogen.litellm.completion")
    def test_strips_slowburn_prefix_in_params(self, mock_completion) -> None:
        mock_completion.return_value = _make_completion_response()
        client, _, _ = _make_client()

        client.create({
            "messages": [{"role": "user", "content": "test"}],
            "model": f"slowburn/{MOCK_MODEL_NAME}",
            "max_tokens": 100,
        })

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == MOCK_MODEL_NAME

    @patch("slowburn.integrations.autogen.litellm.completion")
    def test_multiple_creates_accumulate(self, mock_completion) -> None:
        mock_completion.return_value = _make_completion_response(cost=0.001)
        client, _, reporter = _make_client()

        for _ in range(3):
            client.create({"messages": [{"role": "user", "content": "test"}], "max_tokens": 100})

        assert reporter.num_calls == 3
        assert reporter.total_cost() == pytest.approx(0.003, abs=1e-5)


class TestSlowBurnModelClientCost:

    def test_cost_from_hidden_params(self) -> None:
        client, _, _ = _make_client()
        response = _make_completion_response(cost=0.005)
        assert client.cost(response) == 0.005

    def test_cost_raises_on_missing_data(self) -> None:
        """Broken response should raise, not return stale data."""
        client, _, _ = _make_client()
        with pytest.raises(AttributeError):
            client.cost(SimpleNamespace())

    def test_cost_raises_on_zero_cost(self) -> None:
        """Zero cost in hidden_params should raise."""
        client, _, _ = _make_client()
        response = _make_completion_response(cost=0.0)
        with pytest.raises(AttributeError, match="no valid"):
            client.cost(response)


class TestSlowBurnModelClientGetUsage:

    def test_usage_dict(self) -> None:
        client, _, _ = _make_client()
        response = _make_completion_response(
            prompt_tokens=100, completion_tokens=50, cost=0.004,
        )
        usage = client.get_usage(response)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["cost"] == 0.004

    def test_usage_raises_on_broken_response(self) -> None:
        """Broken response should raise, not return zeros."""
        client, _, _ = _make_client()
        with pytest.raises(AttributeError):
            client.get_usage(SimpleNamespace())


class TestSlowBurnModelClientMessageRetrieval:

    def test_retrieves_messages(self) -> None:
        client, _, _ = _make_client()
        response = _make_completion_response(content="Hello from AG2")
        assert client.message_retrieval(response) == ["Hello from AG2"]

    def test_multiple_choices(self) -> None:
        client, _, _ = _make_client()
        msg1 = SimpleNamespace(content="choice 1")
        msg2 = SimpleNamespace(content="choice 2")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=msg1), SimpleNamespace(message=msg2)],
        )
        assert client.message_retrieval(response) == ["choice 1", "choice 2"]

    def test_raises_on_broken_response(self) -> None:
        """Broken response should raise, not return empty list."""
        client, _, _ = _make_client()
        with pytest.raises(AttributeError):
            client.message_retrieval(SimpleNamespace())
