"""Tests for the create_llm() convenience function."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from slowburn import create_llm
from slowburn.config import slowburn_config, temp_config

from .conftest import MOCK_MODEL_NAME


class TestCreateLLM:
    """Test create_llm() factory function setup and parameter handling."""

    def test_creates_worker_with_defaults(self) -> None:
        """create_llm() with model should produce a live worker.

        Steps:
        1. Call create_llm() with model.
        2. Verify the worker is alive (get_reporter succeeds).
        3. Stop the worker.
        """
        llm = create_llm(model=MOCK_MODEL_NAME)
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 0
        finally:
            llm.stop()

    def test_daily_window_alias(self) -> None:
        """``limits=dict(budget_per_day=...)`` should set a 86400-second budget window."""
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_day=1.0))
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_hourly_window_alias(self) -> None:
        """``limits=dict(budget_per_hour=...)`` should work."""
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_hour=1.0))
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_minutely_window_alias(self) -> None:
        """``limits=dict(budget_per_minute=...)`` should work."""
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_minute=1.0))
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_numeric_window(self) -> None:
        """A canonical CostLimit with a numeric window (seconds) should work."""
        from slowburn import CostLimit

        llm = create_llm(
            model=MOCK_MODEL_NAME,
            limits=dict(budget=[CostLimit(budget_usd=1.0, window=7200)]),
        )
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_invalid_window_alias_raises(self) -> None:
        """An unrecognized shorthand should be rejected."""
        with pytest.raises(Exception, match="biweekly|kwarg|extra"):
            create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_biweekly=1.0))

    def test_custom_name(self) -> None:
        """Passing name= should set the worker name."""
        llm = create_llm(model=MOCK_MODEL_NAME, name="my-custom-worker")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_default_retry_on_includes_litellm_api_errors(self) -> None:
        """create_llm() without explicit retry_on uses _DEFAULT_RETRY_ON.

        Steps:
        1. Import the module-level constant.
        2. Verify all key litellm transient error types are present.
        3. Verify ValueError and asyncio.TimeoutError are still included.
        """
        from slowburn import _DEFAULT_RETRY_ON

        expected = [
            ValueError,
            asyncio.TimeoutError,
            litellm.Timeout,
            litellm.APIError,
            litellm.APIConnectionError,
            litellm.BadRequestError,
            litellm.InternalServerError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
        ]
        for exc_type in expected:
            assert exc_type in _DEFAULT_RETRY_ON, f"{exc_type.__name__} missing from _DEFAULT_RETRY_ON"

    def test_custom_retry_on_accepted(self) -> None:
        """Passing an explicit retry_on list is accepted and worker starts."""
        llm = create_llm(model=MOCK_MODEL_NAME, retry_on=[ValueError])
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_custom_retry_wait_accepted(self) -> None:
        """Passing retry_wait= starts the worker without error."""
        llm = create_llm(model=MOCK_MODEL_NAME, retry_wait=2.0)
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_custom_retry_algorithm_accepted(self) -> None:
        """Passing retry_algorithm='Linear' starts the worker without error."""
        llm = create_llm(model=MOCK_MODEL_NAME, retry_algorithm="Linear")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_custom_retry_jitter_accepted(self) -> None:
        """Passing retry_jitter=0.5 starts the worker without error."""
        llm = create_llm(model=MOCK_MODEL_NAME, retry_jitter=0.5)
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_retry_wait_small_positive_accepted(self) -> None:
        """retry_wait=0.001 (near-instant retry) is a valid configuration."""
        llm = create_llm(model=MOCK_MODEL_NAME, retry_wait=0.001)
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_retry_defaults_flow_from_config(self) -> None:
        """create_llm() reads retry_wait/algorithm/jitter from slowburn_config."""
        from concurry import RetryAlgorithm

        with temp_config(retry_wait=5.0, retry_algorithm=RetryAlgorithm.Linear, retry_jitter=0.0):
            llm = create_llm(model=MOCK_MODEL_NAME)
            try:
                reporter = llm.get_reporter().result(timeout=5.0)
                assert reporter is not None
            finally:
                llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_litellm_api_error_triggers_retry(self, mock_acompletion) -> None:
        """litellm.APIError is retried when included in retry_on.

        Steps:
        1. Mock acompletion to raise APIError twice, then succeed.
        2. Create worker with retry_on=[litellm.APIError], num_retries=3,
           retry_wait=0 (instant).
        3. Call call_llm — should ultimately succeed after 2 failures.
        4. Confirm acompletion was called 3 times total.
        """
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        message = SimpleNamespace(content="recovered", tool_calls=None)
        choice = SimpleNamespace(message=message)
        success_response = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=MOCK_MODEL_NAME,
            _hidden_params={"response_cost": 0.0},
        )
        api_error = litellm.APIError(
            status_code=500,
            message="Internal Server Error",
            llm_provider="openai",
            model=MOCK_MODEL_NAME,
        )
        mock_acompletion.side_effect = [api_error, api_error, success_response]

        llm = create_llm(
            model=MOCK_MODEL_NAME,
            num_retries=3,
            retry_on=[litellm.APIError],
            retry_wait=0.001,
        )
        try:
            result = llm.call_llm(prompt="Hi").result(timeout=15.0)
            assert result == "recovered"
            assert mock_acompletion.call_count == 3
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_litellm_api_error_not_retried_when_excluded(self, mock_acompletion) -> None:
        """litellm.APIError is NOT retried when excluded from retry_on.

        Steps:
        1. Mock acompletion to raise APIError.
        2. Create worker with retry_on=[ValueError] only (excludes APIError).
        3. call_llm should raise immediately without retrying.
        4. Confirm acompletion was called exactly once.
        """
        api_error = litellm.APIError(
            status_code=500,
            message="Internal Server Error",
            llm_provider="openai",
            model=MOCK_MODEL_NAME,
        )
        mock_acompletion.side_effect = api_error

        llm = create_llm(
            model=MOCK_MODEL_NAME,
            num_retries=3,
            retry_on=[ValueError],
            retry_wait=0.001,
        )
        try:
            with pytest.raises(litellm.APIError):
                llm.call_llm(prompt="Hi").result(timeout=10.0)
            assert mock_acompletion.call_count == 1
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_end_to_end_call(self, mock_acompletion) -> None:
        """create_llm -> call_llm -> get report should work end-to-end.

        Steps:
        1. Create worker via create_llm.
        2. Mock acompletion.
        3. Make one call.
        4. Verify reporter recorded it.
        """
        from types import SimpleNamespace

        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        message = SimpleNamespace(content="test output", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=MOCK_MODEL_NAME,
            _hidden_params={"response_cost": 0.0001},
        )
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_hour=1.0))
        try:
            result = llm.call_llm(prompt="Hi").result(timeout=10.0)
            assert result == "test output"

            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()
