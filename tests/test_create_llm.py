"""Tests for the create_llm() convenience function."""

from unittest.mock import AsyncMock, patch

import pytest

from slowburn import create_llm


class TestCreateLLM:
    """Test create_llm() factory function setup and parameter handling."""

    def test_creates_worker_with_defaults(self) -> None:
        """create_llm() with model should produce a live worker.

        Steps:
        1. Call create_llm() with model.
        2. Verify the worker is alive (get_reporter succeeds).
        3. Stop the worker.
        """
        llm = create_llm(model="gpt-4o-mini")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 0
        finally:
            llm.stop()

    def test_daily_window_alias(self) -> None:
        """window='daily' should set a 86400-second budget window."""
        llm = create_llm(model="gpt-4o-mini", window="daily")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_hourly_window_alias(self) -> None:
        """window='hourly' should work."""
        llm = create_llm(model="gpt-4o-mini", window="hourly")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_minutely_window_alias(self) -> None:
        """window='minutely' should work."""
        llm = create_llm(model="gpt-4o-mini", window="minutely")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_numeric_window(self) -> None:
        """A numeric window (seconds) should work."""
        llm = create_llm(model="gpt-4o-mini", window=7200)
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
        finally:
            llm.stop()

    def test_invalid_window_alias_raises(self) -> None:
        """An unrecognized string window should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown window alias"):
            create_llm(model="gpt-4o-mini", window="biweekly")

    def test_custom_name(self) -> None:
        """Passing name= should set the worker name."""
        llm = create_llm(model="gpt-4o-mini", name="my-custom-worker")
        try:
            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter is not None
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
            model="gpt-4o-mini",
            _hidden_params={"response_cost": 0.0001},
        )
        llm = create_llm(model="gpt-4o-mini", budget_usd=1.0, window="hourly")
        try:
            result = llm.call_llm(prompt="Hi").result(timeout=10.0)
            assert result == "test output"

            reporter = llm.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()
