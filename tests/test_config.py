"""Tests for the SlowBurn global configuration system.

Verifies:
- Default values match expected constants
- Runtime mutation works with Pydantic validation
- temp_config scoping, nesting, and exception safety
- Config flows through to all consuming components
- _NO_ARG sentinel semantics
- reset_to_defaults restores original values
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from concurry import RetryAlgorithm

from slowburn.config import (
    _NO_ARG,
    is_no_arg,
    slowburn_config,
    temp_config,
)

from .conftest import MOCK_MODEL_NAME


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure every test starts and ends with pristine defaults."""
    slowburn_config.reset_to_defaults()
    yield
    slowburn_config.reset_to_defaults()


# ===========================================================================
# TestSlowBurnDefaults: verify all defaults match expected values
# ===========================================================================


class TestSlowBurnDefaults:
    def test_default_chars_per_token(self) -> None:
        assert slowburn_config.defaults.chars_per_token == 3.0

    def test_default_input_token_estimate_multiplier(self) -> None:
        assert slowburn_config.defaults.input_token_estimate_multiplier == 1.5

    def test_default_input_token_estimate_overhead(self) -> None:
        assert slowburn_config.defaults.input_token_estimate_overhead == 10

    def test_default_temperature(self) -> None:
        assert slowburn_config.defaults.temperature == 0.7

    def test_default_max_tokens(self) -> None:
        assert slowburn_config.defaults.max_tokens == 1000

    def test_default_timeout(self) -> None:
        assert slowburn_config.defaults.timeout == 120.0

    def test_default_image_tokens_low_detail(self) -> None:
        assert slowburn_config.defaults.image_tokens_low_detail == 85

    def test_default_image_tokens_high_detail(self) -> None:
        assert slowburn_config.defaults.image_tokens_high_detail == 1000

    def test_default_budget_usd(self) -> None:
        assert slowburn_config.defaults.budget_usd == float("inf")

    def test_default_window_seconds(self) -> None:
        assert slowburn_config.defaults.window_seconds == 86400.0

    def test_default_max_rpm(self) -> None:
        assert slowburn_config.defaults.max_rpm == 500

    def test_default_max_input_tpm(self) -> None:
        assert slowburn_config.defaults.max_input_tpm == 1_000_000

    def test_default_max_output_tpm(self) -> None:
        assert slowburn_config.defaults.max_output_tpm == 200_000

    def test_default_num_retries(self) -> None:
        assert slowburn_config.defaults.num_retries == 3

    def test_default_retry_wait(self) -> None:
        assert slowburn_config.defaults.retry_wait == 1.0

    def test_default_retry_algorithm(self) -> None:
        assert slowburn_config.defaults.retry_algorithm == RetryAlgorithm.Exponential

    def test_default_retry_jitter(self) -> None:
        assert slowburn_config.defaults.retry_jitter == 0.3

    def test_default_backpressure_threshold(self) -> None:
        assert slowburn_config.defaults.backpressure_threshold_seconds == 0.5

    def test_default_verbosity(self) -> None:
        assert slowburn_config.defaults.verbosity == 1

    def test_default_openrouter_fetch_timeout(self) -> None:
        assert slowburn_config.defaults.openrouter_fetch_timeout == 60.0


# ===========================================================================
# TestConfigMutability: runtime mutation with validation
# ===========================================================================


class TestConfigMutability:
    def test_config_is_mutable(self) -> None:
        slowburn_config.defaults.temperature = 0.0
        assert slowburn_config.defaults.temperature == 0.0

    def test_mutation_persists(self) -> None:
        slowburn_config.defaults.max_tokens = 500
        assert slowburn_config.defaults.max_tokens == 500

    def test_pydantic_validation_rejects_negative_chars_per_token(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.chars_per_token = -1.0

    def test_pydantic_validation_rejects_zero_chars_per_token(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.chars_per_token = 0.0

    def test_pydantic_validation_rejects_negative_timeout(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.timeout = -5.0

    def test_pydantic_validation_rejects_unknown_field(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.nonexistent_field = 42

    def test_retry_wait_rejects_zero(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.retry_wait = 0.0

    def test_retry_wait_rejects_negative(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.retry_wait = -1.0

    def test_retry_wait_accepts_small_positive(self) -> None:
        slowburn_config.defaults.retry_wait = 0.001
        assert slowburn_config.defaults.retry_wait == pytest.approx(0.001)

    def test_retry_algorithm_accepts_valid_values(self) -> None:
        for algo in (RetryAlgorithm.Exponential, RetryAlgorithm.Linear, RetryAlgorithm.Fibonacci):
            slowburn_config.defaults.retry_algorithm = algo
            assert slowburn_config.defaults.retry_algorithm == algo

    def test_retry_algorithm_rejects_invalid_string(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.retry_algorithm = "Quadratic"

    def test_retry_algorithm_rejects_nonexistent(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.retry_algorithm = "Constant"

    def test_retry_jitter_bounds(self) -> None:
        slowburn_config.defaults.retry_jitter = 0.0
        assert slowburn_config.defaults.retry_jitter == 0.0
        slowburn_config.defaults.retry_jitter = 1.0
        assert slowburn_config.defaults.retry_jitter == 1.0

    def test_retry_jitter_rejects_out_of_bounds(self) -> None:
        with pytest.raises(Exception):
            slowburn_config.defaults.retry_jitter = -0.1
        with pytest.raises(Exception):
            slowburn_config.defaults.retry_jitter = 1.1


# ===========================================================================
# TestTempConfig: scoped overrides
# ===========================================================================


class TestTempConfig:
    def test_basic_override(self) -> None:
        with temp_config(temperature=0.0):
            assert slowburn_config.defaults.temperature == 0.0
        assert slowburn_config.defaults.temperature == 0.7

    def test_multiple_overrides(self) -> None:
        with temp_config(temperature=0.0, max_tokens=500, timeout=30.0):
            assert slowburn_config.defaults.temperature == 0.0
            assert slowburn_config.defaults.max_tokens == 500
            assert slowburn_config.defaults.timeout == 30.0
        assert slowburn_config.defaults.temperature == 0.7
        assert slowburn_config.defaults.max_tokens == 1000
        assert slowburn_config.defaults.timeout == 120.0

    def test_nested_contexts(self) -> None:
        with temp_config(temperature=0.5):
            assert slowburn_config.defaults.temperature == 0.5
            with temp_config(temperature=0.0):
                assert slowburn_config.defaults.temperature == 0.0
            assert slowburn_config.defaults.temperature == 0.5
        assert slowburn_config.defaults.temperature == 0.7

    def test_restores_on_exception(self) -> None:
        try:
            with temp_config(temperature=0.0):
                assert slowburn_config.defaults.temperature == 0.0
                raise ValueError("boom")
        except ValueError:
            pass
        assert slowburn_config.defaults.temperature == 0.7

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config key"):
            with temp_config(nonexistent_param=42):
                pass

    def test_yields_config(self) -> None:
        with temp_config(temperature=0.0) as cfg:
            assert cfg is slowburn_config

    def test_override_chars_per_token(self) -> None:
        with temp_config(chars_per_token=4.0):
            assert slowburn_config.defaults.chars_per_token == 4.0
        assert slowburn_config.defaults.chars_per_token == 3.0

    def test_override_retry_wait(self) -> None:
        with temp_config(retry_wait=5.0):
            assert slowburn_config.defaults.retry_wait == 5.0
        assert slowburn_config.defaults.retry_wait == 1.0

    def test_override_retry_algorithm(self) -> None:
        with temp_config(retry_algorithm=RetryAlgorithm.Linear):
            assert slowburn_config.defaults.retry_algorithm == RetryAlgorithm.Linear
        assert slowburn_config.defaults.retry_algorithm == RetryAlgorithm.Exponential

    def test_override_retry_jitter(self) -> None:
        with temp_config(retry_jitter=0.0):
            assert slowburn_config.defaults.retry_jitter == 0.0
        assert slowburn_config.defaults.retry_jitter == 0.3


# ===========================================================================
# TestNoArgSentinel: verify _NO_ARG semantics
# ===========================================================================


class TestNoArgSentinel:
    def test_no_arg_is_not_none(self) -> None:
        assert _NO_ARG is not None

    def test_no_arg_identity(self) -> None:
        assert _NO_ARG is _NO_ARG

    def test_no_arg_is_falsy(self) -> None:
        assert not _NO_ARG

    def test_is_no_arg_true(self) -> None:
        assert is_no_arg(_NO_ARG) is True

    def test_is_no_arg_false_for_none(self) -> None:
        assert is_no_arg(None) is False

    def test_is_no_arg_false_for_value(self) -> None:
        assert is_no_arg(0.7) is False

    def test_no_arg_repr(self) -> None:
        assert repr(_NO_ARG) == "_NO_ARG"

    def test_no_arg_is_singleton(self) -> None:
        from slowburn.config import _NoArgType

        instance_a = _NoArgType()
        instance_b = _NoArgType()
        assert instance_a is instance_b


# ===========================================================================
# TestResetToDefaults
# ===========================================================================


class TestResetToDefaults:
    def test_reset_restores_all(self) -> None:
        slowburn_config.defaults.temperature = 0.0
        slowburn_config.defaults.max_tokens = 1
        slowburn_config.defaults.chars_per_token = 10.0
        slowburn_config.reset_to_defaults()
        assert slowburn_config.defaults.temperature == 0.7
        assert slowburn_config.defaults.max_tokens == 1000
        assert slowburn_config.defaults.chars_per_token == 3.0

    def test_reset_restores_retry_fields(self) -> None:
        slowburn_config.defaults.retry_wait = 5.0
        slowburn_config.defaults.retry_algorithm = RetryAlgorithm.Linear
        slowburn_config.defaults.retry_jitter = 0.0
        slowburn_config.reset_to_defaults()
        assert slowburn_config.defaults.retry_wait == 1.0
        assert slowburn_config.defaults.retry_algorithm == RetryAlgorithm.Exponential
        assert slowburn_config.defaults.retry_jitter == 0.3


# ===========================================================================
# TestConfigAffectsComponents: verify config flows into consuming code
# ===========================================================================


class TestConfigAffectsComponents:
    def test_config_affects_estimate_input_tokens(self) -> None:
        """Changing token estimation params via config changes the output."""
        from slowburn.cost_accounting import estimate_input_tokens

        result_default, _ = estimate_input_tokens("hello world", 100)
        with temp_config(input_token_estimate_multiplier=1.0, input_token_estimate_overhead=0):
            result_tuned, _ = estimate_input_tokens("hello world", 100)
        assert result_tuned < result_default

    def test_config_affects_estimate_tokens_helper(self) -> None:
        """_estimate_tokens in llm_worker reads chars_per_token from config."""
        from slowburn.llm_worker import _estimate_tokens

        result_default = _estimate_tokens("a" * 300)
        with temp_config(chars_per_token=1.0):
            result_tuned = _estimate_tokens("a" * 300)
        assert result_tuned > result_default

    def test_config_affects_cost_limit_window(self) -> None:
        """CostLimit with no explicit window_seconds reads from config."""
        from slowburn.limits import CostLimit

        default_limit = CostLimit(budget_usd=1.0)
        assert default_limit.window_seconds == 86400.0

        with temp_config(window_seconds=3600.0):
            hourly_limit = CostLimit(budget_usd=1.0)
            assert hourly_limit.window_seconds == 3600.0

    def test_explicit_arg_overrides_config(self) -> None:
        """Explicit window_seconds overrides the config default."""
        from slowburn.limits import CostLimit

        with temp_config(window_seconds=3600.0):
            explicit_limit = CostLimit(budget_usd=1.0, window_seconds=7200.0)
            assert explicit_limit.window_seconds == 7200.0

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_config_affects_create_llm_defaults(self, mock_acompletion) -> None:
        """create_llm() should use config defaults when no explicit values given."""
        from slowburn import create_llm

        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        message = SimpleNamespace(content="test output", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=MOCK_MODEL_NAME,
            _hidden_params={"response_cost": 0.0001},
        )

        with temp_config(temperature=0.0, max_tokens=500, timeout=30.0):
            llm = create_llm(model=MOCK_MODEL_NAME)
            try:
                result = llm.call_llm(prompt="Hi").result(timeout=10.0)
                assert result == "test output"
                call_kwargs = mock_acompletion.call_args.kwargs
                assert call_kwargs["temperature"] == 0.0
                assert call_kwargs["max_tokens"] == 500
            finally:
                llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_explicit_arg_overrides_config_in_create_llm(self, mock_acompletion) -> None:
        """Explicit args to create_llm() should override config defaults."""
        from slowburn import create_llm

        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        message = SimpleNamespace(content="test output", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=MOCK_MODEL_NAME,
            _hidden_params={"response_cost": 0.0001},
        )

        with temp_config(temperature=0.0):
            llm = create_llm(model=MOCK_MODEL_NAME, temperature=0.9)
            try:
                llm.call_llm(prompt="Hi").result(timeout=10.0)
                call_kwargs = mock_acompletion.call_args.kwargs
                assert call_kwargs["temperature"] == 0.9
            finally:
                llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_none_temperature_passes_through(self, mock_acompletion) -> None:
        """Passing temperature=None to create_llm should forward None to litellm."""
        from slowburn import create_llm

        usage = SimpleNamespace(prompt_tokens=30, completion_tokens=15, total_tokens=45)
        message = SimpleNamespace(content="test output", tool_calls=None)
        choice = SimpleNamespace(message=message)
        mock_acompletion.return_value = SimpleNamespace(
            usage=usage,
            choices=[choice],
            model=MOCK_MODEL_NAME,
            _hidden_params={"response_cost": 0.0001},
        )

        llm = create_llm(model=MOCK_MODEL_NAME, temperature=None)
        try:
            llm.call_llm(prompt="Hi").result(timeout=10.0)
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["temperature"] is None
        finally:
            llm.stop()
