"""
Global configuration for SlowBurn.

Provides a single source of truth for all tunable defaults (token estimation,
LLM call parameters, budget/rate limits, backpressure thresholds, etc.).

Pattern follows Concurry's GlobalConfig: a mutable singleton ``slowburn_config``
whose ``defaults`` field can be mutated at runtime or scoped via ``temp_config()``.

The ``_NO_ARG`` sentinel distinguishes "user didn't pass a value" from "user
explicitly passed None." This matters because ``None`` is a legitimate value
for LLM parameters (e.g., ``temperature=None`` means "let the model decide").
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional

from concurry import RateLimitAlgorithm, RetryAlgorithm
from morphic import MutableTyped
from pydantic import ConfigDict, Field, confloat, conint

from .constants import BackpressureNotify, BudgetOverflowAction, ImageDetailLevel


class _NoArgType:
    """Sentinel type for 'argument not provided.'

    ``_NO_ARG`` is identity-distinct from ``None``, allowing three-way
    distinction: not provided / explicitly None / explicit value.
    """

    _instance: Optional["_NoArgType"] = None

    def __new__(cls) -> "_NoArgType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_NO_ARG"

    def __bool__(self) -> bool:
        return False


_NO_ARG = _NoArgType()
_NO_ARG_TYPE = _NoArgType


def is_no_arg(value: Any) -> bool:
    """Check whether a value is the _NO_ARG sentinel."""
    return value is _NO_ARG


class SlowBurnDefaults(MutableTyped):
    """All tunable defaults for SlowBurn, in one place.

    Every hardcoded constant that previously lived in module-level variables,
    Field defaults, or inline magic numbers is now a field here. Call sites
    read from ``slowburn_config.defaults.<field>`` at call time, so changing
    a field here changes the behavior of all subsequent calls.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Token estimation (applied on top of litellm.token_counter for input,
    # and on top of max_tokens for output)
    chars_per_token: confloat(gt=0) = 3.0
    input_token_estimate_multiplier: confloat(ge=1.0) = 1.5
    input_token_estimate_overhead: conint(ge=0) = 10
    output_token_estimate_multiplier: confloat(gt=0.0, le=1.0) = 1.0
    output_token_estimate_overhead: conint(ge=0) = 0

    # LLM call defaults
    temperature: Optional[confloat(ge=0.0, le=2.0)] = 0.7
    max_tokens: conint(ge=1) = 1000
    timeout: confloat(gt=0.0) = 120.0

    # Vision token estimates
    image_tokens_low_detail: conint(ge=1) = 85
    image_tokens_high_detail: conint(ge=1) = 1000
    image_detail: ImageDetailLevel = "auto"

    num_retries: conint(ge=0) = 5

    # Generic transient-error retry backoff. This is intentionally short because
    # request-rate pacing is enforced separately by rate_limit_algorithm. Rate
    # limit errors need provider-aware cooldown (for example Retry-After), not a
    # globally larger base wait for every transient failure.
    retry_wait: confloat(gt=0.0) = 1.0
    retry_algorithm: RetryAlgorithm = RetryAlgorithm.Exponential
    retry_jitter: confloat(ge=0.0, le=1.0) = 0.3

    # Rate-limit algorithm for the per-window CallLimit and token RateLimits.
    # GCRA enforces a steady emission interval (Theoretical Arrival Time) so
    # request *starts* are spaced ~ window / capacity apart. This is
    # robust to heterogeneous call durations because GCRA tracks start times
    # only and avoids the bursty edge cases of SlidingWindow when providers
    # count failed (429) requests against the same window.
    rate_limit_algorithm: RateLimitAlgorithm = RateLimitAlgorithm.GCRA

    # Backpressure
    backpressure_threshold_seconds: confloat(ge=0.0) = 0.5
    backpressure_notify: BackpressureNotify = "ignore"
    on_budget_overflow: BudgetOverflowAction = "warn"

    # Verbosity
    verbosity: conint(ge=0) = 1

    # OpenRouter API
    openrouter_fetch_timeout: confloat(gt=0.0) = 60.0


class SlowBurnConfig(MutableTyped):
    """Top-level config object. The global singleton is ``slowburn_config``.

    Usage::

        from slowburn.config import slowburn_config, temp_config

        # Read a default
        slowburn_config.defaults.temperature

        # Mutate at runtime
        slowburn_config.defaults.temperature = 0.0

        # Scoped override (restores on exit)
        with temp_config(temperature=0.0):
            run_eval()
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    defaults: SlowBurnDefaults = Field(default_factory=SlowBurnDefaults)

    def reset_to_defaults(self) -> None:
        """Restore all defaults to their original values."""
        object.__setattr__(self, "defaults", SlowBurnDefaults())


slowburn_config = SlowBurnConfig()


@contextmanager
def temp_config(**overrides: Any) -> Generator[SlowBurnConfig, None, None]:
    """Temporarily override config defaults within a ``with`` block.

    All keyword arguments must be valid field names on ``SlowBurnDefaults``.
    On exit (normal or exception), the original values are restored.

    Usage::

        with temp_config(temperature=0.0, num_retries=0):
            llm = create_llm(model="gpt-4o-mini")
            # temperature=0.0, num_retries=0
        # restored to previous values

    Raises:
        ValueError: If any key is not a recognized SlowBurnDefaults field.
    """
    defaults = slowburn_config.defaults
    valid_fields = set(SlowBurnDefaults.model_fields.keys())

    unknown = set(overrides.keys()) - valid_fields
    if len(unknown) > 0:
        raise ValueError(f"Unknown config key(s): {sorted(unknown)}. Valid keys: {sorted(valid_fields)}")

    saved = {}
    for key in overrides:
        saved[key] = getattr(defaults, key)

    try:
        for key, value in overrides.items():
            setattr(defaults, key, value)
        yield slowburn_config
    finally:
        for key, value in saved.items():
            setattr(defaults, key, value)
