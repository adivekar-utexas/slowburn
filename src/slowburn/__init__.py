"""
SlowBurn — Cost-Sustainable Concurrent Execution for Long-Horizon LLM Agents.

SlowBurn is a thin layer on top of Concurry that adds:
1. CostLimit — a dollar-denominated rate limit
2. SlowBurnLLM — a cost-tracking asyncio LLM worker
3. CostReporter — experiment-level cost attribution
4. Drop-in integrations for CrewAI and AutoGen (AG2)

Quick start::

    from slowburn import create_llm

    llm = create_llm(model="gpt-4o-mini", budget_usd=5.0)
    result = llm.call_llm(prompt="Hello world").result()
    print(llm.get_reporter().result().total_cost())
    llm.stop()
"""

import asyncio
import math
from typing import Any, Dict, List, Optional, Type, Union

import litellm
from concurry import CallLimit, LimitSet, RateLimit, RateLimitAlgorithm, RetryAlgorithm
from morphic import validate

from .config import (
    _NO_ARG,
    _NO_ARG_TYPE,
    SlowBurnConfig,
    SlowBurnDefaults,
    is_no_arg,
    slowburn_config,
    temp_config,
)
from .constants import (
    WINDOW_ALIAS_SECONDS,
    BackpressureNotify,
    BudgetOverflowAction,
    ExecutionBackend,
    PricingUnavailableAction,
    ToolChoiceOption,
    WindowAlias,
)
from .cost_accounting import CostCallContext, cost_controlled_call, estimate_input_tokens
from .limits import DEFAULT_COST_LIMIT_KEY, CostLimit, dollars_to_microdollars, microdollars_to_dollars
from .llm_worker import ImageInput, SlowBurnLLM
from .pricing import ModelNotFoundError, PricingCache
from .reporter import CostReporter

__all__: List[str] = [
    "create_llm",
    "_DEFAULT_RETRY_ON",
    "CostCallContext",
    "cost_controlled_call",
    "estimate_input_tokens",
    "CostLimit",
    "ImageInput",
    "SlowBurnLLM",
    "PricingCache",
    "ModelNotFoundError",
    "CostReporter",
    "dollars_to_microdollars",
    "microdollars_to_dollars",
    "DEFAULT_COST_LIMIT_KEY",
    "slowburn_config",
    "temp_config",
    "SlowBurnConfig",
    "SlowBurnDefaults",
]


_DEFAULT_RETRY_ON: List[Type[BaseException]] = [
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


@validate
def create_llm(
    model: str,
    budget_usd: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    window: Union[WindowAlias, int, float, _NO_ARG_TYPE] = _NO_ARG,
    max_rpm: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    max_input_tpm: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    max_output_tpm: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    rate_limit_algorithm: Union[str, _NO_ARG_TYPE] = _NO_ARG,
    api_key: str = "",
    backend: ExecutionBackend = "Asyncio",
    name: Optional[str] = None,
    temperature: Union[Optional[float], _NO_ARG_TYPE] = _NO_ARG,
    max_tokens: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    timeout: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    num_retries: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    retry_on: Union[List[Type[BaseException]], _NO_ARG_TYPE] = _NO_ARG,
    retry_wait: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    retry_algorithm: Union[str, RetryAlgorithm, _NO_ARG_TYPE] = _NO_ARG,
    retry_jitter: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    tools: Optional[List[Dict[str, object]]] = None,
    tool_choice: Optional[ToolChoiceOption] = None,
    extra_limits: Optional[List[object]] = None,
    litellm_params: Optional[Dict[str, object]] = None,
    backpressure_notify: Union[BackpressureNotify, _NO_ARG_TYPE] = _NO_ARG,
    on_budget_overflow: Union[BudgetOverflowAction, _NO_ARG_TYPE] = _NO_ARG,
    on_pricing_unavailable: PricingUnavailableAction = "error",
) -> SlowBurnLLM:
    """Create a cost-controlled LLM worker with sensible defaults.

    This is the "two-line setup" entry point. It assembles a CostLimit,
    token rate limits, a call limit, and an asyncio SlowBurnLLM worker
    in one function call.

    All defaults are read from ``slowburn_config.defaults`` at call time,
    so they can be tuned globally via ``temp_config()`` or by mutating
    ``slowburn_config.defaults`` directly.

    Args:
        model: litellm model identifier (e.g. "gpt-4o-mini", "claude-3-5-haiku-20241022").
        budget_usd: Maximum dollar spend per window.
            Defaults to slowburn_config.defaults.budget_usd.
        window: Budget window — "daily", "hourly", "minutely", or seconds (int/float).
            Defaults to slowburn_config.defaults.window.
        max_rpm: Maximum requests per minute (CallLimit capacity).
            Defaults to slowburn_config.defaults.max_rpm.
        max_input_tpm: Maximum input tokens per minute.
            Defaults to slowburn_config.defaults.max_input_tpm.
        max_output_tpm: Maximum output tokens per minute.
            Defaults to slowburn_config.defaults.max_output_tpm.
        rate_limit_algorithm: Concurry rate-limit algorithm for per-minute call and token limits.
            Defaults to slowburn_config.defaults.rate_limit_algorithm ("GCRA").
        api_key: API key string (or set via environment variable for the provider).
        backend: Execution backend — "Asyncio" (default) or "Ray".
        name: Worker name for logging. Defaults to the model name.
        temperature: LLM sampling temperature.
            Defaults to slowburn_config.defaults.temperature.
        max_tokens: Maximum output tokens per call.
            Defaults to slowburn_config.defaults.max_tokens.
        timeout: Per-call timeout in seconds.
            Defaults to slowburn_config.defaults.timeout.
        num_retries: Number of retries on transient errors.
            Defaults to slowburn_config.defaults.num_retries.
        retry_on: Exception types that trigger a retry on ``call_llm``.
            Defaults to a comprehensive list of litellm transient errors:
            ``litellm.APIError``, ``litellm.APIConnectionError``,
            ``litellm.Timeout``, ``litellm.RateLimitError``,
            ``litellm.InternalServerError``, ``litellm.ServiceUnavailableError``,
            ``litellm.BadRequestError``, ``asyncio.TimeoutError``, ``ValueError``.
            Pass an explicit list to restrict or extend this set.
        retry_wait: Base wait time in seconds for generic transient-error retries.
            Defaults to slowburn_config.defaults.retry_wait (1.0s). Request-rate
            pacing is handled separately by ``rate_limit_algorithm``.
        retry_algorithm: Backoff strategy — "Exponential", "Linear", or "Fibonacci".
            Defaults to slowburn_config.defaults.retry_algorithm (Exponential).
        retry_jitter: Jitter factor in [0, 1] added to each retry wait.
            Defaults to slowburn_config.defaults.retry_jitter (0.3).
        tools: Default tool schemas (OpenAI format) for all calls.
            Pass a list of tool dicts. Overridable per-call via
            ``call_llm(tools=...)``.
        tool_choice: Default tool_choice for all calls ("auto", "required",
            "none"). Overridable per-call via ``call_llm(tool_choice=...)``.
        extra_limits: Additional Limit objects to include in the LimitSet.
        litellm_params: Additional parameters passed to every litellm.acompletion()
            call (e.g. response_format, seed, top_p, stop).
        backpressure_notify: When "warn", logs a warning if acquire() blocks
            longer than backpressure_threshold_seconds waiting for budget/rate
            capacity. When "ignore", silent.
            Defaults to slowburn_config.defaults.backpressure_notify.
        on_budget_overflow: Action when a single call's estimated cost exceeds
            the budget capacity. "warn" (default): proceed with the call but
            log a warning. "error": raise ValueError. "ignore": proceed silently.
            Defaults to slowburn_config.defaults.on_budget_overflow.

    Returns:
        A live SlowBurnLLM worker, ready to accept ``call_llm()`` calls.

    Example::

        from slowburn import create_llm

        llm = create_llm(model="gpt-4o-mini", budget_usd=5.0, window="daily")
        result = llm.call_llm(prompt="Summarize this paper...").result()
        reporter = llm.get_reporter().result()
        print(f"Cost so far: ${reporter.total_cost():.4f}")
        llm.stop()
    """
    defaults = slowburn_config.defaults
    if is_no_arg(budget_usd):
        budget_usd = defaults.budget_usd
    if is_no_arg(window):
        window = defaults.window
    if is_no_arg(max_rpm):
        max_rpm = defaults.max_rpm
    if is_no_arg(max_input_tpm):
        max_input_tpm = defaults.max_input_tpm
    if is_no_arg(max_output_tpm):
        max_output_tpm = defaults.max_output_tpm
    if is_no_arg(rate_limit_algorithm):
        rate_limit_algorithm = defaults.rate_limit_algorithm
    rate_limit_algorithm: RateLimitAlgorithm = RateLimitAlgorithm(rate_limit_algorithm)
    if is_no_arg(temperature):
        temperature = defaults.temperature
    if is_no_arg(max_tokens):
        max_tokens = defaults.max_tokens
    if is_no_arg(timeout):
        timeout = defaults.timeout
    if is_no_arg(num_retries):
        num_retries = defaults.num_retries
    if is_no_arg(retry_on):
        retry_on = _DEFAULT_RETRY_ON
    if is_no_arg(retry_wait):
        retry_wait = defaults.retry_wait
    if is_no_arg(retry_algorithm):
        retry_algorithm = defaults.retry_algorithm
    retry_algorithm: RetryAlgorithm = RetryAlgorithm(retry_algorithm)
    if is_no_arg(retry_jitter):
        retry_jitter = defaults.retry_jitter

    if isinstance(window, str):
        window_seconds = WINDOW_ALIAS_SECONDS[window.lower()]
    else:
        window_seconds = float(window)

    if name is None:
        name = model

    limits_list: List[Any] = []
    if not math.isinf(budget_usd):
        limits_list.append(
            CostLimit(budget_usd=budget_usd, window_seconds=window_seconds),
        )
    # Rate limits use 60s windows (per-minute) regardless of the cost budget
    # window. "rpm" = requests per minute, "tpm" = tokens per minute.
    limits_list.extend(
        [
            RateLimit(
                key="input_tokens",
                window_seconds=60,
                capacity=max_input_tpm,
                algorithm=rate_limit_algorithm,
            ),
            RateLimit(
                key="output_tokens",
                window_seconds=60,
                capacity=max_output_tpm,
                algorithm=rate_limit_algorithm,
            ),
            CallLimit(
                window_seconds=60,
                capacity=max_rpm,
                algorithm=rate_limit_algorithm,
            ),
        ]
    )
    if extra_limits is not None:
        limits_list.extend(extra_limits)

    limit_set = LimitSet(
        limits=limits_list,
        mode=backend,
        shared=True,
    )

    llm = SlowBurnLLM.options(
        mode=backend,
        limits=limit_set,
        num_retries={"call_llm": num_retries, "*": 0},
        retry_on={"call_llm": retry_on, "*": []},
        retry_wait={"call_llm": retry_wait, "*": 1},
        retry_algorithm={"call_llm": retry_algorithm, "*": RetryAlgorithm.Exponential},
        retry_jitter={"call_llm": retry_jitter, "*": 0},
    ).init(
        name=name,
        model_name=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        tools=tools,
        tool_choice=tool_choice,
        litellm_params=litellm_params if litellm_params is not None else {},
        backpressure_notify=backpressure_notify,
        on_budget_overflow=on_budget_overflow,
        on_pricing_unavailable=on_pricing_unavailable,
    )
    return llm
