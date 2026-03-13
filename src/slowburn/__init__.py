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
from typing import Any, List, Optional, Union

from concurry import CallLimit, LimitSet, RateLimit

from .backpressure import set_backpressure_warnings, timed_acquire
from .cost_accounting import CostCallContext, cost_controlled_call, estimate_input_tokens
from .limits import DEFAULT_COST_LIMIT_KEY, CostLimit, dollars_to_microdollars, microdollars_to_dollars
from .llm_worker import ImageInput, SlowBurnLLM
from .pricing import ModelNotFoundError, PricingCache
from .reporter import CostReporter

__all__: list[str] = [
    "create_llm",
    "set_backpressure_warnings",
    "timed_acquire",
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
]

_WINDOW_ALIASES = {
    "daily": 86400,
    "hourly": 3600,
    "minutely": 60,
}


def create_llm(
    model: str,
    budget_usd: float = 5.0,
    window: Union[str, int, float] = "daily",
    max_rpm: int = 500,
    max_input_tpm: int = 1_000_000,
    max_output_tpm: int = 200_000,
    api_key: str = "",
    backend: str = "asyncio",
    name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    timeout: float = 120.0,
    num_retries: int = 3,
    extra_limits: Optional[List[Any]] = None,
    litellm_params: Optional[dict] = None,
    **kwargs,
) -> SlowBurnLLM:
    """Create a cost-controlled LLM worker with sensible defaults.

    This is the "two-line setup" entry point. It assembles a CostLimit,
    token rate limits, a call limit, and an asyncio SlowBurnLLM worker
    in one function call.

    Args:
        model: litellm model identifier (e.g. "gpt-4o-mini", "claude-3-5-haiku-20241022").
        budget_usd: Maximum dollar spend per window.
        window: Budget window — "daily", "hourly", "minutely", or seconds (int/float).
        max_rpm: Maximum requests per minute (CallLimit capacity).
        max_input_tpm: Maximum input tokens per minute.
        max_output_tpm: Maximum output tokens per minute.
        api_key: API key string (or set via environment variable for the provider).
        backend: Execution backend — "asyncio" (default) or "ray".
        name: Worker name for logging. Defaults to the model name.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per call.
        timeout: Per-call timeout in seconds.
        num_retries: Number of retries on transient errors.
        extra_limits: Additional Limit objects to include in the LimitSet.
        litellm_params: Additional parameters passed to every litellm.acompletion()
            call (e.g. tools, response_format, seed, top_p, stop).

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
    if isinstance(window, str):
        window_seconds = _WINDOW_ALIASES.get(window.lower())
        if window_seconds is None:
            raise ValueError(
                f"Unknown window alias '{window}'. "
                f"Use one of {list(_WINDOW_ALIASES.keys())} or a number of seconds."
            )
    else:
        window_seconds = float(window)

    if name is None:
        name = model

    limits_list: List[Any] = [
        CostLimit(budget_usd=budget_usd, window_seconds=window_seconds),
        RateLimit(key="input_tokens", window_seconds=60, capacity=max_input_tpm),
        RateLimit(key="output_tokens", window_seconds=60, capacity=max_output_tpm),
        CallLimit(window_seconds=60, capacity=max_rpm),
    ]
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
        retry_on={"call_llm": [ValueError, asyncio.TimeoutError], "*": []},
    ).init(
        name=name,
        model_name=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        litellm_params=litellm_params if litellm_params is not None else {},
    )
    return llm
