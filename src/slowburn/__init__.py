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
from concurry import (
    CallLimit,
    LimitPool,
    LimitSet,
    RateLimit,
    RateLimitAlgorithm,
    RetryAlgorithm,
)
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
from .endpoints import (
    EndpointConfig,
    EndpointResolver,
    coerce_to_endpoint_config,
    passthrough_resolver,
    resolve_concrete_endpoint_config,
)
from .exceptions import (
    BatchInputMismatchError,
    BudgetOverflowError,
    InvalidConfigValueError,
    PricingUnavailableError,
    SlowBurnNonRetryableError,
    ToolCallContractError,
)
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
    "SlowBurnNonRetryableError",
    "PricingUnavailableError",
    "BudgetOverflowError",
    "ToolCallContractError",
    "InvalidConfigValueError",
    "BatchInputMismatchError",
    "CostLimit",
    "ImageInput",
    "SlowBurnLLM",
    "PricingCache",
    "ModelNotFoundError",
    "CostReporter",
    "EndpointConfig",
    "EndpointResolver",
    "passthrough_resolver",
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


def _window_to_seconds(window: Union[WindowAlias, int, float]) -> float:
    """Resolve a window value (alias or seconds) to a float seconds value."""
    if isinstance(window, str):
        return float(WINDOW_ALIAS_SECONDS[window.lower()])
    return float(window)


def _build_limit_set_for_endpoint(
    *,
    endpoint: EndpointConfig,
    backend: ExecutionBackend,
    rate_limit_algorithm: RateLimitAlgorithm,
) -> LimitSet:
    """Construct one ``LimitSet`` from a fully-resolved ``EndpointConfig``.

    The returned ``LimitSet`` has:

    - A ``CostLimit`` if ``budget_usd`` is finite (otherwise no cost dimension).
    - ``RateLimit``\\s for input/output tokens (per-minute windows).
    - A ``CallLimit`` for requests-per-minute.
    - Any user-supplied ``extra_limits`` appended.
    - The endpoint's full ``model_dump()`` stored as ``LimitSet.config`` so the
      worker can rebuild the typed ``EndpointConfig`` at acquisition time.

    The caller must pre-resolve every ``_NO_ARG`` field before calling this.
    """
    limits_list: List[Any] = []
    if not math.isinf(endpoint.budget_usd):
        limits_list.append(
            CostLimit(
                budget_usd=endpoint.budget_usd,
                window_seconds=_window_to_seconds(endpoint.window),
            ),
        )
    # Token RateLimits and the CallLimit always use a 60-second window: "rpm"
    # and "tpm" are per-minute by industry convention regardless of the
    # cost-budget window.
    limits_list.extend(
        [
            RateLimit(
                key="input_tokens",
                window_seconds=60,
                capacity=endpoint.max_input_tpm,
                algorithm=rate_limit_algorithm,
            ),
            RateLimit(
                key="output_tokens",
                window_seconds=60,
                capacity=endpoint.max_output_tpm,
                algorithm=rate_limit_algorithm,
            ),
            CallLimit(
                window_seconds=60,
                capacity=endpoint.max_rpm,
                algorithm=rate_limit_algorithm,
            ),
        ]
    )
    if endpoint.extra_limits:
        limits_list.extend(endpoint.extra_limits)

    return LimitSet(
        limits=limits_list,
        mode=backend,
        shared=True,
        config=endpoint.model_dump(),
    )


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
    api_base: Optional[str] = None,
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
    endpoints: Optional[List[Union[EndpointConfig, Dict[str, Any]]]] = None,
    endpoint_resolver: Optional[EndpointResolver] = None,
    load_balancing: str = "round_robin",
    worker_index: int = 0,
) -> SlowBurnLLM:
    """Create a cost-controlled LLM worker with sensible defaults.

    This is the "two-line setup" entry point. It assembles a Concurry
    ``LimitPool`` of one or more endpoints, a cost-tracking asyncio
    ``SlowBurnLLM`` worker, and per-endpoint accounting in one call.

    All defaults are read from ``slowburn_config.defaults`` at call time,
    so they can be tuned globally via ``temp_config()`` or by mutating
    ``slowburn_config.defaults`` directly.

    Single-endpoint vs multi-endpoint
    ---------------------------------

    SlowBurn always uses a ``LimitPool`` internally; the difference is how
    many endpoints it contains.

    - **Single endpoint** (``endpoints`` is ``None``): the bare kwargs
      (``model``, ``api_key``, ``api_base``, ``max_rpm``, ``budget_usd``,
      ``window``, ``temperature``, ``max_tokens``, ``timeout``,
      ``litellm_params``, ``extra_limits``) define one synthetic
      :class:`EndpointConfig`. The pool has exactly one ``LimitSet``.
    - **Multi-endpoint** (``endpoints=[...]``): the user provides one
      :class:`EndpointConfig` (or a plain dict) per endpoint. Each is
      cascaded against the bare kwargs: any field set to ``_NO_ARG`` on the
      ``EndpointConfig`` falls back to the corresponding bare kwarg, which
      itself falls back to ``slowburn_config.defaults``. The pool has N
      ``LimitSet``\\s, one per endpoint, and the user supplies an
      ``endpoint_resolver`` if request-time data injection is needed
      (e.g., freshly-assumed AWS STS credentials).

    Cascade order for any overridable field
    ---------------------------------------

    For every field that exists at multiple layers (``model``, ``api_key``,
    ``api_base``, ``temperature``, ``max_tokens``, ``timeout``, plus the
    limit-shaping fields), the value used at call time is resolved as:

        ``call_llm(field=...)``
          > resolver-augmented :class:`EndpointConfig` field
            > ``create_llm(field=...)``
              > ``slowburn_config.defaults.field``

    Args:
        model: litellm model identifier (e.g. "gpt-4o-mini",
            "bedrock/us.anthropic.claude-sonnet-4-6"). When ``endpoints``
            contains entries with their own ``model``, this becomes the
            default for any endpoint whose model is unset.
        budget_usd: Maximum dollar spend per window. Treated as the default
            for any endpoint whose ``budget_usd`` is unset. Set to
            ``float('inf')`` (the default) to disable cost limiting.
        window: Budget window — "daily", "hourly", "minutely", or seconds
            (int/float). Default for any endpoint whose ``window`` is unset.
        max_rpm: Default requests-per-minute cap for any endpoint whose
            ``max_rpm`` is unset.
        max_input_tpm: Default input-tokens-per-minute cap for any endpoint
            whose ``max_input_tpm`` is unset.
        max_output_tpm: Default output-tokens-per-minute cap for any
            endpoint whose ``max_output_tpm`` is unset.
        rate_limit_algorithm: Concurry rate-limit algorithm for the
            per-minute call and token limits. "GCRA" (default), "SlidingWindow",
            or "TokenBucket".
        api_key: Default API key for any endpoint whose ``api_key`` is unset.
        api_base: Default API base URL (litellm ``api_base``) for any
            endpoint whose ``api_base`` is unset. Useful for OpenAI-compatible
            self-hosted endpoints, OpenRouter overrides, etc.
        backend: Execution backend — "Asyncio" (default) or "Ray".
        name: Worker name for logging. Defaults to the model name.
        temperature: Default sampling temperature for any endpoint whose
            ``temperature`` is unset.
        max_tokens: Default max-output-tokens for any endpoint whose
            ``max_tokens`` is unset.
        timeout: Default per-call timeout in seconds.
        num_retries: Number of retries on transient errors.
        retry_on: Exception types that trigger a retry on ``call_llm``.
        retry_wait: Base wait time in seconds for transient-error retries.
        retry_algorithm: Backoff strategy ("Exponential" / "Linear" / "Fibonacci").
        retry_jitter: Jitter factor in [0, 1] added to each retry wait.
        tools: Default tool schemas (OpenAI format) for all calls.
            Overridable per-call via ``call_llm(tools=...)``.
        tool_choice: Default tool_choice for all calls.
        extra_limits: Default ``extra_limits`` list applied to any endpoint
            whose ``extra_limits`` is empty. (When ``endpoints=[...]``, each
            endpoint may have its own ``extra_limits``.)
        litellm_params: Worker-level kwargs forwarded to every
            ``litellm.acompletion`` call. Per-endpoint and per-call
            ``litellm_params`` merge ON TOP of these.
        backpressure_notify: When "warn", logs a warning if acquire blocks
            longer than backpressure_threshold_seconds.
        on_budget_overflow: Action when a single call's estimated cost
            exceeds the budget capacity. "warn" (proceed but log) /
            "error" (raise) / "ignore" (proceed silently).
        on_pricing_unavailable: Action when the model is not in litellm's
            pricing database. "error" / "warn" / "ignore".
        endpoints: List of per-endpoint configurations. Each element may be
            an :class:`EndpointConfig` or a plain dict (validated into one).
            When provided, the worker becomes a multi-endpoint LimitPool
            that load-balances across these endpoints. When ``None``
            (default), a single synthetic endpoint is built from the bare
            kwargs.
        endpoint_resolver: Callable ``(config_dict) -> dict`` that runs once
            per call AFTER the LimitPool selects an endpoint, BEFORE the
            litellm call. The dict it receives is the selected endpoint's
            ``EndpointConfig.model_dump()`` (including any unknown extras
            the user attached). The dict it returns is validated into a new
            ``EndpointConfig`` whose fields override the original. Use this
            to inject request-time data such as freshly-assumed AWS STS
            credentials — write a function that reads ``cfg["account_id"]``
            / ``cfg["role_arn"]`` (or whatever you stored on the endpoint)
            and returns them merged with fresh credentials.
        load_balancing: Pool load-balancing algorithm — "round_robin"
            (default) or "random". Only meaningful when ``endpoints`` has
            more than one entry.
        worker_index: Round-robin offset for the load-balancer. Only
            meaningful when ``endpoints`` has more than one entry; lets you
            stagger multiple workers so they pick different starting
            endpoints.

    Returns:
        A live :class:`SlowBurnLLM` worker, ready to accept ``call_llm()`` calls.

    Example (single endpoint)::

        from slowburn import create_llm

        llm = create_llm(model="gpt-4o-mini", budget_usd=5.0, window="daily")
        result = llm.call_llm(prompt="Summarize this paper...").result()
        print(f"Cost so far: ${llm.get_reporter().result().total_cost():.4f}")
        llm.stop()

    Example (multi-account AWS Bedrock with two-hop role chaining)::

        from slowburn import create_llm, EndpointConfig

        endpoints = [
            EndpointConfig(
                model="bedrock/us.anthropic.claude-sonnet-4-6",
                max_rpm=250,
                # Fields not known to EndpointConfig — preserved for resolver:
                account_id="111111111111",
                region="us-east-1",
                role_arn="arn:aws:iam::111111111111:role/BedrockAccess",
            ),
            EndpointConfig(
                model="bedrock/eu.anthropic.claude-sonnet-4-6",
                max_rpm=125,
                account_id="222222222222",
                region="eu-west-2",
                role_arn="arn:aws:iam::222222222222:role/BedrockAccess",
            ),
        ]

        def my_resolver(cfg: dict) -> dict:
            # User-supplied two-hop role chain, with caching inside.
            creds = assume_role_chain(
                base_role_arn="arn:aws:iam::000:role/Hop1",
                target_role_arn=cfg["role_arn"],
                region=cfg["region"],
            )
            return {
                **cfg,
                "litellm_params": {
                    **cfg.get("litellm_params", {}),
                    "aws_access_key_id": creds["AccessKeyId"],
                    "aws_secret_access_key": creds["SecretAccessKey"],
                    "aws_session_token": creds["SessionToken"],
                    "aws_region_name": cfg["region"],
                },
            }

        llm = create_llm(
            model="bedrock/us.anthropic.claude-sonnet-4-6",
            endpoints=endpoints,
            endpoint_resolver=my_resolver,
            budget_usd=10.0,
        )
    """
    defaults = slowburn_config.defaults
    # Resolve every _NO_ARG bare kwarg through slowburn_config.defaults so the
    # value we feed each EndpointConfig's cascade is fully concrete.
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

    if name is None:
        name = model

    # The worker_defaults dict is what each EndpointConfig's _NO_ARG fields
    # cascade into. It contains exactly the EndpointConfig fields that are
    # also set at the create_llm/worker layer.
    worker_defaults: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_rpm": max_rpm,
        "max_input_tpm": max_input_tpm,
        "max_output_tpm": max_output_tpm,
        "budget_usd": budget_usd,
        "window": window,
        "rate_limit_algorithm": rate_limit_algorithm.value
        if isinstance(rate_limit_algorithm, RateLimitAlgorithm)
        else rate_limit_algorithm,
        "extra_limits": list(extra_limits) if extra_limits is not None else [],
    }

    # Build the list of fully-resolved EndpointConfigs.
    if endpoints is None:
        # Single-endpoint path: synthesize one EndpointConfig that is purely
        # the worker defaults. This still goes through the LimitPool
        # machinery so a single-endpoint pool and a multi-endpoint pool
        # share the same code paths.
        resolved_endpoints: List[EndpointConfig] = [
            resolve_concrete_endpoint_config(
                config=EndpointConfig(),
                worker_defaults=worker_defaults,
            )
        ]
    else:
        if len(endpoints) == 0:
            raise ValueError(
                "create_llm(endpoints=[]) is not allowed. Pass at least one EndpointConfig "
                "or omit `endpoints` for a single-endpoint setup."
            )
        resolved_endpoints = [
            resolve_concrete_endpoint_config(
                config=coerce_to_endpoint_config(ep),
                worker_defaults=worker_defaults,
            )
            for ep in endpoints
        ]

    # Build one LimitSet per endpoint. Each LimitSet stores its endpoint's
    # full model_dump() in its `config` field so the worker can rebuild a
    # typed EndpointConfig at acquisition time.
    limit_sets: List[LimitSet] = [
        _build_limit_set_for_endpoint(
            endpoint=ep,
            backend=backend,
            rate_limit_algorithm=RateLimitAlgorithm(ep.rate_limit_algorithm),
        )
        for ep in resolved_endpoints
    ]
    limit_pool = LimitPool(
        limit_sets=limit_sets,
        load_balancing=load_balancing,
        worker_index=worker_index,
    )

    llm = SlowBurnLLM.options(
        mode=backend,
        limits=limit_pool,
        num_retries={"call_llm": num_retries, "*": 0},
        retry_on={"call_llm": retry_on, "*": []},
        retry_wait={"call_llm": retry_wait, "*": 1},
        retry_algorithm={"call_llm": retry_algorithm, "*": RetryAlgorithm.Exponential},
        retry_jitter={"call_llm": retry_jitter, "*": 0},
    ).init(
        name=name,
        model_name=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        tools=tools,
        tool_choice=tool_choice,
        litellm_params=litellm_params if litellm_params is not None else {},
        backpressure_notify=backpressure_notify,
        on_budget_overflow=on_budget_overflow,
        on_pricing_unavailable=on_pricing_unavailable,
        endpoint_resolver=endpoint_resolver,
    )
    return llm
