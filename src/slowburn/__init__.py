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
    ResourceLimit,
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
    passthrough_resolver,
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


def _build_limit_pool(
    *,
    endpoints: List[EndpointConfig],
    endpoint_overrides: List[set],
    backend: ExecutionBackend,
    load_balancing: str,
    worker_index: int,
    global_max_rpm: int,
    global_max_input_tpm: int,
    global_max_output_tpm: int,
    global_max_concurrent_calls: int,
    global_budget_usd: float,
    global_window_seconds: float,
    global_rate_limit_algorithm: RateLimitAlgorithm,
) -> LimitPool:
    """Build a ``LimitPool`` with shared global limits + per-endpoint overrides.

    For each limit-shaping field (``max_rpm``, ``max_input_tpm``,
    ``max_output_tpm``, ``max_concurrent_calls``, ``budget_usd``):

    - If an endpoint did NOT explicitly set that field (i.e., it inherited
      the ``create_llm`` default), this LimitSet receives a reference to a
      single SHARED ``Limit`` instance that is also shared by every other
      endpoint that did not set it. Concurry's ``InMemorySharedLimitSet``
      accesses the limit's internal state directly, so two LimitSets holding
      the same limit instance share its capacity.
    - If an endpoint DID set that field, this LimitSet gets its own private
      ``Limit`` instance with the endpoint-specific capacity.

    This means:

    - With ``max_rpm=300`` at create_llm and 56 endpoints (none overriding),
      total RPM across the pool is **300** (one shared CallLimit).
    - With ``max_rpm=300`` at create_llm and one endpoint overriding to 1000,
      the 55 unset endpoints share a 300-rpm CallLimit and the one explicit
      endpoint has its own 1000-rpm CallLimit.

    The token RateLimits and CallLimit always use a 60-second window (industry
    convention for "rpm" / "tpm"). The CostLimit uses the configured cost
    window.
    """
    # Build a single shared instance for each global limit that endpoints
    # might inherit. These are constructed lazily — we only allocate the ones
    # at least one endpoint actually needs.
    shared_call_limit: Optional[CallLimit] = None
    shared_input_rate_limit: Optional[RateLimit] = None
    shared_output_rate_limit: Optional[RateLimit] = None
    shared_resource_limit: Optional[ResourceLimit] = None
    shared_cost_limit: Optional[CostLimit] = None

    def _global_call_limit() -> CallLimit:
        nonlocal shared_call_limit
        if shared_call_limit is None:
            shared_call_limit = CallLimit(
                window_seconds=60,
                capacity=global_max_rpm,
                algorithm=global_rate_limit_algorithm,
            )
        return shared_call_limit

    def _global_input_rate_limit() -> RateLimit:
        nonlocal shared_input_rate_limit
        if shared_input_rate_limit is None:
            shared_input_rate_limit = RateLimit(
                key="input_tokens",
                window_seconds=60,
                capacity=global_max_input_tpm,
                algorithm=global_rate_limit_algorithm,
            )
        return shared_input_rate_limit

    def _global_output_rate_limit() -> RateLimit:
        nonlocal shared_output_rate_limit
        if shared_output_rate_limit is None:
            shared_output_rate_limit = RateLimit(
                key="output_tokens",
                window_seconds=60,
                capacity=global_max_output_tpm,
                algorithm=global_rate_limit_algorithm,
            )
        return shared_output_rate_limit

    def _global_resource_limit() -> ResourceLimit:
        nonlocal shared_resource_limit
        if shared_resource_limit is None:
            shared_resource_limit = ResourceLimit(
                key="concurrent_calls",
                capacity=global_max_concurrent_calls,
            )
        return shared_resource_limit

    def _global_cost_limit() -> Optional[CostLimit]:
        nonlocal shared_cost_limit
        if math.isinf(global_budget_usd):
            return None
        if shared_cost_limit is None:
            shared_cost_limit = CostLimit(
                budget_usd=global_budget_usd,
                window_seconds=global_window_seconds,
            )
        return shared_cost_limit

    limit_sets: List[LimitSet] = []
    for endpoint, overrides in zip(endpoints, endpoint_overrides):
        endpoint_algo = RateLimitAlgorithm(endpoint.rate_limit_algorithm)
        algorithm_overridden: bool = "rate_limit_algorithm" in overrides

        # CallLimit (max_rpm)
        if "max_rpm" in overrides or algorithm_overridden:
            call_limit_obj: CallLimit = CallLimit(
                window_seconds=60,
                capacity=endpoint.max_rpm,
                algorithm=endpoint_algo,
            )
        else:
            call_limit_obj = _global_call_limit()

        # Input tokens RateLimit (max_input_tpm)
        if "max_input_tpm" in overrides or algorithm_overridden:
            input_rate_obj: RateLimit = RateLimit(
                key="input_tokens",
                window_seconds=60,
                capacity=endpoint.max_input_tpm,
                algorithm=endpoint_algo,
            )
        else:
            input_rate_obj = _global_input_rate_limit()

        # Output tokens RateLimit (max_output_tpm)
        if "max_output_tpm" in overrides or algorithm_overridden:
            output_rate_obj: RateLimit = RateLimit(
                key="output_tokens",
                window_seconds=60,
                capacity=endpoint.max_output_tpm,
                algorithm=endpoint_algo,
            )
        else:
            output_rate_obj = _global_output_rate_limit()

        # ResourceLimit (max_concurrent_calls)
        if "max_concurrent_calls" in overrides:
            resource_limit_obj: ResourceLimit = ResourceLimit(
                key="concurrent_calls",
                capacity=endpoint.max_concurrent_calls,
            )
        else:
            resource_limit_obj = _global_resource_limit()

        # CostLimit (budget_usd / window) — both must be inheritable together
        # because they're a (capacity, window) pair on the same Limit object.
        cost_overridden: bool = "budget_usd" in overrides or "window" in overrides
        if cost_overridden:
            if math.isinf(endpoint.budget_usd):
                cost_limit_obj: Optional[CostLimit] = None
            else:
                cost_limit_obj = CostLimit(
                    budget_usd=endpoint.budget_usd,
                    window_seconds=_window_to_seconds(endpoint.window),
                )
        else:
            cost_limit_obj = _global_cost_limit()

        limits_list: List[Any] = []
        if cost_limit_obj is not None:
            limits_list.append(cost_limit_obj)
        limits_list.append(input_rate_obj)
        limits_list.append(output_rate_obj)
        limits_list.append(call_limit_obj)
        limits_list.append(resource_limit_obj)
        if endpoint.extra_limits:
            limits_list.extend(endpoint.extra_limits)

        limit_sets.append(
            LimitSet(
                limits=limits_list,
                mode=backend,
                shared=True,
                config=endpoint.model_dump(),
            )
        )

    return LimitPool(
        limit_sets=limit_sets,
        load_balancing=load_balancing,
        worker_index=worker_index,
    )


@validate
def create_llm(
    model: str,
    budget_usd: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    window: Union[WindowAlias, int, float, _NO_ARG_TYPE] = _NO_ARG,
    max_rpm: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    max_input_tpm: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    max_output_tpm: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    max_concurrent_calls: Union[int, _NO_ARG_TYPE] = _NO_ARG,
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
    endpoints: Optional[List[Dict[str, Any]]] = None,
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
    - **Multi-endpoint** (``endpoints=[...]``): the user provides a plain
      dict per endpoint. Each is overlaid against the bare kwargs: any field
      omitted from the dict falls back to the corresponding bare kwarg, which
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
          > resolver-augmented endpoint dict field
            > endpoint dict field (set in ``endpoints=[...]``)
              > ``create_llm(field=...)``
                > ``slowburn_config.defaults.field``

    Shared global limits
    --------------------

    For the limit-shaping fields (``max_rpm``, ``max_input_tpm``,
    ``max_output_tpm``, ``max_concurrent_calls``, ``budget_usd``),
    endpoints that DO NOT explicitly override the field share a single
    ``Limit`` instance with every other endpoint that also did not override.
    This means ``create_llm(max_rpm=300, endpoints=[...])`` enforces a
    global 300 rpm across the pool (not 300 rpm per endpoint), unless an
    endpoint sets its own ``max_rpm``, in which case that endpoint gets a
    private limit at the override capacity.

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
        max_concurrent_calls: Default cap on in-flight calls per endpoint
            (Concurry ``ResourceLimit`` capacity). Endpoints that don't
            override this share a single global ResourceLimit at this
            capacity; endpoints that override get their own.
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
        endpoints: List of per-endpoint configurations as plain dicts. Each
            dict may set any of the ``EndpointConfig`` fields plus arbitrary
            extras (``account_id``, ``role_arn``, etc.) for the resolver to
            consume. Fields omitted from the dict fall back to the
            ``create_llm`` kwargs, which themselves fall back to
            ``slowburn_config.defaults``. When ``None`` (default), a single
            synthetic endpoint is built from the bare kwargs.
        endpoint_resolver: Callable ``(config_dict) -> dict`` that runs once
            per call AFTER the LimitPool selects an endpoint, BEFORE the
            litellm call. The dict it receives is the selected endpoint's
            full ``model_dump()`` (including any unknown extras the user
            attached). The dict it returns is validated into a new
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

        from slowburn import create_llm

        endpoints = [
            {
                "model": "bedrock/us.anthropic.claude-sonnet-4-6",
                "max_rpm": 250,
                # Fields not known to EndpointConfig — preserved for resolver:
                "account_id": "111111111111",
                "region": "us-east-1",
                "role_arn": "arn:aws:iam::111111111111:role/BedrockAccess",
            },
            {
                "model": "bedrock/eu.anthropic.claude-sonnet-4-6",
                "max_rpm": 125,
                "account_id": "222222222222",
                "region": "eu-west-2",
                "role_arn": "arn:aws:iam::222222222222:role/BedrockAccess",
            },
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
    # ----- 1. Resolve every _NO_ARG bare kwarg through slowburn_config.defaults
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
    if is_no_arg(max_concurrent_calls):
        max_concurrent_calls = defaults.max_concurrent_calls
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

    # ----- 2. Concrete create_llm-level values (fall-back layer for endpoints)
    worker_extra_limits: List[Any] = list(extra_limits) if extra_limits is not None else []
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
        "max_concurrent_calls": max_concurrent_calls,
        "budget_usd": budget_usd,
        "window": window,
        "rate_limit_algorithm": rate_limit_algorithm,
        "extra_limits": worker_extra_limits,
    }

    # ----- 3. Normalize endpoint dicts. Each entry must be a plain dict; we
    # build the fully-resolved EndpointConfig later, after recording which
    # fields were explicitly overridden (for the shared-limit-object logic).
    if endpoints is None:
        endpoint_dicts: List[Dict[str, Any]] = [{}]
    else:
        if len(endpoints) == 0:
            raise ValueError(
                "create_llm(endpoints=[]) is not allowed. Pass at least one endpoint dict "
                "or omit `endpoints` for a single-endpoint setup."
            )
        endpoint_dicts = []
        for i, ep in enumerate(endpoints):
            if not isinstance(ep, dict):
                raise TypeError(
                    f"create_llm(endpoints=[...]) entry {i} must be a plain dict, "
                    f"got {type(ep).__name__}. EndpointConfig is an internal type "
                    "that SlowBurn constructs from your dicts."
                )
            endpoint_dicts.append(dict(ep))

    # ----- 4. For each endpoint dict, record which fields it explicitly set
    # (before overlaying defaults), then overlay create_llm-level values for
    # everything else, and finally validate into an EndpointConfig.
    resolved_endpoints: List[EndpointConfig] = []
    endpoint_overrides: List[set] = []
    for ep_dict in endpoint_dicts:
        overrides: set = {k for k in ep_dict.keys() if k in worker_defaults}
        for field, default_value in worker_defaults.items():
            if field not in ep_dict:
                ep_dict[field] = default_value
        resolved_endpoints.append(EndpointConfig(**ep_dict))
        endpoint_overrides.append(overrides)

    # ----- 5. Build the LimitPool with shared global limits + per-endpoint
    # overrides where the endpoint set its own value.
    limit_pool: LimitPool = _build_limit_pool(
        endpoints=resolved_endpoints,
        endpoint_overrides=endpoint_overrides,
        backend=backend,
        load_balancing=load_balancing,
        worker_index=worker_index,
        global_max_rpm=max_rpm,
        global_max_input_tpm=max_input_tpm,
        global_max_output_tpm=max_output_tpm,
        global_max_concurrent_calls=max_concurrent_calls,
        global_budget_usd=budget_usd,
        global_window_seconds=_window_to_seconds(window),
        global_rate_limit_algorithm=rate_limit_algorithm,
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
