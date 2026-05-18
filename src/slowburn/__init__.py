"""
SlowBurn — Cost-Sustainable Concurrent Execution for Long-Horizon LLM Agents.

SlowBurn is a thin layer on top of Concurry that adds:
1. CostLimit — a dollar-denominated rate limit
2. SlowBurnLLM — a cost-tracking asyncio LLM worker
3. CostReporter — experiment-level cost attribution
4. Drop-in integrations for CrewAI and AutoGen (AG2)

Quick start::

    from slowburn import create_llm

    llm = create_llm(model="gpt-4o-mini", limits=dict(budget_per_day=5.0))
    result = llm.call_llm(prompt="Hello world").result()
    print(llm.get_reporter().result().total_cost())
    llm.stop()
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import litellm
from concurry import (
    LimitPool,
    LimitSet,
    LoadBalancingAlgorithm,
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
    BackpressureNotify,
    BudgetOverflowAction,
    ExecutionBackend,
    PricingUnavailableAction,
    ToolChoiceOption,
)
from .cost_accounting import CostCallContext, cost_controlled_call, estimate_input_tokens
from .endpoints import (
    EndpointConfig,
    EndpointResolver,
    _build_endpoint_configs,
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
from .limits_spec import SLOT_TO_LIMIT_KEY, SlowBurnLimits, default_slowburn_limits
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
    "build_limit_pool",
    "passthrough_resolver",
    "dollars_to_microdollars",
    "microdollars_to_dollars",
    "DEFAULT_COST_LIMIT_KEY",
    "slowburn_config",
    "temp_config",
    "SlowBurnConfig",
    "SlowBurnDefaults",
    "SlowBurnLimits",
    "default_slowburn_limits",
    "SLOT_TO_LIMIT_KEY",
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


# ---------------------------------------------------------------------------
# build_limit_pool
# ---------------------------------------------------------------------------


def _coerce_to_slowburn_limits(
    limits: Optional[Union[SlowBurnLimits, Dict[str, Any]]],
) -> Optional[SlowBurnLimits]:
    """Accept ``SlowBurnLimits``, dict, or ``None``."""
    if limits is None:
        return None
    if isinstance(limits, SlowBurnLimits):
        return limits
    if isinstance(limits, dict):
        return SlowBurnLimits(**limits)
    raise TypeError(f"`limits` must be a SlowBurnLimits, a dict, or None; got {type(limits).__name__}.")


def build_limit_pool(
    *,
    endpoints: Optional[List[Dict[str, Any]]] = None,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Union[Optional[float], _NO_ARG_TYPE] = _NO_ARG,
    max_tokens: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    timeout: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    limits: Optional[Union[SlowBurnLimits, Dict[str, Any]]] = None,
    backend: ExecutionBackend = "Asyncio",
    load_balancing: Union[LoadBalancingAlgorithm, str] = LoadBalancingAlgorithm.RoundRobin,
    worker_index: int = 0,
) -> LimitPool:
    """Build a Concurry ``LimitPool`` for SlowBurn from endpoint dicts + globals.

    This helper is for power users who construct :class:`SlowBurnLLM` directly
    via ``SlowBurnLLM.options(limits=...).init(...)`` instead of through
    :func:`create_llm`. It performs the same dict-overlay, slot-cascade, and
    shared-vs-private ``Limit`` routing that ``create_llm`` does internally.

    Limits cascade (replace-slot)
    -----------------------------

    For each of the five slots — ``requests``, ``input_tokens``,
    ``output_tokens``, ``budget``, ``concurrency`` — the value used for an
    endpoint is resolved as:

        endpoint's ``limits.<slot>`` if set
            → ``limits.<slot>`` from this function call if set
                → library default from
                  :func:`slowburn.limits_spec.default_slowburn_limits`

    "Replace-slot" means: if an endpoint sets ``limits.requests``, the
    endpoint's ``requests`` slot fully replaces the global slot. There is no
    per-window merging across cascade layers.

    Sharing
    -------

    When two endpoints both inherit the same slot from the global cascade
    (or the library default), they share the SAME ``RateLimit`` /
    ``CostLimit`` / ``ResourceLimit`` Python instance. So
    ``build_limit_pool(limits={"rpm": 300}, endpoints=[<56 endpoints>])``
    enforces a single global 300 requests/min across the whole pool.

    Args:
        endpoints: List of plain endpoint dicts. Each may set any
            ``EndpointConfig`` field plus arbitrary extras. ``None``
            yields a single synthetic endpoint built from the bare kwargs.
        model: Default model identifier (litellm format).
        api_key: Default API key.
        api_base: Default API base URL.
        temperature: Default sampling temperature.
        max_tokens: Default max output tokens.
        timeout: Default per-call timeout in seconds.
        limits: Global limits applied to every endpoint that does not
            override the slot. Accepts a :class:`SlowBurnLimits`, a dict
            (which is coerced via ``SlowBurnLimits(**dict)`` and so accepts
            shorthand kwargs like ``rpm``, ``budget_per_day``,
            ``concurrency``), or ``None`` (every slot inherits the library
            default).
        backend: Concurry execution backend ("Asyncio" or "Ray").
        load_balancing: Pool load-balancing algorithm.
        worker_index: Round-robin offset.

    Returns:
        A :class:`LimitPool` ready to pass to ``SlowBurnLLM.options(limits=...)``.
    """
    defaults = slowburn_config.defaults
    if is_no_arg(temperature):
        temperature = defaults.temperature
    if is_no_arg(max_tokens):
        max_tokens = defaults.max_tokens
    if is_no_arg(timeout):
        timeout = defaults.timeout
    load_balancing_enum: LoadBalancingAlgorithm = (
        load_balancing
        if isinstance(load_balancing, LoadBalancingAlgorithm)
        else LoadBalancingAlgorithm(load_balancing)
    )

    global_limits: SlowBurnLimits = _coerce_to_slowburn_limits(limits) or SlowBurnLimits()
    library_defaults: SlowBurnLimits = default_slowburn_limits()

    worker_defaults: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        # ``limits`` is intentionally NOT in worker_defaults: per-endpoint
        # ``limits`` is treated as "inherit if not set", and the cascade
        # logic in build_endpoint_limit_set looks it up directly.
    }

    # Validate endpoints argument up front for clear errors.
    if endpoints is None:
        endpoint_dicts: List[Dict[str, Any]] = [{}]
    else:
        if len(endpoints) == 0:
            raise ValueError(
                "build_limit_pool(endpoints=[]) is not allowed. Pass at least one endpoint dict "
                "or omit `endpoints` for a single-endpoint pool."
            )
        for i, ep in enumerate(endpoints):
            if not isinstance(ep, dict):
                raise TypeError(
                    f"build_limit_pool(endpoints=[...]) entry {i} must be a plain dict, "
                    f"got {type(ep).__name__}."
                )
        endpoint_dicts = list(endpoints)

    # Build typed EndpointConfigs (this is where each endpoint's `limits` dict
    # gets coerced into a SlowBurnLimits via pydantic).
    resolved_endpoints, _overrides = _build_endpoint_configs(
        endpoints=endpoint_dicts, defaults=worker_defaults
    )

    # Slot-cascade over each endpoint, with sharing.
    from ._pool_builder import _SharedLimitCache, build_endpoint_limit_set

    cache = _SharedLimitCache()
    limit_sets: List[LimitSet] = [
        build_endpoint_limit_set(
            endpoint=ep,
            global_limits=global_limits,
            default_limits=library_defaults,
            cache=cache,
            backend=backend,
        )
        for ep in resolved_endpoints
    ]

    return LimitPool(
        limit_sets=limit_sets,
        load_balancing=load_balancing_enum,
        worker_index=worker_index,
    )


# ---------------------------------------------------------------------------
# create_llm
# ---------------------------------------------------------------------------


@validate
def create_llm(
    model: str,
    limits: Optional[Union[SlowBurnLimits, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
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
    litellm_params: Optional[Dict[str, object]] = None,
    backpressure_notify: Union[BackpressureNotify, _NO_ARG_TYPE] = _NO_ARG,
    on_budget_overflow: Union[BudgetOverflowAction, _NO_ARG_TYPE] = _NO_ARG,
    on_pricing_unavailable: PricingUnavailableAction = "error",
    endpoints: Optional[List[Dict[str, Any]]] = None,
    endpoint_resolver: Optional[EndpointResolver] = None,
    load_balancing: Union[LoadBalancingAlgorithm, str] = LoadBalancingAlgorithm.RoundRobin,
    worker_index: int = 0,
) -> SlowBurnLLM:
    """Create a cost-controlled LLM worker with sensible defaults.

    All limits are funneled through a single ``limits=`` parameter — a
    :class:`SlowBurnLimits` (or a dict that gets coerced into one) with five
    slots: ``requests``, ``input_tokens``, ``output_tokens``, ``budget``,
    ``concurrency``. The dict accepts both canonical entries (``requests=
    [RateLimit(300, "minute")]``) and a wide set of shorthand kwargs
    (``rpm``, ``input_tokens_per_minute``, ``otph``, ``budget_per_day``,
    ``concurrency``, etc. — see :class:`slowburn.SlowBurnLimits`).

    Single-endpoint vs multi-endpoint
    ---------------------------------

    SlowBurn always uses a ``LimitPool`` internally; the difference is how
    many endpoints it contains.

    - **Single endpoint** (``endpoints`` is ``None``): a single synthetic
      endpoint is built from ``model`` / ``api_key`` / etc. The pool has
      exactly one ``LimitSet``.
    - **Multi-endpoint** (``endpoints=[...]``): each entry is a plain dict
      that may carry its own ``limits=`` dict for slot overrides plus
      bookkeeping fields (``account_id``, ``role_arn``, ...) that the
      ``endpoint_resolver`` reads at call time.

    Cascade order for any overridable LLM-call field
    ------------------------------------------------

    For ``model`` / ``api_key`` / ``api_base`` / ``temperature`` /
    ``max_tokens`` / ``timeout`` and ``litellm_params``:

        ``call_llm(field=...)``
          > resolver-augmented endpoint dict field
            > endpoint dict field (set in ``endpoints=[...]``)
              > ``create_llm(field=...)``
                > ``slowburn_config.defaults.field``

    Cascade order for limits (replace-slot)
    ---------------------------------------

    For each of the 5 slots:

        endpoint's ``limits.<slot>`` if set
          > ``create_llm(limits={...})[<slot>]`` if set
            > library default

    "Replace-slot" means: if an endpoint sets ``limits.requests``, the
    endpoint's ``requests`` slot fully replaces the global slot. No
    per-window merging across cascade layers.

    Sharing
    -------

    Endpoints that inherit a slot from the global cascade share the SAME
    ``RateLimit`` / ``CostLimit`` / ``ResourceLimit`` instance — so
    ``create_llm(limits={"rpm": 300}, endpoints=[...])`` enforces a single
    global 300 requests/min across the whole pool.

    Args:
        model: litellm model identifier (e.g. "gpt-4o-mini",
            "bedrock/us.anthropic.claude-sonnet-4-6").
        limits: Global limits. Accepts a :class:`SlowBurnLimits`, a dict
            (which is coerced via ``SlowBurnLimits(**dict)`` and so accepts
            shorthand kwargs like ``rpm``, ``input_tokens_per_minute``,
            ``budget_per_day``, ``concurrency``), or ``None`` (every slot
            inherits the library default).
        api_key: Default API key for any endpoint whose ``api_key`` is unset.
        api_base: Default API base URL for any endpoint whose ``api_base``
            is unset.
        backend: Execution backend — "Asyncio" (default) or "Ray".
        name: Worker name for logging. Defaults to the model name.
        temperature: Default sampling temperature.
        max_tokens: Default max output tokens.
        timeout: Default per-call timeout in seconds.
        num_retries: Number of retries on transient errors.
        retry_on: Exception types that trigger a retry on ``call_llm``.
        retry_wait: Base wait time in seconds for transient-error retries.
        retry_algorithm: Backoff strategy ("Exponential" / "Linear" / "Fibonacci").
        retry_jitter: Jitter factor in [0, 1] added to each retry wait.
        tools: Default tool schemas (OpenAI format) for all calls.
        tool_choice: Default tool_choice for all calls.
        litellm_params: Worker-level kwargs forwarded to every
            ``litellm.acompletion`` call.
        backpressure_notify: When "warn", logs a warning if acquire blocks
            longer than ``backpressure_threshold_seconds``.
        on_budget_overflow: Action when a single call's estimated cost
            exceeds the budget capacity.
        on_pricing_unavailable: Action when the model is not in litellm's
            pricing database.
        endpoints: List of per-endpoint configurations as plain dicts.
        endpoint_resolver: Callable ``(config_dict) -> dict`` that runs once
            per call AFTER the LimitPool selects an endpoint, BEFORE the
            litellm call.
        load_balancing: Pool load-balancing algorithm.
        worker_index: Round-robin offset.

    Returns:
        A live :class:`SlowBurnLLM` worker.

    Examples
    --------

    Single-endpoint with a daily budget::

        from slowburn import create_llm

        llm = create_llm(model="gpt-4o-mini", limits=dict(budget_per_day=5.0))

    Multi-account AWS Bedrock with a global rate cap and per-endpoint overrides::

        endpoints = [
            {
                "model": "bedrock/us.anthropic.claude-sonnet-4-6",
                "limits": dict(rpm=250),
                "account_id": "111111111111",
                "region": "us-east-1",
                "role_arn": "arn:aws:iam::111111111111:role/BedrockAccess",
            },
            {
                "model": "bedrock/eu.anthropic.claude-sonnet-4-6",
                "account_id": "222222222222",
                "region": "eu-west-2",
                "role_arn": "arn:aws:iam::222222222222:role/BedrockAccess",
            },
        ]
        llm = create_llm(
            model="bedrock/us.anthropic.claude-sonnet-4-6",
            endpoints=endpoints,
            endpoint_resolver=my_resolver,
            limits=dict(rpm=500, budget_per_day=10.0, concurrency=20),
        )
    """
    defaults = slowburn_config.defaults
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

    limit_pool: LimitPool = build_limit_pool(
        endpoints=endpoints,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        limits=limits,
        backend=backend,
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
