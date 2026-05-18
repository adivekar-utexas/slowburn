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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import litellm
from concurry import (
    CallLimit,
    LimitPool,
    LimitSet,
    LoadBalancingAlgorithm,
    RateLimit,
    RateLimitAlgorithm,
    RateWindow,
    ResourceLimit,
    RetryAlgorithm,
)
from concurry.core.constants import RATE_WINDOW_SECONDS
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
    RateLike,
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


def build_limit_pool(
    *,
    endpoints: Optional[List[Dict[str, Any]]] = None,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Union[Optional[float], _NO_ARG_TYPE] = _NO_ARG,
    max_tokens: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    timeout: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    max_request_rate: Union[RateLike, _NO_ARG_TYPE] = _NO_ARG,
    max_input_token_rate: Union[RateLike, _NO_ARG_TYPE] = _NO_ARG,
    max_output_token_rate: Union[RateLike, _NO_ARG_TYPE] = _NO_ARG,
    max_concurrent_requests: Union[int, _NO_ARG_TYPE] = _NO_ARG,
    budget_usd: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    budget_usd_window: Union[RateWindow, str, int, float, _NO_ARG_TYPE] = _NO_ARG,
    backend: ExecutionBackend = "Asyncio",
    load_balancing: Union[LoadBalancingAlgorithm, str] = LoadBalancingAlgorithm.RoundRobin,
    worker_index: int = 0,
) -> LimitPool:
    """Build a Concurry ``LimitPool`` for SlowBurn from endpoint dicts + globals.

    This helper is for power users who construct :class:`SlowBurnLLM` directly
    via ``SlowBurnLLM.options(limits=...).init(...)`` instead of through
    :func:`create_llm`. It performs the same dict-overlay, override tracking,
    and shared-vs-private ``Limit`` routing that ``create_llm`` does
    internally — and returns the resulting ``LimitPool`` so the caller can
    pass it to ``SlowBurnLLM.options(limits=...)``.

    Rate-field input shapes:
        Each rate dimension (``max_request_rate``, ``max_input_token_rate``,
        ``max_output_token_rate``) accepts:

        - ``int`` — uses the dimension's default window from
          ``slowburn_config.defaults`` (``"minutely"`` for all three).
        - :class:`concurry.RateLimit` — used as-is.
        - ``dict`` — validated into a ``RateLimit``.
        - ``List`` of any of the above — multiple rates on the same dimension
          (e.g. 300/min AND 50000/day).

    Shared global limits:
        For each rate-shaping field and ``max_concurrent_requests`` /
        ``budget_usd``: endpoints that did NOT explicitly set the field
        share a single ``Limit`` instance per ``(dimension, window_seconds)``
        across the pool; endpoints that DID set the field get their own
        private ``Limit`` instance(s).

        Example: ``build_limit_pool(max_request_rate=300, endpoints=[56
        endpoints, none overriding])`` creates one shared
        ``CallLimit(window=Minutely, capacity=300)`` across all 56 LimitSets.

    Args:
        endpoints: List of plain endpoint dicts. Each may set any
            ``EndpointConfig`` field plus arbitrary extras. When ``None``,
            a single synthetic endpoint is built from the global kwargs.
        model: Default model identifier (litellm format).
        api_key: Default API key. ``None`` means 'fall back to provider env
            vars or to credentials injected by the resolver via
            ``litellm_params``'.
        api_base: Default API base URL.
        temperature: Default sampling temperature.
        max_tokens: Default max output tokens.
        timeout: Default per-call timeout in seconds.
        max_request_rate: Default request-rate limit. See "Rate-field input
            shapes" above. When passed as an ``int``, the window is
            ``slowburn_config.defaults.max_request_rate_window`` (default
            ``RateWindow.Minutely``).
        max_input_token_rate: Default input-token rate. Same input shapes.
            Default window: ``max_input_token_rate_window`` (Minutely).
        max_output_token_rate: Default output-token rate. Same input shapes.
            Default window: ``max_output_token_rate_window`` (Minutely).
        max_concurrent_requests: Default in-flight request cap (ResourceLimit).
        budget_usd: Default dollar budget per ``budget_usd_window``.
            ``float('inf')`` disables cost limiting.
        budget_usd_window: Default cost-budget window. Accepts a
            :class:`concurry.RateWindow` member, a string alias (e.g.
            ``"daily"``), or a positive number of seconds. Defaults to
            ``slowburn_config.defaults.budget_usd_window`` (``Daily``).
        backend: Concurry execution backend ("Asyncio" or "Ray").
        load_balancing: Pool load-balancing algorithm. Accepts the Concurry
            ``LoadBalancingAlgorithm`` enum or its string form (defaults to
            ``RoundRobin``).
        worker_index: Round-robin offset (lets you stagger multiple workers).

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
    if is_no_arg(max_request_rate):
        max_request_rate = defaults.max_request_rate
    if is_no_arg(max_input_token_rate):
        max_input_token_rate = defaults.max_input_token_rate
    if is_no_arg(max_output_token_rate):
        max_output_token_rate = defaults.max_output_token_rate
    if is_no_arg(max_concurrent_requests):
        max_concurrent_requests = defaults.max_concurrent_requests
    if is_no_arg(budget_usd):
        budget_usd = defaults.budget_usd
    if is_no_arg(budget_usd_window):
        budget_usd_window = defaults.budget_usd_window
    load_balancing_enum: LoadBalancingAlgorithm = (
        load_balancing
        if isinstance(load_balancing, LoadBalancingAlgorithm)
        else LoadBalancingAlgorithm(load_balancing)
    )

    worker_defaults: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_request_rate": max_request_rate,
        "max_input_token_rate": max_input_token_rate,
        "max_output_token_rate": max_output_token_rate,
        "max_concurrent_requests": max_concurrent_requests,
        "budget_usd": budget_usd,
        "budget_usd_window": budget_usd_window,
        # Per-dimension default-window keys consumed by _build_endpoint_configs's
        # _normalize_rate calls; they are stripped from `merged` before the
        # ``EndpointConfig`` is constructed.
        "max_request_rate_window": defaults.max_request_rate_window,
        "max_input_token_rate_window": defaults.max_input_token_rate_window,
        "max_output_token_rate_window": defaults.max_output_token_rate_window,
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

    resolved_endpoints, endpoint_overrides = _build_endpoint_configs(
        endpoints=endpoint_dicts, defaults=worker_defaults
    )

    return _build_limit_pool_from_configs(
        endpoints=resolved_endpoints,
        endpoint_overrides=endpoint_overrides,
        backend=backend,
        load_balancing=load_balancing_enum,
        worker_index=worker_index,
        global_max_concurrent_requests=max_concurrent_requests,
        global_budget_usd=budget_usd,
        global_budget_usd_window=budget_usd_window,
    )


def _build_limit_pool_from_configs(
    *,
    endpoints: List[EndpointConfig],
    endpoint_overrides: List[frozenset],
    backend: ExecutionBackend,
    load_balancing: LoadBalancingAlgorithm,
    worker_index: int,
    global_max_concurrent_requests: int,
    global_budget_usd: float,
    global_budget_usd_window: Union[RateWindow, str, int, float],
) -> LimitPool:
    """Internal: build the ``LimitPool`` from already-resolved EndpointConfigs.

    Implements the shared-vs-private routing using the ``endpoint_overrides``
    sets; see :func:`build_limit_pool` for the user-facing entry point.

    For each rate dimension (``max_request_rate``, ``max_input_token_rate``,
    ``max_output_token_rate``), endpoints that did NOT explicitly override
    the field share their ``RateLimit`` instances with every other endpoint
    that also did not override the field. Sharing is per
    ``(dimension, key, window_seconds)`` tuple so multiple windows on the
    same dimension stay distinct.

    Each ``LimitSet`` carries a ``_rate_keys`` mapping in its ``config`` so
    the worker can populate the per-call ``requested`` / ``update`` dict
    under the right keys (e.g. when an endpoint has both
    ``input_tokens@60s`` and ``input_tokens@86400s``, both must be charged
    on every call).
    """
    # Shared limit caches keyed by the unique signature of each global limit.
    shared_rate_cache: Dict[Tuple[str, float], RateLimit] = {}
    shared_resource_limit: Optional[ResourceLimit] = None
    shared_cost_limit: Optional[CostLimit] = None

    def _share_rate(rate: RateLimit) -> RateLimit:
        """Return the shared instance for ``(rate.key, rate.window_seconds)``.

        If we've seen this signature before, return the cached instance so
        all endpoints inheriting the global rate share state. Otherwise
        cache and return ``rate`` itself.
        """
        sig = (rate.key, float(rate.window_seconds))
        if sig in shared_rate_cache:
            return shared_rate_cache[sig]
        shared_rate_cache[sig] = rate
        return rate

    def _global_resource_limit() -> ResourceLimit:
        nonlocal shared_resource_limit
        if shared_resource_limit is None:
            shared_resource_limit = ResourceLimit(
                key="concurrent_requests",
                capacity=global_max_concurrent_requests,
            )
        return shared_resource_limit

    def _global_cost_limit() -> Optional[CostLimit]:
        nonlocal shared_cost_limit
        if math.isinf(global_budget_usd):
            return None
        if shared_cost_limit is None:
            shared_cost_limit = CostLimit(
                budget_usd=global_budget_usd,
                window=global_budget_usd_window,
            )
        return shared_cost_limit

    rate_dimensions = (
        ("max_request_rate", "call_count"),
        ("max_input_token_rate", "input_tokens"),
        ("max_output_token_rate", "output_tokens"),
    )

    limit_sets: List[LimitSet] = []
    for endpoint, overrides in zip(endpoints, endpoint_overrides):
        endpoint_rate_limits: List[RateLimit] = []
        # _rate_keys is a mapping ``{base_key: [actual_key_in_LimitSet, ...]}``
        # consumed by SlowBurnLLM._build_limit_usage at call time so the
        # worker knows under which keys to charge token / call usage.
        rate_keys: Dict[str, List[str]] = {}

        for field, base_key in rate_dimensions:
            field_overridden = field in overrides
            limits_for_field: List[RateLimit] = list(getattr(endpoint, field))

            if len(limits_for_field) == 1:
                # Single rate: use the base key (back-compat for the worker's
                # legacy hardcoded usage map and consumers reading
                # ``LimitSet.config["_rate_keys"][base_key]``).
                rl = limits_for_field[0]
                if rl.key != base_key:
                    limits_for_field = [_rekeyed(rl, base_key)]
            elif len(limits_for_field) > 1:
                # Multi-rate dimension: keys must be unique within the
                # LimitSet (Concurry forbids duplicates). Strategy:
                #   1. If every user-supplied key is already distinct AND
                #      not the default ``base_key``, respect the user's
                #      naming verbatim.
                #   2. Otherwise, regenerate every key as
                #      ``f"{base_key}_{rate.params_signature()}"`` —
                #      deterministic from the rate's distinguishing params.
                user_keys = [rl.key for rl in limits_for_field]
                user_supplied_unique = (
                    len(set(user_keys)) == len(user_keys)
                    and base_key not in user_keys
                )
                if user_supplied_unique:
                    pass  # leave as-is
                else:
                    limits_for_field = [
                        _rekeyed(rl, f"{base_key}_{rl.params_signature()}")
                        for rl in limits_for_field
                    ]
                    # Defense in depth: even params_signature() can collide if
                    # the user passes literal duplicates. Reject loudly.
                    new_keys = [rl.key for rl in limits_for_field]
                    if len(set(new_keys)) != len(new_keys):
                        raise ValueError(
                            f"Multiple rate limits on dimension {field!r} have identical "
                            f"(capacity, window, algorithm); deduplicate the input or assign "
                            f"distinct ``key`` values. Generated keys: {new_keys}"
                        )

            # Decide shared-vs-private. Only share when the endpoint did not
            # override the dimension at all.
            if field_overridden:
                # Private: use the user-provided RateLimit instances as-is.
                pass
            else:
                limits_for_field = [_share_rate(rl) for rl in limits_for_field]

            endpoint_rate_limits.extend(limits_for_field)
            rate_keys[base_key] = [rl.key for rl in limits_for_field]

        # ResourceLimit (max_concurrent_requests): single value per endpoint,
        # so the simple shared-or-private toggle from before applies.
        if "max_concurrent_requests" in overrides:
            resource_limit_obj: ResourceLimit = ResourceLimit(
                key="concurrent_requests",
                capacity=endpoint.max_concurrent_requests,
            )
        else:
            resource_limit_obj = _global_resource_limit()

        # CostLimit (budget_usd / budget_usd_window): tied together because
        # they're both properties of one (capacity, window) Limit object.
        cost_overridden: bool = "budget_usd" in overrides or "budget_usd_window" in overrides
        if cost_overridden:
            if math.isinf(endpoint.budget_usd):
                cost_limit_obj: Optional[CostLimit] = None
            else:
                cost_limit_obj = CostLimit(
                    budget_usd=endpoint.budget_usd,
                    window=endpoint.budget_usd_window,
                )
        else:
            cost_limit_obj = _global_cost_limit()

        limits_list: List[Any] = []
        if cost_limit_obj is not None:
            limits_list.append(cost_limit_obj)
        limits_list.extend(endpoint_rate_limits)
        limits_list.append(resource_limit_obj)

        # Stash _rate_keys on the LimitSet config so the worker can read it
        # at acquisition time. Other code that introspects the config still
        # sees the EndpointConfig dump verbatim under the same keys.
        config_dump = endpoint.model_dump()
        config_dump["_rate_keys"] = rate_keys

        limit_sets.append(
            LimitSet(
                limits=limits_list,
                mode=backend,
                shared=True,
                config=config_dump,
            )
        )

    return LimitPool(
        limit_sets=limit_sets,
        load_balancing=load_balancing,
        worker_index=worker_index,
    )


def _rekeyed(rate: RateLimit, new_key: str) -> RateLimit:
    """Return a new ``RateLimit`` identical to ``rate`` but with ``key=new_key``.

    Used when we need unique keys for multiple windows on the same dimension
    (Concurry forbids duplicate keys in a single LimitSet).
    """
    if rate.key == new_key:
        return rate
    return RateLimit(
        key=new_key,
        window=rate.window,
        capacity=rate.capacity,
        algorithm=rate.algorithm,
    )


@validate
def create_llm(
    model: str,
    budget_usd: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    budget_usd_window: Union[RateWindow, str, int, float, _NO_ARG_TYPE] = _NO_ARG,
    max_request_rate: Union[RateLike, _NO_ARG_TYPE] = _NO_ARG,
    max_input_token_rate: Union[RateLike, _NO_ARG_TYPE] = _NO_ARG,
    max_output_token_rate: Union[RateLike, _NO_ARG_TYPE] = _NO_ARG,
    max_concurrent_requests: Union[int, _NO_ARG_TYPE] = _NO_ARG,
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
      (``model``, ``api_key``, ``api_base``, ``max_request_rate``,
      ``budget_usd``, ``budget_usd_window``, ``temperature``, ``max_tokens``,
      ``timeout``, ``litellm_params``) define one synthetic
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
    rate-shaping fields), the value used at call time is resolved as:

        ``call_llm(field=...)``
          > resolver-augmented endpoint dict field
            > endpoint dict field (set in ``endpoints=[...]``)
              > ``create_llm(field=...)``
                > ``slowburn_config.defaults.field``

    Shared global limits
    --------------------

    For each rate-shaping dimension (``max_request_rate``,
    ``max_input_token_rate``, ``max_output_token_rate``,
    ``max_concurrent_requests``, ``budget_usd``), endpoints that DO NOT
    explicitly override the field share a single ``Limit`` instance with
    every other endpoint that also did not override. This means
    ``create_llm(max_request_rate=300, endpoints=[...])`` enforces a global
    300 requests/min across the pool (not 300 per endpoint), unless an
    endpoint sets its own ``max_request_rate``, in which case that endpoint
    gets a private limit at the override capacity.

    Rate-field input shapes
    -----------------------

    The three rate dimensions accept any of:

    - ``int`` — uses the dimension's default window
      (``slowburn_config.defaults.max_*_rate_window``, ``"minutely"`` by
      default).
    - :class:`concurry.RateLimit` — used as-is.
    - ``dict`` — validated into a ``RateLimit``.
    - ``List`` of any of the above — multiple rates on the same dimension
      (e.g. 300/min AND 50_000/day).

    Args:
        model: litellm model identifier (e.g. "gpt-4o-mini",
            "bedrock/us.anthropic.claude-sonnet-4-6").
        budget_usd: Maximum dollar spend per ``budget_usd_window``. Defaults
            to ``float('inf')`` (no cost limit).
        budget_usd_window: Cost-budget window. Accepts a
            :class:`concurry.RateWindow`, string alias, or seconds.
            Defaults to ``slowburn_config.defaults.budget_usd_window``
            (``Daily``).
        max_request_rate: Default request-rate limit. See "Rate-field input
            shapes" above. ``int`` uses the
            ``max_request_rate_window`` default (Minutely).
        max_input_token_rate: Default input-token rate. Same input shapes.
        max_output_token_rate: Default output-token rate. Same input shapes.
        max_concurrent_requests: Default cap on in-flight requests per
            endpoint (Concurry ``ResourceLimit`` capacity). Endpoints that
            don't override this share a single global ResourceLimit at this
            capacity; endpoints that override get their own.
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
        load_balancing: Pool load-balancing algorithm. Accepts a Concurry
            :class:`LoadBalancingAlgorithm` enum value
            (``LoadBalancingAlgorithm.RoundRobin`` or ``.Random``) or the
            equivalent string ("RoundRobin", "Random"). Defaults to
            ``RoundRobin``. Only meaningful when ``endpoints`` has more
            than one entry.
        worker_index: Round-robin offset for the load-balancer. Only
            meaningful when ``endpoints`` has more than one entry; lets you
            stagger multiple workers so they pick different starting
            endpoints.

    Returns:
        A live :class:`SlowBurnLLM` worker, ready to accept ``call_llm()`` calls.

    Example (single endpoint)::

        from slowburn import create_llm

        llm = create_llm(model="gpt-4o-mini", budget_usd=5.0)
        result = llm.call_llm(prompt="Summarize this paper...").result()
        print(f"Cost so far: ${llm.get_reporter().result().total_cost():.4f}")
        llm.stop()

    Example (multi-account AWS Bedrock with two-hop role chaining)::

        from slowburn import create_llm

        endpoints = [
            {
                "model": "bedrock/us.anthropic.claude-sonnet-4-6",
                "max_request_rate": 250,
                # Fields not known to EndpointConfig — preserved for resolver:
                "account_id": "111111111111",
                "region": "us-east-1",
                "role_arn": "arn:aws:iam::111111111111:role/BedrockAccess",
            },
            {
                "model": "bedrock/eu.anthropic.claude-sonnet-4-6",
                "max_request_rate": 125,
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
    # ----- 1. Resolve only the kwargs that this function itself consumes —
    # i.e., the ones plumbed into ``SlowBurnLLM.options(...)`` for retry
    # behavior. Everything else (limit-shaping kwargs, per-endpoint defaults,
    # and worker-init kwargs that already accept ``_NO_ARG``) is forwarded
    # downstream as-is, where the receiver does its own ``slowburn_config``
    # fall-back.
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

    # ----- 2. Build the LimitPool. ``build_limit_pool`` owns the dict
    # overlay, override tracking, shared-vs-private Limit-instance routing,
    # AND the ``_NO_ARG → slowburn_config.defaults`` fall-back for every
    # field it needs. We forward our kwargs verbatim (sentinels included).
    limit_pool: LimitPool = build_limit_pool(
        endpoints=endpoints,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_request_rate=max_request_rate,
        max_input_token_rate=max_input_token_rate,
        max_output_token_rate=max_output_token_rate,
        max_concurrent_requests=max_concurrent_requests,
        budget_usd=budget_usd,
        budget_usd_window=budget_usd_window,
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
