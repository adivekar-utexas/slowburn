"""
Endpoint configuration for SlowBurn's multi-account routing.

When a SlowBurnLLM serves traffic across multiple accounts, regions, or even
providers, each "endpoint" is described by an :class:`EndpointConfig`. The
worker maintains an internal Concurry ``LimitPool`` whose member ``LimitSet``
objects are 1:1 with the configured endpoints. On every ``call_llm`` the pool
selects an endpoint via its load-balancer; the selected endpoint's
``EndpointConfig`` then drives the actual ``litellm.acompletion`` call.

Design philosophy
-----------------

``EndpointConfig`` is a *fully-resolved* per-endpoint description. Every field
that has a counterpart at the ``create_llm`` layer is **required** and must be
concrete (no ``_NO_ARG`` sentinels). This object is an internal type:
``create_llm`` constructs it from the user's plain endpoint dicts after
overlaying ``create_llm`` kwargs and ``slowburn_config.defaults``. End users
pass plain dicts (or use :func:`build_limit_pool` directly).

- **Known fields** are exactly the kwargs that ``SlowBurnLLM`` and ``call_llm``
  accept directly (``model``, ``api_key``, ``api_base``, ``temperature``,
  ``max_tokens``, ``timeout``, plus the limit-shaping fields
  ``max_request_rate``, ``max_input_token_rate``, ``max_output_token_rate``,
  ``max_concurrent_requests``, ``budget_usd``, ``budget_usd_window``) and a
  ``litellm_params`` dict that is forwarded as-is to ``litellm.acompletion``.
- **Unknown fields** are accepted (``extra="allow"``). They survive on the
  config object so the user's ``endpoint_resolver`` can read them, but they
  are NOT forwarded to ``litellm.acompletion`` (litellm would error on
  unknown kwargs). Use this for bookkeeping fields like ``account_id``,
  ``role_arn``, ``region``, ``provider`` that are only meaningful to the
  resolver.

Rate-field input shapes
-----------------------

Each rate dimension (``max_request_rate``, ``max_input_token_rate``,
``max_output_token_rate``) accepts any of:

- ``int`` — uses the dimension's default window from ``slowburn_config``.
- :class:`concurry.RateLimit` — used as-is.
- ``dict`` — validated into a ``RateLimit``.
- ``List`` of any of the above — multiple rates on the same dimension
  (e.g. 300/min AND 50000/day).

The ``_normalize_rate`` helper converts each user-supplied value into a
canonical ``List[RateLimit]`` before constructing the ``EndpointConfig``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from concurry import RateLimit, RateWindow
from concurry.core.constants import RATE_WINDOW_SECONDS
from morphic import Typed
from pydantic import ConfigDict, Field

from .config import is_no_arg

# A user-supplied rate value before normalization.
RateLike = Union[int, RateLimit, Dict[str, Any], List[Union[int, RateLimit, Dict[str, Any]]]]


class EndpointConfig(Typed):
    """Fully-resolved per-endpoint configuration.

    All fields with a counterpart at the ``create_llm`` layer are required:
    by the time ``EndpointConfig`` is constructed (inside ``build_limit_pool``
    / ``create_llm``), the cascade against the ``create_llm`` kwargs and
    ``slowburn_config.defaults`` has already been applied, so every field is
    guaranteed to have a concrete value.

    ``EndpointConfig`` is **immutable** — once built, its fields don't change.
    Per-call cascading (per-call > resolver-augmented > config) happens at
    call time and produces *new* values without mutating the config. The
    resolver returns a new dict that is re-validated into a *new*
    ``EndpointConfig`` instance.

    Unknown fields are preserved (``extra="allow"``) so the user can attach
    bookkeeping such as ``account_id`` / ``role_arn`` / ``region`` /
    ``provider`` for the resolver to read. Unknown fields are NOT forwarded
    to ``litellm.acompletion`` — only known fields and the contents of
    ``litellm_params`` are.

    Pass plain dicts (with extras) to ``create_llm(endpoints=[...])`` or
    :func:`slowburn.build_limit_pool` rather than constructing
    ``EndpointConfig`` instances yourself.
    """

    # ``arbitrary_types_allowed=True`` because Concurry's ``RateLimit`` is not a
    # pydantic primitive; pydantic still validates dict→RateLimit coercion via
    # the model's own validator since ``RateLimit`` is a Pydantic ``Typed``.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # ----- Per-call overrides (forwarded to litellm.acompletion as named kwargs) -----

    model: str = Field(
        description=(
            "Per-endpoint model identifier (litellm format). When this endpoint "
            "is selected by the LimitPool, this model is passed to litellm.acompletion."
        ),
    )
    api_key: Optional[str] = Field(
        description=(
            "Per-endpoint API key. ``None`` means 'no explicit key — let "
            "litellm fall back to provider env vars (OPENAI_API_KEY, etc.) "
            "or to the resolver-injected credentials in litellm_params'."
        ),
    )
    api_base: Optional[str] = Field(
        description=(
            "Per-endpoint API base URL (litellm api_base). Useful for self-hosted "
            "OpenAI-compatible endpoints, OpenRouter overrides, etc. ``None`` is "
            "a valid value (means 'use the provider default')."
        ),
    )
    temperature: Optional[float] = Field(
        description=(
            "Per-endpoint sampling temperature. ``None`` is valid and means 'let the provider decide'."
        ),
    )
    max_tokens: int = Field(
        description="Per-endpoint max output tokens.",
    )
    timeout: float = Field(
        description="Per-endpoint per-call timeout in seconds.",
    )

    # ----- Rate-shaping (already normalized to List[RateLimit] before construction) -----

    max_request_rate: List[RateLimit] = Field(
        description=(
            "Per-endpoint request rate limit(s). Stored as a list of "
            "``RateLimit`` (one or more) covering different windows."
        ),
    )
    max_input_token_rate: List[RateLimit] = Field(
        description="Per-endpoint input-token rate limit(s).",
    )
    max_output_token_rate: List[RateLimit] = Field(
        description="Per-endpoint output-token rate limit(s).",
    )
    max_concurrent_requests: int = Field(
        description=(
            "Per-endpoint maximum concurrent in-flight requests "
            "(ResourceLimit capacity). Caps how many ``call_llm`` requests "
            "can be running against this endpoint simultaneously, regardless "
            "of ``max_request_rate``."
        ),
    )

    # ----- Cost / budget -----

    budget_usd: float = Field(
        description=(
            "Per-endpoint dollar budget over ``budget_usd_window``. Use "
            "``float('inf')`` to disable cost limiting for this endpoint."
        ),
    )
    budget_usd_window: Union[RateWindow, int, float] = Field(
        description="Per-endpoint cost-budget window (RateWindow member, alias, or seconds).",
    )

    # ----- LiteLLM passthrough -----

    litellm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary kwargs forwarded to litellm.acompletion(**litellm_params). "
            "Use for AWS auth (aws_access_key_id, aws_secret_access_key, "
            "aws_session_token, aws_region_name, aws_role_name, ...), provider-"
            "specific extras (extra_body, extra_headers, response_format), or "
            "anything else that litellm understands but SlowBurn does not "
            "expose as a first-class field."
        ),
    )

    # ----- Reporter label -----

    endpoint_id: Optional[str] = Field(
        default=None,
        description="Optional human-readable label used by CostReporter to attribute cost per endpoint.",
    )


# A resolver receives the endpoint's serialized form (dict from
# ``EndpointConfig.model_dump()``) and returns a (possibly augmented) dict.
# The worker re-validates the result back into an ``EndpointConfig``.
EndpointResolver = Callable[[Dict[str, Any]], Dict[str, Any]]


def passthrough_resolver(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """The default resolver: returns the config dict unchanged.

    With this resolver, the EndpointConfig the user provides is the
    EndpointConfig the worker uses. Custom resolvers exist for cases where
    request-time data (e.g., freshly-assumed STS credentials) needs to be
    injected before the litellm call.
    """
    return config_dict


# Fields that flow into ``litellm.acompletion`` as NAMED kwargs (i.e., handled
# explicitly by the worker, not via ``**litellm_params``).
_PER_CALL_OVERRIDE_FIELDS = (
    "model",
    "api_key",
    "api_base",
    "temperature",
    "max_tokens",
    "timeout",
)

# Fields used to build the per-endpoint ``LimitSet``.
_LIMIT_SHAPING_FIELDS = (
    "max_request_rate",
    "max_input_token_rate",
    "max_output_token_rate",
    "max_concurrent_requests",
    "budget_usd",
    "budget_usd_window",
)


def cascade_field(
    *,
    field: str,
    call_value: Any,
    config_value: Any,
    worker_default: Any,
) -> Any:
    """Three-level cascade for one field.

    Order: per-call > config > worker default.

    A value is "provided" if it is not the ``_NO_ARG`` sentinel. An explicit
    ``None`` IS a provided value (matches SlowBurn semantics).
    """
    if not is_no_arg(call_value):
        return call_value
    if not is_no_arg(config_value):
        return config_value
    return worker_default


def _normalize_rate(
    value: Any,
    *,
    key_base: str,
    default_window: Union[RateWindow, int, float],
    default_algorithm: Optional[Any] = None,
) -> List[RateLimit]:
    """Coerce a user-supplied rate value into a canonical ``List[RateLimit]``.

    Accepts:
    - ``int`` — wraps as ``RateLimit(key=key_base, window=default_window, capacity=int)``.
    - :class:`RateLimit` — used as-is (key may differ from ``key_base``).
    - ``dict`` — validated into a ``RateLimit``. If the dict omits ``key``, we
      inject ``key_base``. If it omits ``window``, we inject ``default_window``.
      If it omits ``algorithm`` and ``default_algorithm`` is given, we inject
      that.
    - ``List`` — each element normalized as above; the result is concatenated.

    Args:
        value: User-supplied rate value.
        key_base: Default ``key`` to use when an entry doesn't specify one.
            Distinct keys per dimension (e.g. ``"call_count"``,
            ``"input_tokens"``) are how the worker tells dimensions apart.
        default_window: Default window when an entry is just an int or
            doesn't specify a window.
        default_algorithm: Default ``RateLimitAlgorithm`` to apply when an
            entry is an int or a dict without ``algorithm``. ``RateLimit``
            instances passed verbatim are NOT modified. ``None`` (the
            default) lets concurry's global config supply the algorithm.

    Returns:
        A list of ``RateLimit`` instances. Empty list is allowed (means "no
        rate limit on this dimension").
    """
    if isinstance(value, list):
        out: List[RateLimit] = []
        for elem in value:
            out.extend(
                _normalize_rate(
                    elem,
                    key_base=key_base,
                    default_window=default_window,
                    default_algorithm=default_algorithm,
                )
            )
        return out
    if isinstance(value, RateLimit):
        return [value]
    if isinstance(value, dict):
        d = dict(value)
        d.setdefault("key", key_base)
        if "window" not in d and "window_seconds" not in d:
            d["window"] = default_window
        if default_algorithm is not None:
            d.setdefault("algorithm", default_algorithm)
        return [RateLimit(**d)]
    if isinstance(value, bool):
        # bool is a subclass of int; reject explicitly to avoid surprises.
        raise TypeError(
            f"rate value must be int / RateLimit / dict / list, got bool: {value!r}"
        )
    if isinstance(value, int):
        kwargs: Dict[str, Any] = dict(key=key_base, window=default_window, capacity=value)
        if default_algorithm is not None:
            kwargs["algorithm"] = default_algorithm
        return [RateLimit(**kwargs)]
    raise TypeError(
        f"rate value must be int / RateLimit / dict / list of those, got "
        f"{type(value).__name__}: {value!r}"
    )


def _build_endpoint_configs(
    endpoints: List[Dict[str, Any]],
    *,
    defaults: Dict[str, Any],
) -> Tuple[List["EndpointConfig"], List[frozenset]]:
    """Internal: convert plain endpoint dicts into ``EndpointConfig`` instances.

    For each endpoint dict, this:

    1. Records which fields the dict explicitly set (the *override set* —
       used by the limit-pool builder to decide whether the endpoint should
       share a global Concurry ``Limit`` instance or get a private one).
    2. Overlays ``defaults`` for any field the dict omits.
    3. Normalizes the three rate dimensions into ``List[RateLimit]`` using
       ``_normalize_rate`` and the per-dimension default windows in
       ``defaults`` (``max_*_rate_window`` keys). When the user passes an
       int or a dict without ``algorithm``, the slowburn default
       ``rate_limit_algorithm`` (from ``slowburn_config.defaults``) is
       applied so all rate limits within a SlowBurn pool have a consistent
       default algorithm rather than picking up concurry's global config.
    4. Validates the merged dict into an :class:`EndpointConfig`.

    Args:
        endpoints: List of plain endpoint dicts.
        defaults: Dict mapping field name to the fallback value used when
            the endpoint dict does not set that field. Must contain every
            required ``EndpointConfig`` field, plus the per-dimension
            ``..._rate_window`` keys used by ``_normalize_rate``. Missing
            fields will cause validation to fail.

    Returns:
        A two-tuple ``(configs, overrides)`` where both lists have the same
        length as ``endpoints``.
    """
    # Lazy import: ``config.py`` already imports from ``constants.py`` and
    # would create a cycle if we imported ``slowburn_config`` at module load.
    from .config import slowburn_config

    rate_dimensions = (
        ("max_request_rate", "call_count", "max_request_rate_window"),
        ("max_input_token_rate", "input_tokens", "max_input_token_rate_window"),
        ("max_output_token_rate", "output_tokens", "max_output_token_rate_window"),
    )
    default_algorithm = slowburn_config.defaults.rate_limit_algorithm

    configs: List[EndpointConfig] = []
    overrides_list: List[frozenset] = []
    for ep_dict in endpoints:
        # Track override set BEFORE we mutate or overlay anything.
        overrides: frozenset = frozenset(k for k in ep_dict.keys() if k in defaults)
        merged: Dict[str, Any] = {**defaults, **ep_dict}

        # Normalize the three rate dimensions to List[RateLimit].
        for field, key_base, window_key in rate_dimensions:
            default_window = merged.get(window_key)
            merged[field] = _normalize_rate(
                merged[field],
                key_base=key_base,
                default_window=default_window,
                default_algorithm=default_algorithm,
            )

        # Strip the per-dimension window keys (they are *defaults*, not fields
        # of EndpointConfig) so EndpointConfig validation does not see them.
        for _, _, window_key in rate_dimensions:
            merged.pop(window_key, None)

        configs.append(EndpointConfig(**merged))
        overrides_list.append(overrides)
    return configs, overrides_list


__all__ = [
    "EndpointConfig",
    "EndpointResolver",
    "RateLike",
    "passthrough_resolver",
    "cascade_field",
]
