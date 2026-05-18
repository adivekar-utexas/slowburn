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
concrete (no ``_NO_ARG`` sentinels) — except for ``limits``, which is
intentionally nullable: a ``None`` slot means "inherit from the cascade"
(global ``create_llm(limits=...)`` → library default).

This object is an internal type: ``create_llm`` constructs it from the user's
plain endpoint dicts after overlaying ``create_llm`` kwargs and
``slowburn_config.defaults``. End users pass plain dicts (or use
:func:`build_limit_pool` directly).

- **Known fields** are exactly the kwargs that ``SlowBurnLLM`` and ``call_llm``
  accept directly (``model``, ``api_key``, ``api_base``, ``temperature``,
  ``max_tokens``, ``timeout``), the unified ``limits`` slot bag, plus a
  ``litellm_params`` dict that is forwarded as-is to ``litellm.acompletion``.
- **Unknown fields** are accepted (``extra="allow"``). They survive on the
  config object so the user's ``endpoint_resolver`` can read them, but they
  are NOT forwarded to ``litellm.acompletion`` (litellm would error on
  unknown kwargs). Use this for bookkeeping fields like ``account_id``,
  ``role_arn``, ``region``, ``provider`` that are only meaningful to the
  resolver.

Limits cascade
--------------

Each endpoint's ``limits`` is either:

- ``None`` — inherit ALL slots from the global cascade.
- A :class:`SlowBurnLimits` (or compatible dict) where each slot is either
  ``None`` (inherit) or a concrete value (override).

When an endpoint specifies a slot, the override is *replace-slot*: the
endpoint's slot fully replaces the global slot. There is no per-window
merging. This keeps the cascade predictable.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from morphic import Typed
from pydantic import ConfigDict, Field

from .config import is_no_arg
from .limits_spec import SlowBurnLimits


class EndpointConfig(Typed):
    """Fully-resolved per-endpoint configuration.

    All fields with a counterpart at the ``create_llm`` layer are required
    and concrete by the time ``EndpointConfig`` is constructed (inside
    ``build_limit_pool`` / ``create_llm``), with one intentional exception:
    ``limits``, which is nullable so individual slots can fall through to
    the global cascade.

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
    # pydantic primitive; pydantic still validates dict→SlowBurnLimits coercion.
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

    # ----- Limits (unified) -----

    limits: Optional[SlowBurnLimits] = Field(
        default=None,
        description=(
            "Per-endpoint limits. ``None`` means inherit every slot from the "
            "global cascade (``create_llm(limits=...)`` → library default). "
            "A :class:`SlowBurnLimits` (or compatible dict) lets you override "
            "specific slots; slots set to ``None`` still inherit from the "
            "global cascade. Slot overrides are *replace-slot*: if you set "
            "``limits.requests``, the entire global ``requests`` slot is "
            "replaced for this endpoint."
        ),
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


def _build_endpoint_configs(
    endpoints: List[Dict[str, Any]],
    *,
    defaults: Dict[str, Any],
) -> Tuple[List["EndpointConfig"], List[frozenset]]:
    """Internal: convert plain endpoint dicts into ``EndpointConfig`` instances.

    For each endpoint dict, this:

    1. Records which fields the dict explicitly set (the *override set* —
       used by the limit-pool builder to decide whether the endpoint should
       inherit global ``Limit`` instances or use private ones).
    2. Overlays ``defaults`` for any field the dict omits.
    3. Validates the merged dict into an :class:`EndpointConfig`.

    The endpoint's ``limits`` field (if set) is left as the user passed it
    (a :class:`SlowBurnLimits` or dict-coerced equivalent). Slot-level
    inheritance is then handled by :func:`build_limit_pool`.

    Args:
        endpoints: List of plain endpoint dicts.
        defaults: Dict mapping field name to the fallback value used when
            the endpoint dict does not set that field. Must contain every
            non-``limits`` required ``EndpointConfig`` field. Missing
            fields will cause validation to fail.

    Returns:
        A two-tuple ``(configs, overrides)`` where both lists have the same
        length as ``endpoints``. ``overrides[i]`` is the set of field names
        that ``endpoints[i]`` explicitly set (i.e., that should NOT fall
        back to ``defaults``).
    """
    configs: List[EndpointConfig] = []
    overrides_list: List[frozenset] = []
    for ep_dict in endpoints:
        # Track override set BEFORE we mutate or overlay anything.
        overrides: frozenset = frozenset(k for k in ep_dict.keys() if k in defaults)
        merged: Dict[str, Any] = {**defaults, **ep_dict}
        configs.append(EndpointConfig(**merged))
        overrides_list.append(overrides)
    return configs, overrides_list


__all__ = [
    "EndpointConfig",
    "EndpointResolver",
    "passthrough_resolver",
    "cascade_field",
]
