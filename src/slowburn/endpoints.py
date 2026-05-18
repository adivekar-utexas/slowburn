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
never construct an ``EndpointConfig`` directly — they pass plain dicts.

- **Known fields** are exactly the kwargs that ``SlowBurnLLM`` and ``call_llm``
  accept directly (``model``, ``api_key``, ``api_base``, ``temperature``,
  ``max_tokens``, ``timeout``, plus the limit-shaping fields ``max_rpm``,
  ``max_input_tpm``, ``max_output_tpm``, ``max_concurrency``, ``budget_usd``,
  ``window``, ``rate_limit_algorithm``, ``extra_limits``) and a
  ``litellm_params`` dict that is forwarded as-is to ``litellm.acompletion``.
- **Unknown fields** are accepted (``extra="allow"``). They survive on the
  config object so the user's ``endpoint_resolver`` can read them, but they
  are NOT forwarded to ``litellm.acompletion`` (litellm would error on
  unknown kwargs). Use this for bookkeeping fields like ``account_id``,
  ``role_arn``, ``region``, ``provider`` that are only meaningful to the
  resolver.

Cascade order
-------------

For every known field, the value used during a call is resolved as:

    per-call ``call_llm`` kwarg
      > resolver-augmented ``EndpointConfig`` field
        > ``EndpointConfig`` field (concrete, set at ``create_llm`` time)

The default resolver is a passthrough that returns the config unchanged. A
custom resolver runs on every call after the LimitPool selects an endpoint;
it receives the endpoint's ``model_dump()`` dict, may add or rewrite any
fields (including unknown ones), and returns the augmented dict. The worker
then re-validates that dict back into an ``EndpointConfig``, applies the
cascade, and proceeds with the call. This is how the user injects
request-time data such as freshly-assumed STS credentials.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from morphic import Typed
from pydantic import ConfigDict, Field

from .config import is_no_arg
from .constants import WindowAlias


class EndpointConfig(Typed):
    """Fully-resolved per-endpoint configuration in a multi-account SlowBurnLLM.

    All fields with a counterpart at the ``create_llm`` layer are required:
    by the time ``EndpointConfig`` is constructed (inside ``create_llm`` or via
    :func:`build_endpoint_configs`), the cascade against the ``create_llm``
    kwargs and ``slowburn_config.defaults`` has already been applied, so every
    field is guaranteed to have a concrete value.

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

    The simplest way to construct ``EndpointConfig`` instances is via
    :func:`create_llm` — pass plain dicts to ``endpoints=[...]`` and SlowBurn
    builds the ``EndpointConfig`` instances internally::

        endpoints = [
            {
                "model": "bedrock/us.anthropic.claude-sonnet-4-6",
                "max_rpm": 250,
                # Unknown fields (preserved for the resolver):
                "account_id": "111111111111",
                "region": "us-east-1",
                "role_arn": "arn:aws:iam::111111111111:role/BedrockAccess",
            },
            ...
        ]

    Power users who construct :class:`SlowBurnLLM` directly via
    ``SlowBurnLLM.options(...).init(...)`` (instead of through ``create_llm``)
    can use :func:`build_endpoint_configs` to perform the same dict → typed
    conversion themselves. See its docstring for an example.
    """

    # ``arbitrary_types_allowed`` is preserved because some fields (e.g.,
    # ``extra_limits``) hold Concurry ``Limit`` objects that Pydantic does not
    # know how to validate.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # ----- Per-call overrides (forwarded to litellm.acompletion as named kwargs) -----

    model: str = Field(
        description=(
            "Per-endpoint model identifier (litellm format). When this endpoint "
            "is selected by the LimitPool, this model is passed to "
            "litellm.acompletion."
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

    # ----- Limit-shaping (consumed at LimitSet construction; not forwarded to litellm) -----

    max_rpm: int = Field(
        description="Per-endpoint requests-per-minute cap (CallLimit capacity).",
    )
    max_input_tpm: int = Field(
        description="Per-endpoint input tokens-per-minute cap.",
    )
    max_output_tpm: int = Field(
        description="Per-endpoint output tokens-per-minute cap.",
    )
    max_concurrent_calls: int = Field(
        description=(
            "Per-endpoint maximum concurrent in-flight calls (ResourceLimit "
            "capacity). This caps how many ``call_llm`` calls can be running "
            "against this endpoint simultaneously, regardless of RPM."
        ),
    )
    budget_usd: float = Field(
        description=(
            "Per-endpoint dollar budget over the cost window. Use "
            "``float('inf')`` to disable cost limiting for this endpoint."
        ),
    )
    window: Union[WindowAlias, int, float] = Field(
        description="Per-endpoint cost-budget window ('daily' / 'hourly' / 'minutely' / seconds).",
    )
    rate_limit_algorithm: str = Field(
        description="Per-endpoint rate-limit algorithm (GCRA / SlidingWindow / TokenBucket).",
    )
    extra_limits: List[Any] = Field(
        default_factory=list,
        description="Additional Concurry Limit objects to include in this endpoint's LimitSet.",
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
        description=("Optional human-readable label used by CostReporter to attribute cost per endpoint."),
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

# Fields used to build the per-endpoint ``LimitSet``. Concurry needs these as
# concrete values at ``create_llm`` time.
_LIMIT_SHAPING_FIELDS = (
    "max_rpm",
    "max_input_tpm",
    "max_output_tpm",
    "max_concurrent_calls",
    "budget_usd",
    "window",
    "rate_limit_algorithm",
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

    With strict ``EndpointConfig`` (all known fields concrete), the
    ``config_value`` will never be ``_NO_ARG`` — but the cascade still works:
    when a per-call value is unset (``_NO_ARG``), it falls through to the
    config value. The ``worker_default`` argument is kept for backwards
    compatibility with call sites that still want a third-level fallback.
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
       share a global Concurry ``Limit`` instance or get a private one).
    2. Overlays ``defaults`` for any field the dict omits.
    3. Validates the merged dict into an :class:`EndpointConfig`.

    Args:
        endpoints: List of plain endpoint dicts.
        defaults: Dict mapping field name to the fallback value used when
            the endpoint dict does not set that field. Must contain every
            required ``EndpointConfig`` field; missing fields will cause
            validation to fail.

    Returns:
        A two-tuple ``(configs, overrides)`` where both lists have the same
        length as ``endpoints``:

        - ``configs[i]``: the fully-resolved :class:`EndpointConfig`.
        - ``overrides[i]``: a frozenset of field names the user explicitly
          set on the input dict (before defaults were overlaid). Two
          endpoints carrying the same numeric value are still distinguished
          by whether the value was inherited or explicitly set.
    """
    configs: List[EndpointConfig] = []
    overrides_list: List[frozenset] = []
    for ep_dict in endpoints:
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
