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

``EndpointConfig`` is a permissive dict-like Typed model:

- **Known fields** are exactly the kwargs that ``SlowBurnLLM`` and ``call_llm``
  accept directly (``model``, ``api_key``, ``api_base``, ``temperature``,
  ``max_tokens``, ``timeout``, plus the limit-shaping fields ``max_rpm``,
  ``max_input_tpm``, ``max_output_tpm``, ``budget_usd``, ``window``,
  ``rate_limit_algorithm``, ``extra_limits``) and a ``litellm_params`` dict
  that is forwarded as-is to ``litellm.acompletion``.
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
        > ``create_llm`` kwarg (worker default)
          > ``slowburn_config.defaults``

The default resolver is a passthrough that returns the config unchanged. A
custom resolver runs on every call after the LimitPool selects an endpoint;
it receives the endpoint's ``model_dump()`` dict, may add or rewrite any
fields (including unknown ones), and returns the augmented dict. The worker
then re-validates that dict back into an ``EndpointConfig``, applies the
cascade, and proceeds with the call. This is how the user injects
request-time data such as freshly-assumed STS credentials.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from morphic import MutableTyped
from pydantic import ConfigDict, Field

from .config import _NO_ARG, _NO_ARG_TYPE, is_no_arg
from .constants import WindowAlias


class EndpointConfig(MutableTyped):
    """Per-endpoint configuration in a multi-account SlowBurnLLM.

    ``EndpointConfig`` mirrors the kwargs of ``create_llm`` and ``call_llm``
    so any value that can be set globally on the worker can also be set
    per-endpoint here. Unset fields use the ``_NO_ARG`` sentinel, allowing
    the cascade to fall through to the worker default.

    Unknown fields are preserved (``extra="allow"``) so the user can attach
    bookkeeping such as ``account_id`` / ``role_arn`` / ``region`` /
    ``provider`` for the resolver to read. Unknown fields are NOT forwarded
    to ``litellm.acompletion`` — only known fields and the contents of
    ``litellm_params`` are.

    Example (single-endpoint, never explicitly constructed by the user)::

        # When create_llm() is called with bare kwargs, SlowBurn builds one
        # EndpointConfig under the hood from those kwargs. The user never
        # sees this object.
        cfg = EndpointConfig(model="gpt-4o-mini", max_rpm=500, budget_usd=5.0)

    Example (multi-endpoint, AWS Bedrock across accounts)::

        endpoints = [
            EndpointConfig(
                model="bedrock/us.anthropic.claude-sonnet-4-6",
                max_rpm=250,
                # Unknown fields, preserved for the resolver:
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
    """

    # ``arbitrary_types_allowed`` is required because ``_NO_ARG`` is a custom
    # sentinel type that Pydantic does not recognize natively.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # ----- Per-call overrides (forwarded to litellm.acompletion as named kwargs) -----

    model: Union[str, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description=(
            "Per-endpoint model identifier (litellm format). When this endpoint "
            "is selected by the LimitPool, this model is passed to "
            "litellm.acompletion. Falls back to create_llm(model=...) when unset."
        ),
    )
    api_key: Union[str, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint API key. Falls back to create_llm(api_key=...).",
    )
    api_base: Union[Optional[str], _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description=(
            "Per-endpoint API base URL (litellm api_base). Useful for self-hosted "
            "OpenAI-compatible endpoints, OpenRouter overrides, etc."
        ),
    )
    temperature: Union[Optional[float], _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint sampling temperature.",
    )
    max_tokens: Union[int, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint max output tokens.",
    )
    timeout: Union[float, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint per-call timeout in seconds.",
    )

    # ----- Limit-shaping (consumed at LimitSet construction; not forwarded to litellm) -----

    max_rpm: Union[int, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description=(
            "Per-endpoint requests-per-minute cap. Builds the CallLimit for this "
            "endpoint's LimitSet."
        ),
    )
    max_input_tpm: Union[int, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint input tokens-per-minute cap.",
    )
    max_output_tpm: Union[int, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint output tokens-per-minute cap.",
    )
    budget_usd: Union[float, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description=(
            "Per-endpoint dollar budget over the cost window. Falls back to "
            "create_llm(budget_usd=...). When both are set, this REPLACES the "
            "global default for this endpoint (not additive)."
        ),
    )
    window: Union[WindowAlias, int, float, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-endpoint cost-budget window.",
    )
    rate_limit_algorithm: Union[str, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
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

    # ----- Reporter label (auto-derived if None) -----

    endpoint_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional human-readable label used by CostReporter to attribute "
            "cost per endpoint. If unset, SlowBurn auto-derives a label from "
            "any account_id/region/provider fields the user added as extras."
        ),
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


# Fields that, when set on an EndpointConfig, MUST be resolved (no remaining
# _NO_ARG) before the LimitSet is built, because Concurry needs concrete
# integers/floats for its limit construction.
_LIMIT_SHAPING_FIELDS = (
    "max_rpm",
    "max_input_tpm",
    "max_output_tpm",
    "budget_usd",
    "window",
    "rate_limit_algorithm",
)

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


def resolve_concrete_endpoint_config(
    *,
    config: EndpointConfig,
    worker_defaults: Dict[str, Any],
) -> EndpointConfig:
    """Replace every ``_NO_ARG`` field with the corresponding worker default.

    This is called ONCE at ``create_llm`` time, before the EndpointConfig is
    handed to Concurry as a ``LimitSet.config``. The resulting object is fully
    concrete (no remaining ``_NO_ARG``), which means Concurry never sees the
    sentinel, the limit construction has concrete numbers, and the resolver at
    call time is given a clean serialized form.

    Args:
        config: The endpoint config with possible ``_NO_ARG`` placeholders.
        worker_defaults: Dict mapping field name to the resolved worker
            default for that field (i.e., the value the user passed to
            ``create_llm`` or the ``slowburn_config.defaults`` value).

    Returns:
        A new ``EndpointConfig`` with all ``_NO_ARG`` fields filled in.
        Unknown extra fields are preserved verbatim.
    """
    raw: Dict[str, Any] = config.model_dump()
    for field, default in worker_defaults.items():
        if field in raw and is_no_arg(raw[field]):
            raw[field] = default
    return EndpointConfig(**raw)


def coerce_to_endpoint_config(value: Union[EndpointConfig, Dict[str, Any]]) -> EndpointConfig:
    """Accept either an ``EndpointConfig`` instance or a plain dict.

    This is the bridge that lets ``create_llm(endpoints=[...])`` accept
    either:

    - ``[EndpointConfig(model=..., max_rpm=...)]``
    - ``[{"model": ..., "max_rpm": ...}]``
    - A mix of both.

    Plain dicts are validated through ``EndpointConfig(**d)``.
    """
    if isinstance(value, EndpointConfig):
        return value
    if isinstance(value, dict):
        return EndpointConfig(**value)
    raise TypeError(
        f"endpoints elements must be EndpointConfig or dict, got {type(value).__name__}"
    )


__all__ = [
    "EndpointConfig",
    "EndpointResolver",
    "passthrough_resolver",
    "cascade_field",
    "resolve_concrete_endpoint_config",
    "coerce_to_endpoint_config",
]
