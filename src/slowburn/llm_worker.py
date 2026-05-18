"""
SlowBurnLLM: AsyncIO worker for cost-controlled LLM calls.

Follows the VLM worker pattern from the astro-llm project:
estimate tokens -> acquire limits (blocks if budget exhausted) ->
execute LLM call -> update limits with actuals -> log cost.

The blocking acquire() IS the "SlowBurn" mechanism: when the dollar
budget or token rate limit is exhausted, the worker's event loop
sleeps (via asyncio.sleep) until capacity is available rather than
crashing with an error.
"""

import asyncio
import base64
import hashlib
import logging
import math
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Self, TypeVar, Union

import litellm
from concurry import async_gather, worker
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from morphic import Typed, validate
from morphic.string import format_exception_msg
from pydantic import Field

from .config import _NO_ARG, _NO_ARG_TYPE, is_no_arg, slowburn_config
from .constants import (
    BackpressureNotify,
    BudgetOverflowAction,
    ImageDetailLevel,
    PricingUnavailableAction,
    ToolChoiceOption,
)
from .endpoints import (
    EndpointConfig,
    EndpointResolver,
    cascade_field,
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
from .limits import DEFAULT_COST_LIMIT_KEY, microdollars_to_dollars
from .pricing import ModelNotFoundError, PricingCache
from .reporter import CostReporter

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []  # SlowBurn tracks cost internally; litellm callbacks unused
litellm.failure_callback = []  # Clearing these also prevents the LoggingWorker race condition
# under high concurrency (litellm bug: coroutine reuse in queue)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


# Disable LiteLLM's GLOBAL_LOGGING_WORKER entirely.
# SlowBurn clears all callbacks and manages its own cost accounting, so the background
# worker serves no purpose. Without this, the singleton binds to Concurry's private
# event loops and crashes with "RuntimeError: Event loop is closed" on worker shutdown.
# We explicitly .close() the coroutine to prevent "RuntimeWarning: coroutine was never awaited".
def _no_op_enqueue(async_coroutine: Any, **kwargs: Any) -> None:
    if hasattr(async_coroutine, "close"):
        async_coroutine.close()


GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue = _no_op_enqueue  # type: ignore[method-assign]

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _hash_prompt(prompt: Any) -> str:
    """Short hash of the prompt for log correlation."""
    text = str(prompt)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _estimate_tokens(text: str) -> int:
    """Rough estimate of token count from character length.

    Reads ``chars_per_token`` from ``slowburn_config.defaults`` at call time.
    """
    cfg = slowburn_config.defaults
    return max(int(len(text) // cfg.chars_per_token), 1)


def _mime_type_for_path(image_path: Path) -> str:
    """Infer MIME type from file extension, defaulting to image/png."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is not None and mime.startswith("image/"):
        return mime
    return "image/png"


def _encode_image_to_data_url(image_path: Path) -> str:
    """Read an image file and return a base64 data-URL string.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is empty or unreadable.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    raw = image_path.read_bytes()
    if len(raw) == 0:
        raise ValueError(f"Image file is empty: {image_path}")
    encoded = base64.b64encode(raw).decode("utf-8")
    mime = _mime_type_for_path(image_path)
    return f"data:{mime};base64,{encoded}"


ImageInput = Union[str, Path]


def _resolve_image_inputs(images: List[ImageInput]) -> List[str]:
    """Convert a list of image inputs to data-URL strings.

    Each element may be:
    - A ``Path`` or path-like string pointing to a local file (encoded to base64).
    - A string starting with ``http://`` or ``https://`` (passed through as-is).
    - A string starting with ``data:`` (already a data-URL, passed through).

    Returns:
        List of URL strings suitable for the ``image_url`` message content part.
    """
    urls: List[str] = []
    for img in images:
        if isinstance(img, Path):
            urls.append(_encode_image_to_data_url(img))
        elif isinstance(img, str):
            if img.startswith(("http://", "https://", "data:")):
                urls.append(img)
            else:
                urls.append(_encode_image_to_data_url(Path(img)))
        else:
            raise TypeError(
                f"Image input must be a Path, URL string, or data-URL string, got {type(img).__name__}"
            )
    return urls


def _build_user_content(
    prompt: str,
    images: Optional[List[ImageInput]],
    image_detail: ImageDetailLevel,
) -> Union[str, List[Dict[str, Any]]]:
    """Build the user message content, with optional vision parts."""
    if images is None or len(images) == 0:
        return prompt
    image_urls = _resolve_image_inputs(images)
    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content_parts.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
    return content_parts


class Usage(Typed):
    """Token and cost quantities for a single LLM attempt."""

    input_tokens: int
    output_tokens: int
    cost_microdollars: int

    @validate
    def with_output_tokens(self, *, output_tokens: int) -> Self:
        """Return the same usage with a different output token count.

        This is used for failure paths where input tokens may have been
        consumed but no reliable response usage exists.
        """
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=output_tokens,
            cost_microdollars=self.cost_microdollars,
        )


@worker(mode="Asyncio")
class SlowBurnLLM(Typed):
    """AsyncIO worker for concurrent LLM calls with dollar-budget backpressure.

    Wraps litellm.acompletion and integrates with Concurry's LimitSet for
    rate limiting on tokens, calls, AND dollars (via CostLimit). When any
    limit is exhausted, acquire() blocks asynchronously until capacity
    replenishes.

    The worker also maintains a CostReporter that accumulates per-call cost
    data for later export to JSON, Markdown, or LaTeX.

    Typical setup::

        from slowburn.limits import CostLimit
        from concurry import LimitSet, RateLimit, CallLimit

        llm = SlowBurnLLM.options(
            limits=LimitSet(
                limits=[
                    CostLimit(budget_usd=5.0, window=86400),
                    RateLimit(key="input_tokens", window=60, capacity=1_000_000),
                    RateLimit(key="output_tokens", window=60, capacity=200_000),
                    CallLimit(window=60, capacity=500),
                ],
                mode="Asyncio",
                shared=True,
            ),
            num_retries={"call_llm": 3, "*": 0},
            retry_on={"call_llm": [ValueError, asyncio.TimeoutError], "*": []},
        ).init(
            name="my-llm",
            model_name="gpt-4o-mini",
            api_key="sk-...",
        )

        result = llm.call_llm(prompt="Hello world").result()
        print(llm.reporter.result().total_cost())
        llm.stop()
    """

    name: str = Field(..., description="Worker name (for logging)")
    model_name: str = Field(..., description="litellm model identifier")
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key. ``None`` (default) means 'fall back to provider "
            "env vars (e.g. OPENAI_API_KEY) or to credentials injected by "
            "the endpoint_resolver via litellm_params'."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Worker-level API base URL (litellm api_base) for OpenAI-compatible "
            "self-hosted endpoints, OpenRouter overrides, etc. Falls through the "
            "cascade like every other overridable field: per-call > endpoint config > "
            "this worker default."
        ),
    )
    temperature: Union[Optional[float], _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="LLM sampling temperature. Defaults to slowburn_config.defaults.temperature.",
    )
    max_tokens: Union[int, _NO_ARG_TYPE] = Field(
        default=_NO_ARG, description="Max output tokens. Defaults to slowburn_config.defaults.max_tokens."
    )
    timeout: Union[float, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description="Per-call timeout in seconds. Defaults to slowburn_config.defaults.timeout.",
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Default tool schemas (OpenAI format) for all calls. Overridable per-call.",
    )
    tool_choice: Optional[ToolChoiceOption] = Field(
        default=None,
        description='Default tool_choice for all calls ("auto", "required", "none"). Overridable per-call.',
    )
    litellm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters passed through to litellm.acompletion(). "
            "Use for response_format, seed, top_p, stop, logprobs, "
            "or any other litellm-supported parameter. Per-call litellm_params "
            "in call_llm() merge on top of these defaults."
        ),
    )
    backpressure_notify: Union[BackpressureNotify, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description=(
            'When "warn", logs a warning if acquire() blocks longer than '
            'backpressure_threshold_seconds. When "ignore", silent. '
            "Defaults to slowburn_config.defaults.backpressure_notify."
        ),
    )
    on_budget_overflow: Union[BudgetOverflowAction, _NO_ARG_TYPE] = Field(
        default=_NO_ARG,
        description=(
            "Action when a single call's estimated cost exceeds the budget capacity. "
            '"warn" (default): proceed with the call but log a warning. '
            '"error": raise BudgetOverflowError. "ignore": proceed silently. '
            "Defaults to slowburn_config.defaults.on_budget_overflow."
        ),
    )
    on_pricing_unavailable: PricingUnavailableAction = Field(
        default="error",
        description=(
            "Action when the model is not in litellm's pricing database. "
            '"error" (default): raise PricingUnavailableError. '
            '"warn": log a warning and skip cost tracking (set cost to 0). '
            '"ignore": silently skip cost tracking. '
            "This only matters when a CostLimit is active (finite budget)."
        ),
    )
    endpoint_resolver: Optional[EndpointResolver] = Field(
        default=None,
        description=(
            "Optional callable run on every call to inject request-time data "
            "(e.g., freshly-assumed AWS STS credentials) into the selected "
            "endpoint's config. Signature: ``(config_dict) -> dict``. The dict "
            "passed in is the selected endpoint's ``EndpointConfig.model_dump()`` "
            "including any unknown extras the user attached. The dict returned "
            "is re-validated into an ``EndpointConfig`` whose fields then "
            "override the original. Default ``None`` means a passthrough is "
            "used (the original config is used unchanged)."
        ),
    )

    def post_initialize(self) -> None:
        defaults = slowburn_config.defaults
        if is_no_arg(self.temperature):
            object.__setattr__(self, "temperature", defaults.temperature)
        if is_no_arg(self.max_tokens):
            object.__setattr__(self, "max_tokens", defaults.max_tokens)
        if is_no_arg(self.timeout):
            object.__setattr__(self, "timeout", defaults.timeout)
        if is_no_arg(self.backpressure_notify):
            object.__setattr__(self, "backpressure_notify", defaults.backpressure_notify)
        if is_no_arg(self.on_budget_overflow):
            object.__setattr__(self, "on_budget_overflow", defaults.on_budget_overflow)
        self._reporter = CostReporter()

    @property
    def reporter(self) -> CostReporter:
        """Access the CostReporter to inspect costs or export reports."""
        return self._reporter

    @validate
    def build_messages(
        self,
        *,
        prompt: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        images: Optional[List[ImageInput]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        image_detail: Union[ImageDetailLevel, _NO_ARG_TYPE] = _NO_ARG,
    ) -> List[Dict[str, Any]]:
        """Build the messages list without calling the LLM.

        If *prompt* is a list of dicts, treat it as pre-built messages.
        If *history* is provided, appends the new user message to it.
        If *history* is None, builds a fresh [system?, user] list.

        When *history* is provided and *prompt* is empty, no new user
        message is appended (useful for re-submitting after tool results).

        Args:
            prompt: The user message string, or a pre-built messages list.
            system_prompt: Optional system message. Prepended only if
                *history* has no existing system message at index 0.
            images: Optional images to include in the user message.
            history: Previous messages list. If provided, the new user
                message is appended to a copy of this list.
            image_detail: Detail level for vision queries ("low"/"high"/"auto").
                Defaults to slowburn_config.defaults.image_detail.

        Returns:
            The complete messages list ready for litellm.acompletion().
        """
        if is_no_arg(image_detail):
            image_detail = slowburn_config.defaults.image_detail
        if isinstance(prompt, list):
            return list(prompt)

        if history is not None:
            messages = list(history)
            if system_prompt is not None and (len(messages) == 0 or messages[0].get("role") != "system"):
                messages.insert(0, {"role": "system", "content": system_prompt})

            if len(prompt) > 0:
                user_content = _build_user_content(prompt, images, image_detail)
                messages.append({"role": "user", "content": user_content})
            return messages

        messages: List[Dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        user_content = _build_user_content(prompt, images, image_detail)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _has_cost_limit(self) -> bool:
        """Return whether this worker has an active (finite) dollar-denominated CostLimit.

        Returns ``False`` if the only CostLimit on the pool is the library
        default ``CostLimit(inf, "daily")`` — that's a "no real budget"
        signal even though the slot is technically populated.
        """
        try:
            for limit_set in self.limits.limit_sets:
                for limit in limit_set.limits:
                    if getattr(limit, "key", None) == DEFAULT_COST_LIMIT_KEY:
                        budget_usd = getattr(limit, "budget_usd", None)
                        if budget_usd is None or math.isinf(budget_usd):
                            continue
                        return True
        except (AttributeError, TypeError):
            return False
        return False

    def _should_track_cost(self) -> bool:
        """Return whether CostLimit accounting should include cost.

        Cost tracking is runtime accounting state, not part of Usage itself. A
        configured CostLimit only becomes enforceable when model pricing is
        available before the call; warn/ignore policies disable CostLimit
        enforcement while leaving token/call limits and reporter logging intact.
        """
        if self._has_cost_limit() is False:
            return False
        try:
            PricingCache.get_token_costs(self.model_name)
            return True
        except ModelNotFoundError as model_not_found_error:
            # Unknown pricing is deterministic configuration state, not a transient
            # LLM failure. Do not raise ValueError here: ValueError is retryable
            # because validators use it for stochastic malformed responses.
            if self.on_pricing_unavailable == "error":
                raise PricingUnavailableError(
                    f"Model {self.model_name!r} is not in the pricing database, "
                    "and on_pricing_unavailable='error'. "
                    "Set on_pricing_unavailable='warn' or provide model pricing to continue."
                ) from model_not_found_error
            elif self.on_pricing_unavailable == "warn":
                logger.warning(
                    f"[{self.name}] Model '{self.model_name}' not in pricing database. "
                    f"Cost estimation disabled; CostLimit will not enforce budget for this model."
                )
                return False
            elif self.on_pricing_unavailable == "ignore":
                return False
            else:
                raise InvalidConfigValueError(
                    f"Unknown on_pricing_unavailable={self.on_pricing_unavailable!r}. "
                    "Must be 'error', 'warn', or 'ignore'."
                ) from model_not_found_error

    def _estimate_usage(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[ToolChoiceOption],
        should_track_cost: bool,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Usage:
        """Estimate token and cost quantities for a pre-call reservation.

        Token estimation uses litellm's local tokenizer over the full messages
        list (history, tool schemas, tool results, images), then applies a
        safety multiplier and overhead so reserved capacity is conservative.

        Cost is only estimated when cost tracking is active. Token/call-only
        limit sets do not require pricing data and receive cost_microdollars=0.

        Args:
            messages: Messages list passed to ``litellm.token_counter``.
            tools, tool_choice: Tool schemas (also fed to the token counter).
            should_track_cost: Whether to compute the cost dimension.
            model: Per-call model used for tokenization and pricing. Falls
                back to ``self.model_name``. Pass a per-call override (already
                cascaded) so reservations match what will actually be billed.
            max_tokens: Per-call max output tokens. Falls back to
                ``self.max_tokens``. Drives the output-token portion of the
                reservation.
        """
        defaults = slowburn_config.defaults
        used_model: str = model if model is not None else self.model_name
        used_max_tokens: int = max_tokens if max_tokens is not None else self.max_tokens
        base_input_tokens: int = litellm.token_counter(
            model=used_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            use_default_image_token_count=True,
        )
        input_tokens: int = (
            int(base_input_tokens * defaults.input_token_estimate_multiplier)
            + defaults.input_token_estimate_overhead
        )
        output_tokens: int = (
            int(used_max_tokens * defaults.output_token_estimate_multiplier)
            + defaults.output_token_estimate_overhead
        )
        cost_microdollars: int = 0
        if should_track_cost:
            cost_microdollars = PricingCache.estimate_cost_microdollars(
                used_model,
                input_tokens,
                output_tokens,
            )
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_microdollars=cost_microdollars,
        )

    def _pre_acquire_rate_keys(self) -> Dict[str, List[str]]:
        """Union of per-slot limit keys across every LimitSet in the pool.

        Cached on first read. We need this BEFORE the LimitPool selects an
        endpoint at acquire time because the worker has to specify amounts for
        each ``RateLimit`` / ``CostLimit`` key the eventually-selected LimitSet
        might carry. Concurry skips unknown keys silently, so it's safe to
        overspecify.
        """
        cached = getattr(self, "_pre_acquire_rate_keys_cache", None)
        if cached is not None:
            return cached
        union: Dict[str, set] = {
            "requests": set(),
            "input_tokens": set(),
            "output_tokens": set(),
            "budget": set(),
        }
        # ``self.limits`` is the LimitPool; each LimitSet's config carries
        # ``_rate_keys`` from build_limit_pool.
        limit_sets = getattr(self.limits, "limit_sets", None)
        if limit_sets is not None:
            for ls in limit_sets:
                cfg = getattr(ls, "config", None) or {}
                rate_keys_for_ls = cfg.get("_rate_keys", {})
                for base, keys in rate_keys_for_ls.items():
                    union.setdefault(base, set()).update(keys)
        # Fall back: when the worker is built directly with a hand-rolled
        # LimitSet (no ``_rate_keys`` config), assume the canonical base
        # keys are present so manually-built fixtures keep working.
        _fallbacks: Dict[str, str] = {
            "requests": "requests",
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "budget": DEFAULT_COST_LIMIT_KEY,
        }
        for base, fallback_key in _fallbacks.items():
            if not union[base]:
                union[base].add(fallback_key)
        result = {base: sorted(keys) for base, keys in union.items()}
        # Stash on the instance via object.__setattr__ since SlowBurnLLM is
        # a Typed model and direct attribute assignment is restricted.
        object.__setattr__(self, "_pre_acquire_rate_keys_cache", result)
        return result

    def _build_limit_usage(
        self,
        *,
        usage: Usage,
        should_track_cost: bool,
        rate_keys: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, int]:
        """Build the usage dict for acquire / acquisition.update().

        The dict is keyed by Concurry limit-set keys. For the four rate-style
        slots (``requests``, ``input_tokens``, ``output_tokens``, ``budget``),
        the worker emits the same numeric value under every actual key the
        relevant LimitSet exposes for that slot — so a slot with two windows
        (e.g. per-minute and per-day) charges both.

        Args:
            usage: Tokens and cost to record.
            should_track_cost: Whether to include the cost-limit key.
            rate_keys: Per-slot key mapping to use. If ``None``, the
                worker's pool-level union (across every LimitSet) is used —
                appropriate for pre-acquire, when we don't yet know which
                LimitSet the pool will pick. When the post-acquisition
                ``acquisition.config["_rate_keys"]`` is available, pass it
                here so ``update()`` charges only the keys actually present
                on the selected LimitSet.
        """
        keys = rate_keys if rate_keys is not None else self._pre_acquire_rate_keys()
        limit_usage: Dict[str, int] = {}
        for k in keys.get("input_tokens", ["input_tokens"]):
            limit_usage[k] = usage.input_tokens
        for k in keys.get("output_tokens", ["output_tokens"]):
            limit_usage[k] = usage.output_tokens
        for k in keys.get("requests", ["requests"]):
            limit_usage[k] = 1
        if should_track_cost:
            for k in keys.get("budget", [DEFAULT_COST_LIMIT_KEY]):
                limit_usage[k] = usage.cost_microdollars
        return limit_usage

    def _extract_actual_cost(self, response: Any, model: Optional[str] = None) -> int:
        """Extract actual cost from a litellm response, falling back to 0.

        Pricing failures here are deliberately swallowed: the response was
        already produced and accounted in tokens; an unknown price should
        not propagate as a retryable error after a successful call.

        Args:
            response: The litellm response object.
            model: The per-call model used to make this request (after the
                cascade has resolved per-call > endpoint > worker default).
                Falls back to ``self.model_name`` when ``None``. Threading
                this through is what makes per-endpoint pricing accurate
                when different endpoints serve different models.
        """
        try:
            return PricingCache.actual_cost_microdollars(
                response, model=model if model is not None else self.model_name
            )
        except (ModelNotFoundError, ValueError, TypeError, KeyError, AttributeError):
            return 0

    def _account_call(
        self,
        *,
        acquisition: Any,
        usage: Usage,
        should_track_cost: bool,
        model: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        """Update both the Concurry acquisition and the CostReporter.

        This is the single point of truth for cost accounting. Every path
        that consumed (or potentially consumed) tokens — success or failure —
        must call this exactly once before re-raising. Otherwise the
        acquisition leaks reserved capacity, or the CostReporter under-reports
        the budget consumed by a failed attempt.

        Args:
            acquisition: The Concurry acquisition handle.
            usage: Tokens and cost to record.
            should_track_cost: Whether to include the cost dimension in the
                limit-set update.
            model: Per-call resolved model (after cascade). Falls back to
                ``self.model_name`` for the reporter row when None.
            endpoint_id: Per-call endpoint label for reporter attribution
                (e.g., "111111111111/us-east-1"). None for single-endpoint
                deployments.
        """
        acquisition_config = getattr(acquisition, "config", None) or {}
        acquisition_rate_keys = acquisition_config.get("_rate_keys")
        acquisition.update(
            usage=self._build_limit_usage(
                usage=usage,
                should_track_cost=should_track_cost,
                rate_keys=acquisition_rate_keys,
            )
        )
        self._reporter.log_call(
            model=model if model is not None else self.model_name,
            cost_usd=microdollars_to_dollars(usage.cost_microdollars),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            endpoint_id=endpoint_id,
        )

    async def _call_with_timeout(
        self,
        *,
        acquisition: Any,
        estimated_usage: Usage,
        should_track_cost: bool,
        merged_params: Dict[str, Any],
        messages: List[Dict[str, Any]],
        prompt_hash: str,
        call_t0: float,
        verbosity: int,
        model: str,
        api_key: Optional[str],
        api_base: Optional[str],
        temperature: Optional[float],
        max_tokens: int,
        timeout: float,
        endpoint_id: Optional[str],
    ) -> Any:
        """Execute the litellm call with timeout and account on timeout failure.

        On timeout, no response object exists, so usage accounting falls back
        to the conservative estimate reserved before the request was sent,
        with output tokens set to zero because no completion usage exists.
        Both the acquisition and the reporter are updated before re-raising.
        """
        api_t0: float = time.monotonic()
        # Build the explicit-named-kwarg portion of the litellm call. Any
        # value resolved via the cascade is passed here; arbitrary
        # passthroughs (aws_access_key_id, extra_body, response_format,
        # etc.) live in merged_params and are spread via **.
        named_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if api_base is not None:
            named_kwargs["api_base"] = api_base

        if verbosity >= 3:
            # Log the FULL kwargs that go to litellm.acompletion, with
            # credentials redacted. This is the source of truth for what
            # the LLM provider actually receives.
            _SECRET_KEYS = {
                "api_key",
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_web_identity_token",
            }
            redacted: Dict[str, Any] = {}
            for k, v in {**named_kwargs, **merged_params}.items():
                if k == "messages":
                    # Messages can be huge; show only the role+length per turn.
                    redacted[k] = [
                        {"role": m.get("role"), "content_len": len(str(m.get("content", "")))} for m in v
                    ]
                elif k in _SECRET_KEYS:
                    if v is None or v == "":
                        redacted[k] = "<empty>"
                    else:
                        # Show prefix only; never log the secret itself.
                        s = str(v)
                        redacted[k] = f"<set len={len(s)} prefix={s[:6]}...>"
                else:
                    redacted[k] = v
            logger.info(
                f"[{model}] [Prompt={prompt_hash}] LITELLM_CALL | endpoint={endpoint_id} | kwargs={redacted}"
            )

        try:
            litellm.drop_params = True
            response = await asyncio.wait_for(
                litellm.acompletion(**named_kwargs, **merged_params),
                timeout=timeout,
            )
        except asyncio.TimeoutError as timeout_error:
            self._account_call(
                acquisition=acquisition,
                usage=estimated_usage.with_output_tokens(output_tokens=0),
                should_track_cost=should_track_cost,
                model=model,
                endpoint_id=endpoint_id,
            )
            if verbosity >= 2:
                logger.warning(
                    f"[{model}] [Prompt={prompt_hash}] TIMEOUT | "
                    f"after {time.monotonic() - call_t0:.2f}s "
                    f"(timeout={timeout}s)"
                )
            raise timeout_error

        if verbosity >= 3:
            actual_input: int = response.usage.prompt_tokens
            actual_output: int = response.usage.completion_tokens
            logger.info(
                f"[{model}] [Prompt={prompt_hash}] RESPONSE | "
                f"api={time.monotonic() - api_t0:.2f}s total={time.monotonic() - call_t0:.2f}s | "
                f"in={actual_input} out={actual_output}"
            )
        return response

    @validate
    async def call_llm(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        images: Optional[List[ImageInput]] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        tools: Union[Optional[List[Dict[str, Any]]], _NO_ARG_TYPE] = _NO_ARG,
        tool_choice: Union[Optional[ToolChoiceOption], _NO_ARG_TYPE] = _NO_ARG,
        return_messages: Optional[bool] = None,
        validator: Optional[Callable[[str], T]] = None,
        image_detail: Union[ImageDetailLevel, _NO_ARG_TYPE] = _NO_ARG,
        verbosity: Union[int, _NO_ARG_TYPE] = _NO_ARG,
        litellm_params: Optional[Dict[str, Any]] = None,
        model: Union[str, _NO_ARG_TYPE] = _NO_ARG,
        api_key: Union[Optional[str], _NO_ARG_TYPE] = _NO_ARG,
        api_base: Union[Optional[str], _NO_ARG_TYPE] = _NO_ARG,
        temperature: Union[Optional[float], _NO_ARG_TYPE] = _NO_ARG,
        max_tokens: Union[int, _NO_ARG_TYPE] = _NO_ARG,
        timeout: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    ) -> Union[str, T, List[Dict[str, Any]]]:
        """Execute a single LLM call with cost-aware backpressure.

        Args:
            prompt: The user message content (string), or a pre-built
                messages list (list of dicts). When a list is passed,
                *system_prompt* and *images* are ignored.
            images: Optional list of images to include in the message.
            system_prompt: Optional system message prepended to the conversation.
            history: Previous messages list for multi-turn conversations.
                When provided, the new user message (from *prompt*) is
                appended to a copy of this list. Pass ``None`` (default)
                for single-turn calls.
            tools: Tool schemas (OpenAI format) for this call. Overrides
                the worker-level ``self.tools``. Pass ``None`` to disable
                tools for this call even if the worker has defaults.
            tool_choice: Tool choice for this call ("auto", "required",
                "none", or a specific tool dict). Overrides
                ``self.tool_choice``.
            return_messages: Controls the return type. ``None`` (default)
                auto-detects: returns a messages list if *history* is
                provided or *prompt* is a list, otherwise returns a string.
                ``True`` forces messages-list output. ``False`` forces
                string output.
            validator: Optional callable that parses/validates the response text.
                When *return_messages* resolves to True, validation is still
                executed for retry/failure semantics, but the returned value
                remains the complete messages list.
            image_detail: Detail level for vision queries ("low"/"high"/"auto").
                Defaults to slowburn_config.defaults.image_detail.
            verbosity: Logging verbosity (0=silent, 1=minimal,
                2=warnings+progress, 3=full debug with per-event logging).
            litellm_params: Per-call parameters passed through to
                ``litellm.acompletion()``. Merged on top of the worker-level
                ``self.litellm_params`` and the selected endpoint's
                ``litellm_params``.
            model: Per-call model override. Cascade: per-call > endpoint
                config > worker default. Useful when one call needs a
                different model than the endpoint's usual one.
            api_key: Per-call API key override.
            api_base: Per-call API base URL override (litellm ``api_base``).
            temperature: Per-call sampling temperature.
            max_tokens: Per-call max output tokens. Affects pre-acquire
                reservation: a higher value reserves more output capacity.
            timeout: Per-call timeout in seconds.

        Returns:
            When *return_messages* resolves to False: the raw response text,
            or the parsed result from *validator* if provided.
            When *return_messages* resolves to True: the complete messages
            list with the assistant response appended.
        """
        if is_no_arg(verbosity):
            verbosity = slowburn_config.defaults.verbosity
        if is_no_arg(image_detail):
            image_detail = slowburn_config.defaults.image_detail

        # Returning messages is the structured-conversation mode. It is required for
        # multi-turn continuations and pre-built message lists because callers need the
        # assistant message appended to the conversation state.
        if return_messages is None:
            return_messages = history is not None or isinstance(prompt, list)

        # Resolve tools/tool_choice from caller or worker defaults
        if is_no_arg(tools):
            tools = self.tools
        if is_no_arg(tool_choice):
            tool_choice = self.tool_choice

        # Tool calls are structured assistant-message data. They cannot be faithfully
        # represented as a response string, so fail before making an API call rather
        # than serializing tool calls into fake text or retrying a deterministic caller error.
        if tools is not None and len(tools) > 0 and return_messages is False:
            raise ToolCallContractError(
                "call_llm() was called with tools but return_messages=False. "
                "Tool-call responses are structured assistant messages, not response text. "
                "Pass return_messages=True when using tools."
            )

        messages = self.build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            history=history,
            image_detail=image_detail,
        )

        # Pre-acquire reservation runs BEFORE the LimitPool selects an
        # endpoint, so we cannot yet know which model the resolver/cascade
        # will pick. We use the worker default model (self.model_name) for
        # tokenization and pricing here. The per-call max_tokens override IS
        # known, so we honor it for the output-token portion of the
        # reservation. After acquisition, the actual cost is re-extracted
        # using the resolved per-call model and any over-reservation flows
        # back to the bucket via Concurry's context-exit refund.
        pre_max_tokens: int = max_tokens if not is_no_arg(max_tokens) else self.max_tokens
        should_track_cost: bool = self._should_track_cost()
        estimated_usage: Usage = self._estimate_usage(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            should_track_cost=should_track_cost,
            max_tokens=pre_max_tokens,
        )
        estimated_limit_usage: Dict[str, int] = self._build_limit_usage(
            usage=estimated_usage,
            should_track_cost=should_track_cost,
        )

        call_t0: float = time.monotonic()
        prompt_hash: str = _hash_prompt(prompt)
        if verbosity >= 3:
            logger.info(
                f"[{self.name}] [Prompt={prompt_hash}] CALL_START | "
                f"model={self.model_name} "
                f"est_in={estimated_usage.input_tokens} est_out={estimated_usage.output_tokens} "
                f"timeout={self.timeout}s tools={tools is not None and len(tools) > 0}"
            )

        try:
            acquire_start: float = time.monotonic()
            context_manager: Any = await self.limits.async_acquire(requested=estimated_limit_usage)
            acquire_elapsed: float = time.monotonic() - acquire_start
        except ValueError as acquire_error:
            # Concurry uses ValueError for both generic acquire failures and the
            # specific "single request exceeds limit capacity" case. Only the
            # latter is handled by SlowBurn's budget-overflow policy.
            if "exceeds capacity" not in str(acquire_error):
                raise acquire_error

            overflow_message: str = (
                f"A single call_llm() call to {self.model_name} is estimated to cost "
                f"${microdollars_to_dollars(estimated_usage.cost_microdollars):.6f} "
                f"(~{estimated_usage.input_tokens} input + {estimated_usage.output_tokens} output tokens), "
                f"which exceeds your budget_usd per window. "
                f"Fix by: (1) increasing the budget while creating the LLM, "
                f"(2) reducing max_tokens (currently {pre_max_tokens}), "
                f"or (3) using a more budget-friendly model."
            )

            # Budget overflow is a caller/configuration problem. Retrying the same
            # request cannot make it fit inside the configured capacity.
            if self.on_budget_overflow == "error":
                raise BudgetOverflowError(overflow_message) from acquire_error
            elif self.on_budget_overflow == "warn":
                logger.warning(f"[{self.name}] Budget overflow: {overflow_message}")
            elif self.on_budget_overflow == "ignore":
                pass
            else:
                raise InvalidConfigValueError(
                    f"Unknown on_budget_overflow={self.on_budget_overflow!r}. "
                    "Must be 'error', 'warn', or 'ignore'."
                ) from acquire_error

            # In warn/ignore mode, acquire the non-cost limits normally but cap the
            # cost request to the smallest positive amount so a single expensive call
            # does not permanently block on a capacity it can never fit into.
            capped_limit_usage: Dict[str, int] = dict(estimated_limit_usage)
            if should_track_cost:
                capped_limit_usage[DEFAULT_COST_LIMIT_KEY] = 1
            acquire_start = time.monotonic()
            context_manager = await self.limits.async_acquire(requested=capped_limit_usage)
            acquire_elapsed = time.monotonic() - acquire_start

        if self.backpressure_notify == "warn":
            threshold: float = slowburn_config.defaults.backpressure_threshold_seconds
            if acquire_elapsed > threshold:
                logger.warning(
                    f"[{self.name}] Backpressure: blocked {acquire_elapsed:.1f}s "
                    f"waiting for budget/rate capacity "
                    f"(estimated ${microdollars_to_dollars(estimated_usage.cost_microdollars):.6f}, "
                    f"~{estimated_usage.input_tokens} input + {estimated_usage.output_tokens} output tokens)"
                )

        # From this point on, every failure path must update both the acquisition and
        # reporter exactly once with either actual usage (after a response exists) or
        # estimated usage (before a response exists).
        async with context_manager as acquisition:
            # ---------------------------------------------------------------
            # Step 1: rebuild the typed EndpointConfig from the acquisition.
            # ---------------------------------------------------------------
            # Concurry stores whatever dict was passed to LimitSet(config=...)
            # on the acquisition. When SlowBurn built the pool via create_llm,
            # this is a fully-populated EndpointConfig.model_dump() (with
            # ``_rate_keys`` extra). When the user constructed a LimitSet
            # manually with no config, it is an empty dict. EndpointConfig is
            # strict (all known fields required) so we backfill any missing
            # fields from the worker's own attributes before validating.
            raw_config: Dict[str, Any] = dict(getattr(acquisition, "config", None) or {})
            # ``_rate_keys`` is metadata for the worker, not an EndpointConfig
            # field — strip before validation.
            raw_config.pop("_rate_keys", None)
            cfg = slowburn_config.defaults
            _resolved_max_tokens_default: int = (
                self.max_tokens if not is_no_arg(self.max_tokens) else cfg.max_tokens
            )
            _resolved_timeout_default: float = self.timeout if not is_no_arg(self.timeout) else cfg.timeout
            _resolved_temperature_default: Optional[float] = (
                self.temperature if not is_no_arg(self.temperature) else cfg.temperature
            )
            # Backfill any missing EndpointConfig fields from the worker's own
            # attributes / library defaults. ``limits`` is left untouched —
            # ``None`` is a valid value (means "inherit"), and the LimitSet
            # the worker is using has already been built and contains the
            # actual Limit objects. The ``EndpointConfig`` is rebuilt here
            # purely so the per-call cascade and the resolver can read its
            # fields; it is not used to re-derive limits.
            _worker_endpoint_defaults: Dict[str, Any] = {
                "model": self.model_name,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "temperature": _resolved_temperature_default,
                "max_tokens": _resolved_max_tokens_default,
                "timeout": _resolved_timeout_default,
            }
            for _field, _default in _worker_endpoint_defaults.items():
                raw_config.setdefault(_field, _default)
            endpoint_config: EndpointConfig = EndpointConfig(**raw_config)

            # ---------------------------------------------------------------
            # Step 2: run the user's resolver (default = passthrough). The
            # resolver receives a serialized dict (so the user can write
            # dict-style code without depending on EndpointConfig's typing).
            # The dict it returns is re-validated into a new EndpointConfig.
            # ---------------------------------------------------------------
            resolver = self.endpoint_resolver if self.endpoint_resolver is not None else passthrough_resolver
            try:
                augmented_dict: Dict[str, Any] = resolver(endpoint_config.model_dump())
            except Exception as resolver_error:
                # Resolver failures are deterministic configuration errors —
                # retrying does not change the resolver. Account the
                # reservation back to the bucket before re-raising so we do
                # not leak capacity.
                self._account_call(
                    acquisition=acquisition,
                    usage=estimated_usage.with_output_tokens(output_tokens=0),
                    should_track_cost=should_track_cost,
                    model=self.model_name,
                    endpoint_id=endpoint_config.endpoint_id,
                )
                raise resolver_error
            endpoint_config = EndpointConfig(**augmented_dict)

            # ---------------------------------------------------------------
            # Step 3: cascade per-call > endpoint > worker default for every
            # field that flows into litellm.acompletion as a named kwarg.
            # ---------------------------------------------------------------
            resolved_model: str = cascade_field(
                field="model",
                call_value=model,
                config_value=endpoint_config.model,
                worker_default=self.model_name,
            )
            resolved_api_key: Optional[str] = cascade_field(
                field="api_key",
                call_value=api_key,
                config_value=endpoint_config.api_key,
                worker_default=self.api_key,
            )
            resolved_api_base: Optional[str] = cascade_field(
                field="api_base",
                call_value=api_base,
                config_value=endpoint_config.api_base,
                worker_default=self.api_base,
            )
            resolved_temperature: Optional[float] = cascade_field(
                field="temperature",
                call_value=temperature,
                config_value=endpoint_config.temperature,
                worker_default=self.temperature,
            )
            resolved_max_tokens: int = cascade_field(
                field="max_tokens",
                call_value=max_tokens,
                config_value=endpoint_config.max_tokens,
                worker_default=self.max_tokens,
            )
            resolved_timeout: float = cascade_field(
                field="timeout",
                call_value=timeout,
                config_value=endpoint_config.timeout,
                worker_default=self.timeout,
            )

            # ---------------------------------------------------------------
            # Step 4: build the merged litellm_params dict. Order (lowest to
            # highest priority): worker defaults, endpoint, per-call.
            # tools/tool_choice are appended last and bypass the cascade.
            # ---------------------------------------------------------------
            merged_params: Dict[str, Any] = {}
            merged_params.update(self.litellm_params)
            merged_params.update(endpoint_config.litellm_params)
            if litellm_params is not None:
                merged_params.update(litellm_params)
            if tools is not None:
                merged_params["tools"] = tools
            if tool_choice is not None:
                merged_params["tool_choice"] = tool_choice

            endpoint_id: Optional[str] = endpoint_config.endpoint_id

            if verbosity >= 3:
                logger.info(
                    f"[{resolved_model}] [Prompt={prompt_hash}] ACQUIRED | "
                    f"wait={time.monotonic() - call_t0:.2f}s | "
                    f"endpoint={endpoint_id} | "
                    f"sending request..."
                )

            # ---------------------------------------------------------------
            # Step 5: execute the litellm call.
            # ---------------------------------------------------------------
            try:
                response: Any = await self._call_with_timeout(
                    acquisition=acquisition,
                    estimated_usage=estimated_usage,
                    should_track_cost=should_track_cost,
                    merged_params=merged_params,
                    messages=messages,
                    prompt_hash=prompt_hash,
                    call_t0=call_t0,
                    verbosity=verbosity,
                    model=resolved_model,
                    api_key=resolved_api_key,
                    api_base=resolved_api_base,
                    temperature=resolved_temperature,
                    max_tokens=resolved_max_tokens,
                    timeout=resolved_timeout,
                    endpoint_id=endpoint_id,
                )
            except asyncio.TimeoutError:
                raise
            except BaseException as base_exception:
                self._account_call(
                    acquisition=acquisition,
                    usage=estimated_usage.with_output_tokens(output_tokens=0),
                    should_track_cost=should_track_cost,
                    model=resolved_model,
                    endpoint_id=endpoint_id,
                )
                if verbosity >= 3:
                    logger.warning(
                        f"[{resolved_model}] [Prompt={prompt_hash}] ERROR | "
                        f"{type(base_exception).__name__} at "
                        f"{time.monotonic() - call_t0:.2f}s (will retry if configured): "
                        f"{format_exception_msg(base_exception)}"
                    )
                raise base_exception

            # ---------------------------------------------------------------
            # Step 6: extract actuals using the per-call resolved model so
            # cost extraction reflects what was actually billed.
            # ---------------------------------------------------------------
            actual_usage: Usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost_microdollars=self._extract_actual_cost(response, model=resolved_model),
            )

            try:
                # LiteLLM normalizes provider responses into an assistant message.
                # SlowBurn only treats textual content as response text; tool_calls
                # remain structured message fields and are returned only in messages mode.
                response_message: Any = response.choices[0].message
                response_text: Optional[str] = response_message.content
                tool_calls: Optional[List[Any]] = response_message.tool_calls

                # Null content with no tool calls is usually a refusal/content-filter
                # or provider anomaly. Keep it retryable because a repeated stochastic
                # call may produce a usable response.
                if response_text is None and tool_calls is None:
                    raise ValueError(
                        f"LLM returned null content with no tool calls "
                        f"(model={resolved_model}). "
                        f"This may indicate a refusal or content filter."
                    )

                # Provider anomaly guard: the LLM returned tool_calls even though
                # the caller did not pass tool schemas (or did so without requesting
                # message-mode output). This is a non-retryable protocol violation.
                if response_text is None and tool_calls is not None and return_messages is False:
                    raise ToolCallContractError(
                        "LLM returned tool_calls with no text content, but return_messages=False. "
                        "Tool-call responses must be consumed as assistant messages. "
                        "Pass return_messages=True when using tools."
                    )

                # Validators parse response text. A tool-only response is a valid
                # assistant message, but it is invalid for a text validator; leave this
                # as retryable ValueError because another model attempt may choose text.
                if response_text is None and tool_calls is not None and validator is not None:
                    raise ValueError(
                        "LLM returned tool_calls with no text content, but a text validator was provided. "
                        "This is retryable because another stochastic LLM call may return text content."
                    )

                result: Union[str, T, List[Dict[str, Any]]]
                if validator is not None:
                    try:
                        result = validator(response_text)
                    except ValueError as validation_error:
                        raise validation_error
                    except Exception as validation_error:
                        raise ValueError(f"Validator error: {validation_error}") from validation_error
                elif response_text is not None:
                    result = response_text
                else:
                    result = messages

                # In messages mode, return LiteLLM's full normalized assistant message
                # so provider extensions such as reasoning_content and tool_calls are preserved.
                if return_messages:
                    assistant_message: Dict[str, Any] = response_message.model_dump(exclude_none=True)
                    if "content" not in assistant_message:
                        assistant_message["content"] = None
                    messages.append(assistant_message)
                    result = messages

                self._account_call(
                    acquisition=acquisition,
                    usage=actual_usage,
                    should_track_cost=should_track_cost,
                    model=resolved_model,
                    endpoint_id=endpoint_id,
                )

                if verbosity >= 2:
                    endpoint_suffix: str = f" via {endpoint_id}" if endpoint_id is not None else ""
                    logger.info(
                        f"[{resolved_model}]{endpoint_suffix}: "
                        f"{actual_usage.input_tokens}+{actual_usage.output_tokens} tokens, "
                        f"${microdollars_to_dollars(actual_usage.cost_microdollars):.6f}"
                    )

                return result
            except (ValueError, SlowBurnNonRetryableError) as post_response_error:
                self._account_call(
                    acquisition=acquisition,
                    usage=actual_usage,
                    should_track_cost=should_track_cost,
                    model=resolved_model,
                    endpoint_id=endpoint_id,
                )
                if verbosity >= 2:
                    logger.warning(
                        f"[{resolved_model}] [Prompt={prompt_hash}] POST_RESPONSE_ERROR | "
                        f"{type(post_response_error).__name__} at {time.monotonic() - call_t0:.2f}s: "
                        f"{format_exception_msg(post_response_error)}"
                    )
                raise post_response_error
            except BaseException as base_exception:
                self._account_call(
                    acquisition=acquisition,
                    usage=estimated_usage.with_output_tokens(output_tokens=0),
                    should_track_cost=should_track_cost,
                    model=resolved_model,
                    endpoint_id=endpoint_id,
                )
                if verbosity >= 2:
                    logger.warning(
                        f"[{resolved_model}] [Prompt={prompt_hash}] POST_RESPONSE_ERROR | "
                        f"{type(base_exception).__name__} at "
                        f"{time.monotonic() - call_t0:.2f}s: "
                        f"{format_exception_msg(base_exception)}"
                    )
                raise base_exception

    @validate
    async def call_llm_batch(
        self,
        prompts: List[Union[str, List[Dict[str, Any]]]],
        *,
        images_per_prompt: Optional[List[Optional[List[ImageInput]]]] = None,
        system_prompt: Optional[str] = None,
        history_per_prompt: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
        tools: Union[Optional[List[Dict[str, Any]]], _NO_ARG_TYPE] = _NO_ARG,
        tool_choice: Union[Optional[ToolChoiceOption], _NO_ARG_TYPE] = _NO_ARG,
        return_messages: Optional[bool] = None,
        validator: Optional[Callable[[str], T]] = None,
        image_detail: Union[ImageDetailLevel, _NO_ARG_TYPE] = _NO_ARG,
        verbosity: Union[int, _NO_ARG_TYPE] = _NO_ARG,
        litellm_params: Optional[Dict[str, Any]] = None,
        model: Union[str, _NO_ARG_TYPE] = _NO_ARG,
        api_key: Union[Optional[str], _NO_ARG_TYPE] = _NO_ARG,
        api_base: Union[Optional[str], _NO_ARG_TYPE] = _NO_ARG,
        temperature: Union[Optional[float], _NO_ARG_TYPE] = _NO_ARG,
        max_tokens: Union[int, _NO_ARG_TYPE] = _NO_ARG,
        timeout: Union[float, _NO_ARG_TYPE] = _NO_ARG,
    ) -> List[Union[str, T, List[Dict[str, Any]]]]:
        """Execute multiple LLM calls concurrently with shared backpressure.

        Args:
            prompts: List of user message strings or pre-built message lists.
            images_per_prompt: Optional list, same length as *prompts*, where
                each element is either ``None`` (text-only) or a list of images.
            system_prompt: Optional system message applied to all calls.
            history_per_prompt: Optional list, same length as *prompts*, where
                each element is either ``None`` (no history) or a messages list.
            tools: Tool schemas shared across all items (overrides worker default).
            tool_choice: Tool choice shared across all items.
            return_messages: Return type override shared across all items.
            validator: Optional callable applied to each response.
            image_detail: Detail level for vision queries ("low"/"high"/"auto").
            verbosity: Logging verbosity.
            litellm_params: Per-call parameters passed through to each call.

        Returns:
            List of results (strings, parsed validator output, or message lists).
        """
        if is_no_arg(verbosity):
            verbosity = slowburn_config.defaults.verbosity
        if len(prompts) == 0:
            return []

        if images_per_prompt is not None and len(images_per_prompt) != len(prompts):
            raise BatchInputMismatchError(
                f"images_per_prompt length ({len(images_per_prompt)}) "
                f"must match prompts length ({len(prompts)})"
            )
        if history_per_prompt is not None and len(history_per_prompt) != len(prompts):
            raise BatchInputMismatchError(
                f"history_per_prompt length ({len(history_per_prompt)}) "
                f"must match prompts length ({len(prompts)})"
            )

        if images_per_prompt is None:
            images_per_prompt = [None] * len(prompts)
        if history_per_prompt is None:
            history_per_prompt = [None] * len(prompts)

        tasks = [
            self.call_llm(
                prompt=prompt_item,
                images=images_item,
                system_prompt=system_prompt,
                history=history_item,
                tools=tools,
                tool_choice=tool_choice,
                return_messages=return_messages,
                validator=validator,
                image_detail=image_detail,
                verbosity=verbosity,
                litellm_params=litellm_params,
                model=model,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            for prompt_item, images_item, history_item in zip(
                prompts,
                images_per_prompt,
                history_per_prompt,
            )
        ]

        results: List[Union[str, T, List[Dict[str, Any]]]] = await async_gather(
            tasks,
            progress=dict(
                disable=verbosity < 2,
                desc=f"{self.name}",
                miniters=max(len(prompts) // 2, 1),
            ),
        )
        return results

    def get_reporter(self) -> CostReporter:
        """Return the CostReporter for external access.

        Since the worker runs in an asyncio event loop, call this method
        via the future pattern: ``reporter = llm.get_reporter().result()``.
        """
        return self._reporter
