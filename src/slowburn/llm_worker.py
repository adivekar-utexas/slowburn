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
                    CostLimit(budget_usd=5.0, window_seconds=86400),
                    RateLimit(key="input_tokens", window_seconds=60, capacity=1_000_000),
                    RateLimit(key="output_tokens", window_seconds=60, capacity=200_000),
                    CallLimit(window_seconds=60, capacity=500),
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
    api_key: str = Field(default="", description="API key (or set via env var)")
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
        """Return whether this worker has an active dollar-denominated CostLimit."""
        try:
            for limit_set in self.limits.limit_sets:
                for limit in limit_set.limits:
                    if getattr(limit, "key", None) == DEFAULT_COST_LIMIT_KEY:
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

    def _build_litellm_params(
        self,
        *,
        worker_params: Dict[str, Any],
        call_params: Optional[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[ToolChoiceOption],
    ) -> Dict[str, Any]:
        """Merge worker-level and call-level LiteLLM parameters.

        Call-level values intentionally override worker defaults so that
        per-call reasoning controls, response_format, etc. do not mutate
        the worker's persistent configuration.
        """
        merged: Dict[str, Any] = {**worker_params}
        if call_params is not None:
            merged.update(call_params)
        if tools is not None:
            merged["tools"] = tools
        if tool_choice is not None:
            merged["tool_choice"] = tool_choice
        return merged

    def _estimate_usage(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[ToolChoiceOption],
        should_track_cost: bool,
    ) -> Usage:
        """Estimate token and cost quantities for a pre-call reservation.

        Token estimation uses litellm's local tokenizer over the full messages
        list (history, tool schemas, tool results, images), then applies a
        safety multiplier and overhead so reserved capacity is conservative.

        Cost is only estimated when cost tracking is active. Token/call-only
        limit sets do not require pricing data and receive cost_microdollars=0.
        """
        defaults = slowburn_config.defaults
        base_input_tokens: int = litellm.token_counter(
            model=self.model_name,
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
            int(self.max_tokens * defaults.output_token_estimate_multiplier)
            + defaults.output_token_estimate_overhead
        )
        cost_microdollars: int = 0
        if should_track_cost:
            cost_microdollars = PricingCache.estimate_cost_microdollars(
                self.model_name,
                input_tokens,
                output_tokens,
            )
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_microdollars=cost_microdollars,
        )

    def _build_limit_usage(
        self,
        *,
        usage: Usage,
        should_track_cost: bool,
    ) -> Dict[str, int]:
        """Build the usage dict for acquisition.update().

        The Concurry limit layer needs a dict keyed by limit names. Usage owns
        token and cost quantities; should_track_cost is separate runtime
        accounting state that controls whether the CostLimit key is included.
        """
        limit_usage: Dict[str, int] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "call_count": 1,
        }
        if should_track_cost:
            limit_usage[DEFAULT_COST_LIMIT_KEY] = usage.cost_microdollars
        return limit_usage

    def _extract_actual_cost(self, response: Any) -> int:
        """Extract actual cost from a litellm response, falling back to 0.

        Pricing failures here are deliberately swallowed: the response was
        already produced and accounted in tokens; an unknown price should
        not propagate as a retryable error after a successful call.
        """
        try:
            return PricingCache.actual_cost_microdollars(response, model=self.model_name)
        except (ModelNotFoundError, ValueError, TypeError, KeyError, AttributeError):
            return 0

    def _account_call(
        self,
        *,
        acquisition: Any,
        usage: Usage,
        should_track_cost: bool,
    ) -> None:
        """Update both the Concurry acquisition and the CostReporter.

        This is the single point of truth for cost accounting. Every path
        that consumed (or potentially consumed) tokens — success or failure —
        must call this exactly once before re-raising. Otherwise the
        acquisition leaks reserved capacity, or the CostReporter under-reports
        the budget consumed by a failed attempt.
        """
        acquisition.update(
            usage=self._build_limit_usage(
                usage=usage,
                should_track_cost=should_track_cost,
            )
        )
        self._reporter.log_call(
            model=self.model_name,
            cost_usd=microdollars_to_dollars(usage.cost_microdollars),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
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
    ) -> Any:
        """Execute the litellm call with timeout and account on timeout failure.

        On timeout, no response object exists, so usage accounting falls back
        to the conservative estimate reserved before the request was sent,
        with output tokens set to zero because no completion usage exists.
        Both the acquisition and the reporter are updated before re-raising.
        """
        api_t0: float = time.monotonic()
        try:
            litellm.drop_params = True
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=self.model_name,
                    messages=messages,
                    api_key=self.api_key if len(self.api_key) > 0 else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **merged_params,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as timeout_error:
            self._account_call(
                acquisition=acquisition,
                usage=estimated_usage.with_output_tokens(output_tokens=0),
                should_track_cost=should_track_cost,
            )
            if verbosity >= 2:
                logger.warning(
                    f"[{self.name}] [Prompt={prompt_hash}] TIMEOUT | "
                    f"after {time.monotonic() - call_t0:.2f}s "
                    f"(timeout={self.timeout}s)"
                )
            raise timeout_error

        if verbosity >= 3:
            actual_input: int = response.usage.prompt_tokens
            actual_output: int = response.usage.completion_tokens
            logger.info(
                f"[{self.name}] [Prompt={prompt_hash}] RESPONSE | "
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
                ``self.litellm_params``.

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

        merged_params: Dict[str, Any] = self._build_litellm_params(
            worker_params=self.litellm_params,
            call_params=litellm_params,
            tools=tools,
            tool_choice=tool_choice,
        )

        should_track_cost: bool = self._should_track_cost()
        estimated_usage: Usage = self._estimate_usage(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            should_track_cost=should_track_cost,
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
                f"(2) reducing max_tokens (currently {self.max_tokens}), "
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
            if verbosity >= 3:
                logger.info(
                    f"[{self.name}] [Prompt={prompt_hash}] ACQUIRED | "
                    f"wait={time.monotonic() - call_t0:.2f}s | sending request..."
                )

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
                )
            except asyncio.TimeoutError:
                raise
            except BaseException as base_exception:
                self._account_call(
                    acquisition=acquisition,
                    usage=estimated_usage.with_output_tokens(output_tokens=0),
                    should_track_cost=should_track_cost,
                )
                if verbosity >= 3:
                    logger.warning(
                        f"[{self.name}] [Prompt={prompt_hash}] ERROR | "
                        f"{type(base_exception).__name__} at "
                        f"{time.monotonic() - call_t0:.2f}s (will retry if configured): "
                        f"{format_exception_msg(base_exception)}"
                    )
                raise base_exception

            actual_usage: Usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost_microdollars=self._extract_actual_cost(response),
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
                        f"(model={self.model_name}). "
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
                )

                if verbosity >= 2:
                    logger.info(
                        f"[{self.name}] {self.model_name}: "
                        f"{actual_usage.input_tokens}+{actual_usage.output_tokens} tokens, "
                        f"${microdollars_to_dollars(actual_usage.cost_microdollars):.6f}"
                    )

                return result
            except (ValueError, SlowBurnNonRetryableError) as post_response_error:
                self._account_call(
                    acquisition=acquisition,
                    usage=actual_usage,
                    should_track_cost=should_track_cost,
                )
                if verbosity >= 2:
                    logger.warning(
                        f"[{self.name}] [Prompt={prompt_hash}] POST_RESPONSE_ERROR | "
                        f"{type(post_response_error).__name__} at {time.monotonic() - call_t0:.2f}s: "
                        f"{format_exception_msg(post_response_error)}"
                    )
                raise post_response_error
            except BaseException as base_exception:
                self._account_call(
                    acquisition=acquisition,
                    usage=estimated_usage.with_output_tokens(output_tokens=0),
                    should_track_cost=should_track_cost,
                )
                if verbosity >= 2:
                    logger.warning(
                        f"[{self.name}] [Prompt={prompt_hash}] POST_RESPONSE_ERROR | "
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
                desc=f"{self.model_name}",
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
