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
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import litellm
from concurry import async_gather, worker
from morphic import Typed, validate
from pydantic import Field

from .config import _NO_ARG, _NO_ARG_TYPE, is_no_arg, slowburn_config
from .constants import BackpressureNotify, BudgetOverflowAction, ImageDetailLevel, ToolChoiceOption
from .limits import DEFAULT_COST_LIMIT_KEY, microdollars_to_dollars
from .pricing import PricingCache
from .reporter import CostReporter

litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
            '"error": raise ValueError. "ignore": proceed silently. '
            "Defaults to slowburn_config.defaults.on_budget_overflow."
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
                if images is not None and len(images) > 0:
                    image_urls = _resolve_image_inputs(images)
                    content_parts: List[Dict[str, Any]] = [
                        {"type": "text", "text": prompt},
                    ]
                    for url in image_urls:
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": url, "detail": image_detail},
                            }
                        )
                    messages.append({"role": "user", "content": content_parts})
                else:
                    messages.append({"role": "user", "content": prompt})
            return messages

        messages: List[Dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        if images is not None and len(images) > 0:
            image_urls = _resolve_image_inputs(images)
            content_parts = [
                {"type": "text", "text": prompt},
            ]
            for url in image_urls:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": image_detail},
                    }
                )
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

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
    ) -> Union[str, List[Dict[str, Any]]]:
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
                Ignored when *return_messages* resolves to True.
            image_detail: Detail level for vision queries ("low"/"high"/"auto").
                Defaults to slowburn_config.defaults.image_detail.
            verbosity: Logging verbosity (0=silent, 1=normal, 2=debug).
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

        should_return_messages: bool
        if return_messages is not None:
            should_return_messages = return_messages
        elif history is not None or isinstance(prompt, list):
            should_return_messages = True
        else:
            should_return_messages = False

        messages = self.build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            history=history,
            image_detail=image_detail,
        )

        merged_params: Dict[str, Any] = {**self.litellm_params}
        if litellm_params is not None:
            merged_params.update(litellm_params)

        resolved_tools = tools if not is_no_arg(tools) else self.tools
        resolved_tool_choice = tool_choice if not is_no_arg(tool_choice) else self.tool_choice

        if resolved_tools is not None:
            merged_params["tools"] = resolved_tools
        if resolved_tool_choice is not None:
            merged_params["tool_choice"] = resolved_tool_choice

        # 1. ESTIMATE tokens using litellm's local tokenizer
        # This accounts for the full messages list (history, tool schemas,
        # tool results, images) rather than just the current prompt string.
        # A safety multiplier and buffer are applied on top to account for
        # differences between litellm's tokenizer and the actual provider.
        defaults = slowburn_config.defaults
        base_input_tokens = litellm.token_counter(
            model=self.model_name,
            messages=messages,
            tools=resolved_tools,
            tool_choice=resolved_tool_choice,
            use_default_image_token_count=True,
        )
        estimated_input_tokens = (
            int(base_input_tokens * defaults.input_token_estimate_multiplier)
            + defaults.input_token_estimate_overhead
        )
        estimated_output_tokens = (
            int(self.max_tokens * defaults.output_token_estimate_multiplier)
            + defaults.output_token_estimate_overhead
        )

        # 2. ESTIMATE cost in microdollars
        estimated_cost = PricingCache.estimate_cost_microdollars(
            self.model_name,
            estimated_input_tokens,
            estimated_output_tokens,
        )

        # 3. ACQUIRE (blocks if budget/rate exhausted)
        requested: Dict[str, int] = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "call_count": 1,
            DEFAULT_COST_LIMIT_KEY: estimated_cost,
        }

        try:
            acquire_start = time.monotonic()
            context_manager = self.limits.acquire(requested=requested)
            acquire_elapsed = time.monotonic() - acquire_start
        except ValueError as acquire_error:
            if "exceeds capacity" not in str(acquire_error):
                raise

            overflow_message = (
                f"A single call_llm() call to {self.model_name} is estimated to cost "
                f"${microdollars_to_dollars(estimated_cost):.6f} "
                f"(~{estimated_input_tokens} input + {estimated_output_tokens} output tokens), "
                f"which exceeds your budget_usd per window. "
                f"Fix by: (1) increasing the budget while creating the LLM, "
                f"(2) reducing max_tokens (currently {self.max_tokens}), "
                f"or (3) using a more budget-friendly model."
            )

            if self.on_budget_overflow == "error":
                raise ValueError(overflow_message) from acquire_error

            if self.on_budget_overflow == "warn":
                logger.warning(f"[{self.name}] Budget overflow: {overflow_message}")

            capped_requested = dict(requested)
            capped_requested[DEFAULT_COST_LIMIT_KEY] = 1
            acquire_start = time.monotonic()
            context_manager = self.limits.acquire(requested=capped_requested)
            acquire_elapsed = time.monotonic() - acquire_start

        if self.backpressure_notify == "warn":
            threshold = slowburn_config.defaults.backpressure_threshold_seconds
            if acquire_elapsed > threshold:
                logger.warning(
                    f"[{self.name}] Backpressure: blocked {acquire_elapsed:.1f}s "
                    f"waiting for budget/rate capacity "
                    f"(estimated ${microdollars_to_dollars(estimated_cost):.6f}, "
                    f"~{estimated_input_tokens} input + {estimated_output_tokens} output tokens)"
                )

        with context_manager as acquisition:
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

                actual_input = response.usage.prompt_tokens
                actual_output = response.usage.completion_tokens

                actual_cost = PricingCache.actual_cost_microdollars(
                    response,
                    model=self.model_name,
                )

                response_message = response.choices[0].message
                response_text = response_message.content
                tool_calls = response_message.tool_calls

                if response_text is None and tool_calls is None:
                    acquisition.update(
                        usage={
                            "input_tokens": actual_input,
                            "output_tokens": actual_output,
                            "call_count": 1,
                            DEFAULT_COST_LIMIT_KEY: actual_cost,
                        }
                    )
                    raise ValueError(
                        f"LLM returned null content with no tool calls "
                        f"(model={self.model_name}). "
                        f"This may indicate a refusal or content filter."
                    )

                if response_text is None and tool_calls is not None:
                    import json as _json

                    response_text = _json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in tool_calls
                            ]
                        }
                    )

                # 5. Apply validator if provided (skipped when returning messages)
                result: Union[str, List[Dict[str, Any]]]
                if should_return_messages:
                    assistant_message: Dict[str, Any] = {"role": "assistant"}
                    if response_message.content is not None:
                        assistant_message["content"] = response_message.content
                    if tool_calls is not None:
                        assistant_message["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ]
                    if "content" not in assistant_message:
                        assistant_message["content"] = None
                    messages.append(assistant_message)
                    result = messages
                elif validator is not None:
                    try:
                        result = validator(response_text)
                    except ValueError:
                        acquisition.update(
                            usage={
                                "input_tokens": actual_input,
                                "output_tokens": actual_output,
                                "call_count": 1,
                                DEFAULT_COST_LIMIT_KEY: actual_cost,
                            }
                        )
                        raise
                    except Exception as e:
                        acquisition.update(
                            usage={
                                "input_tokens": actual_input,
                                "output_tokens": actual_output,
                                "call_count": 1,
                                DEFAULT_COST_LIMIT_KEY: actual_cost,
                            }
                        )
                        raise ValueError(f"Validator error: {e}") from e
                else:
                    result = response_text

                # 6. UPDATE limits with actuals (refunds unused budget)
                acquisition.update(
                    usage={
                        "input_tokens": actual_input,
                        "output_tokens": actual_output,
                        "call_count": 1,
                        DEFAULT_COST_LIMIT_KEY: actual_cost,
                    }
                )

                # 7. LOG to reporter
                self._reporter.log_call(
                    model=self.model_name,
                    cost_usd=microdollars_to_dollars(actual_cost),
                    input_tokens=actual_input,
                    output_tokens=actual_output,
                )

                if verbosity >= 2:
                    logger.info(
                        f"[{self.name}] {self.model_name}: "
                        f"{actual_input}+{actual_output} tokens, "
                        f"${microdollars_to_dollars(actual_cost):.6f}"
                    )

                return result

            except ValueError:
                raise
            except (asyncio.TimeoutError, BaseException):
                acquisition.update(
                    usage={
                        "input_tokens": estimated_input_tokens,
                        "output_tokens": 0,
                        "call_count": 1,
                        DEFAULT_COST_LIMIT_KEY: estimated_cost,
                    }
                )
                raise

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
    ) -> List[Union[str, List[Dict[str, Any]]]]:
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
            raise ValueError(
                f"images_per_prompt length ({len(images_per_prompt)}) "
                f"must match prompts length ({len(prompts)})"
            )
        if history_per_prompt is not None and len(history_per_prompt) != len(prompts):
            raise ValueError(
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

        results: List[Union[str, List[Dict[str, Any]]]] = await async_gather(
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
