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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import litellm
from concurry import async_gather, worker
from morphic import Typed
from pydantic import Field

from .limits import DEFAULT_COST_LIMIT_KEY
from .pricing import PricingCache
from .reporter import CostReporter

litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _estimate_tokens(text: str, *, chars_per_token: float = 3.0) -> int:
    """Rough estimate of token count from character length."""
    return max(int(len(text) // chars_per_token), 1)


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
                f"Image input must be a Path, URL string, or data-URL string, "
                f"got {type(img).__name__}"
            )
    return urls


@worker(mode="asyncio")
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
                mode="asyncio",
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
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    timeout: float = Field(default=120.0, gt=0.0)
    litellm_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional parameters passed through to litellm.acompletion(). "
            "Use for tools, response_format, seed, top_p, stop, logprobs, "
            "or any other litellm-supported parameter. Per-call litellm_params "
            "in call_llm() merge on top of these defaults."
        ),
    )

    def post_initialize(self) -> None:
        self._reporter = CostReporter()

    @property
    def reporter(self) -> CostReporter:
        """Access the CostReporter to inspect costs or export reports."""
        return self._reporter

    async def call_llm(
        self,
        *,
        prompt: str,
        images: Optional[List[ImageInput]] = None,
        system_prompt: Optional[str] = None,
        validator: Optional[Callable[[str], T]] = None,
        image_detail: str = "auto",
        verbosity: int = 1,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a single LLM call with cost-aware backpressure.

        Args:
            prompt: The user message content.
            images: Optional list of images to include in the message. Each
                element can be a ``pathlib.Path`` to a local file, a URL string
                (``http://`` / ``https://``), or an already-encoded data-URL
                string (``data:image/...;base64,...``). Local files are
                automatically base64-encoded. Pass ``None`` (default) for a
                text-only call.
            system_prompt: Optional system message prepended to the conversation.
            validator: Optional callable that parses/validates the response text.
                If it raises ``ValueError``, the error propagates (and Concurry's
                retry mechanism can catch it if configured with ``retry_on``).
            image_detail: Detail level for vision queries sent to the API.
                One of ``"low"``, ``"high"``, or ``"auto"`` (default).
            verbosity: Logging verbosity (0=silent, 1=normal, 2=debug).
            litellm_params: Per-call parameters passed through to
                ``litellm.acompletion()``. Merged on top of the worker-level
                ``self.litellm_params``. Use for call-specific tools,
                response_format, seed, etc.

        Returns:
            The raw response text, or the parsed result from *validator* if provided.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        if images is not None and len(images) > 0:
            image_urls = _resolve_image_inputs(images)
            content_parts: List[Dict[str, Any]] = [
                {"type": "text", "text": prompt},
            ]
            for url in image_urls:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": url, "detail": image_detail},
                })
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        merged_params: Dict[str, Any] = {**self.litellm_params}
        if litellm_params is not None:
            merged_params.update(litellm_params)

        # 1. ESTIMATE tokens
        estimated_text_tokens = _estimate_tokens(prompt)
        if system_prompt is not None:
            estimated_text_tokens += _estimate_tokens(system_prompt)
        estimated_input_tokens = int(estimated_text_tokens * 5.0) + 50

        # Account for image tokens (high-detail images use ~1000 tokens each,
        # low-detail ~85; "auto" is treated as high for safety)
        if images is not None and len(images) > 0:
            tokens_per_image = 85 if image_detail == "low" else 1000
            estimated_input_tokens += tokens_per_image * len(images)

        estimated_output_tokens = self.max_tokens

        # 2. ESTIMATE cost in microdollars
        estimated_cost = PricingCache.estimate_cost_microdollars(
            self.model_name, estimated_input_tokens, estimated_output_tokens,
        )

        # 3. ACQUIRE (blocks if budget/rate exhausted)
        requested: Dict[str, int] = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "call_count": 1,
            DEFAULT_COST_LIMIT_KEY: estimated_cost,
        }

        with self.limits.acquire(requested=requested) as acq:
            try:
                litellm.drop_params = True
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=self.model_name,
                        messages=messages,
                        api_key=self.api_key if self.api_key else None,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        **merged_params,
                    ),
                    timeout=self.timeout,
                )

                actual_input = response.usage.prompt_tokens
                actual_output = response.usage.completion_tokens

                response_message = response.choices[0].message
                response_text = response_message.content
                tool_calls = response_message.tool_calls

                if response_text is None and tool_calls is None:
                    raise ValueError(
                        f"LLM returned null content with no tool calls "
                        f"(model={self.model_name}). "
                        f"This may indicate a refusal or content filter."
                    )

                if response_text is None and tool_calls is not None:
                    import json as _json
                    response_text = _json.dumps({
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
                    })

                # 4. GET actual cost
                actual_cost = PricingCache.actual_cost_microdollars(
                    response, model=self.model_name,
                )

                # 5. Apply validator if provided
                result: Any = response_text
                if validator is not None:
                    try:
                        result = validator(response_text)
                    except ValueError:
                        acq.update(usage={
                            "input_tokens": actual_input,
                            "output_tokens": actual_output,
                            "call_count": 1,
                            DEFAULT_COST_LIMIT_KEY: actual_cost,
                        })
                        raise
                    except Exception as e:
                        acq.update(usage={
                            "input_tokens": actual_input,
                            "output_tokens": actual_output,
                            "call_count": 1,
                            DEFAULT_COST_LIMIT_KEY: actual_cost,
                        })
                        raise ValueError(f"Validator error: {e}") from e

                # 6. UPDATE limits with actuals (refunds unused budget)
                acq.update(usage={
                    "input_tokens": actual_input,
                    "output_tokens": actual_output,
                    "call_count": 1,
                    DEFAULT_COST_LIMIT_KEY: actual_cost,
                })

                # 7. LOG to reporter
                from .limits import microdollars_to_dollars
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

            except (ValueError, asyncio.TimeoutError):
                raise
            except BaseException:
                acq.update(usage={
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": 0,
                    "call_count": 1,
                    DEFAULT_COST_LIMIT_KEY: estimated_cost,
                })
                raise

    async def call_llm_batch(
        self,
        *,
        prompts: List[str],
        images_per_prompt: Optional[List[Optional[List[ImageInput]]]] = None,
        system_prompt: Optional[str] = None,
        validator: Optional[Callable[[str], T]] = None,
        image_detail: str = "auto",
        verbosity: int = 1,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Execute multiple LLM calls concurrently with shared backpressure.

        Args:
            prompts: List of user message strings.
            images_per_prompt: Optional list, same length as *prompts*, where
                each element is either ``None`` (text-only call) or a list of
                ``ImageInput`` for that prompt. Pass ``None`` (default) if no
                prompts use images.
            system_prompt: Optional system message applied to all calls.
            validator: Optional callable applied to each response.
            image_detail: Detail level for vision queries ("low", "high", "auto").
            verbosity: Logging verbosity.
            litellm_params: Per-call parameters passed through to each
                ``litellm.acompletion()`` call in the batch.

        Returns:
            List of results (raw text or parsed validator output).
        """
        if len(prompts) == 0:
            return []

        if images_per_prompt is not None and len(images_per_prompt) != len(prompts):
            raise ValueError(
                f"images_per_prompt length ({len(images_per_prompt)}) "
                f"must match prompts length ({len(prompts)})"
            )

        if images_per_prompt is None:
            images_per_prompt = [None] * len(prompts)

        tasks = [
            self.call_llm(
                prompt=p,
                images=imgs,
                system_prompt=system_prompt,
                validator=validator,
                image_detail=image_detail,
                verbosity=verbosity,
                litellm_params=litellm_params,
            )
            for p, imgs in zip(prompts, images_per_prompt)
        ]

        results: List[Any] = await async_gather(
            tasks,
            progress=dict(
                disable=verbosity < 2,
                desc=f"{self.name}:{self.model_name}",
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
