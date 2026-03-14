"""
Centralized cost-accounting utilities for SlowBurn.

Every SlowBurn integration (llm_worker, autogen, langgraph, langchain, crewai)
repeats the same estimate-acquire-update-log cycle. This module extracts the
shared pieces so the formula lives in one place.

Three utilities:

1. ``estimate_input_tokens`` — the shared chars-to-tokens formula
2. ``CostCallContext`` — lightweight mutable bag that the caller fills with
   actual usage after the LLM call completes
3. ``cost_controlled_call`` — sync context manager that wraps the full
   acquire-execute-update-log cycle for cost-only LimitSets
"""

from contextlib import contextmanager
from typing import Any, Generator, Tuple

from .config import slowburn_config
from .limits import DEFAULT_COST_LIMIT_KEY, microdollars_to_dollars
from .pricing import PricingCache
from .reporter import CostReporter


def estimate_input_tokens(text: str, max_tokens: int) -> Tuple[int, int]:
    """Estimate input and output token counts from raw text.

    Applies the shared formula used across all SlowBurn integrations:
    ``int(max(len(text) / chars_per_token, 1) * token_safety_multiplier) + base_overhead_tokens``

    All three constants are read from ``slowburn_config.defaults`` at call
    time, so they can be tuned globally via ``temp_config()``.

    Args:
        text: Concatenated text of all messages (user + system).
        max_tokens: Maximum output tokens (typically the model's max_tokens setting).

    Returns:
        ``(estimated_input_tokens, estimated_output_tokens)`` where the output
        estimate is simply ``max_tokens`` passed through.
    """
    cfg = slowburn_config.defaults
    raw_estimate = max(int(len(text) / cfg.chars_per_token), 1)
    estimated_input = int(raw_estimate * cfg.token_safety_multiplier) + cfg.base_overhead_tokens
    return estimated_input, max_tokens


class CostCallContext:
    """Mutable context yielded by ``cost_controlled_call``.

    The caller executes the LLM call inside the ``with`` block, then calls
    ``set_actual()`` with the real usage numbers before the block exits.
    The context manager reads these values to update the acquisition and
    log to the reporter.

    If the caller never calls ``set_actual()`` (e.g. because an exception
    was raised), the context manager falls back to the estimated cost.
    """

    __slots__ = (
        "estimated_cost",
        "actual_cost",
        "actual_input",
        "actual_output",
        "updated",
    )

    def __init__(self, estimated_cost: int) -> None:
        self.estimated_cost: int = estimated_cost
        self.actual_cost: int = 0
        self.actual_input: int = 0
        self.actual_output: int = 0
        self.updated: bool = False

    def set_actual(
        self,
        *,
        cost: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record the actual usage from a completed LLM call.

        Must be called exactly once inside the ``cost_controlled_call`` block,
        after the LLM response has been received and parsed.
        """
        self.actual_cost = cost
        self.actual_input = input_tokens
        self.actual_output = output_tokens
        self.updated = True


@contextmanager
def cost_controlled_call(
    limit_set: Any,
    reporter: CostReporter,
    model: str,
    estimated_input: int,
    estimated_output: int,
) -> Generator[CostCallContext, None, None]:
    """Sync context manager for the acquire-execute-update-log cycle.

    Handles the full cost-control lifecycle for **synchronous** callers
    (autogen, langgraph). The caller executes the LLM call inside the
    ``with`` block and reports actual usage via ``ctx.set_actual()``.

    On normal exit (``set_actual`` was called), updates the acquisition
    with actual cost and logs to the reporter. On any exception (including
    ``KeyboardInterrupt``), updates with the estimated cost so the
    acquisition is always cleanly released.

    Usage::

        est_in, est_out = estimate_input_tokens(text, max_tokens)
        with cost_controlled_call(limit_set, reporter, model, est_in, est_out) as ctx:
            response = litellm.completion(...)
            actual_cost = PricingCache.actual_cost_microdollars(response, model=model)
            ctx.set_actual(
                cost=actual_cost,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return response

    Args:
        limit_set: Concurry LimitSet (must contain a CostLimit).
        reporter: CostReporter for logging the call.
        model: Model name string for cost lookup.
        estimated_input: Estimated input tokens (from ``estimate_input_tokens``).
        estimated_output: Estimated output tokens (typically max_tokens).
    """
    estimated_cost = PricingCache.estimate_cost_microdollars(
        model,
        estimated_input,
        estimated_output,
    )

    with limit_set.acquire(requested={DEFAULT_COST_LIMIT_KEY: max(estimated_cost, 1)}) as acq:
        ctx = CostCallContext(estimated_cost=estimated_cost)
        try:
            yield ctx

            if not ctx.updated:
                raise RuntimeError(
                    "cost_controlled_call: ctx.set_actual() was never called. "
                    "The caller must report actual usage before the with-block exits."
                )

            acq.update(usage={DEFAULT_COST_LIMIT_KEY: ctx.actual_cost})
            reporter.log_call(
                model=model,
                cost_usd=microdollars_to_dollars(ctx.actual_cost),
                input_tokens=ctx.actual_input,
                output_tokens=ctx.actual_output,
            )
        except BaseException:
            if not ctx.updated:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: estimated_cost})
            raise
