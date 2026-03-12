"""
SlowBurnCrewAI: Cost-control middleware for CrewAI via LLM call hooks.

Intercepts every LLM call made by any CrewAI agent. Before the call,
estimates cost and acquires budget from a shared LimitSet (blocking if
budget is exhausted). After the call, logs the estimated actual cost
to a CostReporter.

Design note:
    CrewAI hooks split execution across two callbacks: ``before_llm_call``
    and ``after_llm_call``. Concurry's ``acquire()`` returns a context manager
    that MUST be ``update()``-ed before exit. Since the two callbacks are
    separate function calls, we acquire and immediately update with the
    **estimated** cost in the before-hook (which blocks if budget is
    exhausted — this IS the backpressure), then log the estimated actual
    cost from the response text length in the after-hook.

    This means CrewAI integration charges the estimated cost at acquire time,
    not the actual cost. For precise cost tracking, use SlowBurnLLM directly.

Usage::

    from slowburn.integrations.crewai import SlowBurnCrewAI

    sb = SlowBurnCrewAI(budget_usd=5.0, window_seconds=86400)
    sb.install()
    crew.kickoff()
    sb.uninstall()

    print(sb.reporter.to_markdown())
"""

import logging
from typing import Optional

from concurry import LimitSet

from ..limits import DEFAULT_COST_LIMIT_KEY, CostLimit, microdollars_to_dollars
from ..pricing import PricingCache
from ..reporter import CostReporter

logger = logging.getLogger(__name__)


class SlowBurnCrewAI:
    """Cost-control middleware for CrewAI via LLM call hooks.

    Args:
        budget_usd: Maximum dollar spend per window. Ignored if ``limit_set`` is provided.
        window_seconds: Length of the budget window in seconds. Ignored if ``limit_set`` is provided.
        limit_set: Optional pre-created LimitSet to use. Enables sharing a single budget
            across multiple SlowBurn integrations (e.g., CrewAI + AutoGen + SlowBurnLLM).
        reporter: Optional pre-existing CostReporter to share with other components.
    """

    def __init__(
        self,
        budget_usd: float = 0.0,
        window_seconds: float = 86400,
        limit_set: Optional[LimitSet] = None,
        reporter: Optional[CostReporter] = None,
    ):
        if limit_set is not None:
            self.limit_set = limit_set
        else:
            if budget_usd <= 0:
                raise ValueError(
                    "SlowBurnCrewAI requires either a positive budget_usd or a pre-created limit_set."
                )
            self.limit_set = LimitSet(
                limits=[CostLimit(budget_usd=budget_usd, window_seconds=window_seconds)],
                mode="thread",
                shared=True,
            )
        self.reporter = reporter if reporter is not None else CostReporter()
        self._installed = False

    def install(self) -> None:
        """Register hooks with CrewAI. Call once before ``crew.kickoff()``."""
        try:
            from crewai.hooks import (
                LLMCallHookContext,
                register_after_llm_call_hook,
                register_before_llm_call_hook,
            )
        except ImportError as e:
            raise ImportError(
                "CrewAI is not installed. Install with: pip install slowburn[crewai]"
            ) from e

        def check_budget(context: "LLMCallHookContext") -> Optional[bool]:
            model_name = context.llm.model_name
            if not isinstance(model_name, str) or len(model_name) == 0:
                raise RuntimeError(
                    "SlowBurnCrewAI: context.llm.model_name is empty or not a string."
                )

            total_text = " ".join(
                msg.get("content", "")
                for msg in context.messages
                if isinstance(msg.get("content"), str)
            )
            estimated_input = int(max(len(total_text) // 3, 1) * 5.0) + 50
            max_tokens = context.llm.max_tokens
            if max_tokens is None:
                raise RuntimeError(
                    "SlowBurnCrewAI: Could not determine max_tokens from CrewAI's LLM object. "
                    "Ensure the agent's LLM has a 'max_tokens' attribute."
                )
            estimated_output = max_tokens

            estimated_cost = PricingCache.estimate_cost_microdollars(
                model_name, estimated_input, estimated_output,
            )

            with self.limit_set.acquire(
                requested={DEFAULT_COST_LIMIT_KEY: max(estimated_cost, 1)}
            ) as acq:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: max(estimated_cost, 1)})

            return None

        def track_cost(context: "LLMCallHookContext") -> Optional[str]:
            model_name = context.llm.model_name
            if not isinstance(model_name, str) or len(model_name) == 0:
                raise RuntimeError(
                    "SlowBurnCrewAI.track_cost: model_name is empty or not a string."
                )

            response_text = context.response
            if response_text is None:
                raise RuntimeError(
                    "SlowBurnCrewAI.track_cost: context.response is None. "
                    "The LLM call may have failed silently."
                )

            est_output_tokens = max(len(response_text) // 3, 1)
            estimated_cost = PricingCache.estimate_cost_microdollars(
                model_name, 0, est_output_tokens,
            )
            self.reporter.log_call(
                model=model_name,
                cost_usd=microdollars_to_dollars(estimated_cost),
                input_tokens=0,
                output_tokens=est_output_tokens,
            )
            return None

        register_before_llm_call_hook(check_budget)
        register_after_llm_call_hook(track_cost)
        self._installed = True

    def uninstall(self) -> None:
        """Remove all CrewAI LLM hooks."""
        if not self._installed:
            return
        try:
            from crewai.hooks import clear_all_llm_call_hooks
            clear_all_llm_call_hooks()
        except ImportError:
            pass
        self._installed = False
