"""
SlowBurnCrewAI: Cost-control middleware for CrewAI via event bus.

Intercepts every LLM call made by any CrewAI agent. Before the call,
estimates cost and acquires budget from a shared LimitSet (blocking if
budget is exhausted). After the call, logs the estimated actual cost
to a CostReporter.

Design note:
    CrewAI >=1.0 uses an event bus with LLMCallStartedEvent / LLMCallCompletedEvent.
    Since the two events fire in separate callbacks, we acquire and immediately
    update with the **estimated** cost in the started-event (which blocks if budget
    is exhausted — this IS the backpressure), then log the estimated actual cost
    from the response in the completed-event.

Usage::

    from slowburn.integrations.crewai import SlowBurnCrewAI

    sb = SlowBurnCrewAI(budget_usd=5.0, window="daily")
    sb.install()
    crew.kickoff()
    sb.uninstall()

    print(sb.reporter.to_markdown())
"""

import logging
from typing import Optional, Union

from concurry import LimitSet, RateWindow

from ..config import slowburn_config
from ..cost_accounting import estimate_input_tokens
from ..limits import DEFAULT_COST_LIMIT_KEY, CostLimit
from ..pricing import PricingCache
from ..reporter import CostReporter

logger = logging.getLogger(__name__)


class SlowBurnCrewAI:
    """Cost-control middleware for CrewAI via event bus or hooks.

    Supports both CrewAI >=1.0 (event bus) and older versions (hooks API).

    Args:
        budget_usd: Maximum dollar spend per window. Ignored if ``limit_set`` is provided.
        window: Length of the budget window. Accepts a :class:`RateWindow`
            member, a string alias (``"daily"``, ``"hourly"``, ``"weekly"``,
            etc.), or a positive number of seconds. Ignored if ``limit_set``
            is provided.
        limit_set: Optional pre-created LimitSet to use. Enables sharing a single budget
            across multiple SlowBurn integrations (e.g., CrewAI + AutoGen + SlowBurnLLM).
        reporter: Optional pre-existing CostReporter to share with other components.
    """

    def __init__(
        self,
        budget_usd: float = 0.0,
        window: Optional[Union[RateWindow, str, int, float]] = None,
        max_tokens: Optional[int] = None,
        limit_set: Optional[LimitSet] = None,
        reporter: Optional[CostReporter] = None,
    ):
        if window is None:
            window = RateWindow.Daily
        if limit_set is not None:
            self.limit_set = limit_set
        else:
            if budget_usd <= 0:
                raise ValueError(
                    "SlowBurnCrewAI requires either a positive budget_usd or a pre-created limit_set."
                )
            self.limit_set = LimitSet(
                limits=[CostLimit(budget_usd=budget_usd, window=window)],
                mode="Threads",
                shared=True,
            )
        if max_tokens is None:
            raise ValueError(
                "SlowBurnCrewAI requires max_tokens (the max output tokens "
                "configured on your CrewAI LLM). This is needed for cost "
                "estimation because CrewAI's event bus does not expose it."
            )
        self.max_tokens = max_tokens
        self.reporter = reporter if reporter is not None else CostReporter()
        self._installed = False
        self._backend = None

    def install(self) -> None:
        """Register listeners with CrewAI. Call once before ``crew.kickoff()``."""
        if self._try_install_event_bus():
            self._backend = "event_bus"
            self._installed = True
            return
        if self._try_install_hooks():
            self._backend = "hooks"
            self._installed = True
            return
        raise ImportError(
            "CrewAI is not installed or has an unsupported version. "
            "Install with: pip install slowburn[crewai]"
        )

    def _try_install_event_bus(self) -> bool:
        """Try to install via CrewAI >=1.0 event bus API."""
        try:
            from crewai.events import (
                LLMCallCompletedEvent,
                LLMCallStartedEvent,
                crewai_event_bus,
            )
        except ImportError:
            return False

        limit_set = self.limit_set
        reporter = self.reporter
        max_tokens = self.max_tokens

        @crewai_event_bus.on(LLMCallStartedEvent)
        def _on_llm_start(source, event: LLMCallStartedEvent):
            model_name = event.model or "unknown"
            messages = event.messages or []

            total_text = " ".join(
                msg.get("content", "") if isinstance(msg, dict) else str(getattr(msg, "content", ""))
                for msg in messages
            )
            estimated_input, estimated_output = estimate_input_tokens(total_text, max_tokens)

            estimated_cost = PricingCache.estimate_cost_usd(
                model_name,
                estimated_input,
                estimated_output,
            )

            with limit_set.acquire(requested={DEFAULT_COST_LIMIT_KEY: estimated_cost}) as acq:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: estimated_cost})

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _on_llm_end(source, event: LLMCallCompletedEvent):
            model_name = event.model or "unknown"
            response_text = event.response or ""
            if not isinstance(response_text, str):
                response_text = str(response_text)

            est_output_tokens = max(int(len(response_text) / slowburn_config.defaults.chars_per_token), 1)
            estimated_cost = PricingCache.estimate_cost_usd(
                model_name,
                0,
                est_output_tokens,
            )
            reporter.log_call(
                model=model_name,
                cost_usd=estimated_cost,
                input_tokens=0,
                output_tokens=est_output_tokens,
            )

        self._event_handlers = (_on_llm_start, _on_llm_end)
        return True

    def _try_install_hooks(self) -> bool:
        """Try to install via legacy CrewAI hooks API."""
        try:
            from crewai.hooks import (
                LLMCallHookContext,
                register_after_llm_call_hook,
                register_before_llm_call_hook,
            )
        except ImportError:
            return False

        limit_set = self.limit_set
        reporter = self.reporter

        def check_budget(context: "LLMCallHookContext") -> Optional[bool]:
            model_name = context.llm.model_name
            if not isinstance(model_name, str) or len(model_name) == 0:
                return None

            total_text = " ".join(
                msg.get("content", "") for msg in context.messages if isinstance(msg.get("content"), str)
            )
            max_tokens = context.llm.max_tokens
            if max_tokens is None:
                raise RuntimeError(
                    f"SlowBurnCrewAI: max_tokens is not set on LLM '{model_name}'. "
                    f"Set max_tokens on the CrewAI LLM to enable cost estimation."
                )
            estimated_input, estimated_output = estimate_input_tokens(total_text, max_tokens)

            estimated_cost = PricingCache.estimate_cost_usd(
                model_name,
                estimated_input,
                estimated_output,
            )

            with limit_set.acquire(requested={DEFAULT_COST_LIMIT_KEY: estimated_cost}) as acq:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: estimated_cost})

            return None

        def track_cost(context: "LLMCallHookContext") -> Optional[str]:
            model_name = context.llm.model_name
            if not isinstance(model_name, str) or len(model_name) == 0:
                return None

            response_text = context.response or ""
            est_output_tokens = max(int(len(response_text) / slowburn_config.defaults.chars_per_token), 1)
            estimated_cost = PricingCache.estimate_cost_usd(
                model_name,
                0,
                est_output_tokens,
            )
            reporter.log_call(
                model=model_name,
                cost_usd=estimated_cost,
                input_tokens=0,
                output_tokens=est_output_tokens,
            )
            return None

        register_before_llm_call_hook(check_budget)
        register_after_llm_call_hook(track_cost)
        return True

    def uninstall(self) -> None:
        """Remove all CrewAI LLM listeners/hooks."""
        if not self._installed:
            return
        if self._backend == "hooks":
            try:
                from crewai.hooks import clear_all_llm_call_hooks

                clear_all_llm_call_hooks()
            except ImportError:
                pass
        self._installed = False
