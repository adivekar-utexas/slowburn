"""
SlowBurnMiddleware: LangGraph agent middleware for cost-controlled execution.

Uses LangGraph's ``AgentMiddleware.wrap_model_call`` to intercept every
model call with budget-aware backpressure. If the budget is exhausted,
the middleware blocks (via Concurry's ``acquire()``) until the rate limit
window rolls over, then proceeds. The agent slows down rather than crashing.

The ``ModelRequest`` object provides ``request.model`` (a ``BaseChatModel``)
and ``request.messages`` (list of LangChain messages), giving us access to
the model name and message content for cost estimation.

Usage::

    from langchain.agents import create_agent
    from slowburn.integrations.langgraph import SlowBurnMiddleware

    budget = SlowBurnMiddleware(budget_usd=5.0)
    agent = create_agent(
        model="openai:gpt-4o-mini",
        middleware=[budget],
        tools=[...],
    )
    result = agent.invoke({"messages": [("user", "Write a report")]})
    print(budget.reporter.to_markdown())

Requires: ``pip install slowburn[langgraph]``
"""

import logging
from typing import Any, Callable, Optional

from concurry import LimitSet

from ..limits import DEFAULT_COST_LIMIT_KEY, CostLimit, microdollars_to_dollars
from ..pricing import PricingCache
from ..reporter import CostReporter

logger = logging.getLogger(__name__)


def _get_model_name(model: Any) -> str:
    """Extract model name string from a LangChain BaseChatModel.

    BaseChatModel always has model_name. Raises AttributeError if not.
    """
    name = model.model_name
    if isinstance(name, str) and len(name) > 0:
        return name
    raise RuntimeError(
        f"SlowBurnMiddleware: model_name on {type(model).__name__} is "
        f"empty or not a string: {name!r}"
    )


def _extract_text_from_messages(messages: list) -> str:
    """Extract text content from LangChain message objects."""
    parts = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
    return " ".join(parts)


class SlowBurnMiddleware:
    """LangGraph agent middleware for cost-controlled execution.

    Wraps every model call with budget-aware backpressure via
    ``wrap_model_call``. The acquire-execute-update cycle happens
    atomically within the ``with`` block.

    Args:
        budget_usd: Maximum dollar spend per window. Ignored if ``limit_set`` is provided.
        window_seconds: Length of the budget window in seconds. Ignored if ``limit_set`` is provided.
        limit_set: Optional pre-created LimitSet to use. Enables sharing a single budget
            across multiple SlowBurn integrations.
        reporter: Optional pre-existing CostReporter to share.
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
                    "SlowBurnMiddleware requires either a positive budget_usd or a pre-created limit_set."
                )
            self.limit_set = LimitSet(
                limits=[CostLimit(budget_usd=budget_usd, window_seconds=window_seconds)],
                mode="thread",
                shared=True,
            )
        self.reporter = reporter if reporter is not None else CostReporter()

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable,
    ) -> Any:
        """Wrap each model call with cost-aware backpressure.

        Args:
            request: LangGraph ``ModelRequest`` with ``.model`` and ``.messages``.
            handler: Callback that executes the model request.

        Returns:
            The ``ModelResponse`` from handler(request).

        Raises:
            RuntimeError: If model name cannot be determined from request.
            ModelNotFoundError: If model is not in litellm's pricing database.
        """
        model_name = _get_model_name(request.model)
        max_tokens = None
        if request.model_settings is not None:
            max_tokens = request.model_settings.get("max_tokens")
        if max_tokens is None:
            max_tokens = request.model.max_tokens
        if max_tokens is None:
            raise RuntimeError(
                f"SlowBurnMiddleware: Could not determine max_tokens from model "
                f"'{model_name}' or model_settings. Set max_tokens on the model "
                f"or pass it via model_settings."
            )

        total_text = _extract_text_from_messages(request.messages)
        if request.system_message is not None:
            sys_content = request.system_message.content
            if isinstance(sys_content, str):
                total_text += " " + sys_content

        estimated_input = int(max(len(total_text) // 3, 1) * 5.0) + 50
        estimated_output = max_tokens
        estimated_cost = PricingCache.estimate_cost_microdollars(
            model_name, estimated_input, estimated_output,
        )

        with self.limit_set.acquire(
            requested={DEFAULT_COST_LIMIT_KEY: max(estimated_cost, 1)}
        ) as acq:
            try:
                response = handler(request)

                resp_message = response
                content = resp_message.content
                response_text = content if isinstance(content, str) else ""

                usage_metadata = resp_message.usage_metadata
                if usage_metadata is not None:
                    if "input_tokens" not in usage_metadata:
                        raise KeyError(
                            f"usage_metadata missing 'input_tokens': "
                            f"{list(usage_metadata.keys())}"
                        )
                    if "output_tokens" not in usage_metadata:
                        raise KeyError(
                            f"usage_metadata missing 'output_tokens': "
                            f"{list(usage_metadata.keys())}"
                        )
                    actual_input = usage_metadata["input_tokens"]
                    actual_output = usage_metadata["output_tokens"]
                else:
                    actual_input = estimated_input
                    actual_output = max(len(response_text) // 3, 1)

                actual_cost = PricingCache.estimate_cost_microdollars(
                    model_name, actual_input, actual_output,
                )
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: max(actual_cost, 1)})

                self.reporter.log_call(
                    model=model_name,
                    cost_usd=microdollars_to_dollars(actual_cost),
                    input_tokens=actual_input,
                    output_tokens=actual_output,
                )

                return response

            except BaseException:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: estimated_cost})
                raise
