"""
SlowBurnModelClient: AG2 (AutoGen) ModelClient protocol with cost-aware backpressure.

Wraps litellm to provide cost-controlled LLM calls for AutoGen agents.
When the dollar budget is exhausted, ``create()`` blocks (backpressure)
until the rate limit window rolls over — the agent thread simply waits.

Usage::

    from autogen import AssistantAgent, LLMConfig
    from concurry import LimitSet
    from slowburn.limits import CostLimit
    from slowburn.reporter import CostReporter
    from slowburn.integrations.autogen import SlowBurnModelClient

    limit_set = LimitSet(
        limits=[CostLimit(budget_usd=5.0, window="daily")],
        mode="Threads", shared=True,
    )
    reporter = CostReporter()

    assistant = AssistantAgent(
        "assistant",
        llm_config=LLMConfig({"model": "slowburn/claude-3-haiku", "api_key": "sk-..."}),
    )
    assistant.register_model_client(
        model_client_cls=SlowBurnModelClient,
        limit_set=limit_set,
        reporter=reporter,
    )
"""

import logging
from typing import Any, Dict, List, Optional

import litellm

from ..cost_accounting import cost_controlled_call, estimate_input_tokens
from ..limits import DEFAULT_COST_LIMIT_KEY, microdollars_to_dollars
from ..pricing import PricingCache
from ..reporter import CostReporter

logger = logging.getLogger(__name__)


class SlowBurnModelClient:
    """AG2 ModelClient protocol implementation with cost-aware backpressure.

    Implements the four methods required by AutoGen's ModelClient protocol:
    ``create``, ``cost``, ``get_usage``, ``message_retrieval``.

    The ``create()`` method is synchronous and blocking. When the dollar
    budget is exhausted it blocks via ``limit_set.acquire()`` (which uses
    ``time.sleep`` under the hood) until the rate limit window rolls over.

    Args:
        config: AG2 model configuration dict. **Must** contain a ``"model"`` key.
        limit_set: Concurry LimitSet with CostLimit (and optionally other limits).
        reporter: CostReporter instance for accumulating cost data.

    Raises:
        ValueError: If ``config`` does not contain a ``"model"`` key.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        limit_set: Any,
        reporter: Optional[CostReporter] = None,
        **kwargs,
    ):
        if "model" not in config:
            raise ValueError(
                "SlowBurnModelClient requires 'model' in config dict. Got keys: " + str(list(config.keys()))
            )
        self.model = config["model"]
        self.litellm_model = self.model.removeprefix("slowburn/")
        self.limit_set = limit_set
        self.reporter = reporter if reporter is not None else CostReporter()

    def create(self, params: Dict[str, Any]) -> Any:
        """Make an LLM call with cost-aware backpressure.

        This is SYNCHRONOUS. When the budget is exhausted, this method
        BLOCKS until the rate limit window rolls over. AG2's agent thread
        simply waits.

        Args:
            params: AG2 call parameters (messages, model, temperature, max_tokens, etc.).

        Returns:
            litellm completion response object.
        """
        messages = params.get("messages")
        if messages is None:
            raise ValueError("SlowBurnModelClient.create(): 'messages' is required in params.")
        model = params.get("model", self.litellm_model)
        if model.startswith("slowburn/"):
            model = model.removeprefix("slowburn/")
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            raise ValueError(
                "SlowBurnModelClient.create(): 'max_tokens' is required in params. "
                "Set it in the AG2 config_list or pass it per-call."
            )

        # 1. ESTIMATE tokens and cost
        total_text = " ".join(
            msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
        )
        est_input, est_output = estimate_input_tokens(total_text, max_tokens)

        # 2. ACQUIRE → EXECUTE → UPDATE → LOG (via shared context manager)
        with cost_controlled_call(
            self.limit_set,
            self.reporter,
            model,
            est_input,
            est_output,
        ) as ctx:
            # Pass through all params from AG2 to litellm, only overriding
            # model/messages/max_tokens which we already extracted above.
            litellm_params = {k: v for k, v in params.items() if k not in ("messages", "model")}
            response = litellm.completion(
                model=model,
                messages=messages,
                **litellm_params,
            )

            actual_cost_micro = PricingCache.actual_cost_microdollars(response, model=model)
            ctx.set_actual(
                cost=actual_cost_micro,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            return response

    def cost(self, response: Any) -> float:
        """Return cost in USD for this response.

        Raises:
            AttributeError: If the response object does not have cost data.
        """
        cost_usd = response._hidden_params.get("response_cost")
        if cost_usd is not None and cost_usd > 0:
            return cost_usd
        raise AttributeError(
            f"Response has no valid 'response_cost' in _hidden_params. "
            f"Got: {getattr(response, '_hidden_params', 'MISSING')}"
        )

    def get_usage(self, response: Any) -> Dict[str, Any]:
        """Return usage dict for AG2's built-in cost tracking system.

        Raises:
            AttributeError: If the response object does not have usage data.
        """
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": self.cost(response),
            "model": response.model,
        }

    def message_retrieval(self, response: Any) -> List[str]:
        """Return response messages as strings.

        Raises:
            AttributeError: If the response does not have choices with message content.
        """
        return [choice.message.content for choice in response.choices]
