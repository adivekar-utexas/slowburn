"""
SlowBurnCallbackHandler: LangChain callback handler for cost-controlled execution.

Intercepts every LLM call via ``on_llm_start`` (to acquire budget with
backpressure) and ``on_llm_end`` (to update with actual cost and log).

Backpressure mechanism: ``on_llm_start`` blocks (sleeps) via
``limit_set.acquire()`` until budget is available. The LLM call is
delayed, not cancelled. The chain/agent thread simply waits.

The acquire/update cycle uses a proper ``with`` block inside ``on_llm_start``
+ ``on_llm_end`` by storing the acquisition keyed by ``run_id``. If
``on_llm_error`` fires instead of ``on_llm_end``, the acquisition is
still updated with the full estimated cost (worst case).

Usage::

    from langchain_openai import ChatOpenAI
    from slowburn.integrations.langchain import SlowBurnCallbackHandler

    budget_handler = SlowBurnCallbackHandler(budget_usd=5.0)
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[budget_handler])

    result = llm.invoke("Write a market analysis report")
    print(budget_handler.reporter.to_markdown())

Requires: ``pip install slowburn[langchain]``
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from concurry import LimitSet

from ..config import slowburn_config
from ..cost_accounting import estimate_input_tokens
from ..limits import DEFAULT_COST_LIMIT_KEY, CostLimit, microdollars_to_dollars
from ..pricing import PricingCache
from ..reporter import CostReporter

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler as _BaseCallbackHandler
    except ImportError:
        _BaseCallbackHandler = object


class SlowBurnCallbackHandler(_BaseCallbackHandler):
    """LangChain callback handler for cost-controlled LLM execution.

    Implements ``on_llm_start``, ``on_llm_end``, and ``on_llm_error``
    from LangChain's callback protocol. The handler acquires budget
    in ``on_llm_start`` (blocking if exhausted) and updates with
    actual cost in ``on_llm_end``.

    Args:
        budget_usd: Maximum dollar spend per window. Ignored if ``limit_set`` is provided.
        window_seconds: Length of the budget window in seconds. Ignored if ``limit_set`` is provided.
        limit_set: Optional pre-created LimitSet to use. Enables sharing a single budget
            across multiple SlowBurn integrations.
        reporter: Optional pre-existing CostReporter to share.

    Raises:
        RuntimeError: If model name cannot be determined from serialized config.
    """

    raise_error: bool = True

    def __init__(
        self,
        budget_usd: float = 0.0,
        window_seconds: Optional[float] = None,
        limit_set: Optional[LimitSet] = None,
        reporter: Optional[CostReporter] = None,
    ):
        if window_seconds is None:
            from concurry.core.constants import RATE_WINDOW_SECONDS

            window_seconds = RATE_WINDOW_SECONDS[slowburn_config.defaults.budget_usd_window]
        if limit_set is not None:
            self.limit_set = limit_set
        else:
            if budget_usd <= 0:
                raise ValueError(
                    "SlowBurnCallbackHandler requires either a positive budget_usd "
                    "or a pre-created limit_set."
                )
            self.limit_set = LimitSet(
                limits=[CostLimit(budget_usd=budget_usd, window=window_seconds)],
                mode="Threads",
                shared=True,
            )
        self.reporter = reporter if reporter is not None else CostReporter()
        self._pending: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        """Extract model name from LangChain's serialized LLM config.

        LangChain passes the serialized LLM as a dict with 'kwargs'
        containing the model configuration.
        """
        kwargs = serialized.get("kwargs", {})
        for key in ("model_name", "model", "model_id"):
            name = kwargs.get(key)
            if isinstance(name, str) and len(name) > 0:
                return name
        name = serialized.get("id", [None])[-1]
        if isinstance(name, str) and len(name) > 0:
            return name
        raise RuntimeError(
            f"SlowBurnCallbackHandler: Could not determine model name from "
            f"serialized LLM config. Keys available: {list(kwargs.keys())}. "
            f"Ensure the LLM model has a 'model_name' or 'model' attribute."
        )

    def _extract_max_tokens(self, serialized: Dict[str, Any]) -> int:
        """Extract max_tokens from LangChain's serialized LLM config."""
        kwargs = serialized.get("kwargs", {})
        for key in ("max_tokens", "max_output_tokens"):
            val = kwargs.get(key)
            if isinstance(val, int) and val > 0:
                return val
        raise RuntimeError(
            f"SlowBurnCallbackHandler: Could not determine max_tokens from "
            f"serialized LLM config. Keys: {list(kwargs.keys())}. "
            f"Set max_tokens on the LLM model."
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Block until budget is available, then allow LLM to proceed.

        Acquires budget from the LimitSet. If budget is exhausted, this
        method blocks (sleeps) until the rate limit window rolls over.
        The acquisition is stored by run_id for update in on_llm_end.
        """
        model_name = self._extract_model_name(serialized)
        max_tokens = self._extract_max_tokens(serialized)

        total_text = " ".join(prompts)
        estimated_input, estimated_output = estimate_input_tokens(total_text, max_tokens)
        estimated_cost = PricingCache.estimate_cost_microdollars(
            model_name,
            estimated_input,
            estimated_output,
        )

        acq = self.limit_set.acquire(requested={DEFAULT_COST_LIMIT_KEY: max(estimated_cost, 1)})

        run_key = str(run_id) if run_id is not None else "default"
        with self._lock:
            self._pending[run_key] = {
                "acq": acq,
                "model_name": model_name,
                "estimated_input": estimated_input,
                "estimated_cost": estimated_cost,
            }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Update budget with actual cost and log to reporter."""
        run_key = str(run_id) if run_id is not None else "default"
        with self._lock:
            entry = self._pending.pop(run_key, None)

        if entry is None:
            raise RuntimeError(
                f"SlowBurnCallbackHandler.on_llm_end: No pending acquisition "
                f"for run_id={run_key}. This means on_llm_start was not called "
                f"or raised an exception."
            )

        acq = entry["acq"]
        model_name = entry["model_name"]
        estimated_input = entry["estimated_input"]

        token_usage = {}
        if response.llm_output is not None:
            token_usage = response.llm_output.get("token_usage", {})

        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")

        if prompt_tokens is not None and completion_tokens is not None:
            actual_cost = PricingCache.estimate_cost_microdollars(
                model_name,
                prompt_tokens,
                completion_tokens,
            )
        else:
            text = ""
            for gen_list in response.generations:
                for gen in gen_list:
                    text += gen.text
            completion_tokens = max(int(len(text) / slowburn_config.defaults.chars_per_token), 1)
            actual_cost = PricingCache.estimate_cost_microdollars(
                model_name,
                estimated_input,
                completion_tokens,
            )

        acq.update(usage={DEFAULT_COST_LIMIT_KEY: max(actual_cost, 1)})

        self.reporter.log_call(
            model=model_name,
            cost_usd=microdollars_to_dollars(actual_cost),
            input_tokens=prompt_tokens if prompt_tokens is not None else estimated_input,
            output_tokens=completion_tokens if completion_tokens is not None else 0,
        )

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Release budget acquisition on error, charging full estimated cost."""
        run_key = str(run_id) if run_id is not None else "default"
        with self._lock:
            entry = self._pending.pop(run_key, None)

        if entry is None:
            return

        acq = entry["acq"]
        estimated_cost = entry["estimated_cost"]
        acq.update(usage={DEFAULT_COST_LIMIT_KEY: estimated_cost})
