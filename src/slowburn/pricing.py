"""
PricingCache: Cost estimation and tracking using litellm's pricing database.

Provides pre-call estimation and post-call actual cost extraction with a
fallback chain to handle litellm's known failure modes:

    Tier 1: response._hidden_params["response_cost"]  (litellm auto-calculated)
    Tier 2: litellm.completion_cost(completion_response=response)  (explicit recalc)
    Tier 3: Manual calc from response.usage token counts + cost_per_token rates
    Tier 4: Estimate from response text length + cost_per_token rates

All costs are returned in MICRODOLLARS (int) for CostLimit compatibility.
1 microdollar = $0.000001.  $1.00 = 1,000,000 microdollars.
"""

import json
import logging
import urllib.request
from typing import Any, Dict, Optional, Tuple

import litellm

from .limits import MICRODOLLARS_PER_DOLLAR

logger = logging.getLogger(__name__)

_openrouter_cache: Optional[Dict[str, Dict[str, str]]] = None


class ModelNotFoundError(LookupError):
    """Raised when a model is not in litellm's pricing database.

    This is a hard error by design. SlowBurn refuses to guess pricing
    because silent defaults would make budget tracking meaningless.
    To fix, either use a model that litellm knows about, or register
    custom pricing via ``litellm.register_model()``.
    """


def _fetch_openrouter_pricing() -> Dict[str, Dict[str, str]]:
    """Fetch model pricing from OpenRouter's public API.

    Returns a dict mapping model_id -> {"prompt": str, "completion": str}
    where values are cost-per-token in USD as strings.

    Cached for the lifetime of the process.
    """
    global _openrouter_cache
    if _openrouter_cache is not None:
        return _openrouter_cache

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "SlowBurn/0.1"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        result = {}
        for model in data.get("data", []):
            mid = model.get("id", "")
            pricing = model.get("pricing", {})
            if mid and pricing.get("prompt") is not None:
                result[mid] = pricing
        _openrouter_cache = result
        logger.info(f"Fetched pricing for {len(result)} models from OpenRouter API")
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch OpenRouter pricing: {e}")
        _openrouter_cache = {}
        return {}


class PricingCache:
    """Cost estimation and tracking using litellm's pricing database.

    Lookup order for get_token_costs("openrouter/z-ai/glm-4.5"):
    1. Exact match in litellm.model_cost: "openrouter/z-ai/glm-4.5"
    2. With openrouter/ prefix: "openrouter/z-ai/glm-4.5" (for bare model names)
    3. Strip first prefix (litellm's own logic): "z-ai/glm-4.5"
    4. Strip to base model name: "glm-4.5"
    5. OpenRouter API fallback (for openrouter/ models only)
    """

    @staticmethod
    def get_token_costs(model: str) -> Tuple[float, float]:
        """Get per-token costs in USD for a model.

        Tries multiple name resolution strategies mirroring litellm's own
        lookup logic, plus an OpenRouter API fallback for openrouter/ models.

        Returns:
            (input_cost_per_token, output_cost_per_token) in USD.

        Raises:
            ModelNotFoundError: If the model is not found in any pricing source.
        """
        cost_map = litellm.model_cost

        candidates = [model]

        if model.startswith("openrouter/"):
            stripped_once = model.removeprefix("openrouter/")
            candidates.append(stripped_once)
            parts = stripped_once.split("/")
            if len(parts) > 1:
                candidates.append(parts[-1])
        else:
            candidates.append(f"openrouter/{model}")

        for candidate in candidates:
            model_info = cost_map.get(candidate)
            if model_info is not None:
                input_cost = model_info.get("input_cost_per_token")
                output_cost = model_info.get("output_cost_per_token")
                if input_cost is not None and output_cost is not None:
                    return (input_cost, output_cost)

        if model.startswith("openrouter/"):
            or_model_id = model.removeprefix("openrouter/")
            or_pricing = _fetch_openrouter_pricing()
            if or_model_id in or_pricing:
                pricing = or_pricing[or_model_id]
                try:
                    input_cost = float(pricing["prompt"])
                    output_cost = float(pricing["completion"])
                    if input_cost >= 0 and output_cost >= 0:
                        logger.info(
                            f"Resolved '{model}' via OpenRouter API: "
                            f"${input_cost * 1e6:.2f}/${output_cost * 1e6:.2f} per M tokens"
                        )
                        return (input_cost, output_cost)
                except (ValueError, KeyError, TypeError):
                    pass

        raise ModelNotFoundError(
            f"Model '{model}' not found in litellm's pricing database or OpenRouter API. "
            f"SlowBurn cannot estimate or track costs for unknown models. "
            f"Fix: use a model listed in litellm.model_cost, or register "
            f"custom pricing via litellm.register_model({{'model_name': '{model}', "
            f"'input_cost_per_token': <rate>, 'output_cost_per_token': <rate>}})."
        )

    @staticmethod
    def estimate_cost_microdollars(
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> int:
        """Pre-call cost estimation in microdollars.

        Args:
            model: litellm model name (e.g. "gpt-4o-mini", "openrouter/z-ai/glm-4.5").
            estimated_input_tokens: Estimated input token count.
            estimated_output_tokens: Estimated output token count (typically max_tokens).

        Returns:
            Estimated cost in microdollars (int). Minimum 1 microdollar.

        Raises:
            ModelNotFoundError: If the model is not in any pricing source.
        """
        input_rate, output_rate = PricingCache.get_token_costs(model)
        total_usd = (input_rate * estimated_input_tokens) + (output_rate * estimated_output_tokens)
        return max(int(total_usd * MICRODOLLARS_PER_DOLLAR), 1)

    @staticmethod
    def actual_cost_microdollars(response: Any, model: Optional[str] = None) -> int:
        """Post-call actual cost extraction with tiered fallback.

        Tier 1: response._hidden_params["response_cost"]
        Tier 2: litellm.completion_cost(completion_response=response)
        Tier 3: Manual calc from response.usage + get_token_costs(model)
        Tier 4: Estimate from response text length + get_token_costs(model)

        Tiers 1-2 do not require the model to be in the pricing database
        (litellm already computed the cost). Tiers 3-4 call get_token_costs()
        and will raise ModelNotFoundError if the model is unknown.

        Args:
            response: litellm completion response object.
            model: Model name (used for Tier 3-4 fallback). If None, attempts
                to read ``response.model``.

        Returns:
            Actual cost in microdollars (int). Minimum 1 microdollar.

        Raises:
            ModelNotFoundError: If Tiers 1-2 fail and the model is not in
                any pricing source (needed for Tiers 3-4).
        """
        if model is None:
            model = getattr(response, "model", None)

        # Tier 1: _hidden_params (fastest, most common for non-streaming)
        try:
            cost_usd = response._hidden_params.get("response_cost")
            if cost_usd is not None and cost_usd > 0:
                return max(int(cost_usd * MICRODOLLARS_PER_DOLLAR), 1)
        except (AttributeError, TypeError):
            pass

        # Tier 2: Explicit completion_cost recalculation
        try:
            cost_usd = litellm.completion_cost(completion_response=response)
            if cost_usd is not None and cost_usd > 0:
                return max(int(cost_usd * MICRODOLLARS_PER_DOLLAR), 1)
        except Exception:
            pass

        # Tier 3: Manual from usage object + pricing rates (raises if model unknown)
        try:
            usage = response.usage
            if usage is not None and model is not None:
                input_rate, output_rate = PricingCache.get_token_costs(model)
                cost_usd = (
                    input_rate * usage.prompt_tokens
                    + output_rate * usage.completion_tokens
                )
                if cost_usd > 0:
                    return max(int(cost_usd * MICRODOLLARS_PER_DOLLAR), 1)
        except ModelNotFoundError:
            raise
        except (AttributeError, TypeError):
            pass

        # Tier 4: Estimate from response text length (raises if model unknown)
        try:
            text = response.choices[0].message.content or ""
            est_output_tokens = len(text) // 3
            if model is not None:
                _, output_rate = PricingCache.get_token_costs(model)
                cost_usd = output_rate * est_output_tokens
                return max(int(cost_usd * MICRODOLLARS_PER_DOLLAR), 1)
        except ModelNotFoundError:
            raise
        except (AttributeError, TypeError, IndexError):
            pass

        raise ModelNotFoundError(
            f"Could not determine cost for response. Model: {model!r}. "
            f"Neither litellm's auto-cost nor the pricing database produced a result."
        )
