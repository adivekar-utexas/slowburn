"""Tests for PricingCache: cost estimation and extraction with fallbacks.

``estimate_cost_usd`` and ``actual_cost_usd`` return ``float`` dollars.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slowburn.pricing import ModelNotFoundError, PricingCache

from .conftest import MOCK_MODEL_NAME

# ---------------------------------------------------------------------------
# Helper: build fake litellm response objects
# ---------------------------------------------------------------------------


def _make_response(
    *,
    hidden_cost: float = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    content: str = "Hello world",
    model: str = MOCK_MODEL_NAME,
) -> SimpleNamespace:
    """Build a mock litellm response with configurable fields."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    hidden = {"response_cost": hidden_cost} if hidden_cost is not None else {}
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=model,
        _hidden_params=hidden,
    )


# ===========================================================================
# Tests: get_token_costs
# ===========================================================================


class TestGetTokenCosts:
    """Test PricingCache.get_token_costs model lookups."""

    def test_known_model(self) -> None:
        """gpt-4o-mini should be in litellm's cost map and return real rates."""
        input_rate, output_rate = PricingCache.get_token_costs(MOCK_MODEL_NAME)
        assert input_rate > 0
        assert output_rate > 0
        assert output_rate >= input_rate

    def test_unknown_model_raises(self) -> None:
        """An unknown model should raise ModelNotFoundError, not silently guess."""
        with pytest.raises(ModelNotFoundError, match="totally-fake-model-xyz-999"):
            PricingCache.get_token_costs("totally-fake-model-xyz-999")

    def test_openrouter_prefix_fallback(self) -> None:
        """If 'model' is not found, tries 'openrouter/model' as fallback."""
        input_rate, output_rate = PricingCache.get_token_costs("anthropic/claude-3-haiku")
        assert input_rate > 0
        assert output_rate > 0

    def test_error_message_includes_fix(self) -> None:
        """The error message should tell the user how to register custom pricing."""
        with pytest.raises(ModelNotFoundError, match="litellm.register_model"):
            PricingCache.get_token_costs("my-custom-deployment")

    def test_model_cost_get_raises_propagates(self) -> None:
        """If litellm.model_cost.get itself raises, it should propagate."""
        with patch("slowburn.pricing.litellm.model_cost") as mock_cost:
            mock_cost.get.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                PricingCache.get_token_costs(MOCK_MODEL_NAME)


# ===========================================================================
# Tests: estimate_cost_usd
# ===========================================================================


class TestEstimateCostUsd:
    """Test pre-call cost estimation in dollars."""

    def test_positive_result(self) -> None:
        cost = PricingCache.estimate_cost_usd(MOCK_MODEL_NAME, 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_zero_tokens_returns_zero(self) -> None:
        """Zero tokens should produce zero cost (no minimum-clamp)."""
        cost = PricingCache.estimate_cost_usd(MOCK_MODEL_NAME, 0, 0)
        assert cost == 0.0
        assert isinstance(cost, float)

    def test_scales_with_tokens(self) -> None:
        """More tokens should cost more."""
        small = PricingCache.estimate_cost_usd(MOCK_MODEL_NAME, 100, 50)
        large = PricingCache.estimate_cost_usd(MOCK_MODEL_NAME, 10_000, 5_000)
        assert large > small

    def test_unknown_model_raises(self) -> None:
        """Estimating cost for an unknown model should raise, not guess."""
        with pytest.raises(ModelNotFoundError):
            PricingCache.estimate_cost_usd("fake-model-999", 1000, 500)


# ===========================================================================
# Tests: actual_cost_usd (tiered fallback)
# ===========================================================================


class TestActualCostUsd:
    """Test the tiered fallback chain for post-call cost extraction."""

    def test_tier1_hidden_params(self) -> None:
        """Tier 1: Uses response._hidden_params['response_cost'] when available.

        With dollars-native, the cost passes through unchanged."""
        response = _make_response(hidden_cost=0.05)
        cost = PricingCache.actual_cost_usd(response, model=MOCK_MODEL_NAME)
        assert cost == 0.05

    def test_tier1_skipped_when_zero(self) -> None:
        """Tier 1 is skipped when response_cost is 0.0 (falls to lower tier)."""
        response = _make_response(hidden_cost=0.0)
        cost = PricingCache.actual_cost_usd(response, model=MOCK_MODEL_NAME)
        assert cost > 0

    def test_tier2_completion_cost(self) -> None:
        """Tier 2: Falls to litellm.completion_cost when _hidden_params is empty."""
        response = _make_response()
        response._hidden_params = {}
        with patch("slowburn.pricing.litellm.completion_cost", return_value=0.03):
            cost = PricingCache.actual_cost_usd(response, model=MOCK_MODEL_NAME)
        assert cost == 0.03

    def test_tier3_manual_calc(self) -> None:
        """Tier 3: Manual calculation from usage tokens + pricing rates."""
        response = _make_response(prompt_tokens=1000, completion_tokens=500)
        response._hidden_params = {}
        with patch("slowburn.pricing.litellm.completion_cost", side_effect=Exception("nope")):
            cost = PricingCache.actual_cost_usd(response, model=MOCK_MODEL_NAME)
        assert cost > 0

    def test_tier4_text_length(self) -> None:
        """Tier 4: Estimates from response text length when usage is missing."""
        response = SimpleNamespace(
            _hidden_params={},
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="A" * 300))],
            model=MOCK_MODEL_NAME,
        )
        with patch("slowburn.pricing.litellm.completion_cost", side_effect=Exception("nope")):
            cost = PricingCache.actual_cost_usd(response, model=MOCK_MODEL_NAME)
        assert cost > 0

    def test_unknown_model_tier3_raises(self) -> None:
        """If Tiers 1-2 fail and model is unknown, Tier 3 should raise."""
        response = _make_response(prompt_tokens=100, completion_tokens=50)
        response._hidden_params = {}
        with patch("slowburn.pricing.litellm.completion_cost", side_effect=Exception("nope")):
            with pytest.raises(ModelNotFoundError):
                PricingCache.actual_cost_usd(response, model="fake-model-999")

    def test_all_tiers_fail_raises(self) -> None:
        """When all tiers fail (no response data, no model), should raise."""
        response = SimpleNamespace()
        with patch("slowburn.pricing.litellm.completion_cost", side_effect=Exception("nope")):
            with pytest.raises(ModelNotFoundError):
                PricingCache.actual_cost_usd(response, model=None)

    def test_tier1_success_bypasses_model_check(self) -> None:
        """Tier 1 works even for unknown models (litellm already computed cost)."""
        response = _make_response(hidden_cost=0.01, model="some-unknown-thing")
        cost = PricingCache.actual_cost_usd(response, model="some-unknown-thing")
        assert cost == 0.01

    def test_model_from_response(self) -> None:
        """If model arg is None, reads model from response.model."""
        response = _make_response(hidden_cost=0.01, model=MOCK_MODEL_NAME)
        cost = PricingCache.actual_cost_usd(response, model=None)
        assert cost == 0.01
