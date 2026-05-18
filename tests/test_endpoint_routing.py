"""Tests for multi-endpoint routing via LimitPool + EndpointConfig + resolver."""

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from slowburn import create_llm, EndpointConfig
from slowburn.endpoints import (
    cascade_field,
    coerce_to_endpoint_config,
    passthrough_resolver,
    resolve_concrete_endpoint_config,
)
from slowburn.config import _NO_ARG, is_no_arg

from .conftest import MOCK_MODEL_NAME


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_response(*, in_tok: int = 10, out_tok: int = 5, cost_usd: float = 0.001) -> SimpleNamespace:
    """Build a mock litellm response with usage and an _hidden_params cost."""
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        model=MOCK_MODEL_NAME,
        _hidden_params={"response_cost": cost_usd},
    )


def _patch_acompletion(response: Any) -> Any:
    """Return an AsyncMock-patched litellm.acompletion context manager."""
    return patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock, return_value=response)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no worker spawn)
# ---------------------------------------------------------------------------


class TestEndpointConfigStructure:
    """EndpointConfig accepts known + unknown fields, _NO_ARG defaults."""

    def test_known_fields_default_to_no_arg(self) -> None:
        cfg = EndpointConfig()
        assert is_no_arg(cfg.model)
        assert is_no_arg(cfg.api_key)
        assert is_no_arg(cfg.api_base)
        assert is_no_arg(cfg.max_rpm)
        assert is_no_arg(cfg.budget_usd)

    def test_unknown_fields_preserved_via_extra_allow(self) -> None:
        """Extra fields like account_id and role_arn are kept on the model."""
        cfg = EndpointConfig(
            model="bedrock/us.x",
            account_id="111111111111",
            role_arn="arn:aws:iam::111111111111:role/Foo",
            region="us-east-1",
        )
        dump = cfg.model_dump()
        assert dump["account_id"] == "111111111111"
        assert dump["role_arn"] == "arn:aws:iam::111111111111:role/Foo"
        assert dump["region"] == "us-east-1"
        # And accessible via attribute as well.
        assert cfg.account_id == "111111111111"

    def test_dict_coercion(self) -> None:
        """Plain dicts are accepted and validated into EndpointConfig."""
        ec = coerce_to_endpoint_config({"model": "gpt-4o", "max_rpm": 250, "extra": "x"})
        assert isinstance(ec, EndpointConfig)
        assert ec.model == "gpt-4o"
        assert ec.max_rpm == 250
        assert ec.extra == "x"  # type: ignore[attr-defined]

    def test_dict_coercion_rejects_invalid(self) -> None:
        with pytest.raises(TypeError, match="EndpointConfig or dict"):
            coerce_to_endpoint_config([1, 2, 3])  # type: ignore[arg-type]


class TestCascadeField:
    """The three-level cascade resolver."""

    def test_call_value_wins_over_config_and_default(self) -> None:
        result = cascade_field(
            field="model",
            call_value="per-call-model",
            config_value="endpoint-model",
            worker_default="worker-default",
        )
        assert result == "per-call-model"

    def test_config_wins_when_call_is_no_arg(self) -> None:
        result = cascade_field(
            field="model",
            call_value=_NO_ARG,
            config_value="endpoint-model",
            worker_default="worker-default",
        )
        assert result == "endpoint-model"

    def test_worker_default_when_both_above_are_no_arg(self) -> None:
        result = cascade_field(
            field="model",
            call_value=_NO_ARG,
            config_value=_NO_ARG,
            worker_default="worker-default",
        )
        assert result == "worker-default"

    def test_explicit_none_at_call_layer_wins(self) -> None:
        """An explicit None is a 'provided value' and overrides lower layers."""
        result = cascade_field(
            field="temperature",
            call_value=None,
            config_value=0.5,
            worker_default=0.7,
        )
        assert result is None


class TestResolveConcreteEndpointConfig:
    """resolve_concrete_endpoint_config fills _NO_ARG fields from worker defaults."""

    def test_fills_missing_fields(self) -> None:
        cfg = EndpointConfig(model="custom-model", custom_field="hi")
        defaults = {
            "model": "fallback-model",
            "max_rpm": 500,
            "max_input_tpm": 100_000,
            "max_output_tpm": 50_000,
            "budget_usd": 5.0,
            "window": "daily",
            "rate_limit_algorithm": "GCRA",
            "api_key": "fallback-key",
            "api_base": None,
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 120.0,
            "extra_limits": [],
        }
        resolved = resolve_concrete_endpoint_config(config=cfg, worker_defaults=defaults)
        # model was set explicitly on the config, so it stays
        assert resolved.model == "custom-model"
        # api_key was _NO_ARG, so it falls back to the worker default
        assert resolved.api_key == "fallback-key"
        assert resolved.max_rpm == 500
        # custom field is preserved
        assert resolved.custom_field == "hi"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Backwards compatibility: single-endpoint via bare kwargs still works
# ---------------------------------------------------------------------------


class TestSingleEndpointBackwardsCompat:
    """create_llm() with bare kwargs continues to work after the LimitPool refactor."""

    def test_create_llm_no_endpoints_uses_one_synthetic_endpoint(self) -> None:
        """No `endpoints` arg => internal LimitPool with one LimitSet."""
        llm = create_llm(model=MOCK_MODEL_NAME, max_rpm=100, budget_usd=5.0)
        try:
            assert len(llm.limits.limit_sets) == 1
            ls = llm.limits.limit_sets[0]
            assert ls.config["model"] == MOCK_MODEL_NAME
            assert ls.config["max_rpm"] == 100
            assert ls.config["budget_usd"] == 5.0
        finally:
            llm.stop()

    def test_single_endpoint_call_works(self) -> None:
        """A single mocked call goes through the pool path successfully."""
        with _patch_acompletion(_make_response()):
            llm = create_llm(model=MOCK_MODEL_NAME, budget_usd=5.0)
            try:
                result = llm.call_llm(prompt="hi").result(timeout=5.0)
                assert result == "ok"
                rep = llm.get_reporter().result(timeout=5.0)
                assert rep.num_calls == 1
            finally:
                llm.stop()


# ---------------------------------------------------------------------------
# Multi-endpoint routing
# ---------------------------------------------------------------------------


class TestMultiEndpointRouting:
    """Multi-endpoint pools route calls across LimitSets."""

    def test_pool_has_one_limitset_per_endpoint(self) -> None:
        endpoints = [
            EndpointConfig(model="m-A", endpoint_id="A"),
            EndpointConfig(model="m-B", endpoint_id="B"),
            EndpointConfig(model="m-C", endpoint_id="C"),
        ]
        llm = create_llm(model="default-model", endpoints=endpoints)
        try:
            assert len(llm.limits.limit_sets) == 3
            ids = [ls.config["endpoint_id"] for ls in llm.limits.limit_sets]
            assert ids == ["A", "B", "C"]
        finally:
            llm.stop()

    def test_round_robin_visits_each_endpoint(self) -> None:
        """Round-robin picks A, B, C, A, B, C, ... in order."""
        with _patch_acompletion(_make_response()) as mock_call:
            endpoints = [
                EndpointConfig(model="m-A", endpoint_id="A"),
                EndpointConfig(model="m-B", endpoint_id="B"),
                EndpointConfig(model="m-C", endpoint_id="C"),
            ]
            llm = create_llm(model="default-model", endpoints=endpoints)
            try:
                for _ in range(6):
                    llm.call_llm(prompt="x").result(timeout=5.0)
                models_in_order = [c.kwargs["model"] for c in mock_call.call_args_list]
                # Round-robin over 3 endpoints across 6 calls => each model hit twice.
                assert models_in_order.count("m-A") == 2
                assert models_in_order.count("m-B") == 2
                assert models_in_order.count("m-C") == 2
                # And the ordering is deterministic round-robin.
                assert models_in_order == ["m-A", "m-B", "m-C", "m-A", "m-B", "m-C"]
            finally:
                llm.stop()

    def test_endpoint_id_recorded_in_reporter(self) -> None:
        with _patch_acompletion(_make_response()):
            llm = create_llm(
                model="default",
                endpoints=[
                    EndpointConfig(model="m-A", endpoint_id="A"),
                    EndpointConfig(model="m-B", endpoint_id="B"),
                ],
            )
            try:
                for _ in range(4):
                    llm.call_llm(prompt="x").result(timeout=5.0)
                rep = llm.get_reporter().result(timeout=5.0)
                by_ep = rep.summary_by_endpoint()
                assert by_ep["A"]["calls"] == 2
                assert by_ep["B"]["calls"] == 2
            finally:
                llm.stop()

    def test_dict_endpoints_accepted(self) -> None:
        """Plain dicts are coerced to EndpointConfig."""
        with _patch_acompletion(_make_response()):
            llm = create_llm(
                model="default",
                endpoints=[
                    {"model": "m-A", "endpoint_id": "A"},
                    {"model": "m-B", "endpoint_id": "B"},
                ],
            )
            try:
                llm.call_llm(prompt="x").result(timeout=5.0)
                rep = llm.get_reporter().result(timeout=5.0)
                assert rep.num_calls == 1
            finally:
                llm.stop()

    def test_empty_endpoints_list_raises(self) -> None:
        with pytest.raises(ValueError, match="endpoints=\\[\\]"):
            create_llm(model="default", endpoints=[])


# ---------------------------------------------------------------------------
# Cascade precedence in real calls
# ---------------------------------------------------------------------------


class TestCascadePrecedence:
    """Cascade: per-call > endpoint > worker default for actual litellm kwargs."""

    def test_endpoint_model_wins_over_worker(self) -> None:
        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="worker-default-model",
                endpoints=[EndpointConfig(model="endpoint-model", endpoint_id="A")],
            )
            try:
                llm.call_llm(prompt="x").result(timeout=5.0)
                assert mock_call.call_args.kwargs["model"] == "endpoint-model"
            finally:
                llm.stop()

    def test_per_call_model_wins_over_endpoint(self) -> None:
        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="worker-default-model",
                endpoints=[EndpointConfig(model="endpoint-model", endpoint_id="A")],
            )
            try:
                llm.call_llm(prompt="x", model="per-call-model").result(timeout=5.0)
                assert mock_call.call_args.kwargs["model"] == "per-call-model"
            finally:
                llm.stop()

    def test_worker_default_used_when_endpoint_unset(self) -> None:
        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="worker-default-model",
                endpoints=[EndpointConfig(endpoint_id="A")],
            )
            try:
                llm.call_llm(prompt="x").result(timeout=5.0)
                assert mock_call.call_args.kwargs["model"] == "worker-default-model"
            finally:
                llm.stop()

    def test_per_call_temperature_overrides(self) -> None:
        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(model="default", temperature=0.7)
            try:
                llm.call_llm(prompt="x", temperature=0.0).result(timeout=5.0)
                assert mock_call.call_args.kwargs["temperature"] == 0.0
            finally:
                llm.stop()

    def test_api_base_per_endpoint(self) -> None:
        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="default",
                endpoints=[
                    EndpointConfig(model="m-A", api_base="https://a.example/v1", endpoint_id="A"),
                    EndpointConfig(model="m-B", api_base="https://b.example/v1", endpoint_id="B"),
                ],
            )
            try:
                llm.call_llm(prompt="1").result(timeout=5.0)
                llm.call_llm(prompt="2").result(timeout=5.0)
                api_bases = [c.kwargs.get("api_base") for c in mock_call.call_args_list]
                assert api_bases == ["https://a.example/v1", "https://b.example/v1"]
            finally:
                llm.stop()


# ---------------------------------------------------------------------------
# Resolver augmentation
# ---------------------------------------------------------------------------


class TestEndpointResolver:
    """Resolver is invoked per call with the endpoint dict, augments per-call kwargs."""

    def test_passthrough_resolver_default(self) -> None:
        """The default resolver is a passthrough (no augmentation)."""
        cfg_dict = {"model": "x", "account_id": "123"}
        assert passthrough_resolver(cfg_dict) is cfg_dict

    def test_resolver_receives_endpoint_dict_with_extras(self) -> None:
        """Resolver gets the full model_dump including unknown extras."""
        seen: List[Dict[str, Any]] = []

        def my_resolver(d: Dict[str, Any]) -> Dict[str, Any]:
            seen.append(dict(d))
            return d

        with _patch_acompletion(_make_response()):
            llm = create_llm(
                model="default",
                endpoints=[
                    EndpointConfig(
                        model="m-A",
                        endpoint_id="A",
                        account_id="acct-1",
                        custom_thing="X",
                    ),
                ],
                endpoint_resolver=my_resolver,
            )
            try:
                llm.call_llm(prompt="x").result(timeout=5.0)
                assert len(seen) == 1
                assert seen[0]["account_id"] == "acct-1"
                assert seen[0]["custom_thing"] == "X"
                assert seen[0]["model"] == "m-A"
            finally:
                llm.stop()

    def test_resolver_can_override_api_key_and_litellm_params(self) -> None:
        """Resolver output overrides endpoint config; litellm_params merges."""

        def my_resolver(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **cfg,
                "api_key": f"fresh-{cfg['endpoint_id']}",
                "litellm_params": {
                    **cfg.get("litellm_params", {}),
                    "aws_session_token": f"TOKEN-{cfg['endpoint_id']}",
                },
            }

        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="default",
                endpoints=[
                    EndpointConfig(model="m-A", endpoint_id="A"),
                    EndpointConfig(model="m-B", endpoint_id="B"),
                ],
                endpoint_resolver=my_resolver,
            )
            try:
                llm.call_llm(prompt="1").result(timeout=5.0)
                llm.call_llm(prompt="2").result(timeout=5.0)
                api_keys = [c.kwargs.get("api_key") for c in mock_call.call_args_list]
                tokens = [c.kwargs.get("aws_session_token") for c in mock_call.call_args_list]
                assert api_keys == ["fresh-A", "fresh-B"]
                assert tokens == ["TOKEN-A", "TOKEN-B"]
            finally:
                llm.stop()

    def test_resolver_can_override_model(self) -> None:
        """Resolver may rewrite the model for a request."""

        def my_resolver(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return {**cfg, "model": "resolver-overridden-model"}

        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="worker-default",
                endpoints=[EndpointConfig(model="endpoint-model", endpoint_id="A")],
                endpoint_resolver=my_resolver,
            )
            try:
                llm.call_llm(prompt="x").result(timeout=5.0)
                assert mock_call.call_args.kwargs["model"] == "resolver-overridden-model"
            finally:
                llm.stop()

    def test_per_call_model_overrides_resolver_output(self) -> None:
        """call_llm(model=...) wins even over resolver output."""

        def my_resolver(cfg: Dict[str, Any]) -> Dict[str, Any]:
            return {**cfg, "model": "resolver-model"}

        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="worker-default",
                endpoints=[EndpointConfig(model="endpoint-model", endpoint_id="A")],
                endpoint_resolver=my_resolver,
            )
            try:
                llm.call_llm(prompt="x", model="per-call-model").result(timeout=5.0)
                assert mock_call.call_args.kwargs["model"] == "per-call-model"
            finally:
                llm.stop()


# ---------------------------------------------------------------------------
# Per-endpoint vs global budget
# ---------------------------------------------------------------------------


class TestPerEndpointBudget:
    """Per-endpoint budget_usd overrides the global default for that endpoint only."""

    def test_global_budget_applied_to_endpoints_without_one(self) -> None:
        """An endpoint that does not specify budget_usd inherits the global value."""
        from slowburn.limits import DEFAULT_COST_LIMIT_KEY

        llm = create_llm(
            model="default",
            budget_usd=5.0,
            endpoints=[
                EndpointConfig(model="m-A", endpoint_id="A"),  # no budget_usd
                EndpointConfig(model="m-B", endpoint_id="B", budget_usd=10.0),  # override
            ],
        )
        try:
            # Each endpoint's LimitSet should have a CostLimit
            for ls in llm.limits.limit_sets:
                keys = [getattr(lim, "key", None) for lim in ls.limits]
                assert DEFAULT_COST_LIMIT_KEY in keys
            # Endpoint A inherits 5.0; endpoint B has its own 10.0
            assert llm.limits.limit_sets[0].config["budget_usd"] == 5.0
            assert llm.limits.limit_sets[1].config["budget_usd"] == 10.0
        finally:
            llm.stop()

    def test_inf_budget_skips_cost_limit(self) -> None:
        """budget_usd=inf means no CostLimit is added to the endpoint's LimitSet."""
        from slowburn.limits import DEFAULT_COST_LIMIT_KEY

        llm = create_llm(
            model="default",
            endpoints=[EndpointConfig(model="m-A", endpoint_id="A", budget_usd=float("inf"))],
        )
        try:
            ls = llm.limits.limit_sets[0]
            keys = [getattr(lim, "key", None) for lim in ls.limits]
            assert DEFAULT_COST_LIMIT_KEY not in keys
        finally:
            llm.stop()
