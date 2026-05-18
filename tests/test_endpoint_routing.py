"""Tests for multi-endpoint routing via LimitPool + EndpointConfig + resolver."""

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from slowburn import create_llm
from slowburn.endpoints import EndpointConfig, cascade_field, passthrough_resolver
from slowburn.config import _NO_ARG

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


class TestEndpointConfigStrictness:
    """EndpointConfig is an internal type with strict required fields."""

    def test_extra_fields_preserved(self) -> None:
        """Unknown fields like account_id survive on the model."""
        from concurry import RateLimit

        cfg = EndpointConfig(
            model="bedrock/x",
            api_key="k",
            api_base=None,
            temperature=0.7,
            max_tokens=100,
            timeout=10.0,
            max_request_rate=[RateLimit(key="call_count", window="minutely", capacity=100)],
            max_input_token_rate=[RateLimit(key="input_tokens", window="minutely", capacity=1000)],
            max_output_token_rate=[RateLimit(key="output_tokens", window="minutely", capacity=1000)],
            max_concurrent_requests=10,
            budget_usd=5.0,
            budget_usd_window="daily",
            account_id="111111111111",
        )
        dump = cfg.model_dump()
        assert dump["account_id"] == "111111111111"
        assert cfg.account_id == "111111111111"  # type: ignore[attr-defined]

    def test_missing_required_field_raises(self) -> None:
        """Missing required field on direct construction raises a validation error."""
        # morphic wraps pydantic's ValidationError in a ValueError, so accept either.
        with pytest.raises((ValueError, Exception)) as excinfo:
            EndpointConfig(model="m")
        msg = str(excinfo.value)
        assert "api_key" in msg or "required" in msg.lower()


# ---------------------------------------------------------------------------
# Backwards compatibility: single-endpoint via bare kwargs still works
# ---------------------------------------------------------------------------


class TestSingleEndpointBackwardsCompat:
    """create_llm() with bare kwargs continues to work after the LimitPool refactor."""

    def test_create_llm_no_endpoints_uses_one_synthetic_endpoint(self) -> None:
        """No `endpoints` arg => internal LimitPool with one LimitSet."""
        llm = create_llm(model=MOCK_MODEL_NAME, max_request_rate=100, budget_usd=5.0)
        try:
            assert len(llm.limits.limit_sets) == 1
            ls = llm.limits.limit_sets[0]
            assert ls.config["model"] == MOCK_MODEL_NAME
            # max_request_rate is normalized to a list of one RateLimit.
            request_rates = ls.config["max_request_rate"]
            assert len(request_rates) == 1
            assert request_rates[0]["capacity"] == 100
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
            {"model": "m-A", "endpoint_id": "A"},
            {"model": "m-B", "endpoint_id": "B"},
            {"model": "m-C", "endpoint_id": "C"},
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
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
                {"model": "m-C", "endpoint_id": "C"},
            ]
            llm = create_llm(model="default-model", endpoints=endpoints)
            try:
                for _ in range(6):
                    llm.call_llm(prompt="x").result(timeout=5.0)
                models_in_order = [c.kwargs["model"] for c in mock_call.call_args_list]
                assert models_in_order.count("m-A") == 2
                assert models_in_order.count("m-B") == 2
                assert models_in_order.count("m-C") == 2
                assert models_in_order == ["m-A", "m-B", "m-C", "m-A", "m-B", "m-C"]
            finally:
                llm.stop()

    def test_endpoint_id_recorded_in_reporter(self) -> None:
        with _patch_acompletion(_make_response()):
            llm = create_llm(
                model="default",
                endpoints=[
                    {"model": "m-A", "endpoint_id": "A"},
                    {"model": "m-B", "endpoint_id": "B"},
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

    def test_empty_endpoints_list_raises(self) -> None:
        with pytest.raises(ValueError, match=r"endpoints=\[\]"):
            create_llm(model="default", endpoints=[])

    def test_endpointconfig_instance_rejected(self) -> None:
        """EndpointConfig instances must NOT be passed; only plain dicts."""
        from concurry import RateLimit

        cfg = EndpointConfig(
            model="m-A", api_key="", api_base=None, temperature=0.7, max_tokens=100,
            timeout=10.0,
            max_request_rate=[RateLimit(key="call_count", window="minutely", capacity=100)],
            max_input_token_rate=[RateLimit(key="input_tokens", window="minutely", capacity=1000)],
            max_output_token_rate=[RateLimit(key="output_tokens", window="minutely", capacity=1000)],
            max_concurrent_requests=10, budget_usd=5.0, budget_usd_window="daily",
        )
        # @validate rejects at the pydantic layer with ValidationError; our
        # explicit TypeError raise inside build_limit_pool is the fallback path.
        with pytest.raises(Exception) as excinfo:
            create_llm(model="default", endpoints=[cfg])  # type: ignore[list-item]
        msg = str(excinfo.value).lower()
        assert "dict" in msg


# ---------------------------------------------------------------------------
# Cascade precedence in real calls
# ---------------------------------------------------------------------------


class TestCascadePrecedence:
    """Cascade: per-call > endpoint > worker default for actual litellm kwargs."""

    def test_endpoint_model_wins_over_worker(self) -> None:
        with _patch_acompletion(_make_response()) as mock_call:
            llm = create_llm(
                model="worker-default-model",
                endpoints=[{"model": "endpoint-model", "endpoint_id": "A"}],
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
                endpoints=[{"model": "endpoint-model", "endpoint_id": "A"}],
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
                endpoints=[{"endpoint_id": "A"}],
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
                    {"model": "m-A", "api_base": "https://a.example/v1", "endpoint_id": "A"},
                    {"model": "m-B", "api_base": "https://b.example/v1", "endpoint_id": "B"},
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
                    {
                        "model": "m-A",
                        "endpoint_id": "A",
                        "account_id": "acct-1",
                        "custom_thing": "X",
                    },
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
                    {"model": "m-A", "endpoint_id": "A"},
                    {"model": "m-B", "endpoint_id": "B"},
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
                endpoints=[{"model": "endpoint-model", "endpoint_id": "A"}],
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
                endpoints=[{"model": "endpoint-model", "endpoint_id": "A"}],
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
                {"model": "m-A", "endpoint_id": "A"},  # no budget_usd
                {"model": "m-B", "endpoint_id": "B", "budget_usd": 10.0},  # override
            ],
        )
        try:
            for ls in llm.limits.limit_sets:
                keys = [getattr(lim, "key", None) for lim in ls.limits]
                assert DEFAULT_COST_LIMIT_KEY in keys
            assert llm.limits.limit_sets[0].config["budget_usd"] == 5.0
            assert llm.limits.limit_sets[1].config["budget_usd"] == 10.0
        finally:
            llm.stop()

    def test_inf_budget_skips_cost_limit(self) -> None:
        """budget_usd=inf means no CostLimit is added to the endpoint's LimitSet."""
        from slowburn.limits import DEFAULT_COST_LIMIT_KEY

        llm = create_llm(
            model="default",
            endpoints=[{"model": "m-A", "endpoint_id": "A", "budget_usd": float("inf")}],
        )
        try:
            ls = llm.limits.limit_sets[0]
            keys = [getattr(lim, "key", None) for lim in ls.limits]
            assert DEFAULT_COST_LIMIT_KEY not in keys
        finally:
            llm.stop()


# ---------------------------------------------------------------------------
# Shared limit object semantics (the key new behavior)
# ---------------------------------------------------------------------------


class TestSharedLimitObjects:
    """Endpoints that don't override a limit field share a single Limit instance.

    Endpoints that do override get their own private Limit instance.
    """

    def test_unset_endpoints_share_one_call_limit_instance(self) -> None:
        """All endpoints inheriting max_request_rate share the same RateLimit object."""
        llm = create_llm(
            model="default",
            max_request_rate=300,
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
                {"model": "m-C", "endpoint_id": "C"},
            ],
        )
        try:
            call_limits = []
            for ls in llm.limits.limit_sets:
                for lim in ls.limits:
                    if getattr(lim, "key", None) == "call_count":
                        call_limits.append(lim)
                        break
            assert len(call_limits) == 3
            # All three are the SAME instance (object identity).
            assert call_limits[0] is call_limits[1]
            assert call_limits[1] is call_limits[2]
            # And the shared RateLimit has the global capacity.
            assert call_limits[0].capacity == 300
        finally:
            llm.stop()

    def test_overriding_endpoint_gets_private_call_limit(self) -> None:
        """An endpoint that sets max_request_rate gets its own RateLimit; the rest share."""
        llm = create_llm(
            model="default",
            max_request_rate=300,
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},  # inherits 300
                {"model": "m-B", "endpoint_id": "B", "max_request_rate": 1000},  # override
                {"model": "m-C", "endpoint_id": "C"},  # inherits 300
            ],
        )
        try:
            call_limits = []
            for ls in llm.limits.limit_sets:
                for lim in ls.limits:
                    if getattr(lim, "key", None) == "call_count":
                        call_limits.append(lim)
                        break
            # A and C share the global; B has its own.
            assert call_limits[0] is call_limits[2]
            assert call_limits[0] is not call_limits[1]
            assert call_limits[0].capacity == 300
            assert call_limits[1].capacity == 1000
        finally:
            llm.stop()

    def test_unset_endpoints_share_one_resource_limit(self) -> None:
        """max_concurrent_requests inheritance shares a single ResourceLimit."""
        llm = create_llm(
            model="default",
            max_concurrent_requests=50,
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
            ],
        )
        try:
            res_limits = []
            for ls in llm.limits.limit_sets:
                for lim in ls.limits:
                    if getattr(lim, "key", None) == "concurrent_requests":
                        res_limits.append(lim)
                        break
            assert len(res_limits) == 2
            assert res_limits[0] is res_limits[1]
            assert res_limits[0].capacity == 50
        finally:
            llm.stop()

    def test_overriding_endpoint_gets_private_resource_limit(self) -> None:
        """An endpoint that sets max_concurrent_requests gets its own ResourceLimit."""
        llm = create_llm(
            model="default",
            max_concurrent_requests=50,
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B", "max_concurrent_requests": 200},
            ],
        )
        try:
            res_limits = []
            for ls in llm.limits.limit_sets:
                for lim in ls.limits:
                    if getattr(lim, "key", None) == "concurrent_requests":
                        res_limits.append(lim)
                        break
            assert res_limits[0] is not res_limits[1]
            assert res_limits[0].capacity == 50
            assert res_limits[1].capacity == 200
        finally:
            llm.stop()

    def test_unset_endpoints_share_one_cost_limit(self) -> None:
        """Endpoints inheriting budget_usd share a single CostLimit instance."""
        llm = create_llm(
            model="default",
            budget_usd=5.0,
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
            ],
        )
        try:
            from slowburn.limits import DEFAULT_COST_LIMIT_KEY

            cost_limits = []
            for ls in llm.limits.limit_sets:
                for lim in ls.limits:
                    if getattr(lim, "key", None) == DEFAULT_COST_LIMIT_KEY:
                        cost_limits.append(lim)
                        break
            assert len(cost_limits) == 2
            assert cost_limits[0] is cost_limits[1]
        finally:
            llm.stop()

    def test_overriding_endpoint_gets_private_cost_limit(self) -> None:
        """An endpoint that sets budget_usd gets its own CostLimit."""
        from slowburn.limits import DEFAULT_COST_LIMIT_KEY

        llm = create_llm(
            model="default",
            budget_usd=5.0,
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B", "budget_usd": 10.0},
            ],
        )
        try:
            cost_limits = []
            for ls in llm.limits.limit_sets:
                for lim in ls.limits:
                    if getattr(lim, "key", None) == DEFAULT_COST_LIMIT_KEY:
                        cost_limits.append(lim)
                        break
            assert cost_limits[0] is not cost_limits[1]
        finally:
            llm.stop()


# ---------------------------------------------------------------------------
# New rate-shape tests: int / RateLimit / dict / list polymorphism
# ---------------------------------------------------------------------------


class TestRateInputShapes:
    """The three rate dimensions accept int / RateLimit / dict / list."""

    def test_int_uses_default_window(self) -> None:
        """``max_request_rate=300`` -> one RateLimit @ Minutely (default)."""
        llm = create_llm(model=MOCK_MODEL_NAME, max_request_rate=300)
        try:
            ls = llm.limits.limit_sets[0]
            call_limits = [lim for lim in ls.limits if getattr(lim, "key", None) == "call_count"]
            assert len(call_limits) == 1
            assert call_limits[0].capacity == 300
            assert call_limits[0].window_seconds == 60.0
        finally:
            llm.stop()

    def test_ratelimit_passthrough(self) -> None:
        """A bare RateLimit instance is used as-is (key gets normalized)."""
        from concurry import RateLimit, RateWindow

        llm = create_llm(
            model=MOCK_MODEL_NAME,
            max_request_rate=RateLimit(key="custom", window=RateWindow.Hourly, capacity=5000),
        )
        try:
            ls = llm.limits.limit_sets[0]
            call_limits = [lim for lim in ls.limits if getattr(lim, "key", None) == "call_count"]
            assert len(call_limits) == 1
            assert call_limits[0].capacity == 5000
            assert call_limits[0].window_seconds == 3600.0
        finally:
            llm.stop()

    def test_dict_validated_into_ratelimit(self) -> None:
        """A dict is validated into a RateLimit (key auto-injected if missing)."""
        llm = create_llm(
            model=MOCK_MODEL_NAME,
            max_request_rate={"capacity": 50, "window": "hourly"},
        )
        try:
            ls = llm.limits.limit_sets[0]
            call_limits = [lim for lim in ls.limits if getattr(lim, "key", None) == "call_count"]
            assert len(call_limits) == 1
            assert call_limits[0].capacity == 50
            assert call_limits[0].window_seconds == 3600.0
        finally:
            llm.stop()

    def test_list_of_rates_creates_unique_keys(self) -> None:
        """A list of rates produces unique keys derived from each rate's params."""
        llm = create_llm(
            model=MOCK_MODEL_NAME,
            max_request_rate=[300, {"capacity": 50000, "window": "daily"}],
        )
        try:
            ls = llm.limits.limit_sets[0]
            call_keys = {
                lim.key
                for lim in ls.limits
                if isinstance(getattr(lim, "key", None), str) and lim.key.startswith("call_count")
            }
            # Keys are ``call_count_{rate.params_signature()}``; the suffix
            # encodes ``cap`` / ``window_seconds`` / first-3-letters-of-algo.
            assert len(call_keys) == 2
            # Both should embed the base name.
            assert all(k.startswith("call_count_c") for k in call_keys)
            # _rate_keys metadata records BOTH keys under the base name.
            assert set(ls.config["_rate_keys"]["call_count"]) == call_keys
        finally:
            llm.stop()

    def test_user_supplied_unique_keys_respected(self) -> None:
        """When the user passes RateLimits with distinct non-default keys, keep them."""
        from concurry import RateLimit

        llm = create_llm(
            model=MOCK_MODEL_NAME,
            max_request_rate=[
                RateLimit(key="rps_window", window=60, capacity=300),
                RateLimit(key="daily_burst", window="daily", capacity=50_000),
            ],
        )
        try:
            ls = llm.limits.limit_sets[0]
            # The user's chosen keys survive verbatim.
            assert set(ls.config["_rate_keys"]["call_count"]) == {"rps_window", "daily_burst"}
        finally:
            llm.stop()

    def test_same_window_different_capacity_disambiguates(self) -> None:
        """Two rates with the same window but different capacities still get unique keys."""
        llm = create_llm(
            model=MOCK_MODEL_NAME,
            max_request_rate=[300, {"capacity": 600, "window": "minutely"}],
        )
        try:
            ls = llm.limits.limit_sets[0]
            keys = ls.config["_rate_keys"]["call_count"]
            assert len(set(keys)) == 2  # two unique keys
        finally:
            llm.stop()

    def test_truly_identical_rates_rejected(self) -> None:
        """Two rates that are byte-identical should raise (unwinnable collision)."""
        with pytest.raises(ValueError, match="identical"):
            create_llm(model=MOCK_MODEL_NAME, max_request_rate=[300, 300])


class TestSharedAcrossWindows:
    """Shared-instance caching is keyed by (dimension, window_seconds)."""

    def test_two_windows_per_dim_share_per_window(self) -> None:
        """Two endpoints inheriting a multi-window dimension share both RateLimits."""
        llm = create_llm(
            model="default",
            max_request_rate=[300, {"capacity": 50000, "window": "daily"}],
            endpoints=[
                {"endpoint_id": "A"},
                {"endpoint_id": "B"},
            ],
        )
        try:
            # Collect all call_count_* RateLimits per LimitSet.
            per_ls: List[Dict[str, Any]] = []
            for ls in llm.limits.limit_sets:
                d: Dict[str, Any] = {}
                for lim in ls.limits:
                    if isinstance(getattr(lim, "key", None), str) and lim.key.startswith("call_count"):
                        d[lim.key] = lim
                per_ls.append(d)
            # Two distinct keys per LimitSet (params-signature derived).
            assert len(per_ls[0]) == 2
            assert per_ls[0].keys() == per_ls[1].keys()
            for k in per_ls[0]:
                # Both endpoints share the same RateLimit instance per key.
                assert per_ls[0][k] is per_ls[1][k]
            keys = list(per_ls[0].keys())
            # ... but the two windows are distinct from each other.
            assert per_ls[0][keys[0]] is not per_ls[0][keys[1]]
        finally:
            llm.stop()

    def test_multi_window_call_charges_all_keys(self) -> None:
        """A successful call must charge ALL keys for a multi-window dimension."""
        from concurry import RateLimit

        # Use TINY capacities so we can detect that multiple keys advanced.
        # Specifically: a 1m capacity of 100 + a 1h capacity of 100. After a
        # call charging 50 input tokens, both instances should report 50
        # consumed.
        with _patch_acompletion(_make_response(in_tok=50, out_tok=50, cost_usd=0.0001)):
            llm = create_llm(
                model=MOCK_MODEL_NAME,
                max_input_token_rate=[
                    RateLimit(key="input_tokens", window="minutely", capacity=10_000),
                    RateLimit(key="input_tokens_hourly_explicit", window="hourly", capacity=20_000),
                ],
            )
            try:
                llm.call_llm(prompt="x").result(timeout=5.0)
                ls = llm.limits.limit_sets[0]
                # Both input-token RateLimits should have advanced past zero
                # available capacity (i.e., they were charged).
                input_rates = [
                    lim
                    for lim in ls.limits
                    if isinstance(getattr(lim, "key", None), str) and "input_tokens" in lim.key
                ]
                # Two RateLimits, both should have current usage > 0.
                assert len(input_rates) == 2
                for rl in input_rates:
                    stats = rl.get_stats()
                    # available_tokens drops below capacity once we charge.
                    available = stats.get("available_tokens")
                    if available is not None:
                        assert available < rl.capacity, f"{rl.key}: not charged ({stats})"
            finally:
                llm.stop()
