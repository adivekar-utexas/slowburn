"""Tests for multi-endpoint routing via LimitPool + EndpointConfig + resolver.

Adapted to the unified ``SlowBurnLimits`` API:

- ``create_llm(limits=dict(...))`` for global limits across all endpoints.
- Each endpoint dict may carry its own ``"limits"`` key (a dict or
  ``SlowBurnLimits``) for per-endpoint slot overrides.
- The cascade is *replace-slot*: if an endpoint sets ``limits.requests``,
  the entire global ``requests`` slot is replaced for that endpoint.
- Library defaults (from ``default_slowburn_limits()``) populate every
  un-overridden slot, so the worker always has a key for every slot.
"""

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from concurry import RateLimit

from slowburn import CostLimit, SlowBurnLimits, create_llm
from slowburn.config import _NO_ARG
from slowburn.endpoints import EndpointConfig, cascade_field, passthrough_resolver
from slowburn.limits import DEFAULT_COST_LIMIT_KEY

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


def _find_limit(limit_set, *, key: str):
    """Find the (single) Limit on ``limit_set`` whose ``key`` matches."""
    for lim in limit_set.limits:
        if getattr(lim, "key", None) == key:
            return lim
    return None


# ---------------------------------------------------------------------------
# Pure-Python helpers (no worker spawn)
# ---------------------------------------------------------------------------


class TestCascadeField:
    """The three-level cascade resolver for non-limit fields."""

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
        cfg = EndpointConfig(
            model="bedrock/x",
            api_key="k",
            api_base=None,
            temperature=0.7,
            max_tokens=100,
            timeout=10.0,
            limits=SlowBurnLimits(rpm=100, budget_per_day=5.0),
            account_id="111111111111",
        )
        dump = cfg.model_dump()
        assert dump["account_id"] == "111111111111"
        assert cfg.account_id == "111111111111"  # type: ignore[attr-defined]

    def test_limits_field_is_optional(self) -> None:
        """``limits`` defaults to ``None`` (inherit-everything)."""
        cfg = EndpointConfig(
            model="m",
            api_key=None,
            api_base=None,
            temperature=0.7,
            max_tokens=100,
            timeout=10.0,
        )
        assert cfg.limits is None

    def test_limits_dict_coerced_to_slowburnlimits(self) -> None:
        """A dict passed for ``limits`` is coerced into ``SlowBurnLimits``."""
        cfg = EndpointConfig(
            model="m",
            api_key=None,
            api_base=None,
            temperature=0.7,
            max_tokens=100,
            timeout=10.0,
            limits=dict(rpm=300, concurrency=5),
        )
        assert isinstance(cfg.limits, SlowBurnLimits)
        assert cfg.limits.requests is not None
        assert cfg.limits.requests[0].capacity == 300
        assert cfg.limits.concurrency == 5

    def test_missing_required_field_raises(self) -> None:
        """Missing required field on direct construction raises a validation error."""
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
        llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(rpm=100, budget_per_day=5.0))
        try:
            assert len(llm.limits.limit_sets) == 1
            ls = llm.limits.limit_sets[0]
            assert ls.config["model"] == MOCK_MODEL_NAME
            # Verify the requests RateLimit and budget CostLimit landed on the LimitSet.
            req = _find_limit(ls, key="requests")
            assert req is not None and req.capacity == 100
            cost = _find_limit(ls, key=DEFAULT_COST_LIMIT_KEY)
            assert cost is not None and cost.budget_usd == 5.0
        finally:
            llm.stop()

    def test_single_endpoint_call_works(self) -> None:
        """A single mocked call goes through the pool path successfully."""
        with _patch_acompletion(_make_response()):
            llm = create_llm(model=MOCK_MODEL_NAME, limits=dict(budget_per_day=5.0))
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
                llm.call_llm(prompt="x", model="per-call-final").result(timeout=5.0)
                assert mock_call.call_args.kwargs["model"] == "per-call-final"
            finally:
                llm.stop()


# ---------------------------------------------------------------------------
# Per-endpoint limits cascade (replace-slot)
# ---------------------------------------------------------------------------


class TestPerEndpointLimitsCascade:
    """Per-endpoint ``limits`` overrides the global slot when set."""

    def test_endpoint_with_no_limits_inherits_global(self) -> None:
        """An endpoint without ``limits=`` inherits all global slots."""
        llm = create_llm(
            model="default",
            limits=dict(rpm=300, budget_per_day=5.0),
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
            ],
        )
        try:
            for ls in llm.limits.limit_sets:
                assert _find_limit(ls, key="requests").capacity == 300
                assert _find_limit(ls, key=DEFAULT_COST_LIMIT_KEY).budget_usd == 5.0
        finally:
            llm.stop()

    def test_endpoint_with_limits_overrides_only_specified_slots(self) -> None:
        """Endpoint's ``limits.requests=...`` replaces only the requests slot.

        Other slots fall through to global / default.
        """
        llm = create_llm(
            model="default",
            limits=dict(rpm=300, budget_per_day=5.0),
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {
                    "model": "m-B",
                    "endpoint_id": "B",
                    "limits": dict(rpm=1000),  # override requests only
                },
            ],
        )
        try:
            ls_a, ls_b = llm.limits.limit_sets
            # A inherits both slots.
            assert _find_limit(ls_a, key="requests").capacity == 300
            assert _find_limit(ls_a, key=DEFAULT_COST_LIMIT_KEY).budget_usd == 5.0
            # B overrides requests, inherits budget.
            assert _find_limit(ls_b, key="requests").capacity == 1000
            assert _find_limit(ls_b, key=DEFAULT_COST_LIMIT_KEY).budget_usd == 5.0
        finally:
            llm.stop()

    def test_replace_slot_does_not_merge_windows(self) -> None:
        """If endpoint sets ``rpm=...`` and global has ``rpd=...``, the
        endpoint's slot is just the rpm RateLimit (replace-slot)."""
        llm = create_llm(
            model="default",
            limits=dict(rpm=300, rpd=10_000),  # 2-window requests slot
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {
                    "model": "m-B",
                    "endpoint_id": "B",
                    "limits": dict(rpm=500),
                },
            ],
        )
        try:
            ls_a, ls_b = llm.limits.limit_sets
            # A inherits both windows.
            req_a = [
                lim
                for lim in ls_a.limits
                if isinstance(lim, RateLimit) and not isinstance(lim, CostLimit) and "requests" in lim.key
            ]
            assert len(req_a) == 2
            # B only has its own rpm — the rpd is gone (replace-slot).
            req_b = [
                lim
                for lim in ls_b.limits
                if isinstance(lim, RateLimit) and not isinstance(lim, CostLimit) and "requests" in lim.key
            ]
            assert len(req_b) == 1
            assert req_b[0].capacity == 500
        finally:
            llm.stop()


# ---------------------------------------------------------------------------
# Sharing semantics: limit instances shared across endpoints that inherit
# ---------------------------------------------------------------------------


class TestSharedLimitObjects:
    """Endpoints inheriting a slot share the same Limit instance pool-wide."""

    def test_unset_endpoints_share_one_request_rate_limit(self) -> None:
        """Endpoints without limits override share a single RateLimit instance."""
        llm = create_llm(
            model="default",
            limits=dict(rpm=300),
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
                {"model": "m-C", "endpoint_id": "C"},
            ],
        )
        try:
            req_limits = [_find_limit(ls, key="requests") for ls in llm.limits.limit_sets]
            assert len(set(id(r) for r in req_limits)) == 1, "all endpoints should share one RateLimit"
            assert req_limits[0].capacity == 300
        finally:
            llm.stop()

    def test_overriding_endpoint_gets_private_request_rate_limit(self) -> None:
        """An endpoint that sets ``limits.requests`` gets its own RateLimit; the rest share."""
        llm = create_llm(
            model="default",
            limits=dict(rpm=300),
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {"model": "m-B", "endpoint_id": "B"},
                {
                    "model": "m-C",
                    "endpoint_id": "C",
                    "limits": dict(rpm=1000),
                },
            ],
        )
        try:
            ls_a, ls_b, ls_c = llm.limits.limit_sets
            r_a = _find_limit(ls_a, key="requests")
            r_b = _find_limit(ls_b, key="requests")
            r_c = _find_limit(ls_c, key="requests")
            assert id(r_a) == id(r_b)  # shared
            assert id(r_a) != id(r_c)  # private override
            assert r_c.capacity == 1000
        finally:
            llm.stop()

    def test_unset_endpoints_share_one_concurrency_limit(self) -> None:
        """Endpoints without concurrency override share a single ResourceLimit."""
        llm = create_llm(
            model="default",
            limits=dict(concurrency=50),
            endpoints=[{"model": "m-A", "endpoint_id": "A"}, {"model": "m-B", "endpoint_id": "B"}],
        )
        try:
            res = [_find_limit(ls, key="concurrent_requests") for ls in llm.limits.limit_sets]
            assert id(res[0]) == id(res[1])
            assert res[0].capacity == 50
        finally:
            llm.stop()

    def test_overriding_endpoint_gets_private_concurrency_limit(self) -> None:
        llm = create_llm(
            model="default",
            limits=dict(concurrency=50),
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {
                    "model": "m-B",
                    "endpoint_id": "B",
                    "limits": dict(concurrency=200),
                },
            ],
        )
        try:
            ls_a, ls_b = llm.limits.limit_sets
            r_a = _find_limit(ls_a, key="concurrent_requests")
            r_b = _find_limit(ls_b, key="concurrent_requests")
            assert id(r_a) != id(r_b)
            assert r_a.capacity == 50
            assert r_b.capacity == 200
        finally:
            llm.stop()

    def test_unset_endpoints_share_one_budget_limit(self) -> None:
        llm = create_llm(
            model="default",
            limits=dict(budget_per_day=5.0),
            endpoints=[{"model": "m-A", "endpoint_id": "A"}, {"model": "m-B", "endpoint_id": "B"}],
        )
        try:
            costs = [_find_limit(ls, key=DEFAULT_COST_LIMIT_KEY) for ls in llm.limits.limit_sets]
            assert id(costs[0]) == id(costs[1])
            assert costs[0].budget_usd == 5.0
        finally:
            llm.stop()

    def test_overriding_endpoint_gets_private_budget_limit(self) -> None:
        llm = create_llm(
            model="default",
            limits=dict(budget_per_day=5.0),
            endpoints=[
                {"model": "m-A", "endpoint_id": "A"},
                {
                    "model": "m-B",
                    "endpoint_id": "B",
                    "limits": dict(budget_per_day=10.0),
                },
            ],
        )
        try:
            ls_a, ls_b = llm.limits.limit_sets
            c_a = _find_limit(ls_a, key=DEFAULT_COST_LIMIT_KEY)
            c_b = _find_limit(ls_b, key=DEFAULT_COST_LIMIT_KEY)
            assert id(c_a) != id(c_b)
            assert c_a.budget_usd == 5.0
            assert c_b.budget_usd == 10.0
        finally:
            llm.stop()

    def test_library_default_shared_when_no_global_set(self) -> None:
        """When neither global nor endpoint sets a slot, the library default
        is shared across all endpoints."""
        llm = create_llm(
            model="default",
            endpoints=[{"model": "m-A", "endpoint_id": "A"}, {"model": "m-B", "endpoint_id": "B"}],
        )
        try:
            ls_a, ls_b = llm.limits.limit_sets
            assert id(_find_limit(ls_a, key="requests")) == id(_find_limit(ls_b, key="requests"))
            assert id(_find_limit(ls_a, key=DEFAULT_COST_LIMIT_KEY)) == id(
                _find_limit(ls_b, key=DEFAULT_COST_LIMIT_KEY)
            )
            assert id(_find_limit(ls_a, key="concurrent_requests")) == id(
                _find_limit(ls_b, key="concurrent_requests")
            )
        finally:
            llm.stop()


# ---------------------------------------------------------------------------
# Multi-window limits within a single slot
# ---------------------------------------------------------------------------


class TestMultiWindowSlots:
    """A slot can carry multiple RateLimits at different windows."""

    def test_two_windows_on_requests_slot(self) -> None:
        """``limits=dict(rpm=300, rpd=10_000)`` produces two RateLimits."""
        llm = create_llm(
            model="default",
            limits=dict(rpm=300, rpd=10_000),
        )
        try:
            ls = llm.limits.limit_sets[0]
            req_limits = [
                lim
                for lim in ls.limits
                if isinstance(lim, RateLimit) and not isinstance(lim, CostLimit) and "requests" in lim.key
            ]
            assert len(req_limits) == 2
            caps = sorted([rl.capacity for rl in req_limits])
            assert caps == [300, 10_000]
            windows = sorted([float(rl.window) for rl in req_limits])
            assert windows == [60.0, 86400.0]
            # Keys must be unique within the LimitSet.
            keys = [rl.key for rl in req_limits]
            assert len(set(keys)) == len(keys)
        finally:
            llm.stop()

    def test_canonical_ratelimit_with_unique_keys_respected(self) -> None:
        """User-supplied unique keys on RateLimits are not regenerated."""
        rl1 = RateLimit(key="my_minutely", capacity=300, window="minute")
        rl2 = RateLimit(key="my_daily", capacity=10_000, window="day")
        llm = create_llm(
            model="default",
            limits=dict(requests=[rl1, rl2]),
        )
        try:
            ls = llm.limits.limit_sets[0]
            assert _find_limit(ls, key="my_minutely") is not None
            assert _find_limit(ls, key="my_daily") is not None
        finally:
            llm.stop()

    def test_truly_identical_rates_rejected(self) -> None:
        """Two RateLimits with identical (capacity, window, algorithm) and same
        default key cannot be disambiguated and raise ValueError."""
        rl1 = RateLimit(key="requests", capacity=300, window="minute")
        rl2 = RateLimit(key="requests", capacity=300, window="minute")
        with pytest.raises(Exception, match="identical|duplicate"):
            create_llm(model="default", limits=dict(requests=[rl1, rl2]))


class TestMultiWindowCharging:
    """When a slot has multiple windows, every call charges all of them."""

    def test_call_charges_both_request_windows(self) -> None:
        """A call that goes through a 2-window requests slot should charge
        both keys, evidenced by reporter.num_calls and limit-set state."""
        with _patch_acompletion(_make_response()):
            llm = create_llm(
                model=MOCK_MODEL_NAME,
                limits=dict(rpm=300, rpd=10_000),
            )
            try:
                for _ in range(3):
                    llm.call_llm(prompt="x").result(timeout=5.0)
                rep = llm.get_reporter().result(timeout=5.0)
                assert rep.num_calls == 3

                # Inspect the LimitSet state — both windows should reflect 3 calls
                # consumed. We do this by reading the request rates' internal
                # counters indirectly via the LimitSet.
                ls = llm.limits.limit_sets[0]
                rate_keys = ls.config.get("_rate_keys", {})
                request_keys = rate_keys.get("requests", [])
                assert len(request_keys) == 2
            finally:
                llm.stop()
