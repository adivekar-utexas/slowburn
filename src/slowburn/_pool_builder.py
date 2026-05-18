"""Slot-cascade helpers for build_limit_pool.

Splits the per-endpoint limit construction into small, testable pieces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from concurry import LimitSet, RateLimit, ResourceLimit

from .endpoints import EndpointConfig
from .limits import CostLimit
from .limits_spec import SLOT_TO_LIMIT_KEY, SlowBurnLimits

# ---------------------------------------------------------------------------
# Slot identity
# ---------------------------------------------------------------------------

# The three rate slots that share the same per-window-key handling.
RATE_SLOTS: Tuple[str, str, str] = ("requests", "input_tokens", "output_tokens")


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


def resolve_slot_limits(
    *,
    endpoint_limits: Optional[SlowBurnLimits],
    global_limits: SlowBurnLimits,
    default_limits: SlowBurnLimits,
) -> Tuple[SlowBurnLimits, Dict[str, str]]:
    """Resolve the per-endpoint slot values via replace-slot cascade.

    For each slot:
    - If ``endpoint_limits`` has the slot set (non-``None``), use that.
      Source = ``"endpoint"`` (private; the user's intent is to override).
    - Else if ``global_limits`` has the slot set, use that.
      Source = ``"global"`` (shared across all endpoints that don't override).
    - Else use ``default_limits``. Source = ``"default"`` (shared).

    Returns:
        A two-tuple ``(resolved, sources)`` where:

        - ``resolved`` is a ``SlowBurnLimits`` whose every slot is populated
          (the cascade always lands on at least the library default).
        - ``sources`` maps slot name → ``"endpoint"`` / ``"global"`` /
          ``"default"`` so the limit-pool builder can decide whether the
          slot's limit object should be a private or shared instance.
    """
    sources: Dict[str, str] = {}
    out: Dict[str, Any] = {}
    for slot in ("requests", "input_tokens", "output_tokens", "budget", "concurrency"):
        ep_value = getattr(endpoint_limits, slot, None) if endpoint_limits is not None else None
        if ep_value is not None:
            out[slot] = ep_value
            sources[slot] = "endpoint"
            continue
        gl_value = getattr(global_limits, slot)
        if gl_value is not None:
            out[slot] = gl_value
            sources[slot] = "global"
            continue
        # Library default — always populated.
        out[slot] = getattr(default_limits, slot)
        sources[slot] = "default"
    return SlowBurnLimits.model_construct(**out), sources


# ---------------------------------------------------------------------------
# Sharing cache
# ---------------------------------------------------------------------------


class _SharedLimitCache:
    """Caches limit instances keyed by ``(slot, source, distinguishing_param)``.

    When two endpoints both use the global ``requests`` slot at 60s, they
    must share the *same* ``RateLimit`` instance so the rate enforcement is
    pool-wide. This cache returns the same Python object on subsequent
    requests.

    "Source" is one of ``"global"`` / ``"default"`` / ``"endpoint"``.
    Endpoint-sourced limits never share (each endpoint has its own private
    instance) — we still pass them through here so the call-site code is
    uniform; the cache just returns the value verbatim with a fresh key
    every call.
    """

    def __init__(self) -> None:
        # Keyed by (slot, source, dedup_key); value = the canonical instance.
        self._cache: Dict[Tuple[str, str, str], Any] = {}
        # Counter for endpoint-sourced limits (never sharable; we still want
        # them to round-trip through the same code path, so we issue a unique
        # dedup key for each call).
        self._endpoint_counter = 0

    def share(self, *, slot: str, source: str, instance: Any, dedup_key: str) -> Any:
        """Return the canonical instance for ``(slot, source, dedup_key)``.

        ``source = "endpoint"`` always returns ``instance`` verbatim and
        never caches (private instances can't share).

        Otherwise: if the cache already has an instance for this triple,
        return that; else cache ``instance`` and return it.
        """
        if source == "endpoint":
            self._endpoint_counter += 1
            return instance
        sig = (slot, source, dedup_key)
        if sig in self._cache:
            return self._cache[sig]
        self._cache[sig] = instance
        return instance


# ---------------------------------------------------------------------------
# Per-slot RateLimit construction (with unique-key generation)
# ---------------------------------------------------------------------------


def _make_unique_keys(
    base_key: str,
    rates: List[RateLimit],
) -> List[RateLimit]:
    """Ensure every ``RateLimit`` in the list has a unique ``key``.

    Single-element lists get the canonical ``base_key``. For multi-element
    lists:

    - If the user already supplied distinct, non-``base_key`` keys, keep them.
    - Otherwise, regenerate every key as ``f"{base_key}_{rate.params_signature()}"``.
    - If even that produces duplicates (= two truly-identical rates), raise.
    """
    if len(rates) == 1:
        rate = rates[0]
        if rate.key != base_key:
            return [_rekeyed(rate, base_key)]
        return rates

    user_keys = [r.key for r in rates]
    if len(set(user_keys)) == len(user_keys) and base_key not in user_keys:
        return rates  # User supplied distinct keys; respect them.

    new_rates = [_rekeyed(r, f"{base_key}_{r.params_signature()}") for r in rates]
    new_keys = [r.key for r in new_rates]
    if len(set(new_keys)) != len(new_keys):
        raise ValueError(
            f"Multiple rate limits on slot {base_key!r} have identical "
            f"(capacity, window, algorithm); deduplicate the input or assign "
            f"distinct ``key`` values. Generated keys: {new_keys}"
        )
    return new_rates


def _rekeyed(rate: RateLimit, new_key: str) -> RateLimit:
    """Return a new ``RateLimit`` identical to ``rate`` but with ``key=new_key``."""
    if rate.key == new_key:
        return rate
    return RateLimit(
        key=new_key,
        window=rate.window,
        capacity=rate.capacity,
        algorithm=rate.algorithm,
    )


# ---------------------------------------------------------------------------
# Build a single endpoint's LimitSet
# ---------------------------------------------------------------------------


def build_endpoint_limit_set(
    *,
    endpoint: EndpointConfig,
    global_limits: SlowBurnLimits,
    default_limits: SlowBurnLimits,
    cache: _SharedLimitCache,
    backend: str,
) -> LimitSet:
    """Build one ``LimitSet`` for a single endpoint via slot-cascade.

    Implements:
    - Replace-slot cascade: endpoint > global > library default.
    - Shared instances: when two endpoints both inherit from the global (or
      default) slot, they get the SAME ``RateLimit`` / ``CostLimit`` /
      ``ResourceLimit`` Python object so enforcement is pool-wide.
    - Unique-key generation for multi-window rate slots.

    Stashes a ``_rate_keys`` dict on the LimitSet's ``config`` so the worker
    can populate ``acquire(requested=...)`` / ``acq.update(usage=...)`` under
    every relevant key (one per (slot, window) pair on the LimitSet).
    """
    resolved, sources = resolve_slot_limits(
        endpoint_limits=endpoint.limits,
        global_limits=global_limits,
        default_limits=default_limits,
    )

    limits_list: List[Any] = []
    rate_keys: Dict[str, List[str]] = {}

    # --- Rate slots ---
    for slot in RATE_SLOTS:
        base_key = SLOT_TO_LIMIT_KEY[slot]
        rates: List[RateLimit] = list(getattr(resolved, slot))  # noqa: F841
        rates = _make_unique_keys(base_key, rates)
        # Share each rate instance per (slot, source, key) so global slots
        # are shared across endpoints. Endpoint-sourced rates are private.
        shared_rates: List[RateLimit] = []
        for rate in rates:
            shared_rates.append(
                cache.share(
                    slot=slot,
                    source=sources[slot],
                    instance=rate,
                    dedup_key=rate.key,
                )
            )
        limits_list.extend(shared_rates)
        rate_keys[base_key] = [r.key for r in shared_rates]

    # --- Budget slot ---
    budget_list: List[CostLimit] = list(resolved.budget)
    budget_list = _make_unique_budget_keys(budget_list)
    shared_budgets: List[CostLimit] = []
    for cl in budget_list:
        shared_budgets.append(
            cache.share(
                slot="budget",
                source=sources["budget"],
                instance=cl,
                dedup_key=cl.key,
            )
        )
    limits_list.extend(shared_budgets)
    rate_keys["budget"] = [cl.key for cl in shared_budgets]

    # --- Concurrency slot (single int → ResourceLimit) ---
    capacity = int(resolved.concurrency)
    res_limit = ResourceLimit(key="concurrent_requests", capacity=capacity)
    res_limit = cache.share(
        slot="concurrency",
        source=sources["concurrency"],
        instance=res_limit,
        dedup_key=str(capacity),
    )
    limits_list.append(res_limit)

    # Build LimitSet
    config_dump = endpoint.model_dump()
    config_dump["_rate_keys"] = rate_keys

    return LimitSet(
        limits=limits_list,
        mode=backend,
        shared=True,
        config=config_dump,
    )


def _make_unique_budget_keys(budgets: List[CostLimit]) -> List[CostLimit]:
    """Same idea as ``_make_unique_keys`` but for ``CostLimit``s.

    Single-element list: keep the default key (``"cost_microdollars"``).
    Multi-element list: ensure unique keys via ``params_signature()``.
    """
    if len(budgets) <= 1:
        return budgets
    user_keys = [b.key for b in budgets]
    default_key = SLOT_TO_LIMIT_KEY["budget"]
    if len(set(user_keys)) == len(user_keys) and default_key not in user_keys:
        return budgets
    # Re-key with params_signature — CostLimit subclasses RateLimit so it
    # inherits the same method.
    out: List[CostLimit] = []
    for b in budgets:
        new_key = f"{default_key}_{b.params_signature()}"
        if b.key == new_key:
            out.append(b)
        else:
            out.append(
                CostLimit(
                    budget_usd=b.budget_usd,
                    window=b.window,
                    key=new_key,
                    algorithm=b.algorithm,
                )
            )
    new_keys = [b.key for b in out]
    if len(set(new_keys)) != len(new_keys):
        raise ValueError(
            f"Multiple CostLimits on the budget slot have identical "
            f"(capacity, window, algorithm); deduplicate or assign distinct "
            f"``key`` values. Generated keys: {new_keys}"
        )
    return out


__all__ = [
    "RATE_SLOTS",
    "build_endpoint_limit_set",
    "resolve_slot_limits",
    "_SharedLimitCache",
]
