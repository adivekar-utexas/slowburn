"""
SlowBurnLimits: a unified limits container for ``create_llm`` and per-endpoint
configuration.

Five slots:

- ``requests`` — a list of :class:`concurry.RateLimit`, keyed ``"requests"``.
- ``input_tokens`` — a list of :class:`concurry.RateLimit`, keyed ``"input_tokens"``.
- ``output_tokens`` — a list of :class:`concurry.RateLimit`, keyed ``"output_tokens"``.
- ``budget`` — a list of :class:`slowburn.CostLimit`, keyed ``"cost_usd"``.
- ``concurrency`` — a single ``int`` (the capacity of a :class:`concurry.ResourceLimit`).

Each slot is *optional*. ``None`` means "inherit" — at the global level, ``None``
falls through to ``slowburn_config.defaults.default_limits``; at the per-endpoint
level, ``None`` falls through to the global ``limits=`` passed to ``create_llm``,
and from there to the library default.

Shorthand kwargs
----------------

The constructor accepts a flexible set of shorthand kwargs that get parsed
into the right slot in :py:meth:`pre_initialize`. ``max_`` is always stripped
first.

============================================================================  ===============  ====================
Pattern                                                                       Slot             Window
============================================================================  ===============  ====================
``rps`` / ``rpm`` / ``rph`` / ``rpd`` / ``rpw``                               ``requests``     second/minute/hour/day/week
``requests_per_{second,minute,hour,day,week}``                                ``requests``     from suffix
``itps`` / ``itpm`` / ``itph`` / ``itpd`` / ``itpw``                          ``input_tokens`` second/minute/hour/day/week
``input_tps`` / ``input_tpm`` / ``input_tph`` / ``input_tpd`` / ``input_tpw`` ``input_tokens`` second/minute/hour/day/week
``input_tokens_per_{second,minute,hour,day,week}``                            ``input_tokens`` from suffix
``otps`` / ``otpm`` / ``otph`` / ``otpd`` / ``otpw``                          ``output_tokens`` second/minute/hour/day/week
``output_tps`` / ``output_tpm`` / ``output_tph`` / ``output_tpd`` / ``output_tpw`` ``output_tokens`` second/minute/hour/day/week
``output_tokens_per_{second,minute,hour,day,week}``                           ``output_tokens`` from suffix
``budget_per_{second,minute,hour,day,week}``                                  ``budget``       from suffix
``concurrency``                                                               ``concurrency``  —
============================================================================  ===============  ====================

Conflict rules:

- Two shorthand kwargs that resolve to the same ``(slot, window)`` raise
  ``ValueError`` (e.g. ``rpm=300, requests_per_minute=500``).
- A canonical field (e.g. ``requests=[RateLimit(300, "minute")]``) plus a
  shorthand that resolves to the same ``(slot, window)`` raises ``ValueError``.
- Different windows on the same slot are merged into a list.

Examples
--------

Shorthand only::

    SlowBurnLimits(rpm=300, rpd=10_000, itpm=50_000, otph=2_000_000,
                   budget_per_day=5.0, concurrency=10)

Canonical only::

    SlowBurnLimits(
        requests=[RateLimit(300, "minute"), RateLimit(10_000, "day")],
        budget=[CostLimit(budget_usd=5.0, window="day")],
        concurrency=10,
    )

Mixed (different windows on the same slot are fine)::

    SlowBurnLimits(rpm=300, requests=[RateLimit(10_000, "day")])
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from concurry import RateLimit, RateLimitAlgorithm, RateWindow
from morphic import Typed
from pydantic import ConfigDict

from .limits import CostLimit

# ----------------------------------------------------------------------------
# Slot/window resolution table
# ----------------------------------------------------------------------------

# Canonical slot names (the actual SlowBurnLimits fields).
_CANONICAL_SLOTS = ("requests", "input_tokens", "output_tokens", "budget", "concurrency")

# Internal limit-key strings used for the underlying RateLimit/CostLimit objects.
# These also serve as the keys the worker uses when populating
# ``acquire(requested=...)`` / ``acq.update(usage=...)``.
SLOT_TO_LIMIT_KEY: Dict[str, str] = {
    "requests": "requests",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "budget": "cost_usd",  # matches CostLimit's DEFAULT_COST_LIMIT_KEY
}

# Maps single-letter window suffix → RateWindow.
_LETTER_TO_WINDOW: Dict[str, RateWindow] = {
    "s": RateWindow.Secondly,
    "m": RateWindow.Minutely,
    "h": RateWindow.Hourly,
    "d": RateWindow.Daily,
    "w": RateWindow.Weekly,
}

# Maps verbose window suffix → RateWindow.
_VERBOSE_TO_WINDOW: Dict[str, RateWindow] = {
    "second": RateWindow.Secondly,
    "minute": RateWindow.Minutely,
    "hour": RateWindow.Hourly,
    "day": RateWindow.Daily,
    "week": RateWindow.Weekly,
}


def _resolve_shorthand(kwarg_name: str) -> Optional[Tuple[str, Optional[RateWindow]]]:
    """Resolve a single shorthand kwarg into ``(slot, window)``.

    Returns ``None`` if ``kwarg_name`` does not match any shorthand pattern.
    Returns ``(slot, None)`` for ``concurrency``-style kwargs that don't
    carry a window.

    The ``max_`` prefix is stripped before matching.
    """
    name = kwarg_name
    if name.startswith("max_"):
        name = name[len("max_") :]

    if name == "concurrency":
        return ("concurrency", None)

    # Compact request forms: rpm/rps/rph/rpd/rpw
    m = re.fullmatch(r"rp([smhdw])", name)
    if m is not None:
        return ("requests", _LETTER_TO_WINDOW[m.group(1)])

    # Verbose request form: requests_per_{second,minute,hour,day,week}
    m = re.fullmatch(r"requests_per_(second|minute|hour|day|week)", name)
    if m is not None:
        return ("requests", _VERBOSE_TO_WINDOW[m.group(1)])

    # Input tokens: itpm/itps/itph/itpd/itpw  OR  input_tps/_tpm/_tph/_tpd/_tpw
    m = re.fullmatch(r"itp([smhdw])", name)
    if m is not None:
        return ("input_tokens", _LETTER_TO_WINDOW[m.group(1)])
    m = re.fullmatch(r"input_tp([smhdw])", name)
    if m is not None:
        return ("input_tokens", _LETTER_TO_WINDOW[m.group(1)])
    # Verbose: input_tokens_per_{second,minute,hour,day,week}
    m = re.fullmatch(r"input_tokens_per_(second|minute|hour|day|week)", name)
    if m is not None:
        return ("input_tokens", _VERBOSE_TO_WINDOW[m.group(1)])

    # Output tokens: otpm/otps/otph/otpd/otpw  OR  output_tps/_tpm/_tph/_tpd/_tpw
    m = re.fullmatch(r"otp([smhdw])", name)
    if m is not None:
        return ("output_tokens", _LETTER_TO_WINDOW[m.group(1)])
    m = re.fullmatch(r"output_tp([smhdw])", name)
    if m is not None:
        return ("output_tokens", _LETTER_TO_WINDOW[m.group(1)])
    # Verbose: output_tokens_per_{second,minute,hour,day,week}
    m = re.fullmatch(r"output_tokens_per_(second|minute|hour|day|week)", name)
    if m is not None:
        return ("output_tokens", _VERBOSE_TO_WINDOW[m.group(1)])

    # Budget: budget_per_{second,minute,hour,day,week}
    m = re.fullmatch(r"budget_per_(second|minute|hour|day|week)", name)
    if m is not None:
        return ("budget", _VERBOSE_TO_WINDOW[m.group(1)])

    return None


# ----------------------------------------------------------------------------
# SlowBurnLimits
# ----------------------------------------------------------------------------

# Type aliases for the slot value spaces.
RateSlotInput = Union[RateLimit, List[RateLimit]]
BudgetSlotInput = Union[CostLimit, List[CostLimit]]


class SlowBurnLimits(Typed):
    """Unified limits container.

    Normalized form (after ``pre_initialize`` runs):

    - Each rate slot (``requests``, ``input_tokens``, ``output_tokens``) is
      either ``None`` (meaning "inherit") or a non-empty ``List[RateLimit]``.
    - The ``budget`` slot is either ``None`` or a non-empty ``List[CostLimit]``.
    - The ``concurrency`` slot is either ``None`` or a positive ``int``.

    The constructor accepts both the canonical fields (``requests``, etc.)
    and a wide set of shorthand kwargs (``rpm``, ``input_tokens_per_minute``,
    ``budget_per_day``, etc. — see module docstring). Shorthand kwargs are
    resolved into the canonical slots in :py:meth:`pre_initialize`.

    Empty input (no fields set) yields a fully-inheriting ``SlowBurnLimits``
    with all slots ``None``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    requests: Optional[List[RateLimit]] = None
    input_tokens: Optional[List[RateLimit]] = None
    output_tokens: Optional[List[RateLimit]] = None
    budget: Optional[List[CostLimit]] = None
    concurrency: Optional[int] = None

    @classmethod
    def pre_initialize(cls, data: Dict[str, Any]) -> None:
        """Parse shorthand kwargs and merge them into the canonical slots.

        Steps:

        1. Walk every kwarg. If it's a canonical name, leave it. Otherwise,
           try to resolve it as a shorthand. If it doesn't match any known
           shorthand pattern, leave it (let pydantic raise a clear "unknown
           field" error since ``model_config = extra='forbid'``).
        2. For each shorthand match, build the corresponding ``RateLimit`` /
           ``CostLimit`` and append to the matched slot. Track which
           ``(slot, window)`` pairs are already populated; raise on duplicates.
        3. Normalize the canonical slot values: a single ``RateLimit`` /
           ``CostLimit`` becomes a one-element list. Detect canonical vs.
           shorthand collisions on the same ``(slot, window)``.
        """
        if not isinstance(data, dict):
            return

        # First pass: pull canonical fields out of `data`. We take ownership
        # of them so we can append shorthand-derived entries below.
        canonical: Dict[str, Any] = {}
        for slot in _CANONICAL_SLOTS:
            if slot in data:
                canonical[slot] = data.pop(slot)

        # Normalize canonical slot values to lists (or int / None).
        rate_slots: Dict[str, List[RateLimit]] = {
            "requests": [],
            "input_tokens": [],
            "output_tokens": [],
        }
        budget_list: List[CostLimit] = []
        concurrency_value: Optional[int] = None
        # Track which (slot, window_seconds) pairs are already populated so
        # we can reject duplicates from canonical/shorthand collisions.
        windows_seen: Dict[Tuple[str, float], str] = {}

        def _record(slot: str, limit_obj: Any, source: str) -> None:
            """Add ``limit_obj`` to its slot, raising on duplicate windows."""
            if slot == "concurrency":
                # int slot — handled separately
                return
            window_seconds = float(limit_obj.window)  # already-resolved seconds
            sig = (slot, window_seconds)
            if sig in windows_seen:
                raise ValueError(
                    f"SlowBurnLimits: duplicate ({slot}, window={window_seconds}s) "
                    f"from {windows_seen[sig]!r} and {source!r}. Each "
                    f"(slot, window) pair may appear at most once."
                )
            windows_seen[sig] = source
            if slot == "budget":
                budget_list.append(limit_obj)
            else:
                rate_slots[slot].append(limit_obj)

        for slot, value in canonical.items():
            if value is None:
                continue
            if slot == "concurrency":
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"SlowBurnLimits.concurrency must be a positive int, got {value!r}")
                concurrency_value = value
            elif slot == "budget":
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if not isinstance(item, CostLimit):
                        raise TypeError(
                            f"SlowBurnLimits.budget must contain CostLimit "
                            f"instances; got {type(item).__name__}: {item!r}. "
                            f"Use ``budget_per_<window>=<usd>`` for shorthand."
                        )
                    _record("budget", item, source=f"budget=[..., {item!r}]")
            else:
                # rate slot
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if not isinstance(item, RateLimit):
                        raise TypeError(
                            f"SlowBurnLimits.{slot} must contain RateLimit "
                            f"instances; got {type(item).__name__}: {item!r}. "
                            f"Use shorthand kwargs like ``rpm=...`` for "
                            f"capacity-and-window pairs."
                        )
                    _record(slot, item, source=f"{slot}=[..., {item!r}]")

        # Second pass: walk remaining kwargs in `data`. Anything that matches a
        # shorthand pattern is consumed and routed to its slot. Anything else
        # is left for pydantic to flag (extra='forbid' will raise).
        # We collect shorthand keys first so we can mutate `data` without
        # iterating-and-deleting.
        shorthand_keys = list(data.keys())
        for kwarg in shorthand_keys:
            resolved = _resolve_shorthand(kwarg)
            if resolved is None:
                continue
            slot, window = resolved
            value = data.pop(kwarg)
            if slot == "concurrency":
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(
                        f"SlowBurnLimits.concurrency (from {kwarg!r}) must be a positive int, got {value!r}"
                    )
                if concurrency_value is not None:
                    raise ValueError(
                        f"SlowBurnLimits: concurrency was set twice (canonical + shorthand {kwarg!r})."
                    )
                concurrency_value = value
                continue
            # Rate / budget slot
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"Shorthand {kwarg!r} expects a numeric capacity, got {type(value).__name__}: {value!r}"
                )
            if slot == "budget":
                limit_obj = CostLimit(budget_usd=float(value), window=window)
            else:
                if value < 1:
                    raise ValueError(f"Shorthand {kwarg!r} capacity must be >= 1, got {value!r}")
                limit_obj = RateLimit(
                    key=SLOT_TO_LIMIT_KEY[slot],
                    capacity=int(value),
                    window=window,
                )
            _record(slot, limit_obj, source=kwarg)

        # Write back canonical slots into ``data`` so pydantic validation
        # sees the normalized form.
        for slot in ("requests", "input_tokens", "output_tokens"):
            data[slot] = rate_slots[slot] if len(rate_slots[slot]) > 0 else None
        data["budget"] = budget_list if len(budget_list) > 0 else None
        data["concurrency"] = concurrency_value


# ----------------------------------------------------------------------------
# Library defaults
# ----------------------------------------------------------------------------


def default_slowburn_limits() -> SlowBurnLimits:
    """Return the library-default limits.

    Every slot is populated with a permissive limit so the worker's
    ``_build_limit_usage`` always has a key for every slot:

    - ``requests``: 100,000 / minute
    - ``input_tokens``: 1,000,000,000 / minute (effectively unlimited)
    - ``output_tokens``: 100,000,000 / minute (effectively unlimited)
    - ``budget``: ``CostLimit(inf, "daily")``
    - ``concurrency``: 1,000,000 (effectively unlimited)
    """
    return SlowBurnLimits(
        requests=[RateLimit(key=SLOT_TO_LIMIT_KEY["requests"], capacity=100_000, window=RateWindow.Minutely)],
        input_tokens=[
            RateLimit(
                key=SLOT_TO_LIMIT_KEY["input_tokens"],
                capacity=1_000_000_000,
                window=RateWindow.Minutely,
            )
        ],
        output_tokens=[
            RateLimit(
                key=SLOT_TO_LIMIT_KEY["output_tokens"],
                capacity=100_000_000,
                window=RateWindow.Minutely,
            )
        ],
        budget=[
            CostLimit(
                budget_usd=float("inf"),
                window=RateWindow.Daily,
                # Cost is dollar-denominated; we need a continuous-accumulator
                # algorithm (GCRA, TokenBucket) that admits fractional acquires.
                # SlidingWindow / FixedWindow / LeakyBucket count discrete
                # records and would reject fractional cost values.
                algorithm=RateLimitAlgorithm.GCRA,
            )
        ],
        concurrency=1_000_000,
    )


__all__ = [
    "SlowBurnLimits",
    "SLOT_TO_LIMIT_KEY",
    "default_slowburn_limits",
]
