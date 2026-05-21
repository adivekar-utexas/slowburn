"""End-to-end multi-account, multi-region AWS Bedrock test.

Builds a SlowBurnLLM with N_accounts x N_regions endpoints. Each call is
dispatched by the LimitPool's round-robin to one endpoint; the user's
resolver does an N-hop STS role chain (with TTL caching) to obtain temporary
credentials for the target account, then SlowBurn forwards those creds to
``litellm.acompletion``.

Auth chain (all hops must succeed):

    <caller (whatever your local mechanism puts on the default profile)>
        -> hop_role_chain[0]
            -> hop_role_chain[1]
                -> ... -> target_role_template (per target account)

All account IDs, role ARNs, regions, and inference profile prefixes are
loaded from ``tests/configs/bedrock_e2e.json``. That file is gitignored.
A committed ``tests/configs/bedrock_e2e.template.json`` has the schema with
``<FILL_IN>`` placeholders. To run this test:

    1. Copy the template:
         cp tests/configs/bedrock_e2e.template.json tests/configs/bedrock_e2e.json
    2. Fill in your own AWS account IDs / role ARNs.
    3. Refresh the caller credentials onto your default AWS profile using
       whatever mechanism your organisation uses (aws-vault, aws sso login,
       ``aws configure``, an internal credential helper, etc.).
    4. Run the test:
         SLOWBURN_RUN_BEDROCK_E2E=1 pytest -s --log-cli-level=INFO \\
             tests/test_e2e_bedrock_multi_region.py

The ``-s --log-cli-level=INFO`` is what produces the per-call evidence:
- ``[ENDPOINT_RESOLVER]`` prints the augmented dict the resolver returned.
- ``LITELLM_CALL`` (from the worker, verbosity=3) prints the EXACT kwargs
  that ``litellm.acompletion`` received, with credentials redacted to the
  first 6 chars so you can see they actually changed per account.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "configs" / "bedrock_e2e.json"
_PLACEHOLDER_PREFIX = "<FILL_IN"


def _has_placeholder(value: Any) -> bool:
    """Return True if any string in the (possibly nested) value still looks
    like an unfilled ``<FILL_IN: ...>`` placeholder from the template."""
    if isinstance(value, str):
        return value.startswith(_PLACEHOLDER_PREFIX)
    if isinstance(value, dict):
        return any(_has_placeholder(v) for k, v in value.items() if not k.startswith("_"))
    if isinstance(value, list):
        return any(_has_placeholder(v) for v in value)
    return False


def _load_config() -> Optional[Dict[str, Any]]:
    """Load ``bedrock_e2e.json`` if present and fully filled in.

    Returns ``None`` if the file is missing or still contains placeholder
    strings, so the test can skip cleanly with a helpful message.
    """
    if not _CONFIG_PATH.exists():
        return None
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
    except json.JSONDecodeError:
        return None
    if _has_placeholder(cfg):
        return None
    return cfg


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _aws_credentials_available() -> bool:
    try:
        import boto3
    except ImportError:
        return False
    try:
        # Use a fresh session each time so we re-read ``~/.aws/credentials``
        # (the cached default session would otherwise hold stale creds if the
        # file was rewritten between calls).
        sts = boto3.session.Session().client("sts")
        sts.get_caller_identity()
        return True
    except Exception:
        return False


def _try_refresh_aws_credentials(refresh_command: Optional[str]) -> bool:
    """Run ``refresh_command`` (if non-null) and return True iff credentials
    are available afterwards. ``boto3`` reads its default-profile
    credentials at each ``Session``/``client`` construction, so calling
    ``_aws_credentials_available()`` again after the refresh is the right
    way to verify success.

    NOTE: This function and the ``ada_credentials_command`` field it reads
    are LOCAL-ONLY plumbing — they exist in this file on the developer's
    machine but are not part of the committed test code (the published
    version asks the user to refresh credentials manually).
    """
    if not refresh_command:
        return False
    import shlex
    import subprocess

    print(f"\n[bedrock_e2e] AWS creds expired or missing; running: {refresh_command}")
    try:
        result = subprocess.run(
            shlex.split(refresh_command),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as e:
        print(f"[bedrock_e2e] credentials refresh command failed to launch: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("[bedrock_e2e] credentials refresh command timed out after 120s.")
        return False
    if result.returncode != 0:
        print(
            f"[bedrock_e2e] credentials refresh command exited {result.returncode}.\n"
            f"  stdout: {result.stdout.strip()[:400]}\n"
            f"  stderr: {result.stderr.strip()[:400]}"
        )
        return False
    return _aws_credentials_available()


_RUN_E2E = os.environ.get("SLOWBURN_RUN_BEDROCK_E2E", "").lower() in {"1", "true", "yes"}
_CFG = _load_config()


def _compute_skip_reason() -> Optional[str]:
    if not _RUN_E2E:
        return "Set SLOWBURN_RUN_BEDROCK_E2E=1 to enable this test."
    if _CFG is None:
        return (
            f"Config file {_CONFIG_PATH} is missing or still contains "
            f"'<FILL_IN: ...>' placeholders. Copy "
            f"tests/configs/bedrock_e2e.template.json to "
            f"tests/configs/bedrock_e2e.json and fill in your own AWS "
            f"account IDs / role ARNs."
        )
    if not _aws_credentials_available():
        # Try to refresh via the user-provided command if set in the config.
        if _try_refresh_aws_credentials(_CFG.get("ada_credentials_command")):
            return None
        return (
            "AWS default-profile credentials are missing or expired, and "
            "auto-refresh either is disabled (`ada_credentials_command` is "
            "null) or failed. Refresh them manually (via aws-vault, aws sso "
            "login, or your organisation's credential helper) before "
            "running this test."
        )
    return None


# Cache the skip decision so the AWS-credentials probe runs at most once
# per test session.
_SKIP_REASON: Optional[str] = _compute_skip_reason()


pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# ---------------------------------------------------------------------------
# Endpoint construction
# ---------------------------------------------------------------------------


def _build_endpoints(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_accounts: List[str] = cfg["target_accounts"]
    region_to_prefix: List[List[str]] = cfg["region_to_prefix"]
    model_template: str = cfg["model_template"]
    rpm: int = cfg["test_params"]["endpoint_rpm"]
    concurrency: int = cfg["test_params"]["endpoint_concurrency"]

    out: List[Dict[str, Any]] = []
    for account in target_accounts:
        for region, prefix in region_to_prefix:
            out.append(
                {
                    "endpoint_id": f"{account}/{region}",
                    "account_id": account,
                    "region": region,
                    "model": model_template.format(prefix=prefix),
                    "limits": dict(
                        rpm=rpm,
                        # Cap each endpoint at ``concurrency`` in-flight requests.
                        # Setting this on the endpoint dict (rather than at
                        # create_llm level) gives each endpoint its own private
                        # ResourceLimit at this capacity.
                        concurrency=concurrency,
                    ),
                }
            )
    return out


# ---------------------------------------------------------------------------
# N-hop STS resolver with TTL caching
# ---------------------------------------------------------------------------


_LOG = logging.getLogger("slowburn.test.e2e_bedrock")


class _CredCache:
    """TTL-aware thread-safe credentials cache keyed by an arbitrary string."""

    def __init__(self, refresh_margin_seconds: int = 120) -> None:
        self._refresh_margin = refresh_margin_seconds
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

    @staticmethod
    def _expiry_unix(creds: Dict[str, Any]) -> float:
        exp: datetime = creds["Expiration"]
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return exp.timestamp()

    def get_or_assume(self, *, key: str, assume_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                creds, expiry = cached
                if expiry - now > self._refresh_margin:
                    return creds
            creds = assume_fn()
            self._cache[key] = (creds, self._expiry_unix(creds))
            return creds


def _make_n_hop_resolver(cfg: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Closure-captured resolver and cache, generalized over the hop chain.

    Walks ``cfg["hop_role_chain"]`` in order, then assumes the final
    per-account target role from ``cfg["target_role_template"]``.
    """
    import boto3

    hop_chain: List[str] = list(cfg["hop_role_chain"])
    target_role_template: str = cfg["target_role_template"]
    cache = _CredCache()

    def _sts_with(creds: Optional[Dict[str, Any]]) -> Any:
        if creds is None:
            return boto3.client("sts")
        return boto3.client(
            "sts",
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )

    def _assume_with(prev_creds: Optional[Dict[str, Any]], role_arn: str, session: str) -> Dict[str, Any]:
        return _sts_with(prev_creds).assume_role(RoleArn=role_arn, RoleSessionName=session)[
            "Credentials"
        ]

    def _walk_hops_to_target(account_id: str) -> Dict[str, Any]:
        prev: Optional[Dict[str, Any]] = None
        for i, role_arn in enumerate(hop_chain):
            key = f"hop{i}:{role_arn}"
            captured_prev = prev
            prev = cache.get_or_assume(
                key=key,
                assume_fn=lambda role=role_arn, session=f"slowburn-hop{i}", c=captured_prev: _assume_with(
                    c, role, session
                ),
            )
        # Final hop: the per-account target role.
        target_arn = target_role_template.format(account=account_id)
        captured_prev = prev
        return cache.get_or_assume(
            key=f"target:{account_id}",
            assume_fn=lambda c=captured_prev: _assume_with(
                c, target_arn, f"slowburn-target-{account_id}"
            ),
        )

    def resolver(ep_cfg: Dict[str, Any]) -> Dict[str, Any]:
        account_id = ep_cfg["account_id"]
        target_creds = _walk_hops_to_target(account_id)
        augmented: Dict[str, Any] = {
            **ep_cfg,
            "litellm_params": {
                **ep_cfg.get("litellm_params", {}),
                "aws_region_name": ep_cfg["region"],
                "aws_access_key_id": target_creds["AccessKeyId"],
                "aws_secret_access_key": target_creds["SecretAccessKey"],
                "aws_session_token": target_creds["SessionToken"],
            },
        }
        # Log the augmented config with credentials redacted.
        _redact = lambda v: f"<len={len(v)} prefix={v[:6]}...>" if v else "<empty>"
        loggable = {
            **{k: v for k, v in augmented.items() if k != "litellm_params"},
            "litellm_params": {
                "aws_region_name": augmented["litellm_params"]["aws_region_name"],
                "aws_access_key_id": _redact(augmented["litellm_params"]["aws_access_key_id"]),
                "aws_secret_access_key": _redact(augmented["litellm_params"]["aws_secret_access_key"]),
                "aws_session_token": _redact(augmented["litellm_params"]["aws_session_token"]),
            },
        }
        _LOG.info(f"[ENDPOINT_RESOLVER] returning {loggable}")
        return augmented

    return resolver


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_bedrock_multi_account_multi_region(caplog: pytest.LogCaptureFixture) -> None:
    """Real Bedrock calls fanned out across N accounts x N regions endpoints."""
    from slowburn import create_llm

    assert _CFG is not None  # guarded by pytestmark.skipif, but be explicit.

    # Show worker + resolver logs in the pytest output.
    caplog.set_level(logging.INFO, logger="slowburn.llm_worker")
    caplog.set_level(logging.INFO, logger="slowburn.test.e2e_bedrock")

    endpoints = _build_endpoints(_CFG)
    expected_n_endpoints = len(_CFG["target_accounts"]) * len(_CFG["region_to_prefix"])
    assert len(endpoints) == expected_n_endpoints

    test_params = _CFG["test_params"]

    # Pass plain dicts (the only accepted form). create_llm validates each
    # dict into an internal EndpointConfig with the create_llm kwargs and
    # slowburn_config.defaults filling in any field the dict omits.
    #
    # Pass a deliberately INVALID worker-default model. If the per-endpoint
    # model override actually flows through the cascade, AWS will see the
    # correct regional model id (us./eu./jp./au.) and every call will
    # succeed. If the cascade is broken and the worker default leaks
    # through, AWS will reject every call with a model-not-found error
    # and the test will explode.
    llm = create_llm(
        model="bedrock/INVALID-WORKER-DEFAULT-MODEL-DOES-NOT-EXIST",
        name="bedrock-multi-account-pool",
        on_pricing_unavailable="warn",
        endpoints=endpoints,  # list of plain dicts
        endpoint_resolver=_make_n_hop_resolver(_CFG),
        max_tokens=test_params["max_tokens"],
        timeout=test_params["timeout_seconds"],
    )
    try:
        n_calls = test_params["n_calls"]
        prompts = [
            f"Write a five-paragraph explanation (about 600 words) on quantum "
            f"entanglement, suitable for a curious physics undergraduate. "
            f"Vary your wording slightly each time; this is essay #{i}."
            for i in range(n_calls)
        ]

        # Measure wall-clock for the entire batch dispatch.
        t_start = time.monotonic()
        results = llm.call_llm_batch(prompts=prompts, verbosity=3).result(
            timeout=test_params["batch_timeout_seconds"]
        )
        t_end = time.monotonic()
        wall_time = t_end - t_start

        assert len(results) == n_calls
        for i, r in enumerate(results):
            assert isinstance(r, str), f"result {i} not a string: {r!r}"
            assert len(r) > 0

        rep = llm.get_reporter().result(timeout=10.0)
        # The reporter records every API attempt, including retries on
        # transient errors (e.g. AWS Bedrock ServiceUnavailableError /
        # ThrottlingException). Use >= rather than == so the test passes
        # when AWS occasionally retries one or two calls under load.
        assert rep.num_calls >= n_calls, (
            f"Reporter recorded {rep.num_calls} calls but expected at least {n_calls}"
        )
        assert rep.total_cost() > 0.0
        if rep.num_calls > n_calls:
            print(
                f"  (note: reporter saw {rep.num_calls - n_calls} retried "
                f"call(s) due to transient AWS errors)"
            )

        # ------------------------------------------------------------------
        # Pull per-call API durations from the captured worker log records.
        # The worker emits 'RESPONSE | api=<X>s total=<Y>s | ...' at INFO
        # for every call when verbosity >= 3. We parse those out of caplog
        # so we can compute min/median/p95/max API time without touching
        # the worker.
        # ------------------------------------------------------------------
        import re
        from statistics import median

        api_times: List[float] = []
        total_times: List[float] = []
        pat = re.compile(r"RESPONSE \| api=([\d.]+)s total=([\d.]+)s")
        for record in caplog.records:
            m = pat.search(record.getMessage())
            if m is not None:
                api_times.append(float(m.group(1)))
                total_times.append(float(m.group(2)))

        # Print per-endpoint reporter summary.
        print("\n=== Per-endpoint cost summary ===")
        for ep_id, info in sorted(rep.summary_by_endpoint().items()):
            print(
                f"  {ep_id:<32} calls={info['calls']} "
                f"in={info['input_tokens']:>4} out={info['output_tokens']:>4} "
                f"cost=${info['cost_usd']:.6f}"
            )
        print(f"Total cost: ${rep.total_cost():.6f}")

        # Print timing summary.
        print("\n=== Latency summary ===")
        print(f"  Wall-clock for {n_calls}-call batch: {wall_time:.2f}s")
        print(f"  Effective throughput: {n_calls / wall_time:.1f} calls/s")
        if api_times:
            n = len(api_times)
            api_sorted = sorted(api_times)
            tot_sorted = sorted(total_times)
            p95_idx = int(0.95 * (n - 1))
            print(f"  Calls with timing in log: {n}/{n_calls}")
            print(
                f"  API time          min={min(api_sorted):.2f}s "
                f"median={median(api_sorted):.2f}s "
                f"p95={api_sorted[p95_idx]:.2f}s "
                f"max={max(api_sorted):.2f}s"
            )
            print(
                f"  Total (acq+api)   min={min(tot_sorted):.2f}s "
                f"median={median(tot_sorted):.2f}s "
                f"p95={tot_sorted[p95_idx]:.2f}s "
                f"max={max(tot_sorted):.2f}s"
            )
        else:
            print("  (no per-call timings captured \u2014 caplog level too high)")
    finally:
        llm.stop()
