"""Test that async_acquire prevents event-loop deadlocks in SlowBurnLLM.

The critical scenario: call_llm_batch with more prompts than rate-limit
capacity per window. With sync acquire(), time.sleep() blocks the event
loop, preventing ALL coroutines from making progress. With async_acquire(),
await asyncio.sleep() yields control so completed coroutines can finish
and the window can advance.

NOTE: CallLimit/RateLimit use time-window-based capacity (token bucket),
not resource-based capacity (semaphore). Capacity refills over the time
window, not when calls "release." The deadlock occurs because sync
time.sleep() blocks the event loop thread, preventing the loop from
servicing ANY other coroutines while waiting for the window to refill.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from concurry import CallLimit, LimitSet, RateLimit

from slowburn import create_llm
from slowburn.limits import CostLimit
from slowburn.llm_worker import SlowBurnLLM

from .conftest import MOCK_MODEL_NAME


def _make_response():
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    message = SimpleNamespace(content="Hello", tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=MOCK_MODEL_NAME,
        _hidden_params={"response_cost": 0.0001},
    )


def _build_worker_with_call_limit(capacity: int, window_seconds: float = 1.0) -> SlowBurnLLM:
    """Create a SlowBurnLLM with a tight CallLimit.

    Uses a short window (1s default) so capacity refills quickly during tests.
    """
    limit_set = LimitSet(
        limits=[
            CostLimit(budget_usd=100.0, window_seconds=3600),
            RateLimit(key="input_tokens", window_seconds=60, capacity=1_000_000),
            RateLimit(key="output_tokens", window_seconds=60, capacity=200_000),
            CallLimit(window_seconds=window_seconds, capacity=capacity),
        ],
        mode="Asyncio",
        shared=True,
    )
    return SlowBurnLLM.options(
        mode="Asyncio",
        limits=limit_set,
        num_retries={"call_llm": 0, "*": 0},
    ).init(
        name="deadlock-test",
        model_name=MOCK_MODEL_NAME,
        api_key="test-key",
        temperature=0.5,
        max_tokens=100,
        timeout=10.0,
    )


class TestAsyncAcquireDeadlockPrevention:
    """Verify that batches exceeding CallLimit capacity complete without deadlock.

    CallLimit uses token-bucket rate limiting: capacity refills over the time
    window. With sync acquire, the event loop blocks during the refill wait,
    deadlocking all coroutines. With async_acquire, the event loop stays
    responsive so coroutines can complete and the window can advance.
    """

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_at_capacity_succeeds(self, mock_acompletion) -> None:
        """Batch size == capacity should succeed immediately (baseline)."""
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=5, window_seconds=1.0)
        try:
            results = w.call_llm_batch(prompts=["Hi"] * 5).result(timeout=30.0)
            assert len(results) == 5
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_2x_capacity_no_deadlock(self, mock_acompletion) -> None:
        """Batch 2x capacity must complete: first wave fills capacity, second
        wave waits for the 1s window to refill, then proceeds.

        This is the core deadlock scenario. With sync acquire(), calls 6-10
        would block the event loop in time.sleep() while waiting for refill,
        preventing calls 1-5 from completing. With async_acquire(), the event
        loop stays responsive.
        """
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=5, window_seconds=1.0)
        try:
            start = time.monotonic()
            results = w.call_llm_batch(prompts=["Hi"] * 10).result(timeout=30.0)
            elapsed = time.monotonic() - start
            assert len(results) == 10
            assert elapsed >= 0.5, "Should take at least ~1s for second wave"
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_3x_capacity_no_deadlock(self, mock_acompletion) -> None:
        """Batch 3x capacity completes across three refill waves."""
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=3, window_seconds=1.0)
        try:
            results = w.call_llm_batch(prompts=["Hi"] * 9).result(timeout=30.0)
            assert len(results) == 9
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_single_calls_below_capacity_fast(self, mock_acompletion) -> None:
        """Sequential single calls below capacity should complete quickly."""
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=10, window_seconds=1.0)
        try:
            for i in range(5):
                result = w.call_llm(prompt=f"Hi {i}").result(timeout=10.0)
                assert len(result) > 0

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 5
        finally:
            w.stop()


class TestCreateLLMBatchCapacity:
    """Test that create_llm's default limits don't deadlock on real batch sizes.

    create_llm uses max_rpm=500 by default (CallLimit capacity=500,
    window=60s). Batches <= 500 should complete in one wave.
    """

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_create_llm_batch_within_rpm(self, mock_acompletion) -> None:
        """A batch within the default max_rpm should complete without delay."""
        mock_acompletion.return_value = _make_response()
        llm = create_llm(
            model=MOCK_MODEL_NAME,
            budget_usd=100.0,
            window="hourly",
            max_rpm=500,
        )
        try:
            results = llm.call_llm_batch(prompts=["Hi"] * 20).result(timeout=30.0)
            assert len(results) == 20
        finally:
            llm.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_create_llm_small_rpm_batch_exceeds(self, mock_acompletion) -> None:
        """Batch exceeding call-rate capacity completes via async_acquire.

        Uses the low-level worker builder with a short-window CallLimit
        to verify that create_llm-style workers don't deadlock.
        """
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=4, window_seconds=1.0)
        try:
            results = w.call_llm_batch(prompts=["Hi"] * 8).result(timeout=30.0)
            assert len(results) == 8
        finally:
            w.stop()


class TestSyncAcquireDeadlockProof:
    """Prove that sync acquire() DOES deadlock when N > capacity on an event loop.

    These tests run at the concurry level (no SlowBurnLLM), directly
    demonstrating the deadlock mechanism that async_acquire fixes.
    """

    def test_sync_acquire_deadlocks_over_capacity(self) -> None:
        """Sync acquire() on an asyncio event loop deadlocks when N > capacity.

        Runs asyncio.run() in a daemon thread with a hard 5s join timeout.
        With a 300s window and capacity=3, coroutines 4-6 enter sync
        acquire()'s time.sleep() loop, blocking the event loop thread.
        The thread.join(5.0) returns with the thread still alive, proving
        the deadlock: 6 trivial coroutines would otherwise finish in <1s.

        This is the negative test documenting the bug that async_acquire fixes.
        """
        import threading

        ls = LimitSet(
            limits=[CallLimit(window_seconds=300.0, capacity=3)],
            mode="Asyncio",
            shared=True,
        )
        completed = []

        async def sync_worker(i: int) -> int:
            acq = ls.acquire(requested={"call_count": 1})
            try:
                await asyncio.sleep(0.01)
                acq.update(usage={"call_count": 1})
                completed.append(i)
                return i
            finally:
                acq.release()

        async def run_all():
            return await asyncio.gather(*[sync_worker(i) for i in range(6)])

        def run_in_thread():
            asyncio.run(run_all())

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

        deadlocked = thread.is_alive()
        assert deadlocked, (
            "Expected sync acquire to deadlock the event loop, but it completed. "
            f"completed={completed}"
        )
        assert len(completed) <= 3, (
            f"Expected at most 3 completions (capacity), got {len(completed)}"
        )

    def test_async_acquire_no_deadlock_over_capacity(self) -> None:
        """async_acquire() on an asyncio event loop does NOT deadlock.

        Same scenario as above, but using async_acquire. All 6 coroutines
        complete: the first 3 run immediately, the remaining 3 wait via
        await asyncio.sleep() (yielding the event loop), and proceed
        once the 1s window refills capacity.
        """
        ls = LimitSet(
            limits=[CallLimit(window_seconds=1.0, capacity=3)],
            mode="Asyncio",
            shared=True,
        )
        completed = []

        async def async_worker(i: int) -> int:
            acq = await ls.async_acquire(requested={"call_count": 1})
            try:
                await asyncio.sleep(0.01)
                acq.update(usage={"call_count": 1})
                completed.append(i)
                return i
            finally:
                acq.release()

        async def run_all():
            await asyncio.wait_for(
                asyncio.gather(*[async_worker(i) for i in range(6)]),
                timeout=10.0,
            )

        asyncio.run(run_all())
        assert len(completed) == 6


class TestBatchCostReporterAccuracy:
    """Verify that cost tracking remains accurate with async_acquire batches."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_reporter_counts_all_calls_in_batch(self, mock_acompletion) -> None:
        """Every call in a multi-wave batch is logged to the CostReporter."""
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=4, window_seconds=1.0)
        try:
            results = w.call_llm_batch(prompts=["Hi"] * 8).result(timeout=30.0)
            assert len(results) == 8

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 8
            assert reporter.total_cost() > 0
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_reporter_correct_after_sequential_batches(self, mock_acompletion) -> None:
        """Two sequential batches accumulate cost correctly."""
        mock_acompletion.return_value = _make_response()
        w = _build_worker_with_call_limit(capacity=5, window_seconds=1.0)
        try:
            w.call_llm_batch(prompts=["Hi"] * 3).result(timeout=30.0)
            w.call_llm_batch(prompts=["Hi"] * 4).result(timeout=30.0)

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 7
        finally:
            w.stop()
