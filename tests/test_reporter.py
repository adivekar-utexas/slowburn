"""Tests for CostReporter: logging, aggregation, and export formats."""

import json
import threading
from pathlib import Path

import pytest

from slowburn.reporter import CostReporter


class TestCostReporterBasics:
    """Test basic logging and aggregation."""

    def test_empty_reporter(self) -> None:
        r = CostReporter()
        assert r.num_calls == 0
        assert r.total_cost() == 0.0
        assert len(r.summary()) == 0

    def test_single_call(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        assert r.num_calls == 1
        assert r.total_cost() == pytest.approx(0.001)

    def test_multiple_calls_same_model(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        r.log_call(model="gpt-4o-mini", cost_usd=0.002, input_tokens=200, output_tokens=100)
        assert r.num_calls == 2
        assert r.total_cost() == pytest.approx(0.003)

        s = r.summary()
        assert "gpt-4o-mini" in s
        assert s["gpt-4o-mini"]["calls"] == 2
        assert s["gpt-4o-mini"]["input_tokens"] == 300
        assert s["gpt-4o-mini"]["output_tokens"] == 150
        assert s["gpt-4o-mini"]["total_tokens"] == 450

    def test_multiple_models(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        r.log_call(model="claude-3-haiku", cost_usd=0.005, input_tokens=500, output_tokens=200)
        assert r.num_calls == 2
        assert r.total_cost() == pytest.approx(0.006)

        s = r.summary()
        assert len(s) == 2
        assert "gpt-4o-mini" in s
        assert "claude-3-haiku" in s

    def test_metadata_stored(self) -> None:
        r = CostReporter()
        r.log_call(
            model="gpt-4o-mini", cost_usd=0.001,
            input_tokens=100, output_tokens=50,
            metadata={"agent": "researcher", "task_id": 42},
        )
        assert r.calls[0]["metadata"]["agent"] == "researcher"
        assert r.calls[0]["metadata"]["task_id"] == 42

    def test_timestamp_recorded(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        assert "timestamp" in r.calls[0]
        assert isinstance(r.calls[0]["timestamp"], float)
        assert r.calls[0]["timestamp"] > 0

    def test_total_tokens_calculated(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        assert r.calls[0]["total_tokens"] == 150

    def test_reset(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        assert r.num_calls == 1
        r.reset()
        assert r.num_calls == 0
        assert r.total_cost() == 0.0


class TestCostReporterMarkdown:
    """Test Markdown table output."""

    def test_empty_markdown(self) -> None:
        r = CostReporter()
        md = r.to_markdown()
        assert "No LLM calls recorded" in md

    def test_single_model_markdown(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        md = r.to_markdown()
        assert "gpt-4o-mini" in md
        assert "| Model |" in md
        assert "**Total**" in md

    def test_multi_model_markdown_sorted(self) -> None:
        """Models should appear in sorted order in the table."""
        r = CostReporter()
        r.log_call(model="z-model", cost_usd=0.01, input_tokens=100, output_tokens=50)
        r.log_call(model="a-model", cost_usd=0.02, input_tokens=200, output_tokens=100)
        md = r.to_markdown()
        a_pos = md.index("a-model")
        z_pos = md.index("z-model")
        assert a_pos < z_pos


class TestCostReporterJSON:
    """Test JSON output."""

    def test_json_string(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        json_str = r.to_json()
        data = json.loads(json_str)
        assert data["num_calls"] == 1
        assert data["total_cost_usd"] == pytest.approx(0.001)
        assert "summary" in data
        assert "calls" in data
        assert len(data["calls"]) == 1

    def test_json_to_file(self, tmp_path: Path) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        out = tmp_path / "report.json"
        returned = r.to_json(path=out)
        assert returned == str(out)
        assert out.exists()

        data = json.loads(out.read_text())
        assert data["num_calls"] == 1

    def test_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        out = tmp_path / "sub" / "dir" / "report.json"
        r.to_json(path=out)
        assert out.exists()


class TestCostReporterLatex:
    """Test LaTeX table output."""

    def test_empty_latex(self) -> None:
        r = CostReporter()
        tex = r.to_latex()
        assert "No LLM calls recorded" in tex

    def test_latex_structure(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt-4o-mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        tex = r.to_latex()
        assert r"\\begin{tabular}" in tex or r"\begin{tabular}" in tex
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex
        assert "gpt-4o-mini" in tex
        assert r"\textbf{Total}" in tex

    def test_latex_escapes_underscores(self) -> None:
        r = CostReporter()
        r.log_call(model="gpt_4o_mini", cost_usd=0.001, input_tokens=100, output_tokens=50)
        tex = r.to_latex()
        assert r"gpt\_4o\_mini" in tex


class TestCostReporterThreadSafety:
    """Test that concurrent logging does not corrupt data."""

    def test_concurrent_logging(self) -> None:
        """Log from 10 threads simultaneously, verify all calls are recorded.

        Steps:
        1. Create a reporter.
        2. Spawn 10 threads, each logging 100 calls.
        3. Wait for all threads to finish.
        4. Verify exactly 1000 calls are recorded.
        5. Verify total cost is correct.
        """
        r = CostReporter()
        num_threads = 10
        calls_per_thread = 100
        cost_per_call = 0.001

        def log_many():
            for _ in range(calls_per_thread):
                r.log_call(
                    model="gpt-4o-mini",
                    cost_usd=cost_per_call,
                    input_tokens=100,
                    output_tokens=50,
                )

        threads = [threading.Thread(target=log_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert r.num_calls == num_threads * calls_per_thread
        expected_total = num_threads * calls_per_thread * cost_per_call
        assert r.total_cost() == pytest.approx(expected_total, rel=1e-6)
