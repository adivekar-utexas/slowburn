"""
CostReporter: Accumulates per-call cost data and exports reports.

Supports JSON, Markdown table, and LaTeX table output formats for
use in experiment logs and conference papers.
"""

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class CostReporter:
    """Thread-safe cost accumulator with multi-format export.

    Each LLM call is logged with its model, token counts, dollar cost,
    and timestamp. The reporter can produce per-model breakdowns in
    JSON, Markdown, or LaTeX formats.

    Thread safety is provided via a lock so that multiple workers
    (or CrewAI/AutoGen hooks running in different threads) can log
    concurrently without data races.

    Example::

        reporter = CostReporter()
        reporter.log_call(model="gpt-4o-mini", cost_usd=0.0003,
                          input_tokens=150, output_tokens=80)
        print(reporter.to_markdown())
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.calls: List[Dict[str, Any]] = []
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._lock = threading.Lock()

    def log_call(
        self,
        model: str,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single LLM call.

        Args:
            model: Model name (e.g. "gpt-4o-mini").
            cost_usd: Actual dollar cost of this call.
            input_tokens: Number of input (prompt) tokens.
            output_tokens: Number of output (completion) tokens.
            metadata: Arbitrary extra fields (agent name, task id, etc.).
        """
        record: Dict[str, Any] = {
            "model": model,
            "cost_usd": cost_usd,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "timestamp": time.time(),
        }
        if metadata is not None:
            record["metadata"] = metadata
        with self._lock:
            self.calls.append(record)

    @property
    def num_calls(self) -> int:
        with self._lock:
            return len(self.calls)

    def total_cost(self) -> float:
        """Total dollar cost across all logged calls."""
        with self._lock:
            return sum(c["cost_usd"] for c in self.calls)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-model breakdown: total calls, total tokens, total cost.

        Returns:
            Dict mapping model name to a summary dict with keys
            ``calls``, ``input_tokens``, ``output_tokens``,
            ``total_tokens``, and ``cost_usd``.
        """
        with self._lock:
            snapshot = list(self.calls)

        agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        )
        for c in snapshot:
            m = agg[c["model"]]
            m["calls"] += 1
            m["input_tokens"] += c["input_tokens"]
            m["output_tokens"] += c["output_tokens"]
            m["total_tokens"] += c["total_tokens"]
            m["cost_usd"] += c["cost_usd"]
        return dict(agg)

    def to_markdown(self) -> str:
        """Generate a Markdown cost table suitable for papers or READMEs."""
        s = self.summary()
        if len(s) == 0:
            return "_No LLM calls recorded._"

        lines = [
            "| Model | Calls | Input Tokens | Output Tokens | Total Tokens | Cost (USD) |",
            "|-------|------:|-------------:|--------------:|-------------:|-----------:|",
        ]
        grand_calls = 0
        grand_input = 0
        grand_output = 0
        grand_total = 0
        grand_cost = 0.0

        for model, info in sorted(s.items()):
            lines.append(
                f"| {model} | {info['calls']:,} | {info['input_tokens']:,} "
                f"| {info['output_tokens']:,} | {info['total_tokens']:,} "
                f"| ${info['cost_usd']:.6f} |"
            )
            grand_calls += info["calls"]
            grand_input += info["input_tokens"]
            grand_output += info["output_tokens"]
            grand_total += info["total_tokens"]
            grand_cost += info["cost_usd"]

        lines.append(
            f"| **Total** | **{grand_calls:,}** | **{grand_input:,}** "
            f"| **{grand_output:,}** | **{grand_total:,}** "
            f"| **${grand_cost:.6f}** |"
        )
        return "\n".join(lines)

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize the full call log to JSON.

        If *path* is given, writes to disk and returns the path as a string.
        Otherwise, returns the JSON string directly.
        """
        with self._lock:
            snapshot = list(self.calls)

        payload = {
            "summary": self.summary(),
            "total_cost_usd": self.total_cost(),
            "num_calls": len(snapshot),
            "calls": snapshot,
        }
        json_str = json.dumps(payload, indent=2, default=str)

        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
            return str(p)
        return json_str

    def to_latex(self) -> str:
        r"""Generate a LaTeX table for conference papers (LNCS-compatible).

        Returns a standalone ``tabular`` environment that can be wrapped in
        a ``table`` float. Uses ``\toprule``, ``\midrule``, ``\bottomrule``
        from the ``booktabs`` package.
        """
        s = self.summary()
        if len(s) == 0:
            return "% No LLM calls recorded."

        lines = [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Model & Calls & Input Tok. & Output Tok. & Total Tok. & Cost (USD) \\",
            r"\midrule",
        ]
        grand_calls = 0
        grand_input = 0
        grand_output = 0
        grand_total = 0
        grand_cost = 0.0

        for model, info in sorted(s.items()):
            safe_model = model.replace("_", r"\_")
            lines.append(
                f"{safe_model} & {info['calls']:,} & {info['input_tokens']:,} "
                f"& {info['output_tokens']:,} & {info['total_tokens']:,} "
                f"& \\${info['cost_usd']:.6f} \\\\"
            )
            grand_calls += info["calls"]
            grand_input += info["input_tokens"]
            grand_output += info["output_tokens"]
            grand_total += info["total_tokens"]
            grand_cost += info["cost_usd"]

        lines.append(r"\midrule")
        lines.append(
            f"\\textbf{{Total}} & \\textbf{{{grand_calls:,}}} "
            f"& \\textbf{{{grand_input:,}}} & \\textbf{{{grand_output:,}}} "
            f"& \\textbf{{{grand_total:,}}} & \\textbf{{\\${grand_cost:.6f}}} \\\\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded calls."""
        with self._lock:
            self.calls.clear()
