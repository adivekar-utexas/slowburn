"""
Demo: Deep Research Agent with real web search + file writing.

A proper ReAct agent that:
1. Receives a research task
2. Uses DuckDuckGo web search to find real information
3. Takes notes by writing to files in a sandboxed workspace
4. Synthesizes findings into a final report

All file operations are sandboxed to runs/research_agent/<timestamp>/.
Every LLM call is cost-tracked via SlowBurn's CostLimit.

Usage:
    python demos/demo_research_agent.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    print("Set OPENROUTER_API_KEY in .env to run this demo.")
    sys.exit(1)

from concurry import CallLimit, LimitSet, RateLimit  # noqa: E402
from lib.agent_loop import run_agent  # noqa: E402
from lib.tools import TOOL_SCHEMAS, execute_tool_call  # noqa: E402

from slowburn.limits import CostLimit  # noqa: E402
from slowburn.reporter import CostReporter  # noqa: E402

MODEL = "openrouter/z-ai/glm-4.5-air"
BUDGET_USD = 0.15
MAX_TOKENS = 600
MAX_STEPS = 15

RESEARCH_TASKS = [
    (
        "Research the cost of running LLM agents in production. "
        "Search the web for real data on API costs for GPT-4, Claude, and Gemini. "
        "Find specific dollar amounts from benchmarks like SWE-bench and Tau-bench. "
        "Write your findings to a file called 'cost_analysis.md' with sources."
    ),
    (
        "Research backpressure mechanisms in distributed systems and how they "
        "apply to LLM rate limiting. Search the web for how systems like Kafka, "
        "gRPC, and TCP handle backpressure. Write a comparison to 'backpressure.md'."
    ),
]

SYSTEM_PROMPT = """\
You are a thorough research agent. You have access to these tools:
- search_web: Search the web with DuckDuckGo to find real information
- write_file: Save your research notes and reports to files
- read_file: Read files you've previously written
- list_dir: See what files exist in your workspace

For each research task:
1. Use search_web 2-3 times with different queries to gather information
2. Use write_file to save a structured report with real sources and URLs
3. Be specific: cite actual numbers, paper names, and URLs from search results

IMPORTANT: You MUST use search_web to find real data. Do NOT make up facts."""


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).parent / "runs" / "research_agent" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("  Deep Research Agent (Real Web Search)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print(f"  Tasks: {len(RESEARCH_TASKS)}")
    print(f"  Max steps/task: {MAX_STEPS}")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    limit_set = LimitSet(
        limits=[
            CostLimit(budget_usd=BUDGET_USD, window_seconds=3600),
            RateLimit(key="input_tokens", window_seconds=60, capacity=500_000),
            RateLimit(key="output_tokens", window_seconds=60, capacity=100_000),
            CallLimit(window_seconds=60, capacity=100),
        ],
        mode="thread",
        shared=True,
    )
    reporter = CostReporter()

    def tool_executor(name, args):
        return execute_tool_call(name, args, workspace=runs_dir)

    start_time = time.time()

    for i, task in enumerate(RESEARCH_TASKS, 1):
        print(f"\n  --- Task {i}/{len(RESEARCH_TASKS)} ---")
        print(f"  {task[:80]}...")

        task_log_dir = runs_dir / f"task_{i:02d}"
        result = asyncio.run(run_agent(
            model=MODEL,
            task=task,
            tools=TOOL_SCHEMAS,
            tool_executor=tool_executor,
            limit_set=limit_set,
            reporter=reporter,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            max_steps=MAX_STEPS,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            verbose=True,
            log_dir=task_log_dir,
        ))

        print(f"  Result: {result['result'][:150]}...")

    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total LLM calls: {reporter.num_calls}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    files = list(runs_dir.rglob("*"))
    print(f"\n  Files in workspace ({runs_dir}):")
    for f in sorted(files):
        if f.is_file():
            print(f"    {f.relative_to(runs_dir)}: {f.stat().st_size} bytes")

    reporter.to_json(path=runs_dir / "_cost_report.json")
    print(f"\n  Cost report: {runs_dir / '_cost_report.json'}")


if __name__ == "__main__":
    main()
