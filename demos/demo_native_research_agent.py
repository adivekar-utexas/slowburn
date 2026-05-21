"""
Demo: Deep Research Agent with real web search + file writing.

A ReAct agent that:
1. Receives a research task
2. Uses DuckDuckGo web search to find real information
3. Takes notes by writing to files in a sandboxed workspace
4. Synthesizes findings into a final report

All LLM calls go through a SlowBurnLLM worker with dollar-budget backpressure.

Usage:
    cd demos && python demo_native_research_agent.py
"""

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

from lib.agent_loop import run_agent  # noqa: E402
from lib.tools import TOOL_SCHEMAS, execute_tool_call  # noqa: E402

from slowburn import create_llm  # noqa: E402

MODEL = "openrouter/z-ai/glm-4.5-air"
BUDGET_USD = 0.15
MAX_TOKENS = 600
MAX_STEPS = 15

RESEARCH_TASKS = [
    (
        "cost_analysis.md",
        "Research the cost of running LLM agents in production. "
        "Search the web for real data on API costs for GPT-4, Claude, and Gemini. "
        "Find specific dollar amounts from benchmarks like SWE-bench and Tau-bench. "
        "Write your findings to 'cost_analysis.md' with sources.",
    ),
    (
        "backpressure.md",
        "Research backpressure mechanisms in distributed systems and how they "
        "apply to LLM rate limiting. Search the web for how systems like Kafka, "
        "gRPC, and TCP handle backpressure. Write a comparison to 'backpressure.md'.",
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
    runs_dir = Path(__file__).parent / "runs" / "native_research_agent" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print("  Deep Research Agent (Real Web Search)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print(f"  Tasks: {len(RESEARCH_TASKS)}")
    print(f"  Max steps/task: {MAX_STEPS}")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    llm = create_llm(
        model=MODEL,
        limits=dict(budget_per_hour=BUDGET_USD),
        api_key=api_key,
        max_tokens=MAX_TOKENS,
        temperature=0.3,
    )

    def tool_executor(name, args):
        return execute_tool_call(name, args, workspace=runs_dir)

    start_time = time.time()

    for i, (output_file, task) in enumerate(RESEARCH_TASKS, 1):
        print(f"\n  --- Task {i}/{len(RESEARCH_TASKS)}: {output_file} ---")
        print(f"  {task[:80]}...")

        task_name = Path(output_file).stem
        task_log_dir = runs_dir / task_name
        result = run_agent(
            llm=llm,
            task=task,
            tools=TOOL_SCHEMAS,
            tool_executor=tool_executor,
            system_prompt=SYSTEM_PROMPT,
            output_file=output_file,
            max_steps=MAX_STEPS,
            verbose=True,
            log_dir=task_log_dir,
            workspace=runs_dir,
        )

        print(f"  Result: {result['result'][:150]}...")

    total_elapsed = time.time() - start_time

    reporter = llm.get_reporter().result(timeout=5.0)
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

    llm.stop()


if __name__ == "__main__":
    main()
