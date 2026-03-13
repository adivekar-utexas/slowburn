"""
Demo: Autonomous Code Agent with real file operations.

A proper ReAct agent that:
1. Reads code files from its workspace
2. Searches the web for best practices
3. Writes improved code back to files
4. Iterates until the code is well-structured

All file operations are sandboxed to runs/code_agent/<timestamp>/.
Every LLM call is cost-tracked via SlowBurn's CostLimit.

Usage:
    python demos/demo_code_agent.py
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

import logging  # noqa: E402

from concurry import CallLimit, LimitSet  # noqa: E402
from lib.agent_loop import run_agent  # noqa: E402
from lib.tools import TOOL_SCHEMAS, execute_tool_call  # noqa: E402

from slowburn.limits import CostLimit  # noqa: E402
from slowburn.reporter import CostReporter  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")  # noqa: E402

MODEL = "openrouter/z-ai/glm-4.5"
BUDGET_USD = 0.02
MAX_TOKENS = 800
MAX_STEPS = 15

SEED_CODE = '''\
import re

def extract_emails(text):
    """Extract email addresses from text."""
    emails = []
    for word in text.split():
        if "@" in word and "." in word:
            emails.append(word)
    return emails

def count_words(text):
    """Count words in text, excluding common stop words."""
    stop_words = ["the", "a", "an", "is", "are", "was", "were", "in", "on", "at"]
    words = text.lower().split()
    count = 0
    for w in words:
        if w not in stop_words:
            count = count + 1
    return count

def find_duplicates(items):
    """Find duplicate items in a list."""
    seen = []
    dupes = []
    for item in items:
        if item in seen:
            if item not in dupes:
                dupes.append(item)
        else:
            seen.append(item)
    return dupes
'''

SYSTEM_PROMPT = """\
You are an autonomous code improvement agent. You have these tools:
- read_file: Read files in your workspace
- write_file: Write/update files in your workspace
- list_dir: See what files exist
- search_web: Search for Python best practices and patterns

Your workspace has a file called 'solution.py' with Python functions that need improvement.

Your task for each iteration:
1. Read solution.py
2. Search the web for relevant Python best practices
3. Write an improved version of solution.py incorporating what you learned
4. Write a CHANGELOG.md documenting what you changed and why

Focus on: correctness, performance (use set/Counter), type hints, proper regex, docstrings, edge cases.
IMPORTANT: Always use your tools. Do NOT just describe changes — actually write the improved code."""


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).parent / "runs" / "code_agent" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    (runs_dir / "solution.py").write_text(SEED_CODE)

    print(f"{'=' * 70}")
    print("  Autonomous Code Agent (Real Tools)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    limit_set = LimitSet(
        limits=[
            CostLimit(budget_usd=BUDGET_USD, window_seconds=30),
            CallLimit(window_seconds=60, capacity=100),
        ],
        mode="thread",
        shared=True,
    )
    reporter = CostReporter()

    def tool_executor(name, args):
        return execute_tool_call(name, args, workspace=runs_dir)

    tasks = [
        (
            "Read solution.py, search the web for Python best practices for "
            "email extraction (proper regex), word counting (Counter), and "
            "finding duplicates (using sets). Then write an improved version "
            "of solution.py and write a CHANGELOG.md explaining your changes."
        ),
        (
            "Read the current solution.py. Search the web for Python type hints "
            "best practices. Add comprehensive type hints, improve docstrings "
            "with examples, and add input validation. Write the updated file."
        ),
        (
            "Read solution.py one more time. Search for edge cases in email "
            "regex patterns. Make sure extract_emails handles edge cases like "
            "emails in angle brackets <user@example.com>, trailing punctuation, "
            "and international domains. Update solution.py and append to CHANGELOG.md."
        ),
    ]

    start_time = time.time()

    for i, task in enumerate(tasks, 1):
        print(f"\n  --- Iteration {i}/{len(tasks)} ---")
        print(f"  {task[:80]}...")

        iter_log_dir = runs_dir / f"iteration_{i:02d}"
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
            log_dir=iter_log_dir,
        ))

        print(f"  Steps: {result['steps']}, Tool calls: {result['tool_calls']}")

    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total LLM calls: {reporter.num_calls}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    print(f"\n  Files in workspace ({runs_dir}):")
    for f in sorted(runs_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(runs_dir)}: {f.stat().st_size} bytes")

    solution = (runs_dir / "solution.py").read_text()
    print(f"\n  Final solution.py ({len(solution)} chars):")
    for line in solution.strip().split("\n")[:20]:
        print(f"    {line}")
    if solution.count("\n") > 20:
        print(f"    ... ({solution.count(chr(10)) - 20} more lines)")

    changelog = runs_dir / "CHANGELOG.md"
    if changelog.exists():
        print("\n  CHANGELOG.md:")
        for line in changelog.read_text().strip().split("\n")[:10]:
            print(f"    {line}")

    reporter.to_json(path=runs_dir / "_cost_report.json")
    print(f"\n  Cost report: {runs_dir / '_cost_report.json'}")


if __name__ == "__main__":
    main()
