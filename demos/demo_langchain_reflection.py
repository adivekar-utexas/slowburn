"""
Demo: LangChain Reflection/Self-Critique Writing Agent with SlowBurn cost control.

Inspired by: Reflexion (Shinn et al., NeurIPS 2023)

Pattern: Reflection / Self-Critique Loop — a Generator agent produces content,
then a Critic agent evaluates the output against explicit criteria, and the
Generator revises based on the critique. This iterative refinement loop continues
until quality criteria are met or budget is exhausted.

The SlowBurnCallbackHandler is attached to the ChatOpenAI instance, so every
LLM call (both generator and critic) is cost-tracked. When the budget runs low,
the reflection loop is forced to terminate early — a natural demonstration of
budget-constrained quality.

Usage:
    pip install slowburn[langchain] langchain-openai
    export OPENAI_API_KEY="sk-..."
    python demos/demo_langchain_reflection.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY in .env to run this demo.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

from lib.tools import search_web, write_file  # noqa: E402

MODEL = "gpt-4o-mini"
BUDGET_USD = 0.08
WINDOW_SECONDS = 300
MAX_TOKENS = 500
MAX_ROUNDS = 3
TOPIC = "How backpressure works in distributed systems"


def parse_scores(critique_text: str) -> dict:
    """Extract numeric scores from critique text.

    Looks for lines like "Factual Accuracy: 4/5" or "Clarity: 3/5".
    Returns dict mapping criterion -> score (int).
    """
    import re

    scores = {}
    for line in critique_text.split("\n"):
        match = re.search(
            r"(factual accuracy|completeness|clarity|source quality)\s*:\s*(\d)\s*/\s*5", line, re.IGNORECASE
        )
        if match:
            scores[match.group(1).lower()] = int(match.group(2))
    return scores


def main():
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("langchain-openai not installed. Run: pip install langchain-openai")
        return

    from slowburn.integrations.langchain import SlowBurnCallbackHandler
    from slowburn.reporter import CostReporter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).parent / "runs" / "langchain_reflection" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    reporter = CostReporter()
    budget_handler = SlowBurnCallbackHandler(
        budget_usd=BUDGET_USD,
        window_seconds=WINDOW_SECONDS,
        reporter=reporter,
    )

    llm = ChatOpenAI(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        callbacks=[budget_handler],
    )

    print(f"{'=' * 70}")
    print("  LangChain Reflection Agent (Reflexion-inspired)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f} / {WINDOW_SECONDS}s window")
    print(f"  Topic: {TOPIC}")
    print(f"  Max rounds: {MAX_ROUNDS}")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    # Step 0: Search the web for background information
    print("\n  [0] Searching the web for background...")
    search_results = search_web(f"{TOPIC} explanation examples", max_results=5)
    search_data = json.loads(search_results)
    sources_text = ""
    if "results" in search_data:
        for r in search_data["results"]:
            sources_text += f"- {r['title']}: {r['snippet']} ({r['url']})\n"
        print(f"      Found {len(search_data['results'])} sources")
    write_file("search_results.md", sources_text, workspace=runs_dir)

    start_time = time.time()
    draft = None
    critique_text = None

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n  --- Round {round_num}/{MAX_ROUNDS} ---")

        # GENERATE
        generate_prompt = (
            f"Write a well-researched, factually accurate 300-word technical explanation of: {TOPIC}\n\n"
            f"Use these web search results as sources:\n{sources_text}\n\n"
            "Include specific technical details, cite URLs where relevant, "
            "and make the explanation accessible to a senior engineer."
        )
        if critique_text:
            generate_prompt += f"\n\nAddress this critique from the previous round:\n{critique_text}"

        print(f"    [Gen] Generating draft (round {round_num})...")
        gen_response = llm.invoke(generate_prompt)
        draft = gen_response.content
        print(f"          Draft: {len(draft)} chars")

        write_file(f"draft_round_{round_num}.md", draft, workspace=runs_dir)

        # CRITIQUE
        critique_prompt = (
            f"Evaluate the following technical explanation on 4 criteria. "
            f"Score each 1-5 (5=excellent). Provide specific, actionable revision "
            f"instructions for any score below 4.\n\n"
            f"Criteria:\n"
            f"- Factual Accuracy: Are claims supported by the provided sources?\n"
            f"- Completeness: Are key aspects of {TOPIC} covered?\n"
            f"- Clarity: Is the explanation accessible to a senior engineer?\n"
            f"- Source Quality: Are citations real and relevant?\n\n"
            f"Format each score as 'CRITERION: N/5' on its own line.\n\n"
            f"Draft to evaluate:\n{draft}\n\n"
            f"Available sources:\n{sources_text}"
        )

        print(f"    [Crit] Evaluating draft...")
        crit_response = llm.invoke(critique_prompt)
        critique_text = crit_response.content

        scores = parse_scores(critique_text)
        print(f"          Scores: {scores}")

        write_file(f"critique_round_{round_num}.md", critique_text, workspace=runs_dir)

        cost_so_far = reporter.total_cost()
        print(f"          Cost so far: ${cost_so_far:.6f} / ${BUDGET_USD:.2f}")

        if len(scores) >= 3 and all(s >= 4 for s in scores.values()):
            print(f"    All criteria >= 4/5. Quality threshold met!")
            break

        if round_num < MAX_ROUNDS:
            # Additional search to address gaps
            print(f"    [Search] Searching for additional info to address gaps...")
            gap_query = f"{TOPIC} " + " ".join(k for k, v in scores.items() if v < 4)
            extra_results = search_web(gap_query, max_results=3)
            extra_data = json.loads(extra_results)
            if "results" in extra_data:
                for r in extra_data["results"]:
                    sources_text += f"- {r['title']}: {r['snippet']} ({r['url']})\n"
                print(f"          Found {len(extra_data['results'])} additional sources")

    # Write final output
    write_file("final_output.md", draft, workspace=runs_dir)

    total_elapsed = time.time() - start_time

    # Report
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Rounds completed: {round_num}")
    print(f"  Total LLM calls: {reporter.num_calls}")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    print(f"\n  Files in workspace:")
    for f in sorted(runs_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(runs_dir)}: {f.stat().st_size} bytes")

    if draft:
        print(f"\n  Final output ({len(draft)} chars):")
        for line in draft.strip().split("\n")[:10]:
            print(f"    {line}")
        remaining = draft.count("\n") - 10
        if remaining > 0:
            print(f"    ... ({remaining} more lines)")

    reporter.to_json(path=runs_dir / "_cost_report.json")
    print(f"\n  Cost report: {runs_dir / '_cost_report.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
