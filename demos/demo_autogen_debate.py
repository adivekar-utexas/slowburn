"""
Demo: AutoGen/AG2 Multi-Agent Debate with SlowBurn cost control.

Inspired by: "Improving Factuality and Reasoning through Multiagent Debate"
(Du et al., ICML 2024), 1,354 citations.

Pattern: Multi-Agent Debate — 3 solver agents independently research the same
question, then argue across rounds until converging on a verified answer. An
Aggregator agent collects final answers and produces a verdict using majority
voting.

The shared budget ($0.15) is split across all 3 debaters + aggregator. As debate
rounds progress, the budget tightens and SlowBurn's backpressure forces later
rounds to slow down, demonstrating the cost-quality tradeoff in debate systems.

Usage:
    pip install slowburn[autogen] pyautogen
    export OPENAI_API_KEY="sk-..."
    python demos/demo_autogen_debate.py
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
BUDGET_USD = 0.15
WINDOW_SECONDS = 300
MAX_TOKENS = 500
NUM_DEBATE_ROUNDS = 2

QUESTION = (
    "What are the most effective strategies for reducing LLM inference costs "
    "in production, and what are the tradeoffs of each approach? "
    "Consider techniques like model distillation, prompt caching, batching, "
    "quantization, and routing to cheaper models."
)


def run_debate_round(
    debaters: list,
    round_num: int,
    question: str,
    previous_arguments: dict,
    runs_dir: Path,
    reporter,
) -> dict:
    """Run one round of debate where each agent researches and argues."""
    round_arguments = {}

    for debater_name, debater_config in debaters:
        print(f"\n      [{debater_name}] Researching and formulating argument...")

        search_query = f"{question} {debater_config['focus']}"
        results = search_web(search_query, max_results=3)
        search_data = json.loads(results)
        sources = ""
        if "results" in search_data:
            for r in search_data["results"]:
                sources += f"- {r['title']}: {r['snippet']} ({r['url']})\n"

        other_args = ""
        if len(previous_arguments) > 0:
            other_args = "\n\nOther debaters' arguments from the previous round:\n"
            for name, arg in previous_arguments.items():
                if name != debater_name:
                    other_args += f"\n--- {name} ---\n{arg}\n"

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    f"You are '{debater_name}', a debate participant. "
                    f"{debater_config['stance']} "
                    f"In each round, present your strongest argument with evidence. "
                    f"If other debaters make valid points, acknowledge them and "
                    f"update your position. Cite specific sources."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Debate Round {round_num}.\n\n"
                    f"Question: {question}\n\n"
                    f"Your web search results:\n{sources}"
                    f"{other_args}\n\n"
                    f"Present your argument in 150-200 words. Cite sources. "
                    f"If this is round 2+, respond to other debaters' arguments."
                ),
            },
        ]

        response = debater_config["client"].create(
            {
                "model": f"slowburn/{MODEL}",
                "messages": prompt_messages,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7,
            }
        )

        argument = response.choices[0].message.content
        round_arguments[debater_name] = argument
        print(f"      [{debater_name}] Argument: {argument[:100]}...")

        write_file(
            f"round_{round_num}_{debater_name.lower().replace(' ', '_')}.md",
            argument,
            workspace=runs_dir,
        )

    cost_so_far = reporter.total_cost()
    print(f"\n      Cost after round {round_num}: ${cost_so_far:.6f} / ${BUDGET_USD:.2f}")

    return round_arguments


def main():
    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        print("AutoGen (AG2) not installed. Run: pip install pyautogen")
        return

    from concurry import LimitSet
    from slowburn.integrations.autogen import SlowBurnModelClient
    from slowburn.limits import CostLimit
    from slowburn.reporter import CostReporter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).parent / "runs" / "autogen_debate" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    limit_set = LimitSet(
        limits=[CostLimit(budget_usd=BUDGET_USD, window_seconds=WINDOW_SECONDS)],
        mode="Threads",
        shared=True,
    )
    reporter = CostReporter()

    print(f"{'=' * 70}")
    print("  AutoGen Multi-Agent Debate (Du et al. inspired)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f} / {WINDOW_SECONDS}s window")
    print(f"  Debate rounds: {NUM_DEBATE_ROUNDS}")
    print(f"  Question: {QUESTION[:70]}...")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    # Create SlowBurn model clients for each debater (sharing a single budget)
    def make_client(name: str) -> SlowBurnModelClient:
        config = {"model": f"slowburn/{MODEL}", "api_key": api_key}
        return SlowBurnModelClient(
            config=config,
            limit_set=limit_set,
            reporter=reporter,
        )

    debaters = [
        (
            "Alice",
            {
                "focus": "model distillation quantization smaller models",
                "stance": (
                    "You focus on model-level optimizations: distillation, quantization, "
                    "and using smaller specialized models. You argue these provide the "
                    "best cost-per-quality tradeoff."
                ),
                "client": make_client("Alice"),
            },
        ),
        (
            "Bob",
            {
                "focus": "prompt caching batching infrastructure optimization",
                "stance": (
                    "You focus on infrastructure optimizations: prompt caching, request "
                    "batching, KV-cache reuse, and smart routing. You argue these provide "
                    "cost savings without model quality degradation."
                ),
                "client": make_client("Bob"),
            },
        ),
        (
            "Carol",
            {
                "focus": "prompt engineering cost-aware routing cascading models",
                "stance": (
                    "You focus on prompt-level and routing strategies: shorter prompts, "
                    "cascading from cheap to expensive models, and adaptive model "
                    "selection. You argue these are the easiest to implement."
                ),
                "client": make_client("Carol"),
            },
        ),
    ]

    start_time = time.time()
    previous_arguments = {}

    # Run debate rounds
    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        print(f"\n  === Debate Round {round_num}/{NUM_DEBATE_ROUNDS} ===")
        previous_arguments = run_debate_round(
            debaters,
            round_num,
            QUESTION,
            previous_arguments,
            runs_dir,
            reporter,
        )

    # Aggregator: synthesize the debate
    print(f"\n  === Aggregation ===")
    print(f"    [Aggregator] Synthesizing debate results...")

    all_arguments = "\n\n".join(
        f"--- {name} (Final Round) ---\n{arg}" for name, arg in previous_arguments.items()
    )

    aggregator_client = make_client("Aggregator")
    agg_response = aggregator_client.create(
        {
            "model": f"slowburn/{MODEL}",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a neutral debate moderator and aggregator. "
                        "Synthesize the debaters' arguments into a consensus verdict. "
                        "Report: (1) areas of agreement, (2) areas of disagreement, "
                        "(3) the strongest evidence-backed strategies, and "
                        "(4) a final recommendation with confidence level."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {QUESTION}\n\n"
                        f"Final arguments from all debaters:\n{all_arguments}\n\n"
                        f"Write a structured verdict (200-300 words) with consensus "
                        f"findings and a ranked recommendation."
                    ),
                },
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.3,
        }
    )

    verdict = agg_response.choices[0].message.content
    write_file("verdict.md", verdict, workspace=runs_dir)
    print(f"    [Aggregator] Verdict: {len(verdict)} chars")

    total_elapsed = time.time() - start_time

    # Report
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Debate rounds: {NUM_DEBATE_ROUNDS}")
    print(f"  Total LLM calls: {reporter.num_calls}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    print(f"\n  Files in workspace:")
    for f in sorted(runs_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(runs_dir)}: {f.stat().st_size} bytes")

    print(f"\n  Verdict ({len(verdict)} chars):")
    for line in verdict.strip().split("\n")[:15]:
        print(f"    {line}")
    remaining = verdict.count("\n") - 15
    if remaining > 0:
        print(f"    ... ({remaining} more lines)")

    reporter.to_json(path=runs_dir / "_cost_report.json")
    print(f"\n  Cost report: {runs_dir / '_cost_report.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
