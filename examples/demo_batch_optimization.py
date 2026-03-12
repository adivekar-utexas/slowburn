"""
Demo: Batch prompt optimization under a daily dollar budget.

Shows how SlowBurn automatically slows down (backpressure) when
approaching the budget limit, allowing long-running experiments
to complete overnight without crashing.

Without SlowBurn:
    Runs at full speed, burns $15 in 20 minutes, crashes on rate limit.

With SlowBurn (budget_usd=5.0, window="daily"):
    Automatically paces API calls. All iterations complete within budget.
    Cost report shows per-step breakdown.

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python examples/demo_batch_optimization.py
"""

import os
import time


def main():
    from slowburn import create_llm

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this demo with real API calls.")
        print("Showing the setup pattern instead.\n")

    # === 1. Create a cost-controlled LLM worker ===
    llm = create_llm(
        model="gpt-4o-mini",
        budget_usd=5.0,
        window="daily",
        max_rpm=500,
        max_input_tpm=1_000_000,
        max_output_tpm=200_000,
        api_key=api_key,
        temperature=0.7,
        max_tokens=512,
    )

    # === 2. Simulate a prompt optimization loop ===
    base_prompt = "Write a concise summary of the following concept: {concept}"
    concepts = [
        "gradient descent",
        "attention mechanisms",
        "reinforcement learning",
        "diffusion models",
        "mixture of experts",
    ]

    num_iterations = 3
    print(f"Running {num_iterations} optimization iterations over {len(concepts)} concepts...\n")

    for iteration in range(num_iterations):
        print(f"--- Iteration {iteration + 1}/{num_iterations} ---")
        start = time.time()

        # Generate candidate prompts (batch LLM call)
        prompts = [base_prompt.format(concept=c) for c in concepts]

        if api_key:
            results = llm.call_llm_batch(prompts=prompts).result()
            elapsed = time.time() - start
            print(f"  Completed {len(results)} calls in {elapsed:.1f}s")
        else:
            print(f"  [dry run] Would send {len(prompts)} prompts to gpt-4o-mini")

    # === 3. Report costs ===
    reporter = llm.get_reporter().result()
    print(f"\n{'=' * 60}")
    print("COST REPORT")
    print(f"{'=' * 60}")
    print(f"Total calls: {reporter.num_calls}")
    print(f"Total cost:  ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())
    print()

    # Optionally save to JSON
    # reporter.to_json(Path("cost_report.json"))

    llm.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
