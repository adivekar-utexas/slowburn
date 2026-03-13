"""
Demo: LangGraph Plan-and-Execute Research Pipeline with SlowBurn cost control.

Inspired by: Plan-and-Solve Prompting (Wang et al., ACL 2023) + LLM+P (Liu et al., 2023)

Pattern: Plan-and-Execute — a Planner node creates an explicit multi-step plan,
then an Executor node carries out each step using tools, with a Replanner that
modifies the plan based on intermediate results. This separates strategic
thinking from tactical execution.

SlowBurn's SlowBurnMiddleware wraps every model call in the graph. The tight
budget ($0.12) means later executor steps visibly slow down as the budget
tightens — a natural demonstration of backpressure.

Usage:
    pip install slowburn[langgraph] langgraph langchain-openai
    export OPENAI_API_KEY="sk-..."
    python demos/demo_langgraph_plan_execute.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY in .env to run this demo.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

from lib.tools import search_web, write_file  # noqa: E402

MODEL = "gpt-4o-mini"
BUDGET_USD = 0.12
WINDOW_SECONDS = 300
MAX_TOKENS = 500
TASK = "Compare the pricing, features, and market reception of the top 3 LLM inference providers (OpenAI, Anthropic, Google)"


class PlanState(TypedDict):
    task: str
    plan: List[str]
    completed: List[Dict[str, Any]]
    current_step: int
    final_report: str
    workspace: str


def parse_plan(text: str) -> List[str]:
    """Extract numbered steps from LLM-generated plan text."""
    steps = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix_len in range(1, 4):
            for sep in [".", ")", ":", "-"]:
                prefix = line[:prefix_len + 1]
                if prefix.rstrip(sep).isdigit() and sep in prefix:
                    step_text = line[prefix_len + 1:].strip()
                    if len(step_text) > 5:
                        steps.append(step_text)
                    break
    if len(steps) == 0:
        steps = [line.strip("- ").strip() for line in text.strip().split("\n") if len(line.strip()) > 10]
    return steps[:6]


def main():
    try:
        from langgraph.graph import StateGraph, END
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("LangGraph/LangChain not installed. Run: pip install langgraph langchain-openai")
        return

    from slowburn.integrations.langchain import SlowBurnCallbackHandler
    from slowburn.reporter import CostReporter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).parent / "runs" / "langgraph_plan_execute" / timestamp
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
        temperature=0.3,
        callbacks=[budget_handler],
    )

    print(f"{'=' * 70}")
    print("  LangGraph Plan-and-Execute Pipeline (Plan-and-Solve inspired)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f} / {WINDOW_SECONDS}s window")
    print(f"  Task: {TASK[:70]}...")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    # --- Define graph nodes ---

    def planner(state: PlanState) -> Dict[str, Any]:
        """Create an explicit multi-step plan for the task."""
        print("\n    [Planner] Creating research plan...")
        response = llm.invoke(
            f"You are a research planner. Break this task into 4-6 concrete, "
            f"actionable research steps. Each step should be a specific action "
            f"like 'Search for X' or 'Compare Y and Z' or 'Write summary of W'.\n\n"
            f"Task: {state['task']}\n\n"
            f"Output ONLY numbered steps (1. ..., 2. ..., etc.), nothing else."
        )
        steps = parse_plan(response.content)
        print(f"          Plan ({len(steps)} steps):")
        for i, step in enumerate(steps, 1):
            print(f"            {i}. {step[:70]}")

        write_file("plan.md", "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
                    workspace=Path(state["workspace"]))

        return {"plan": steps, "current_step": 0, "completed": []}

    def executor(state: PlanState) -> Dict[str, Any]:
        """Execute the current step using web search and LLM."""
        idx = state["current_step"]
        step = state["plan"][idx]
        print(f"\n    [Executor] Step {idx + 1}/{len(state['plan'])}: {step[:60]}...")

        search_results = search_web(step, max_results=5)
        search_data = json.loads(search_results)
        context_text = ""
        if "results" in search_data:
            for r in search_data["results"]:
                context_text += f"- {r['title']}: {r['snippet']} ({r['url']})\n"
            print(f"              Found {len(search_data['results'])} search results")

        prior_context = ""
        if len(state["completed"]) > 0:
            prior_context = "\n\nPrior findings:\n"
            for c in state["completed"]:
                prior_context += f"- Step '{c['step']}': {c['result'][:200]}\n"

        response = llm.invoke(
            f"You are a research executor. Complete this step using the provided "
            f"search results. Be specific with numbers, names, and URLs.\n\n"
            f"Step to execute: {step}\n\n"
            f"Search results:\n{context_text}"
            f"{prior_context}\n\n"
            f"Write a concise but detailed summary of your findings for this step."
        )

        result_text = response.content
        print(f"              Result: {len(result_text)} chars")

        completed = list(state["completed"])
        completed.append({"step": step, "result": result_text, "sources": context_text})

        write_file(f"step_{idx + 1}_result.md", result_text,
                    workspace=Path(state["workspace"]))

        cost_so_far = reporter.total_cost()
        print(f"              Cost so far: ${cost_so_far:.6f} / ${BUDGET_USD:.2f}")

        return {"completed": completed, "current_step": idx + 1}

    def route_after_executor(state: PlanState) -> str:
        """Routing function: decide whether to continue executing or synthesize."""
        idx = state["current_step"]
        total_steps = len(state["plan"])

        if idx >= total_steps:
            print(f"\n    [Replanner] All {total_steps} steps complete. Moving to synthesis.")
            return "synthesize"

        cost_so_far = reporter.total_cost()
        budget_remaining = BUDGET_USD - cost_so_far
        cost_per_step = cost_so_far / max(idx, 1)

        if budget_remaining < cost_per_step * 1.5:
            print(f"\n    [Replanner] Budget tight (${budget_remaining:.4f} remaining, "
                  f"~${cost_per_step:.4f}/step). Skipping to synthesis.")
            return "synthesize"

        print(f"\n    [Replanner] Continuing to step {idx + 1} "
              f"(${budget_remaining:.4f} remaining)")
        return "execute"

    def synthesizer(state: PlanState) -> Dict[str, Any]:
        """Write the final report from all completed steps."""
        print(f"\n    [Synthesizer] Writing final report from {len(state['completed'])} steps...")

        findings = "\n\n".join(
            f"### {c['step']}\n{c['result']}"
            for c in state["completed"]
        )

        response = llm.invoke(
            f"You are a research synthesizer. Write a comprehensive comparison "
            f"report based on the following research findings. Include a summary "
            f"table, key takeaways, and recommendations.\n\n"
            f"Original task: {state['task']}\n\n"
            f"Research findings:\n{findings}\n\n"
            f"Write a structured markdown report with: "
            f"(1) Executive Summary, (2) Comparison Table, (3) Key Takeaways, "
            f"(4) Recommendation."
        )

        report = response.content
        write_file("final_report.md", report, workspace=Path(state["workspace"]))
        print(f"              Report: {len(report)} chars")

        return {"final_report": report}

    # --- Build the graph ---
    graph = StateGraph(PlanState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "execute": "executor",
            "synthesize": "synthesizer",
        },
    )
    graph.add_edge("synthesizer", END)

    app = graph.compile()

    # --- Run ---
    start_time = time.time()

    initial_state: PlanState = {
        "task": TASK,
        "plan": [],
        "completed": [],
        "current_step": 0,
        "final_report": "",
        "workspace": str(runs_dir),
    }

    final_state = app.invoke(initial_state)

    total_elapsed = time.time() - start_time

    # Report
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Steps completed: {len(final_state.get('completed', []))}")
    print(f"  Total LLM calls: {reporter.num_calls}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    print(f"\n  Files in workspace:")
    for f in sorted(runs_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(runs_dir)}: {f.stat().st_size} bytes")

    report = final_state.get("final_report", "")
    if report:
        print(f"\n  Final Report ({len(report)} chars):")
        for line in report.strip().split("\n")[:15]:
            print(f"    {line}")
        remaining = report.count("\n") - 15
        if remaining > 0:
            print(f"    ... ({remaining} more lines)")

    reporter.to_json(path=runs_dir / "_cost_report.json")
    print(f"\n  Cost report: {runs_dir / '_cost_report.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
