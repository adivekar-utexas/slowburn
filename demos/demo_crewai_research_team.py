"""
Demo: CrewAI Role-Playing Research Team with SlowBurn cost control.

Inspired by: CAMEL (Li et al., NeurIPS 2023) + MetaGPT (Hong et al., ICLR 2024)

Pattern: Role-Playing with Structured Handoffs — 3 agents with distinct personas
collaborate through artifact passing. The Research Analyst searches the web and
writes a fact sheet. The Critical Reviewer searches for contradicting evidence
and writes a critique. The Executive Synthesizer reads both and writes a balanced
brief with confidence ratings.

The tight budget ($0.10) forces agents to be concise. SlowBurn's backpressure
visibly slows the third agent when the first two consume most of the budget.

Usage:
    pip install slowburn[crewai] crewai crewai-tools
    export OPENAI_API_KEY="sk-..."
    python demos/demo_crewai_research_team.py
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

MODEL = "gpt-4o-mini"
BUDGET_USD = 0.10
WINDOW_SECONDS = 300
TOPIC = "cost optimization strategies for LLM agents in production"


def main():
    try:
        from crewai import Agent, Crew, Task, LLM
    except ImportError:
        print("CrewAI not installed. Run: pip install crewai")
        return

    try:
        from crewai_tools import tool as crewai_tool
    except ImportError:
        crewai_tool = None

    from slowburn.integrations.crewai import SlowBurnCrewAI
    from slowburn.reporter import CostReporter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(__file__).parent.parent / "runs" / "crewai_research_team" / timestamp
    runs_dir.mkdir(parents=True, exist_ok=True)

    reporter = CostReporter()
    sb = SlowBurnCrewAI(
        budget_usd=BUDGET_USD,
        window_seconds=WINDOW_SECONDS,
        reporter=reporter,
    )
    sb.install()

    print(f"{'=' * 70}")
    print("  CrewAI Research Team (CAMEL/MetaGPT-inspired)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f} / {WINDOW_SECONDS}s window")
    print(f"  Topic: {TOPIC}")
    print(f"  Workspace: {runs_dir}")
    print(f"{'=' * 70}")

    # Define the LLM for all agents
    llm = LLM(model=MODEL, max_tokens=500, temperature=0.7)

    # --- Agent 1: Research Analyst ---
    analyst = Agent(
        role="Research Analyst",
        goal=(
            f"Find 5 key facts and recent developments about: {TOPIC}. "
            "Use web search to find real data, specific dollar amounts, "
            "and named tools/frameworks. Write a structured fact sheet."
        ),
        backstory=(
            "You are a meticulous technology researcher who always backs "
            "claims with specific data points and URLs. You never speculate "
            "without evidence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --- Agent 2: Critical Reviewer ---
    reviewer = Agent(
        role="Critical Reviewer",
        goal=(
            "Read the Research Analyst's fact sheet and critically evaluate it. "
            "Search the web for contradicting evidence or missing perspectives. "
            "Write a structured critique identifying gaps, unsupported claims, "
            "and areas needing more evidence."
        ),
        backstory=(
            "You are a skeptical fact-checker and peer reviewer. You always "
            "look for what's missing, what's overstated, and what evidence "
            "contradicts the claims. You search for disconfirming evidence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --- Agent 3: Executive Synthesizer ---
    synthesizer = Agent(
        role="Executive Synthesizer",
        goal=(
            "Read both the research fact sheet and the critical review. "
            "Write a balanced 200-word executive brief that includes: "
            "(1) the most well-supported findings, "
            "(2) areas of uncertainty, and "
            "(3) confidence ratings (High/Medium/Low) for each claim."
        ),
        backstory=(
            "You are a senior technology strategist who distills complex "
            "research into actionable briefs for leadership. You always "
            "distinguish between well-supported and speculative claims."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --- Tasks (sequential pipeline with artifact passing) ---
    research_task = Task(
        description=(
            f"Research the topic: '{TOPIC}'. "
            "Search the web for recent developments, specific cost data, "
            "and named tools/frameworks. Write a structured fact sheet with "
            "at least 5 key findings, each with a source URL. "
            "Format as markdown with headers and bullet points."
        ),
        expected_output=(
            "A structured markdown fact sheet with 5+ key findings, "
            "each backed by a specific source URL."
        ),
        agent=analyst,
        output_file=str(runs_dir / "facts.md"),
    )

    review_task = Task(
        description=(
            "Read the Research Analyst's fact sheet (provided as context). "
            "Search the web for contradicting evidence or alternative perspectives. "
            "Write a structured critique that identifies: "
            "(1) unsupported or weakly supported claims, "
            "(2) missing perspectives or data, "
            "(3) areas where the evidence is strong. "
            "Format as markdown."
        ),
        expected_output=(
            "A structured critique with specific gaps, unsupported claims, "
            "and areas of strong evidence."
        ),
        agent=reviewer,
        context=[research_task],
        output_file=str(runs_dir / "critique.md"),
    )

    brief_task = Task(
        description=(
            "Read both the Research Analyst's fact sheet and the Critical Reviewer's "
            "critique (both provided as context). Write a balanced 200-word executive "
            "brief that synthesizes the research, acknowledges uncertainties, and "
            "assigns confidence ratings (High/Medium/Low) to each major claim. "
            "Format as markdown."
        ),
        expected_output=(
            "A balanced 200-word executive brief with confidence ratings "
            "for each major claim."
        ),
        agent=synthesizer,
        context=[research_task, review_task],
        output_file=str(runs_dir / "executive_brief.md"),
    )

    # --- Run the crew ---
    crew = Crew(
        agents=[analyst, reviewer, synthesizer],
        tasks=[research_task, review_task, brief_task],
        verbose=True,
    )

    start_time = time.time()
    print("\n  Starting crew execution...\n")

    result = crew.kickoff()

    total_elapsed = time.time() - start_time

    sb.uninstall()

    # Report
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total LLM calls: {reporter.num_calls}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    print(f"\n  Files in workspace:")
    for f in sorted(runs_dir.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(runs_dir)}: {f.stat().st_size} bytes")

    brief_path = runs_dir / "executive_brief.md"
    if brief_path.exists():
        brief_text = brief_path.read_text()
        print(f"\n  Executive Brief ({len(brief_text)} chars):")
        for line in brief_text.strip().split("\n")[:15]:
            print(f"    {line}")

    reporter.to_json(path=runs_dir / "_cost_report.json")
    print(f"\n  Cost report: {runs_dir / '_cost_report.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
