"""
Demo: CrewAI multi-agent workflow with SlowBurn budget control.

Shows how to add cost control to an existing CrewAI workflow with
zero code changes to agents or tasks — just install the hooks.

Usage:
    pip install slowburn[crewai]
    export OPENAI_API_KEY="sk-..."
    python examples/demo_crewai_budget.py
"""

import os


def main():
    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        print("CrewAI not installed. Run: pip install slowburn[crewai]")
        return

    from slowburn.integrations.crewai import SlowBurnCrewAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY to run this demo.")
        print("Showing the setup pattern instead.\n")

    # === 1. Create SlowBurn cost controller ===
    sb = SlowBurnCrewAI(budget_usd=2.0, window_seconds=3600)  # $2/hour
    sb.install()

    # === 2. Define CrewAI agents (unchanged from normal CrewAI code) ===
    researcher = Agent(
        role="Research Analyst",
        goal="Find key insights about the given topic",
        backstory="You are a thorough research analyst.",
        verbose=True,
    )
    writer = Agent(
        role="Content Writer",
        goal="Write a compelling summary from the research",
        backstory="You are a skilled technical writer.",
        verbose=True,
    )

    # === 3. Define tasks (unchanged) ===
    research_task = Task(
        description="Research the topic: 'Cost optimization in LLM agents'. "
        "Identify 3 key challenges and 3 promising approaches.",
        expected_output="A structured list of challenges and approaches.",
        agent=researcher,
    )
    writing_task = Task(
        description="Write a 200-word executive summary based on the research.",
        expected_output="A polished executive summary paragraph.",
        agent=writer,
    )

    # === 4. Run crew (SlowBurn automatically paces all LLM calls) ===
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    if api_key:
        result = crew.kickoff()
        print(f"\n{'=' * 60}")
        print("RESULT:")
        print(result)
    else:
        print("[dry run] Would run crew with 2 agents and 2 tasks")

    # === 5. Report costs ===
    print(f"\n{'=' * 60}")
    print("COST REPORT")
    print(f"{'=' * 60}")
    print(sb.reporter.to_markdown())
    print(f"\nTotal cost: ${sb.reporter.total_cost():.6f}")

    sb.uninstall()
    print("\nDone.")


if __name__ == "__main__":
    main()
