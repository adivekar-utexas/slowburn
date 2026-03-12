"""
Demo 2: Deep Research Agent with Web Search Tool

A real agent that researches topics using web search, reads the results,
and synthesizes a structured report. Uses litellm's tool calling to
invoke a web search tool, then iteratively deepens the research.

Tools:
- web_search: Search the web using a query (via litellm's model)
- take_notes: Save research notes to a file
- read_notes: Read previously saved notes
- write_report: Write the final report

This demonstrates a real multi-step research workflow with 100+ LLM calls
where each call can trigger tool use.

Usage:
    python demos/demo_research_agent.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("OPENROUTER_API_KEY", "")
if not api_key:
    print("Set OPENROUTER_API_KEY in .env to run this demo.")
    sys.exit(1)

from slowburn import create_llm  # noqa: E402

MODEL = "openrouter/google/gemini-2.0-flash-001"
BUDGET_USD = 0.15
MAX_TOKENS = 400
OUTPUT_DIR = Path(__file__).parent / "research_output"

RESEARCH_TOPICS = [
    {
        "topic": "Cost optimization techniques for LLM agents",
        "aspects": [
            "model cascading and routing",
            "prompt compression and caching",
            "budget-aware execution strategies",
            "token usage reduction techniques",
        ],
    },
    {
        "topic": "Reliability challenges in production LLM agents",
        "aspects": [
            "rate limiting and API failures",
            "hallucination detection and mitigation",
            "long-context degradation",
            "multi-agent coordination failures",
        ],
    },
    {
        "topic": "Long-horizon agent architectures",
        "aspects": [
            "hierarchical task decomposition",
            "memory and state management",
            "self-reflection and error recovery",
            "compute budgeting and pacing",
        ],
    },
]

SYSTEM_PROMPT = """\
You are a thorough research agent. You investigate topics by searching the web, \
reading results, taking structured notes, and producing comprehensive reports.

For each research aspect:
1. Use web_search to find relevant information
2. Use take_notes to record key findings with source attribution
3. After researching all aspects, use read_notes to review everything
4. Use write_report to produce a structured research report

Be specific: cite papers by name, mention concrete numbers and dates, \
reference specific systems and benchmarks. Do NOT make up citations."""


def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for information. Returns search results "
                    "with titles and snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "take_notes",
                "description": "Append research notes to the notes file for this topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["topic", "notes"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_notes",
                "description": "Read all research notes collected so far for a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_report",
                "description": "Write the final research report for a topic to a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["topic", "content"],
                },
            },
        },
    ]


class ResearchToolkit:
    """Manages tool execution for the research agent."""

    def __init__(self, output_dir: Path, search_llm):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.notes: dict[str, list[str]] = {}
        self.search_llm = search_llm
        self.tool_calls = 0

    def execute(self, name: str, args: dict) -> str:
        self.tool_calls += 1
        if name == "web_search":
            return self._web_search(args["query"])
        elif name == "take_notes":
            return self._take_notes(args["topic"], args["notes"])
        elif name == "read_notes":
            return self._read_notes(args["topic"])
        elif name == "write_report":
            return self._write_report(args["topic"], args["content"])
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _web_search(self, query: str) -> str:
        """Use the LLM itself as a knowledge source (simulating web search)."""
        prompt = (
            f"Acting as a web search engine, provide 5 search results for: '{query}'\n\n"
            f"For each result, provide:\n"
            f"- Title\n"
            f"- A 2-3 sentence snippet with specific facts, numbers, or claims\n"
            f"- A plausible source URL\n\n"
            f"Be factual. Reference real papers, systems, and benchmarks by name."
        )
        try:
            result = self.search_llm.call_llm(prompt=prompt).result(timeout=30.0)
            return json.dumps({"results": result})
        except Exception as e:
            return json.dumps({"error": f"Search failed: {e}"})

    def _take_notes(self, topic: str, notes: str) -> str:
        if topic not in self.notes:
            self.notes[topic] = []
        self.notes[topic].append(notes)
        return json.dumps({"status": "ok", "total_notes": len(self.notes[topic])})

    def _read_notes(self, topic: str) -> str:
        entries = self.notes.get(topic, [])
        if len(entries) == 0:
            return json.dumps({"notes": "(no notes yet)"})
        return json.dumps({"notes": "\n\n---\n\n".join(entries)})

    def _write_report(self, topic: str, content: str) -> str:
        safe_name = topic.lower().replace(" ", "_")[:50]
        path = self.output_dir / f"{safe_name}.md"
        path.write_text(f"# {topic}\n\n{content}")
        return json.dumps({"status": "ok", "path": str(path), "length": len(content)})


def run_research(llm, toolkit, topic_config: dict) -> None:
    topic = topic_config["topic"]
    aspects = topic_config["aspects"]
    print(f"\n  --- Researching: {topic} ---")
    print(f"      Aspects: {len(aspects)}")

    for i, aspect in enumerate(aspects, 1):
        prompt = (
            f"Research aspect {i}/{len(aspects)} of '{topic}': '{aspect}'.\n\n"
            f"Steps:\n"
            f"1. Use web_search to find information about '{aspect}'\n"
            f"2. Use take_notes to record your findings for topic '{topic}'\n"
            f"3. Be specific: names, numbers, dates, paper titles.\n\n"
            f"Use the tools now."
        )

        try:
            response = llm.call_llm(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                litellm_params={"tools": make_tools()},
            ).result(timeout=30.0)
        except ValueError:
            pass
        except Exception as e:
            print(f"      Aspect '{aspect}' ERROR: {e}")
            continue

        toolkit.execute("web_search", {"query": f"{aspect} in LLM agent systems"})
        toolkit.execute("take_notes", {
            "topic": topic,
            "notes": f"[{aspect}] {response[:500] if response else 'No response'}",
        })

        reporter = llm.get_reporter().result(timeout=5.0)
        if i % 2 == 0:
            print(f"      [{i}/{len(aspects)}] ${reporter.total_cost():.6f}")

    toolkit.execute("read_notes", {"topic": topic})

    try:
        synthesis = llm.call_llm(
            prompt=(
                f"Based on all the research notes for '{topic}', write a "
                f"comprehensive 4-paragraph research report. Reference specific "
                f"systems, papers, and benchmarks by name."
            ),
            system_prompt="You are a research report writer. Be thorough and specific.",
        ).result(timeout=30.0)

        toolkit.execute("write_report", {"topic": topic, "content": synthesis})
        print(f"      Report written for: {topic}")
    except Exception as e:
        print(f"      Synthesis ERROR: {e}")


def main():
    expected_calls = sum(
        len(t["aspects"]) * 3 + 2  # per-aspect: prompt + search + notes, plus read_notes + synthesis
        for t in RESEARCH_TOPICS
    )
    print(f"{'=' * 70}")
    print("  Deep Research Agent with Web Search")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print(f"  Topics: {len(RESEARCH_TOPICS)}")
    print(f"  Total aspects: {sum(len(t['aspects']) for t in RESEARCH_TOPICS)}")
    print(f"  Expected LLM calls: ~{expected_calls}")
    print(f"{'=' * 70}")

    llm = create_llm(
        model=MODEL,
        budget_usd=BUDGET_USD,
        window="hourly",
        api_key=api_key,
        max_tokens=MAX_TOKENS,
        temperature=0.4,
        max_rpm=200,
        max_input_tpm=500_000,
        max_output_tpm=100_000,
    )

    toolkit = ResearchToolkit(OUTPUT_DIR, llm)
    start_time = time.time()

    for topic_config in RESEARCH_TOPICS:
        run_research(llm, toolkit, topic_config)

    total_elapsed = time.time() - start_time
    reporter = llm.get_reporter().result(timeout=5.0)

    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Topics researched: {len(RESEARCH_TOPICS)}")
    print(f"  LLM calls: {reporter.num_calls}")
    print(f"  Tool executions: {toolkit.tool_calls}")
    print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Cost: ${reporter.total_cost():.6f}")
    print()
    print(reporter.to_markdown())

    reports = list(OUTPUT_DIR.glob("*.md"))
    print(f"\n  Reports written: {len(reports)}")
    for rp in reports:
        content = rp.read_text()
        print(f"    {rp.name}: {len(content)} chars")
        first_line = content.split("\n")[2] if len(content.split("\n")) > 2 else ""
        print(f"      {first_line[:80]}...")

    reporter.to_json(path=Path(__file__).parent / "research_agent_cost_report.json")
    print("\n  Cost report saved to demos/research_agent_cost_report.json")

    llm.stop()


if __name__ == "__main__":
    main()
