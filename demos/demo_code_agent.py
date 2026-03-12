"""
Demo 1: Autonomous Code Improvement Agent (AutoResearch-style)

A real agent with tool use that iteratively improves a Python file.
Each iteration: reads the file, asks the LLM for improvements, writes
the improved version back, then runs tests to verify it works.

Tools:
- read_file: Read a file's contents
- write_file: Write/overwrite a file
- run_command: Execute a shell command (for running tests)
- list_dir: List directory contents

This makes 100+ real LLM calls with tool calling, not just prompt loops.

Usage:
    python demos/demo_code_agent.py
"""

import json
import os
import subprocess
import sys
import tempfile
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
BUDGET_USD = 0.10
MAX_TOKENS = 300
NUM_ITERATIONS = 50

SEED_CODE = '''\
def sort_and_deduplicate(items):
    """Sort a list and remove duplicates."""
    result = []
    for item in sorted(items):
        if item not in result:
            result.append(item)
    return result


def word_frequency(text):
    """Count word frequencies in text."""
    words = text.split()
    freq = {}
    for w in words:
        if w in freq:
            freq[w] = freq[w] + 1
        else:
            freq[w] = 1
    return freq


def flatten(nested_list):
    """Flatten a nested list."""
    result = []
    for item in nested_list:
        if type(item) == list:
            for sub in item:
                result.append(sub)
        else:
            result.append(item)
    return result
'''

SEED_TESTS = '''\
from solution import sort_and_deduplicate, word_frequency, flatten

def test_sort_deduplicate_basic():
    assert sort_and_deduplicate([3, 1, 2, 1, 3]) == [1, 2, 3]

def test_sort_deduplicate_empty():
    assert sort_and_deduplicate([]) == []

def test_sort_deduplicate_strings():
    assert sort_and_deduplicate(["b", "a", "b"]) == ["a", "b"]

def test_word_frequency_basic():
    assert word_frequency("the cat sat on the mat") == {
        "the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1
    }

def test_word_frequency_empty():
    assert word_frequency("") == {}

def test_flatten_basic():
    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]

def test_flatten_no_nesting():
    assert flatten([1, 2, 3]) == [1, 2, 3]

def test_flatten_deep():
    assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, [4, 5]]

if __name__ == "__main__":
    test_sort_deduplicate_basic()
    test_sort_deduplicate_empty()
    test_sort_deduplicate_strings()
    test_word_frequency_basic()
    test_word_frequency_empty()
    test_flatten_basic()
    test_flatten_no_nesting()
    test_flatten_deep()
    print("All tests passed!")
'''

SYSTEM_PROMPT = """\
You are an autonomous code improvement agent. You have tools to read files, \
write files, list directories, and run shell commands.

Your task: iteratively improve the Python code in solution.py. On each iteration:
1. Read the current solution.py
2. Run the tests (python test_solution.py) to see what passes/fails
3. Analyze the code for improvements (performance, readability, edge cases, \
type hints, docstrings, Pythonic patterns)
4. Write an improved version of solution.py
5. Run the tests again to verify your changes don't break anything

Be concrete. Make real changes. Use set() for deduplication, Counter for \
word frequency, recursion for deep flatten, add type hints, etc.

IMPORTANT: Always use the tools. Do not just describe what you would do — \
actually do it by calling read_file, write_file, and run_command."""


def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file at the given path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file at the given path (creates or overwrites).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command. Returns stdout, stderr, and return code.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def execute_tool(name: str, args: dict, workdir: str) -> str:
    if name == "read_file":
        p = Path(workdir) / args["path"]
        if not p.exists():
            return json.dumps({"error": f"File not found: {args['path']}"})
        return json.dumps({"content": p.read_text()})
    elif name == "write_file":
        p = Path(workdir) / args["path"]
        p.write_text(args["content"])
        return json.dumps({"status": "ok", "bytes_written": len(args["content"])})
    elif name == "run_command":
        cp = subprocess.run(
            args["command"], shell=True, capture_output=True, text=True,
            timeout=15, cwd=workdir,
        )
        return json.dumps({
            "stdout": cp.stdout[:2000],
            "stderr": cp.stderr[:1000],
            "returncode": cp.returncode,
        })
    elif name == "list_dir":
        p = Path(workdir) / args["path"]
        if not p.is_dir():
            return json.dumps({"error": f"Not a directory: {args['path']}"})
        files = [f.name for f in sorted(p.iterdir())]
        return json.dumps({"files": files})
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def main():
    print(f"{'=' * 70}")
    print("  Autonomous Code Improvement Agent")
    print(f"  Model: {MODEL}")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"{'=' * 70}\n")

    with tempfile.TemporaryDirectory() as workdir:
        (Path(workdir) / "solution.py").write_text(SEED_CODE)
        (Path(workdir) / "test_solution.py").write_text(SEED_TESTS)

        llm = create_llm(
            model=MODEL,
            budget_usd=BUDGET_USD,
            window="hourly",
            api_key=api_key,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            max_rpm=200,
            max_input_tpm=500_000,
            max_output_tpm=100_000,
            litellm_params={"tools": make_tools()},
        )

        start_time = time.time()
        total_tool_calls = 0
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for iteration in range(1, NUM_ITERATIONS + 1):
            iter_start = time.time()

            messages.append({
                "role": "user",
                "content": (
                    f"Iteration {iteration}/{NUM_ITERATIONS}. "
                    f"Improve solution.py. Read it, run tests, make changes, "
                    f"run tests again. Use your tools."
                ),
            })

            agent_turns = 0
            max_turns = 8

            while agent_turns < max_turns:
                agent_turns += 1
                try:
                    response_text = llm.call_llm(
                        prompt=messages[-1]["content"] if messages[-1]["role"] == "user" else "Continue.",
                        system_prompt=SYSTEM_PROMPT if agent_turns == 1 else None,
                        litellm_params={"tools": make_tools()},
                    ).result(timeout=30.0)
                except ValueError as e:
                    if "null content" in str(e).lower():
                        break
                    raise
                except Exception as e:
                    print(f"  [{iteration:3d}] Turn {agent_turns} ERROR: {type(e).__name__}: {e}")
                    break

                if not response_text:
                    break

                try:
                    parsed = json.loads(response_text) if response_text.strip().startswith("{") else None
                except (json.JSONDecodeError, ValueError):
                    parsed = None

                if parsed and "tool_calls" in parsed:
                    for tc in parsed["tool_calls"]:
                        fn_name = tc.get("function", {}).get("name", "")
                        fn_args = tc.get("function", {}).get("arguments", {})
                        if isinstance(fn_args, str):
                            fn_args = json.loads(fn_args)
                        result = execute_tool(fn_name, fn_args, workdir)
                        total_tool_calls += 1
                        messages.append({"role": "tool", "content": result})
                else:
                    messages.append({"role": "assistant", "content": response_text})
                    break

            if len(messages) > 40:
                messages = messages[:2] + messages[-20:]

            elapsed = time.time() - iter_start
            reporter = llm.get_reporter().result(timeout=5.0)

            if iteration % 5 == 0 or iteration <= 2:
                print(
                    f"  [{iteration:3d}/{NUM_ITERATIONS}] "
                    f"{elapsed:5.1f}s | "
                    f"calls={reporter.num_calls:3d} | "
                    f"tools={total_tool_calls:3d} | "
                    f"${reporter.total_cost():.6f}"
                )

        total_elapsed = time.time() - start_time
        reporter = llm.get_reporter().result(timeout=5.0)

        final_code = (Path(workdir) / "solution.py").read_text()
        cp = subprocess.run(
            "python test_solution.py", shell=True, capture_output=True,
            text=True, cwd=workdir,
        )

        print(f"\n{'=' * 70}")
        print("  RESULTS")
        print(f"{'=' * 70}")
        print(f"  Iterations: {NUM_ITERATIONS}")
        print(f"  LLM calls: {reporter.num_calls}")
        print(f"  Tool calls: {total_tool_calls}")
        print(f"  Time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        print(f"  Cost: ${reporter.total_cost():.6f}")
        print(f"  Tests: {'PASS' if cp.returncode == 0 else 'FAIL'}")
        if cp.stdout.strip():
            print(f"  Test output: {cp.stdout.strip()}")
        if cp.returncode != 0 and cp.stderr.strip():
            print(f"  Test stderr: {cp.stderr.strip()[:200]}")
        print()
        print(reporter.to_markdown())
        print("\n  Final solution.py:\n")
        for line in final_code.strip().split("\n")[:25]:
            print(f"    {line}")

        reporter.to_json(path=Path(__file__).parent / "code_agent_cost_report.json")
        print("\n  Cost report saved to demos/code_agent_cost_report.json")

        llm.stop()


if __name__ == "__main__":
    main()
