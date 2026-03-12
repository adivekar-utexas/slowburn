"""
End-to-end integration test with real LLM calls.

Loads API keys from .env, makes actual calls via litellm,
and verifies the full SlowBurn pipeline: CostLimit -> acquire ->
litellm.acompletion -> update -> CostReporter.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load API keys from _env
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    print(f"SKIP: {env_path} not found. Copy _env from PromptMOO to run this test.")
    sys.exit(0)
load_dotenv(env_path)

# Verify we have at least one key
api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("SKIP: No OPENROUTER_API_KEY or OPENAI_API_KEY found in _env.")
    sys.exit(0)

import time  # noqa: E402

from concurry import LimitSet  # noqa: E402

from slowburn import CostLimit, CostReporter, create_llm  # noqa: E402
from slowburn.integrations.autogen import SlowBurnModelClient  # noqa: E402


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ===================================================================
# Test 1: create_llm convenience API — single call
# ===================================================================
separator("Test 1: create_llm() single call")

# Use openrouter model if we have that key, otherwise OpenAI
if os.getenv("OPENROUTER_API_KEY"):
    model = "openrouter/google/gemini-2.0-flash-001"
    key = os.getenv("OPENROUTER_API_KEY")
else:
    model = "gpt-4o-mini"
    key = os.getenv("OPENAI_API_KEY")

llm = create_llm(
    model=model,
    budget_usd=1.0,
    window="hourly",
    api_key=key,
    max_tokens=150,
    temperature=0.3,
)

start = time.time()
result = llm.call_llm(prompt="What is 2+2? Answer in one word.").result(timeout=30.0)
elapsed = time.time() - start

reporter = llm.get_reporter().result(timeout=5.0)
print(f"Response: {result!r}")
print(f"Time: {elapsed:.2f}s")
print(f"Calls: {reporter.num_calls}")
print(f"Cost:  ${reporter.total_cost():.6f}")
assert reporter.num_calls == 1, f"Expected 1 call, got {reporter.num_calls}"
assert reporter.total_cost() > 0, "Cost should be positive"
assert len(result) > 0, "Response should not be empty"
print("PASS")

# ===================================================================
# Test 2: Multiple sequential calls — cost accumulates
# ===================================================================
separator("Test 2: Multiple sequential calls")

prompts = [
    "Name one planet.",
    "Name one color.",
    "Name one animal.",
]
for p in prompts:
    r = llm.call_llm(prompt=p).result(timeout=30.0)
    print(f"  Q: {p} -> A: {r[:60]!r}")

reporter = llm.get_reporter().result(timeout=5.0)
print(f"Total calls: {reporter.num_calls}")
print(f"Total cost:  ${reporter.total_cost():.6f}")
assert reporter.num_calls == 4, f"Expected 4 calls, got {reporter.num_calls}"
print("PASS")

# ===================================================================
# Test 3: Batch call — concurrent execution
# ===================================================================
separator("Test 3: Batch call (concurrent)")

start = time.time()
batch_results = llm.call_llm_batch(
    prompts=[
        "Capital of France?",
        "Capital of Japan?",
        "Capital of Brazil?",
    ],
).result(timeout=30.0)
elapsed = time.time() - start

print(f"Batch results ({elapsed:.2f}s):")
for i, r in enumerate(batch_results):
    print(f"  [{i}] {r[:80]!r}")

reporter = llm.get_reporter().result(timeout=5.0)
print(f"Total calls: {reporter.num_calls}")
print(f"Total cost:  ${reporter.total_cost():.6f}")
assert reporter.num_calls == 7, f"Expected 7 calls, got {reporter.num_calls}"
assert len(batch_results) == 3
print("PASS")

llm.stop()

# ===================================================================
# Test 4: Validator — parse structured output
# ===================================================================
separator("Test 4: Validator (parse int from response)")

llm2 = create_llm(
    model=model,
    budget_usd=0.50,
    window="hourly",
    api_key=key,
    max_tokens=50,
    temperature=0.0,
    num_retries=2,
)

def extract_number(text: str) -> int:
    """Extract the first integer from the response."""
    import re
    match = re.search(r'\d+', text)
    if match is None:
        raise ValueError(f"No number found in: {text!r}")
    return int(match.group())

result = llm2.call_llm(
    prompt="What is 17 * 3? Reply with just the number.",
    validator=extract_number,
).result(timeout=30.0)

print(f"Result: {result} (type: {type(result).__name__})")
assert isinstance(result, int), f"Expected int, got {type(result)}"
assert result == 51, f"Expected 51, got {result}"
print("PASS")

llm2.stop()

# ===================================================================
# Test 5: SlowBurnModelClient (AutoGen integration) — direct test
# ===================================================================
separator("Test 5: SlowBurnModelClient (AG2 protocol)")

limit_set = LimitSet(
    limits=[CostLimit(budget_usd=0.50, window_seconds=3600)],
    mode="thread",
    shared=True,
)
ag_reporter = CostReporter()

client = SlowBurnModelClient(
    config={"model": model},
    limit_set=limit_set,
    reporter=ag_reporter,
)

response = client.create({
    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
    "model": model,
    "max_tokens": 50,
    "temperature": 0.0,
})

print(f"Response: {response.choices[0].message.content!r}")
print(f"Cost: ${client.cost(response):.6f}")
usage = client.get_usage(response)
print(f"Usage: {usage}")
msgs = client.message_retrieval(response)
print(f"Messages: {msgs}")

assert ag_reporter.num_calls == 1
assert ag_reporter.total_cost() > 0
assert len(msgs) > 0
print("PASS")

# ===================================================================
# Test 6: CostReporter output formats
# ===================================================================
separator("Test 6: Reporter outputs (from accumulated real calls)")

# Merge the two reporters for a combined view
combined = CostReporter()
for r in [reporter, ag_reporter]:
    for call in r.calls:
        combined.calls.append(call)

print("\n--- Markdown ---")
print(combined.to_markdown())

print("\n--- LaTeX ---")
print(combined.to_latex())

print(f"\nGrand total: ${combined.total_cost():.6f} across {combined.num_calls} calls")
print("PASS")

# ===================================================================
separator("ALL TESTS PASSED")
print(f"Total real LLM calls: {combined.num_calls}")
print(f"Total real cost: ${combined.total_cost():.6f}")
