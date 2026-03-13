"""
End-to-end integration test: real VLM calls with images.

Sends each of the 10 test fixture images to a vision-capable LLM,
prints the description it returns, and validates that the description
contains keywords consistent with the actual image content.

Run manually:
    python tests/test_e2e_vision.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    print(f"SKIP: {env_path} not found.")
    sys.exit(0)
load_dotenv(env_path)

api_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("SKIP: No OPENROUTER_API_KEY or OPENAI_API_KEY found in .env.")
    sys.exit(0)

from slowburn import create_llm  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
ALL_TEST_IMAGES = sorted(FIXTURES_DIR.glob("test_image_*.jpg"))

# What I (the developer) actually see in each image, used for validation.
# Each entry is a list of keywords — the LLM description must contain at
# least 2 of them to pass. These are intentionally generous to avoid
# false failures from phrasing differences.
EXPECTED_KEYWORDS = {
    "test_image_1.jpg": ["person", "woman", "sunset", "sun", "silhouette", "back", "plaid", "shirt", "golden", "light"],
    "test_image_2.jpg": ["prayer flags", "flags", "mountain", "colorful", "valley", "mist", "cloud", "tibet", "nepal"],
    "test_image_3.jpg": ["mountain", "aerial", "haze", "fog", "ridge", "landscape", "above", "clouds", "range"],
    "test_image_4.jpg": ["chair", "window", "curtain", "wood", "porch", "dusk", "deck", "bench", "wooden"],
    "test_image_5.jpg": ["ocean", "cliff", "rock", "coast", "sea", "wave", "water", "cove", "stack"],
    "test_image_6.jpg": ["autumn", "fall", "lake", "road", "foliage", "reflection", "orange", "leaves", "tree"],
    "test_image_7.jpg": ["coffee", "cup", "mug", "red", "drink", "tea", "beverage", "white"],
    "test_image_8.jpg": ["beach", "sand", "coast", "town", "cliff", "water", "tide", "shore", "overcast"],
    "test_image_9.jpg": ["fog", "tree", "mist", "sun", "silhouette", "haze", "morning", "light"],
    "test_image_10.jpg": ["person", "man", "hat", "car", "vehicle", "steering", "vintage", "driving", "green"],
}

MIN_KEYWORD_MATCHES = 2


def separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ===================================================================
separator("Vision E2E: Setting up LLM worker")
# ===================================================================

if os.getenv("OPENROUTER_API_KEY"):
    model = "openrouter/google/gemini-2.0-flash-001"
    key = os.getenv("OPENROUTER_API_KEY")
else:
    model = "gpt-4o-mini"
    key = os.getenv("OPENAI_API_KEY")

llm = create_llm(
    model=model,
    budget_usd=2.0,
    window="hourly",
    api_key=key,
    max_tokens=300,
    temperature=0.2,
)

print(f"Model: {model}")
print(f"Test images: {len(ALL_TEST_IMAGES)} in {FIXTURES_DIR}")

# ===================================================================
separator("Vision E2E: Single-image descriptions (all 10 images)")
# ===================================================================

SYSTEM_PROMPT = (
    "You are a precise image description assistant. "
    "Describe what you see in the image in one detailed paragraph. "
    "Focus on the main subject, setting, colors, and mood. "
    "Be specific and factual."
)

all_passed = True
descriptions = {}

for img_path in ALL_TEST_IMAGES:
    img_name = img_path.name
    expected = EXPECTED_KEYWORDS.get(img_name, [])

    start = time.time()
    description = llm.call_llm(
        prompt="Describe this image in detail.",
        images=[img_path],
        system_prompt=SYSTEM_PROMPT,
        image_detail="high",
    ).result(timeout=60.0)
    elapsed = time.time() - start

    descriptions[img_name] = description
    desc_lower = description.lower()

    matched = [kw for kw in expected if kw.lower() in desc_lower]
    match_count = len(matched)
    passed = match_count >= MIN_KEYWORD_MATCHES

    print(f"\n--- {img_name} ({elapsed:.1f}s) ---")
    print(f"  Description: {description}")
    print(f"  Keywords matched: {matched} ({match_count}/{len(expected)})")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"  EXPECTED at least {MIN_KEYWORD_MATCHES} of: {expected}")
        all_passed = False

# ===================================================================
separator("Vision E2E: Multi-image batch call")
# ===================================================================

start = time.time()
batch_results = llm.call_llm_batch(
    prompts=[f"In one sentence, what is in this image?" for _ in ALL_TEST_IMAGES[:5]],
    images_per_prompt=[[img] for img in ALL_TEST_IMAGES[:5]],
    system_prompt="Answer in exactly one sentence.",
).result(timeout=120.0)
elapsed = time.time() - start

print(f"\nBatch of 5 images completed in {elapsed:.1f}s:")
for i, (img, result) in enumerate(zip(ALL_TEST_IMAGES[:5], batch_results)):
    print(f"  [{img.name}] {result}")

assert len(batch_results) == 5, f"Expected 5 results, got {len(batch_results)}"
print("PASS")

# ===================================================================
separator("Vision E2E: Cost report")
# ===================================================================

reporter = llm.get_reporter().result(timeout=5.0)
print(f"Total calls: {reporter.num_calls}")
print(f"Total cost:  ${reporter.total_cost():.6f}")
print(reporter.to_markdown())

llm.stop()

# ===================================================================
if all_passed:
    separator("ALL VISION TESTS PASSED")
else:
    separator("SOME VISION TESTS FAILED — see details above")
    sys.exit(1)
