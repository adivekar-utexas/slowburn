"""
End-to-end integration tests: real VLM calls with images.

Sends test fixture images to a vision-capable LLM, prints the descriptions,
and validates that each description contains keywords matching the actual
image content.

API keys are loaded from .env by conftest.py at session startup.
"""

import time
from pathlib import Path
from typing import Dict, List

import pytest

from slowburn import create_llm

from .conftest import skip_no_api_key

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
ALL_TEST_IMAGES = sorted(FIXTURES_DIR.glob("test_image_*.jpg"))

EXPECTED_KEYWORDS: Dict[str, List[str]] = {
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

SYSTEM_PROMPT = (
    "You are a precise image description assistant. "
    "Describe what you see in the image in one detailed paragraph. "
    "Focus on the main subject, setting, colors, and mood. "
    "Be specific and factual."
)


@skip_no_api_key
class TestVisionSingleImage:
    """Send each of the 10 test images to a vision LLM and validate descriptions."""

    @pytest.fixture(autouse=True)
    def _setup_llm(self, llm_model_and_key):
        model, key = llm_model_and_key
        self.llm = create_llm(
            model=model,
            budget_usd=2.0,
            window="hourly",
            api_key=key,
            max_tokens=300,
            temperature=0.2,
        )
        yield
        self.llm.stop()

    @pytest.mark.parametrize(
        "img_path",
        ALL_TEST_IMAGES,
        ids=[p.name for p in ALL_TEST_IMAGES],
    )
    def test_describe_image(self, img_path: Path) -> None:
        """LLM should describe the image with keywords matching actual content.

        Steps:
        1. Send image to the LLM with a description prompt.
        2. Print the full description (always visible for manual inspection).
        3. Check that at least MIN_KEYWORD_MATCHES keywords from the expected
           list appear in the description.
        """
        img_name = img_path.name
        expected = EXPECTED_KEYWORDS.get(img_name, [])

        start = time.time()
        description = self.llm.call_llm(
            prompt="Describe this image in detail.",
            images=[img_path],
            system_prompt=SYSTEM_PROMPT,
            image_detail="high",
        ).result(timeout=60.0)
        elapsed = time.time() - start

        desc_lower = description.lower()
        matched = [kw for kw in expected if kw.lower() in desc_lower]

        print(f"\n--- {img_name} ({elapsed:.1f}s) ---")
        print(f"  Description: {description}")
        print(f"  Keywords matched: {matched} ({len(matched)}/{len(expected)})")

        assert len(matched) >= MIN_KEYWORD_MATCHES, (
            f"{img_name}: only {len(matched)} keyword matches ({matched}), "
            f"expected >= {MIN_KEYWORD_MATCHES} of {expected}"
        )


@skip_no_api_key
class TestVisionBatch:
    """Batch vision calls with multiple images."""

    def test_batch_five_images(self, llm_model_and_key) -> None:
        """Batch of 5 image descriptions should all return non-empty results."""
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            budget_usd=2.0,
            window="hourly",
            api_key=key,
            max_tokens=100,
            temperature=0.2,
        )
        try:
            start = time.time()
            results = llm.call_llm_batch(
                prompts=["In one sentence, what is in this image?"] * 5,
                images_per_prompt=[[img] for img in ALL_TEST_IMAGES[:5]],
                system_prompt="Answer in exactly one sentence.",
            ).result(timeout=120.0)
            elapsed = time.time() - start

            print(f"\nBatch of 5 images completed in {elapsed:.1f}s:")
            for img, result in zip(ALL_TEST_IMAGES[:5], results):
                print(f"  [{img.name}] {result.strip()}")

            assert len(results) == 5
            for r in results:
                assert len(r.strip()) > 0
        finally:
            llm.stop()


@skip_no_api_key
class TestVisionCostReport:
    """Verify cost tracking works for vision calls."""

    def test_cost_tracked(self, llm_model_and_key) -> None:
        """A vision call should log positive cost to the reporter."""
        model, key = llm_model_and_key
        llm = create_llm(
            model=model,
            budget_usd=1.0,
            window="hourly",
            api_key=key,
            max_tokens=100,
            temperature=0.2,
        )
        try:
            llm.call_llm(
                prompt="What is in this image?",
                images=[ALL_TEST_IMAGES[0]],
            ).result(timeout=60.0)

            reporter = llm.get_reporter().result(timeout=5.0)
            print(f"Vision call cost: ${reporter.total_cost():.6f}")
            print(reporter.to_markdown())

            assert reporter.num_calls == 1
            assert reporter.total_cost() > 0
        finally:
            llm.stop()
