"""Tests for SlowBurnLLM image/vision support with mocked litellm calls.

Tests cover:
- Image encoding utilities (_mime_type_for_path, _encode_image_to_data_url, _resolve_image_inputs)
- Vision message formatting (single image, multiple images, mixed batches)
- Token estimation with images
- Error handling (missing files, empty files, invalid types)
- Batch calls with images_per_prompt
"""

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from concurry import CallLimit, LimitSet, RateLimit

from slowburn.limits import CostLimit
from slowburn.llm_worker import (
    ImageInput,
    SlowBurnLLM,
    _encode_image_to_data_url,
    _estimate_tokens,
    _mime_type_for_path,
    _resolve_image_inputs,
)

from .conftest import MOCK_MODEL_NAME

# ---------------------------------------------------------------------------
# Paths to test fixtures
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
ALL_TEST_IMAGES = sorted(FIXTURES_DIR.glob("test_image_*.jpg"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_acompletion_response(
    content: str = "I see an image",
    prompt_tokens: int = 1100,
    completion_tokens: int = 30,
    model: str = MOCK_MODEL_NAME,
    cost: float = 0.002,
):
    """Build a mock litellm acompletion response."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        usage=usage,
        choices=[choice],
        model=model,
        _hidden_params={"response_cost": cost},
    )


def _build_worker(budget_usd: float = 10.0) -> SlowBurnLLM:
    """Create a SlowBurnLLM worker with a reasonable limit set."""
    limit_set = LimitSet(
        limits=[
            CostLimit(budget_usd=budget_usd, window_seconds=3600),
            RateLimit(key="input_tokens", window_seconds=60, capacity=10_000_000),
            RateLimit(key="output_tokens", window_seconds=60, capacity=2_000_000),
            CallLimit(window_seconds=60, capacity=500),
        ],
        mode="asyncio",
        shared=True,
    )
    return SlowBurnLLM.options(
        mode="asyncio",
        limits=limit_set,
        num_retries={"call_llm": 0, "*": 0},
    ).init(
        name="test-vision-llm",
        model_name=MOCK_MODEL_NAME,
        api_key="test-key",
        temperature=0.5,
        max_tokens=100,
        timeout=10.0,
    )


def _extract_messages(mock_acompletion: AsyncMock) -> List[Dict[str, Any]]:
    """Pull the messages kwarg from the most recent acompletion call."""
    call_kwargs = mock_acompletion.call_args.kwargs
    return call_kwargs.get("messages", [])


# ===========================================================================
# Precondition: fixture images exist
# ===========================================================================

class TestFixtureImages:
    """Verify that the 10 picsum test images are present and valid."""

    def test_all_ten_images_present(self) -> None:
        assert len(ALL_TEST_IMAGES) == 10, (
            f"Expected 10 test images in {FIXTURES_DIR}, found {len(ALL_TEST_IMAGES)}"
        )

    @pytest.mark.parametrize("img", ALL_TEST_IMAGES, ids=[p.name for p in ALL_TEST_IMAGES])
    def test_each_image_is_nonempty_jpeg(self, img: Path) -> None:
        assert img.exists()
        raw = img.read_bytes()
        assert len(raw) > 1000, f"{img.name} is suspiciously small ({len(raw)} bytes)"
        assert raw[:2] == b"\xff\xd8", f"{img.name} does not start with JPEG magic bytes"


# ===========================================================================
# Tests: _mime_type_for_path
# ===========================================================================

class TestMimeTypeForPath:

    def test_jpeg(self, tmp_path: Path) -> None:
        p = tmp_path / "photo.jpg"
        p.write_bytes(b"\xff\xd8dummy")
        assert _mime_type_for_path(p) == "image/jpeg"

    def test_png(self, tmp_path: Path) -> None:
        p = tmp_path / "diagram.png"
        p.write_bytes(b"\x89PNGdummy")
        assert _mime_type_for_path(p) == "image/png"

    def test_webp(self, tmp_path: Path) -> None:
        p = tmp_path / "photo.webp"
        p.write_bytes(b"RIFFwebp")
        assert _mime_type_for_path(p) == "image/webp"

    def test_unknown_extension_falls_back_to_png(self, tmp_path: Path) -> None:
        p = tmp_path / "data.bin"
        p.write_bytes(b"binary")
        assert _mime_type_for_path(p) == "image/png"

    def test_gif(self, tmp_path: Path) -> None:
        p = tmp_path / "anim.gif"
        p.write_bytes(b"GIF89a")
        assert _mime_type_for_path(p) == "image/gif"


# ===========================================================================
# Tests: _encode_image_to_data_url
# ===========================================================================

class TestEncodeImageToDataUrl:

    def test_encodes_jpeg_fixture(self) -> None:
        img = ALL_TEST_IMAGES[0]
        data_url = _encode_image_to_data_url(img)
        assert data_url.startswith("data:image/jpeg;base64,")
        b64_part = data_url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded[:2] == b"\xff\xd8"

    def test_round_trip_preserves_bytes(self) -> None:
        img = ALL_TEST_IMAGES[2]
        original_bytes = img.read_bytes()
        data_url = _encode_image_to_data_url(img)
        b64_part = data_url.split(",", 1)[1]
        assert base64.b64decode(b64_part) == original_bytes

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Image not found"):
            _encode_image_to_data_url(tmp_path / "nonexistent.jpg")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="Image file is empty"):
            _encode_image_to_data_url(empty)

    def test_png_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
        data_url = _encode_image_to_data_url(p)
        assert data_url.startswith("data:image/png;base64,")


# ===========================================================================
# Tests: _resolve_image_inputs
# ===========================================================================

class TestResolveImageInputs:

    def test_local_path_objects(self) -> None:
        paths = ALL_TEST_IMAGES[:3]
        urls = _resolve_image_inputs(paths)
        assert len(urls) == 3
        for url in urls:
            assert url.startswith("data:image/jpeg;base64,")

    def test_local_path_strings(self) -> None:
        path_strs = [str(p) for p in ALL_TEST_IMAGES[:2]]
        urls = _resolve_image_inputs(path_strs)
        assert len(urls) == 2
        for url in urls:
            assert url.startswith("data:image/jpeg;base64,")

    def test_http_urls_passed_through(self) -> None:
        input_urls = [
            "https://example.com/image1.png",
            "http://example.com/image2.jpg",
        ]
        urls = _resolve_image_inputs(input_urls)
        assert urls == input_urls

    def test_data_urls_passed_through(self) -> None:
        data_url = "data:image/png;base64,iVBORw0KGgo="
        urls = _resolve_image_inputs([data_url])
        assert urls == [data_url]

    def test_mixed_inputs(self) -> None:
        inputs: List[ImageInput] = [
            ALL_TEST_IMAGES[0],
            "https://example.com/remote.png",
            "data:image/gif;base64,R0lGODlh",
            str(ALL_TEST_IMAGES[1]),
        ]
        urls = _resolve_image_inputs(inputs)
        assert len(urls) == 4
        assert urls[0].startswith("data:image/jpeg;base64,")
        assert urls[1] == "https://example.com/remote.png"
        assert urls[2] == "data:image/gif;base64,R0lGODlh"
        assert urls[3].startswith("data:image/jpeg;base64,")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Image input must be"):
            _resolve_image_inputs([123])  # type: ignore[list-item]

    def test_empty_list_returns_empty(self) -> None:
        assert _resolve_image_inputs([]) == []


# ===========================================================================
# Tests: call_llm with images (mocked litellm)
# ===========================================================================

class TestCallLLMWithImages:
    """Test that call_llm correctly builds multimodal messages when images are provided."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_single_image_path(self, mock_acompletion) -> None:
        """A single local image should produce a multimodal user message.

        Steps:
        1. Call call_llm with one image Path.
        2. Verify the user message has content parts (text + image_url).
        3. Verify the image_url contains a base64 data-URL.
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            result = w.call_llm(
                prompt="Describe this image",
                images=[ALL_TEST_IMAGES[0]],
            ).result(timeout=10.0)
            assert result == "I see an image"

            messages = _extract_messages(mock_acompletion)
            assert len(messages) == 1
            user_msg = messages[0]
            assert user_msg["role"] == "user"
            assert isinstance(user_msg["content"], list)
            assert len(user_msg["content"]) == 2
            assert user_msg["content"][0]["type"] == "text"
            assert user_msg["content"][0]["text"] == "Describe this image"
            assert user_msg["content"][1]["type"] == "image_url"
            assert user_msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_multiple_images(self, mock_acompletion) -> None:
        """Multiple images should produce one text part + N image_url parts.

        Steps:
        1. Call call_llm with 3 image Paths.
        2. Verify 4 content parts total (1 text + 3 image_url).
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            w.call_llm(
                prompt="Compare these images",
                images=ALL_TEST_IMAGES[:3],
            ).result(timeout=10.0)

            messages = _extract_messages(mock_acompletion)
            user_content = messages[0]["content"]
            assert len(user_content) == 4
            assert user_content[0]["type"] == "text"
            for i in range(1, 4):
                assert user_content[i]["type"] == "image_url"
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_image_with_system_prompt(self, mock_acompletion) -> None:
        """System prompt + image should produce 2 messages: system + multimodal user.

        Steps:
        1. Call with system_prompt and one image.
        2. Verify system message is first, multimodal user message is second.
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            w.call_llm(
                prompt="What do you see?",
                images=[ALL_TEST_IMAGES[4]],
                system_prompt="You are a helpful vision assistant.",
            ).result(timeout=10.0)

            messages = _extract_messages(mock_acompletion)
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful vision assistant."
            assert messages[1]["role"] == "user"
            assert isinstance(messages[1]["content"], list)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_image_detail_parameter(self, mock_acompletion) -> None:
        """image_detail should be forwarded to the image_url content part.

        Steps:
        1. Call with image_detail="low".
        2. Verify the image_url part has detail="low".
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            w.call_llm(
                prompt="Quick look",
                images=[ALL_TEST_IMAGES[0]],
                image_detail="low",
            ).result(timeout=10.0)

            messages = _extract_messages(mock_acompletion)
            img_part = messages[0]["content"][1]
            assert img_part["image_url"]["detail"] == "low"
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_http_url_image(self, mock_acompletion) -> None:
        """An HTTP URL image should be passed through without encoding.

        Steps:
        1. Call with a URL string instead of a file Path.
        2. Verify the image_url contains the URL as-is.
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            url = "https://picsum.photos/id/237/500/500"
            w.call_llm(
                prompt="Describe this dog",
                images=[url],
            ).result(timeout=10.0)

            messages = _extract_messages(mock_acompletion)
            img_part = messages[0]["content"][1]
            assert img_part["image_url"]["url"] == url
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_no_images_produces_text_only_message(self, mock_acompletion) -> None:
        """images=None should produce a standard text-only user message (no regression).

        Steps:
        1. Call without images.
        2. Verify user content is a plain string.
        """
        mock_acompletion.return_value = _make_acompletion_response(content="text reply")
        w = _build_worker()
        try:
            result = w.call_llm(prompt="Hello, no images").result(timeout=10.0)
            assert result == "text reply"

            messages = _extract_messages(mock_acompletion)
            user_msg = messages[0]
            assert isinstance(user_msg["content"], str)
            assert user_msg["content"] == "Hello, no images"
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_empty_images_list_produces_text_only(self, mock_acompletion) -> None:
        """images=[] should behave identically to images=None."""
        mock_acompletion.return_value = _make_acompletion_response(content="text reply")
        w = _build_worker()
        try:
            result = w.call_llm(prompt="Hi", images=[]).result(timeout=10.0)
            assert result == "text reply"

            messages = _extract_messages(mock_acompletion)
            assert isinstance(messages[0]["content"], str)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_image_call_logs_to_reporter(self, mock_acompletion) -> None:
        """Image calls should be tracked in the CostReporter like text calls."""
        mock_acompletion.return_value = _make_acompletion_response(cost=0.003)
        w = _build_worker()
        try:
            w.call_llm(
                prompt="Describe", images=[ALL_TEST_IMAGES[0]],
            ).result(timeout=10.0)

            reporter = w.get_reporter().result(timeout=5.0)
            assert reporter.num_calls == 1
            assert reporter.total_cost() == pytest.approx(0.003, abs=1e-6)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_image_call_with_validator(self, mock_acompletion) -> None:
        """Validator should work identically for vision calls."""
        mock_acompletion.return_value = _make_acompletion_response(content="42")
        w = _build_worker()
        try:
            result = w.call_llm(
                prompt="How many objects?",
                images=[ALL_TEST_IMAGES[5]],
                validator=lambda text: int(text.strip()),
            ).result(timeout=10.0)
            assert result == 42
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_all_ten_fixture_images(self, mock_acompletion) -> None:
        """Verify all 10 fixture images can be encoded and sent in a single call.

        Steps:
        1. Pass all 10 test images.
        2. Verify 11 content parts (1 text + 10 image_url).
        3. Verify each image_url is a valid base64 data-URL.
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            w.call_llm(
                prompt="Describe all ten images",
                images=ALL_TEST_IMAGES,
            ).result(timeout=15.0)

            messages = _extract_messages(mock_acompletion)
            user_content = messages[0]["content"]
            assert len(user_content) == 11
            assert user_content[0]["type"] == "text"
            for i in range(1, 11):
                part = user_content[i]
                assert part["type"] == "image_url"
                url = part["image_url"]["url"]
                assert url.startswith("data:image/jpeg;base64,")
                b64_part = url.split(",", 1)[1]
                decoded = base64.b64decode(b64_part)
                assert len(decoded) > 1000
        finally:
            w.stop()


# ===========================================================================
# Tests: Token estimation with images
# ===========================================================================

class TestTokenEstimationWithImages:
    """Verify that image token overhead is included in limit requests."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_image_increases_estimated_tokens(self, mock_acompletion) -> None:
        """A call with images should request more input tokens than without.

        Steps:
        1. Make a text-only call, capture requested input_tokens.
        2. Make an image call with same prompt, capture requested input_tokens.
        3. Verify image call requested more tokens.
        """
        mock_acompletion.return_value = _make_acompletion_response()

        requested_tokens = []
        original_acquire = None

        class TokenCapture:
            def __init__(self):
                self.captured = []

            def wrap(self, limit_set):
                original_method = limit_set.acquire

                def patched_acquire(requested=None, **kwargs):
                    if requested is not None:
                        self.captured.append(requested.get("input_tokens", 0))
                    return original_method(requested=requested, **kwargs)
                return patched_acquire

        w = _build_worker()
        try:
            w.call_llm(prompt="Hello world").result(timeout=10.0)
            w.call_llm(
                prompt="Hello world",
                images=[ALL_TEST_IMAGES[0]],
                image_detail="high",
            ).result(timeout=10.0)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_low_detail_uses_fewer_tokens(self, mock_acompletion) -> None:
        """image_detail='low' should add fewer estimated tokens than 'high'."""
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            w.call_llm(
                prompt="Quick scan",
                images=[ALL_TEST_IMAGES[0]],
                image_detail="low",
            ).result(timeout=10.0)
        finally:
            w.stop()


# ===========================================================================
# Tests: call_llm_batch with images
# ===========================================================================

class TestCallLLMBatchWithImages:
    """Test batch calls with per-prompt images."""

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_mixed_batch(self, mock_acompletion) -> None:
        """Batch with some image prompts and some text-only prompts.

        Steps:
        1. Submit 3 prompts: first with image, second text-only, third with 2 images.
        2. Verify 3 results.
        3. Verify message formats differ for each call.
        """
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            results = w.call_llm_batch(
                prompts=["Describe img", "Text only", "Compare two"],
                images_per_prompt=[
                    [ALL_TEST_IMAGES[0]],
                    None,
                    [ALL_TEST_IMAGES[1], ALL_TEST_IMAGES[2]],
                ],
            ).result(timeout=15.0)
            assert len(results) == 3

            calls = mock_acompletion.call_args_list
            assert len(calls) == 3

            msg0 = calls[0].kwargs["messages"]
            assert isinstance(msg0[0]["content"], list)
            assert len(msg0[0]["content"]) == 2

            msg1 = calls[1].kwargs["messages"]
            assert isinstance(msg1[0]["content"], str)

            msg2 = calls[2].kwargs["messages"]
            assert isinstance(msg2[0]["content"], list)
            assert len(msg2[0]["content"]) == 3
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_all_images(self, mock_acompletion) -> None:
        """Batch where every prompt has one image."""
        mock_acompletion.return_value = _make_acompletion_response()
        w = _build_worker()
        try:
            results = w.call_llm_batch(
                prompts=[f"Describe image {i}" for i in range(5)],
                images_per_prompt=[[img] for img in ALL_TEST_IMAGES[:5]],
            ).result(timeout=15.0)
            assert len(results) == 5

            for call in mock_acompletion.call_args_list:
                user_content = call.kwargs["messages"][0]["content"]
                assert isinstance(user_content, list)
                assert len(user_content) == 2
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_no_images(self, mock_acompletion) -> None:
        """Batch with images_per_prompt=None should be text-only (no regression)."""
        mock_acompletion.return_value = _make_acompletion_response(content="text")
        w = _build_worker()
        try:
            results = w.call_llm_batch(
                prompts=["p1", "p2"],
                images_per_prompt=None,
            ).result(timeout=15.0)
            assert len(results) == 2

            for call in mock_acompletion.call_args_list:
                user_content = call.kwargs["messages"][0]["content"]
                assert isinstance(user_content, str)
        finally:
            w.stop()

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_batch_length_mismatch_raises(self, mock_acompletion) -> None:
        """images_per_prompt with wrong length should raise ValueError."""
        w = _build_worker()
        try:
            with pytest.raises(ValueError, match="images_per_prompt length"):
                w.call_llm_batch(
                    prompts=["a", "b", "c"],
                    images_per_prompt=[[ALL_TEST_IMAGES[0]]],
                ).result(timeout=10.0)
        finally:
            w.stop()


# ===========================================================================
# Tests: Error handling for images
# ===========================================================================

class TestImageErrorHandling:

    def test_nonexistent_file_raises_on_resolve(self) -> None:
        with pytest.raises(FileNotFoundError):
            _resolve_image_inputs([Path("/tmp/does_not_exist_abc123.jpg")])

    def test_empty_file_raises_on_resolve(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            _resolve_image_inputs([empty])

    def test_invalid_type_raises_on_resolve(self) -> None:
        with pytest.raises(TypeError, match="Image input must be"):
            _resolve_image_inputs([42])  # type: ignore[list-item]

    @patch("slowburn.llm_worker.litellm.acompletion", new_callable=AsyncMock)
    def test_nonexistent_image_in_call_llm(self, mock_acompletion) -> None:
        """Passing a non-existent image to call_llm should raise FileNotFoundError."""
        w = _build_worker()
        try:
            with pytest.raises(FileNotFoundError):
                w.call_llm(
                    prompt="Describe",
                    images=[Path("/tmp/no_such_image_xyz.jpg")],
                ).result(timeout=10.0)
        finally:
            w.stop()
