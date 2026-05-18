import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from slowburn import create_llm

from .conftest import skip_no_api_key

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
TEST_IMAGE = FIXTURES_DIR / "test_image_1.jpg"

ENHANCE_IMAGE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "suggest_image_enhancement",
        "description": (
            "Describe a single change to the main object in the provided image "
            "that would significantly enhance its visual appeal. "
            "Be specific about what the object is and what change to make."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_in_image": {
                    "type": "string",
                    "description": (
                        "A concise description of the primary object or subject "
                        "visible in the image (e.g., 'a person with short hair seen from behind')."
                    ),
                },
                "suggested_change": {
                    "type": "string",
                    "description": (
                        "A specific, actionable change that would enhance the image. "
                        "Must reference the object described above."
                    ),
                },
            },
            "required": ["object_in_image", "suggested_change"],
        },
    },
}


@skip_no_api_key
class TestSonnet46ThinkingBudget:
    """Targeted tests for Claude Sonnet 4.6 reasoning-budget control via OpenRouter.

    Verifies that when we pass `reasoning.max_tokens` through `extra_body`
    to OpenRouter, the model:
    1. Produces reasoning_content.
    2. Produces final text content.
    3. Uses a bounded reasoning budget.
    """

    @pytest.mark.parametrize("budget_tokens", [1024, 2048])
    @pytest.mark.timeout(300)
    def test_sonnet_thinking_budget(self, budget_tokens: int) -> None:
        model_id = "openrouter/anthropic/claude-sonnet-4.6"
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            pytest.skip(f"Missing OPENROUTER_API_KEY for {model_id}")

        # OpenRouter normalizes reasoning control across providers.
        # For Anthropic models, `reasoning.max_tokens` maps to Anthropic's
        # `thinking.budget_tokens`, limiting how many tokens the model spends
        # on reasoning. This is the recommended approach over `thinking.effort`
        # when the goal is to cap the total tokens used for thinking.
        litellm_params = {
            "extra_body": {
                "reasoning": {
                    "max_tokens": budget_tokens,
                    "exclude": False,
                },
            },
        }

        llm = create_llm(
            model=model_id,
            api_key=api_key,
            budget_usd=5.0,
            window="daily",
            max_tokens=16000,  # Ensure enough headroom for thinking + content
            temperature=1.0,  # Temperature must be 1.0 when thinking is enabled
            litellm_params=litellm_params,
            on_pricing_unavailable="warn",
        )

        try:
            prompt = (
                "Solve this complex mathematical riddle step-by-step: "
                "I am a number between 1 and 100. I am a prime number. "
                "If you reverse my digits, you get another prime number. "
                "The sum of my digits is 10. What number am I?"
            )

            start_time = time.time()
            messages: List[Dict[str, Any]] = llm.call_llm(
                prompt=prompt,
                history=[],
                return_messages=True,
            ).result(timeout=240.0)
            elapsed = time.time() - start_time

            assistant_msg = messages[-1]
            reasoning = assistant_msg.get("reasoning_content")
            content = assistant_msg.get("content", "")

            print(f"\n[BUDGET-TEST] Budget: {budget_tokens}, Elapsed: {elapsed:.1f}s")
            print(
                f"  REASONING ({len(reasoning) if reasoning else 0} chars):\n"
                f"{'═' * 50}\n{reasoning}\n{'═' * 50}"
            )
            print(f"  CONTENT ({len(content)} chars):\n{'═' * 50}\n{content}\n{'═' * 50}")

            assert reasoning is not None and len(reasoning) > 0, "Model did not produce reasoning content"
            assert len(content) > 0, "Model did not produce final content"

            # Simple check that the riddle is solved (37 is the answer)
            assert "37" in content or "37" in reasoning

        finally:
            llm.stop()


@skip_no_api_key
class TestImageAwareToolCalling:
    """Verify that vision-capable models actually see the image when tool calling is enabled.

    The tool `suggest_image_enhancement` requires the model to:
    1. Identify the primary object in the image.
    2. Suggest a concrete change to that object.

    If the tool call arguments contain a plausible description of the test image
    (sunset, golden light, person from behind), the model saw the image.  If the
    description is generic or missing, the model did not process the image content.

    Both vision=True and vision=False are exercised so we can see whether models
    hallucinate image content when no image is provided.
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "openrouter/anthropic/claude-sonnet-4.6",
            "openrouter/moonshotai/kimi-k2.5",
            "azure/responses/gpt-5.4-mini",
        ],
    )
    @pytest.mark.parametrize("vision", [True, False])
    @pytest.mark.parametrize("tool_mode", ["auto", "required"])
    @pytest.mark.parametrize("reasoning_level", ["none", "low"])
    @pytest.mark.timeout(300)
    def test_model_sees_image_via_tool_args(
        self,
        model_id: str,
        vision: bool,
        tool_mode: str,
        reasoning_level: str,
    ) -> None:
        """Run a single point and assert the tool arguments mention image content (when vision=True)."""

        if "openrouter" in model_id:
            api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
        elif "azure" in model_id:
            api_key = os.getenv("AZURE_API_KEY")
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

        if api_key is None:
            pytest.skip(f"Missing API key for {model_id}")

        litellm_params: Dict[str, Any] = {}
        if "openrouter" in model_id:
            litellm_params["extra_body"] = {
                "reasoning": {
                    "effort": reasoning_level,
                    "exclude": False,
                },
            }
        elif "azure" in model_id:
            if reasoning_level != "none":
                litellm_params["reasoning_effort"] = {
                    "effort": reasoning_level,
                    "summary": "detailed",
                }
        elif "anthropic/" in model_id and "openrouter" not in model_id:
            if reasoning_level != "none":
                litellm_params["output_config"] = {"effort": reasoning_level}

        tool_schemas: List[Dict[str, Any]] = [ENHANCE_IMAGE_TOOL]

        llm = create_llm(
            model=model_id,
            api_key=api_key,
            budget_usd=5.0,
            window="daily",
            max_tokens=1000,
            temperature=0.0,
            tools=tool_schemas,
            tool_choice=tool_mode,
            litellm_params=litellm_params,
            on_pricing_unavailable="warn",
        )

        try:
            prompt: str = (
                "Look at the image and suggest a change that would enhance it. "
                "Use the tool to provide your suggestion."
            )
            images: Optional[List[Path]] = [TEST_IMAGE] if vision else None

            start_time: float = time.time()
            messages: List[Dict[str, Any]] = llm.call_llm(
                prompt=prompt,
                images=images,
                history=[],
                return_messages=True,
            ).result(timeout=240.0)
            elapsed: float = time.time() - start_time

            assistant_msg: Dict[str, Any] = messages[-1]
            reasoning: Optional[str] = assistant_msg.get("reasoning_content")
            tool_calls: Optional[List[Dict[str, Any]]] = assistant_msg.get("tool_calls")
            content_raw: Optional[str] = assistant_msg.get("content")
            content: str = content_raw if content_raw is not None else ""

            print(
                f"\n{'<' * 50}{'>' * 50}"
                f"\n{'<' * 50}{'>' * 50}"
                f"\n{'<' * 50}{'>' * 50}"
                f"\n[IMAGE-TOOL] Model: {model_id}, Image: {vision}, "
                f"Tools: {tool_mode}, Reasoning: {reasoning_level} ({elapsed:.1f}s)"
            )

            # --- DUMP FINAL ASSISTANT MESSAGE ---
            print("\n  >>> FINAL ASSISTANT MESSAGE <<<")
            print(f"  REASONING:\n{'═' * 50}\n{reasoning}\n{'═' * 50}")
            print(f"  TOOL_CALLS:\n{'═' * 50}\n{tool_calls}\n{'═' * 50}")
            print(f"  CONTENT:\n{'═' * 50}\n{content}\n{'═' * 50}")

            # --- Validations ---
            assert assistant_msg["role"] == "assistant"
            if tool_mode == "required":
                assert tool_calls is not None and len(tool_calls) > 0, (
                    f"Expected at least one tool call with tool_mode=required, got tool_calls={tool_calls}"
                )

            # When the model did call a tool, inspect the arguments.
            if tool_calls is not None and len(tool_calls) > 0:
                first_tc: Dict[str, Any] = tool_calls[0]
                assert first_tc["function"]["name"] == "suggest_image_enhancement", (
                    f"Expected tool name 'suggest_image_enhancement', got {first_tc['function']['name']}"
                )

                args_json: str = first_tc["function"]["arguments"]
                args: Dict[str, Any] = json.loads(args_json)
                object_desc: str = args.get("object_in_image", "")
                suggested_change: str = args.get("suggested_change", "")

                print(f"\n  >> TOOL ARGS: object={object_desc!r}, change={suggested_change!r}")

                # When vision=True, assert the model actually saw the image.
                if vision:
                    image_keywords: List[str] = [
                        "sunset",
                        "sunrise",
                        "golden",
                        "sun",
                        "horizon",
                        "person",
                        "hair",
                        "silhouette",
                        "light",
                        "sky",
                        "field",
                        "back",
                        "lens flare",
                        "warm",
                        "orange",
                        "trees",
                    ]

                    combined: str = (object_desc + " " + suggested_change).lower()
                    has_image_reference: bool = any(kw in combined for kw in image_keywords)
                    assert has_image_reference, (
                        f"Tool arguments do not reference any recognizable image content. "
                        f"object_in_image={object_desc!r}, suggested_change={suggested_change!r}. "
                        f"The model likely did not process the image."
                    )

            # Cost reporting
            reporter = llm.get_reporter().result()
            print(f"\n  >> Cost: ${reporter.total_cost():.6f}, Calls: {reporter.num_calls}")
            if tool_mode == "required":
                assert reporter.num_calls == 1, (
                    f"Expected exactly 1 call with tool_mode=required, got {reporter.num_calls}"
                )

        except Exception as e:
            print(f"\nFAILED: {model_id}, tools={tool_mode}, reason={reasoning_level}")
            raise e
        finally:
            llm.stop()


@skip_no_api_key
class TestReasoningToolsGrid:
    """Comprehensive E2E grid for reasoning models.

    Models: Sonnet 4.6 (OpenRouter), Kimi 2.5 (OpenRouter), GPT 5.4 mini (Azure)
    Vision: Yes/No
    Tools: Auto/Required/None
    Reasoning: None/Low

    This test executes a SINGLE turn to verify that the model correctly
    produces reasoning, content, and tool calls (if applicable) in its
    initial response.
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "openrouter/anthropic/claude-sonnet-4.6",
            "openrouter/moonshotai/kimi-k2.5",
            "azure/responses/gpt-5.4-mini",
        ],
    )
    @pytest.mark.parametrize("vision", [True, False])
    @pytest.mark.parametrize("tool_mode", ["auto", "required", "none"])
    @pytest.mark.parametrize("reasoning_level", ["none", "low"])
    @pytest.mark.timeout(300)
    def test_call_grid(
        self,
        model_id: str,
        vision: bool,
        tool_mode: str,
        reasoning_level: str,
    ) -> None:
        """Run a single point in the grid and dump the assistant message."""

        # Determine API key based on provider
        if "openrouter" in model_id:
            api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
        elif "azure" in model_id:
            api_key = os.getenv("AZURE_API_KEY")
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

        if api_key is None:
            pytest.skip(f"Missing API key for {model_id}")

        # Build litellm_params (simulating what experiment_config.py does)
        litellm_params: Dict[str, Any] = {}
        if "openrouter" in model_id:
            litellm_params["extra_body"] = {
                "reasoning": {
                    "effort": reasoning_level,
                    "exclude": False,
                },
            }
        elif "azure" in model_id:
            if reasoning_level != "none":
                litellm_params["reasoning_effort"] = {
                    "effort": reasoning_level,
                    "summary": "detailed",
                }
        elif "anthropic/" in model_id and "openrouter" not in model_id:
            if reasoning_level != "none":
                litellm_params["output_config"] = {"effort": reasoning_level}

        tool_schemas: Optional[List[Dict[str, Any]]] = None
        if tool_mode != "none":
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Get the current time.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ]

        # Create LLM worker
        llm = create_llm(
            model=model_id,
            api_key=api_key,
            budget_usd=5.0,
            window="daily",
            max_tokens=1000,
            temperature=0.0,
            tools=tool_schemas,
            tool_choice=tool_mode if tool_mode != "none" else None,
            litellm_params=litellm_params,
            on_pricing_unavailable="warn",
        )

        try:
            prompt: str = (
                "What time is it? Use a tool if available. If there is an image, describe it briefly."
            )
            images: Optional[List[Path]] = [TEST_IMAGE] if vision else None

            start_time: float = time.time()
            # return_messages=True so we get the message object
            messages: List[Dict[str, Any]] = llm.call_llm(
                prompt=prompt,
                images=images,
                history=[],
                return_messages=True,
            ).result(timeout=240.0)
            elapsed: float = time.time() - start_time

            assistant_msg: Dict[str, Any] = messages[-1]
            reasoning: Optional[str] = assistant_msg.get("reasoning_content")
            tool_calls: Optional[List[Dict[str, Any]]] = assistant_msg.get("tool_calls")
            content_raw: Optional[str] = assistant_msg.get("content")
            content: str = content_raw if content_raw is not None else ""

            print(
                f"\n{'<' * 50}{'>' * 50}"
                f"\n{'<' * 50}{'>' * 50}"
                f"\n{'<' * 50}{'>' * 50}"
                f"\n[GRID] Model: {model_id}, Image: {vision}, "
                f"Tools: {tool_mode}, Reasoning: {reasoning_level} "
                f"({elapsed:.1f}s)"
            )

            # --- DUMP FINAL ASSISTANT MESSAGE ---
            print("\n  >>> FINAL ASSISTANT MESSAGE <<<")
            print(f"  REASONING:\n{'═' * 50}\n{reasoning}\n{'═' * 50}")
            print(f"  TOOL_CALLS:\n{'═' * 50}\n{tool_calls}\n{'═' * 50}")
            print(f"  CONTENT:\n{'═' * 50}\n{content}\n{'═' * 50}")

            # --- Validations ---
            assert assistant_msg["role"] == "assistant"

            if tool_mode == "required":
                assert tool_calls is not None and len(tool_calls) > 0

            # For non-tool modes, we expect content.
            # For tool modes, content might be empty if the model only emits tool_calls.
            if tool_mode == "none":
                assert len(content) > 0

            # Cost reporting
            reporter = llm.get_reporter().result()
            print(f"\n  >> Cost: ${reporter.total_cost():.6f}, Calls: {reporter.num_calls}")
            assert reporter.num_calls == 1

        except Exception as e:
            print(f"\nFAILED: {model_id}, vis={vision}, tools={tool_mode}, reason={reasoning_level}")
            raise e
        finally:
            llm.stop()
