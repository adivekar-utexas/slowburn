"""
ReAct agent loop powered by a SlowBurnLLM worker.

Implements the standard tool-calling cycle:
1. Send messages + tool schemas to LLM (via SlowBurnLLM.call_llm with history=)
2. If LLM returns tool_calls -> execute each tool -> append results -> goto 1
3. If LLM returns text content -> return it (agent is done)

The SlowBurnLLM worker handles all cost tracking, backpressure, and rate
limiting internally. This loop just drives the conversation and tools.

Each step is optionally logged to disk at ``<log_dir>/step_<N>/``.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from slowburn.llm_worker import SlowBurnLLM


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def run_agent(
    *,
    llm: SlowBurnLLM,
    task: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], str],
    system_prompt: str = "You are a helpful agent. Use your tools to complete the task.",
    max_steps: int = 20,
    verbose: bool = True,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a ReAct agent loop with a SlowBurnLLM worker.

    Uses the multi-turn conversation API: ``call_llm(history=..., tools=...)``
    returns a messages list with the assistant response appended. Tool results
    are appended to the same list and re-submitted.

    Args:
        llm: A live SlowBurnLLM worker (from ``create_llm()``).
        task: The user's task description.
        tools: Tool schemas in OpenAI format (list of dicts).
        tool_executor: Callable(tool_name, tool_args) -> str that executes tools.
        system_prompt: System prompt for the agent.
        max_steps: Maximum number of LLM calls before forcing termination.
        verbose: Print step-by-step progress.
        log_dir: Directory to save step-level logs. If None, no logs are saved.

    Returns:
        Dict with keys: "result" (final text), "steps" (int), "tool_calls" (int),
        "messages" (full conversation).
    """
    messages: List[Dict[str, Any]] = []
    total_tool_calls = 0
    previous_cost = 0.0

    for step in range(1, max_steps + 1):
        step_dir: Optional[Path] = None
        if log_dir is not None:
            step_dir = log_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)

        step_start = time.time()

        prompt = task if step == 1 else ""

        if step_dir is not None:
            input_messages = llm.build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=messages,
            ).result(timeout=10.0)
            _save_json(
                step_dir / "input.json",
                {
                    "step": step,
                    "num_messages": len(input_messages),
                    "messages": input_messages,
                },
            )

        messages = llm.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history=messages,
            tools=tools,
            tool_choice="auto",
        ).result(timeout=120.0)

        step_elapsed = time.time() - step_start

        assistant_message = messages[-1]
        tool_calls_data = assistant_message.get("tool_calls")

        if tool_calls_data is not None:
            tool_call_log: List[Dict[str, Any]] = []
            for tool_call_index, tool_call in enumerate(tool_calls_data, 1):
                function_name = tool_call["function"]["name"]
                try:
                    function_arguments = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    function_arguments = {}

                if verbose:
                    arguments_preview = json.dumps(function_arguments, ensure_ascii=False)
                    if len(arguments_preview) > 100:
                        arguments_preview = arguments_preview[:97] + "..."
                    print(f"    [{step:2d}] tool: {function_name}({arguments_preview})")

                if step_dir is not None:
                    _save_json(
                        step_dir / f"tool_{tool_call_index:02d}_call.json",
                        {
                            "tool_call_id": tool_call["id"],
                            "function": function_name,
                            "arguments": function_arguments,
                        },
                    )

                tool_result = tool_executor(function_name, function_arguments)
                total_tool_calls += 1

                if step_dir is not None:
                    try:
                        result_data = json.loads(tool_result)
                    except (json.JSONDecodeError, TypeError):
                        result_data = {"raw": tool_result}
                    _save_json(step_dir / f"tool_{tool_call_index:02d}_result.json", result_data)

                tool_call_log.append(
                    {
                        "function": function_name,
                        "arguments": function_arguments,
                        "result_length": len(tool_result),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    }
                )

            if step_dir is not None:
                _save_json(
                    step_dir / "output.json",
                    {
                        "step": step,
                        "type": "tool_calls",
                        "tool_calls": tool_call_log,
                        "elapsed_seconds": step_elapsed,
                    },
                )

            if verbose:
                reporter = llm.get_reporter().result(timeout=5.0)
                current_cost = reporter.total_cost()
                step_cost = current_cost - previous_cost
                previous_cost = current_cost

                assistant_content = assistant_message.get("content") or ""
                if len(assistant_content) > 0:
                    content_preview = assistant_content[:120].replace("\n", " ")
                    print(f"         thought: {content_preview}")

                print(
                    f"         step ${step_cost:.6f} | "
                    f"total ${current_cost:.6f} | "
                    f"{reporter.num_calls} LLM calls | "
                    f"{total_tool_calls} tool calls"
                )
        else:
            final_text = assistant_message.get("content", "")

            if step_dir is not None:
                _save_json(
                    step_dir / "output.json",
                    {
                        "step": step,
                        "type": "final_answer",
                        "content": final_text,
                        "elapsed_seconds": step_elapsed,
                    },
                )

            if verbose:
                reporter = llm.get_reporter().result(timeout=5.0)
                current_cost = reporter.total_cost()
                step_cost = current_cost - previous_cost

                print(
                    f"    [{step:2d}] DONE | step ${step_cost:.6f} | "
                    f"total ${current_cost:.6f} | "
                    f"{step} LLM calls | {total_tool_calls} tool calls"
                )
                preview = (final_text or "")[:150].replace("\n", " ")
                print(f"         {preview}...")

            return {
                "result": final_text,
                "steps": step,
                "tool_calls": total_tool_calls,
                "messages": messages,
            }

    if log_dir is not None:
        _save_json(
            log_dir / "max_steps_reached.json",
            {
                "max_steps": max_steps,
                "total_tool_calls": total_tool_calls,
            },
        )

    final_text = ""
    if len(messages) > 0:
        final_text = messages[-1].get("content", "") or ""
    return {
        "result": f"[Agent reached max_steps={max_steps}] {final_text}",
        "steps": max_steps,
        "tool_calls": total_tool_calls,
        "messages": messages,
    }
