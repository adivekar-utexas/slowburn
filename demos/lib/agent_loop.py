"""
ReAct agent loop powered by a SlowBurnLLM worker.

Implements the standard tool-calling cycle:
1. Send messages + tool schemas to LLM (via SlowBurnLLM.call_llm)
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


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def run_agent(
    *,
    llm: Any,
    task: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], str],
    system_prompt: str = "You are a helpful agent. Use your tools to complete the task.",
    max_steps: int = 20,
    verbose: bool = True,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a ReAct agent loop with a SlowBurnLLM worker.

    The worker's built-in cost tracking, backpressure, and rate limiting
    apply to every LLM call automatically.

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
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    total_tool_calls = 0

    for step in range(1, max_steps + 1):
        step_dir = None
        if log_dir is not None:
            step_dir = log_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)

        step_start = time.time()

        if step_dir:
            _save_json(step_dir / "input.json", {
                "step": step,
                "num_messages": len(messages),
                "messages": messages,
            })

        total_text = " ".join(
            m.get("content", "") or ""
            for m in messages
            if isinstance(m.get("content"), str)
        )

        response_text = llm.call_llm(
            prompt=total_text,
            litellm_params={
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            },
        ).result(timeout=120.0)

        step_elapsed = time.time() - step_start

        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls_data = parsed["tool_calls"]
            else:
                tool_calls_data = None
        except (json.JSONDecodeError, TypeError):
            tool_calls_data = None

        if tool_calls_data is not None:
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls_data
                ],
            }
            messages.append(assistant_msg)

            tool_call_log = []
            for tc_idx, tc in enumerate(tool_calls_data, 1):
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                if verbose:
                    args_preview = json.dumps(fn_args, ensure_ascii=False)
                    if len(args_preview) > 100:
                        args_preview = args_preview[:97] + "..."
                    print(f"    [{step:2d}] tool: {fn_name}({args_preview})")

                if step_dir:
                    _save_json(step_dir / f"tool_{tc_idx:02d}_call.json", {
                        "tool_call_id": tc["id"],
                        "function": fn_name,
                        "arguments": fn_args,
                    })

                tool_result = tool_executor(fn_name, fn_args)
                total_tool_calls += 1

                if step_dir:
                    try:
                        result_data = json.loads(tool_result)
                    except (json.JSONDecodeError, TypeError):
                        result_data = {"raw": tool_result}
                    _save_json(step_dir / f"tool_{tc_idx:02d}_result.json", result_data)

                tool_call_log.append({
                    "function": fn_name,
                    "arguments": fn_args,
                    "result_length": len(tool_result),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            if step_dir:
                _save_json(step_dir / "output.json", {
                    "step": step,
                    "type": "tool_calls",
                    "tool_calls": tool_call_log,
                    "elapsed_seconds": step_elapsed,
                })

            if verbose:
                reporter = llm.get_reporter().result(timeout=5.0)
                print(
                    f"         ${reporter.total_cost():.6f} | "
                    f"{reporter.num_calls} LLM calls | "
                    f"{total_tool_calls} tool calls"
                )
        else:
            final_text = response_text

            if step_dir:
                _save_json(step_dir / "output.json", {
                    "step": step,
                    "type": "final_answer",
                    "content": final_text,
                    "elapsed_seconds": step_elapsed,
                })

            if verbose:
                reporter = llm.get_reporter().result(timeout=5.0)
                print(
                    f"    [{step:2d}] DONE (${reporter.total_cost():.6f}, "
                    f"{step} LLM calls, {total_tool_calls} tool calls)"
                )
                preview = final_text[:100].replace("\n", " ")
                print(f"         {preview}...")

            return {
                "result": final_text,
                "steps": step,
                "tool_calls": total_tool_calls,
                "messages": messages,
            }

    if log_dir:
        _save_json(log_dir / "max_steps_reached.json", {
            "max_steps": max_steps,
            "total_tool_calls": total_tool_calls,
        })

    final_text = messages[-1].get("content", "") if len(messages) > 0 else ""
    return {
        "result": f"[Agent reached max_steps={max_steps}] {final_text}",
        "steps": max_steps,
        "tool_calls": total_tool_calls,
        "messages": messages,
    }
