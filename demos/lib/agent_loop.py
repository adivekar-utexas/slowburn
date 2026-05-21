"""
ReAct agent loop powered by a SlowBurnLLM worker.

Implements the standard tool-calling cycle:
1. Send messages + tool schemas to LLM (via SlowBurnLLM.call_llm with history=)
2. If LLM returns tool_calls -> execute each tool -> append results -> goto 1
3. If LLM returns text content -> return it (agent is done)

The SlowBurnLLM worker handles all cost tracking, backpressure, and rate
limiting internally. This loop just drives the conversation and tools.

To keep context small on long-running agents, each step after the first
sends only: the original task + current workspace file contents + the
last turn of history (not the entire conversation).

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


def _read_workspace_files(workspace: Path, exclude_dirs: Optional[List[Path]] = None) -> str:
    """Read user-created files in the workspace and return a summary string.

    Skips hidden files, files starting with '_', and files under any
    directories listed in *exclude_dirs* (e.g., log directories).
    """
    excluded_prefixes: List[str] = []
    if exclude_dirs is not None:
        for excluded_directory in exclude_dirs:
            prefix = str(excluded_directory.resolve())
            if not prefix.endswith("/"):
                prefix += "/"
            excluded_prefixes.append(prefix)

    lines: List[str] = []
    for file_path in sorted(workspace.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith(".") or file_path.name.startswith("_"):
            continue
        resolved = str(file_path.resolve())
        if any(resolved.startswith(prefix) for prefix in excluded_prefixes):
            continue
        relative = file_path.relative_to(workspace)
        try:
            content = file_path.read_text(errors="replace")
            lines.append(f"--- {relative} ---\n{content}")
        except (OSError, UnicodeDecodeError):
            lines.append(f"--- {relative} --- (unreadable)")
    if len(lines) == 0:
        return "(no files in workspace yet)"
    return "\n\n".join(lines)


def _extract_last_turn_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract assistant and tool messages from the last turn.

    Returns everything AFTER the last user message (the assistant response
    and any tool call/result exchanges). The user message itself is excluded
    because the caller provides a fresh prompt with updated context.

    Returns an empty list if there are no user messages.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return messages[index + 1 :]
    return []


def _get_last_tool_name(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Return the function name of the last tool call in the conversation.

    Scans backward for the most recent assistant message with tool_calls.
    Returns None if no tool calls found.
    """
    for index in range(len(messages) - 1, -1, -1):
        tool_calls = messages[index].get("tool_calls")
        if tool_calls is not None and len(tool_calls) > 0:
            return tool_calls[-1]["function"]["name"]
    return None


def run_agent(
    *,
    llm: SlowBurnLLM,
    task: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], str],
    system_prompt: str = "You are a helpful agent. Use your tools to complete the task.",
    output_file: Optional[str] = None,
    max_steps: int = 20,
    verbose: bool = True,
    log_dir: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a ReAct agent loop with a SlowBurnLLM worker.

    Uses the multi-turn conversation API: ``call_llm(history=..., tools=...)``
    returns a messages list with the assistant response appended. Tool results
    are appended to the same list and re-submitted.

    To prevent context from growing unboundedly, each step after the first
    sends only the original task, current workspace file contents, and the
    last turn of history. The full conversation is still tracked internally
    for logging and the return value.

    Args:
        llm: A live SlowBurnLLM worker (from ``create_llm()``).
        task: The user's task description.
        tools: Tool schemas in OpenAI format (list of dicts).
        tool_executor: Callable(tool_name, tool_args) -> str that executes tools.
        system_prompt: System prompt for the agent.
        output_file: Filename for the agent's output report (e.g., "report.md").
            If provided and *workspace* is set, the file is created empty at
            start and the agent is explicitly instructed to update it each step.
        max_steps: Maximum number of LLM calls before forcing termination.
        verbose: Print step-by-step progress.
        log_dir: Directory to save step-level logs. If None, no logs are saved.
        workspace: Directory where the agent reads/writes files. If provided,
            current file contents are included as context at each step,
            allowing history truncation without losing state.

    Returns:
        Dict with keys: "result" (final text), "steps" (int), "tool_calls" (int),
        "messages" (full conversation).
    """
    if output_file is not None and workspace is not None:
        output_path = workspace / output_file
        if not output_path.exists():
            output_path.write_text("")

        append_findings_schema = {
            "type": "function",
            "function": {
                "name": "append_findings",
                "description": (
                    f"Append new research findings to '{output_file}'. "
                    f"Write ONLY the new content to add. It will be appended "
                    f"to the end of the existing file automatically."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The new findings to append to the report file.",
                        },
                    },
                    "required": ["content"],
                },
            },
        }
        effective_tools = [
            tool for tool in tools if tool.get("function", {}).get("name") not in ("write_file", "read_file")
        ] + [append_findings_schema]

        original_executor = tool_executor

        def appending_executor(function_name: str, function_arguments: Dict[str, Any]) -> str:
            if function_name == "append_findings":
                new_content = function_arguments.get("content", "")
                existing = output_path.read_text()
                separator = "\n\n" if len(existing.strip()) > 0 else ""
                output_path.write_text(existing + separator + new_content)
                return json.dumps(
                    {
                        "status": "appended",
                        "file": output_file,
                        "chars_added": len(new_content),
                    }
                )
            return original_executor(function_name, function_arguments)

        tool_executor = appending_executor
        tools = effective_tools

    all_messages: List[Dict[str, Any]] = []
    total_tool_calls = 0
    previous_cost = 0.0

    for step in range(1, max_steps + 1):
        step_dir: Optional[Path] = None
        if log_dir is not None:
            step_dir = log_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)

        step_start = time.time()

        if step == 1:
            prompt = task
            context_history: List[Dict[str, Any]] = []
            if verbose:
                print(f"\n    ── Step {step}/{max_steps} ──")
                print("    [loop] Sending initial task to LLM")
        else:
            last_turn_responses = _extract_last_turn_responses(all_messages)
            context_history = []
            if len(last_turn_responses) > 0:
                context_history.append(
                    {
                        "role": "user",
                        "content": "Here is what you did in the previous step:",
                    }
                )
                context_history.extend(last_turn_responses)

            if verbose:
                print(f"\n    ── Step {step}/{max_steps} ──")
                print(f"    [loop] Including last turn ({len(last_turn_responses)} messages) as context")

            prompt_parts = [f"ORIGINAL TASK:\n{task}"]
            prompt_parts.append(f"PROGRESS: Step {step}/{max_steps}, {total_tool_calls} tool calls so far.")
            if workspace is not None:
                exclude_dirs = [log_dir] if log_dir is not None else None
                workspace_content = _read_workspace_files(workspace, exclude_dirs=exclude_dirs)
                prompt_parts.append(f"CURRENT WORKSPACE FILES:\n{workspace_content}")
                if verbose:
                    file_count = workspace_content.count("--- ")
                    if workspace_content == "(no files in workspace yet)":
                        file_count = 0
                    print(f"    [loop] Reading workspace: {file_count} file(s) included as context")

            if output_file is not None:
                last_tool_name = _get_last_tool_name(all_messages)
                if last_tool_name == "append_findings":
                    prompt_parts.append(
                        f"INSTRUCTIONS FOR THIS STEP:\n"
                        f"Use search_web to find ONE new piece of information "
                        f"relevant to the task that is not yet in '{output_file}'.\n"
                        f"If the report is complete, respond with a text summary (no tool calls)."
                    )
                    if verbose:
                        print("    [loop] Instructing LLM: search for new information")
                else:
                    prompt_parts.append(
                        f"INSTRUCTIONS FOR THIS STEP:\n"
                        f"You just gathered new information. Now you MUST call append_findings "
                        f"to add your new findings to '{output_file}'. "
                        f"Write ONLY the new content; it will be appended automatically. "
                        f"Do NOT search again. Call append_findings now."
                    )
                    if verbose:
                        print(f"    [loop] Instructing LLM: append findings to '{output_file}'")
            else:
                prompt_parts.append(
                    "Continue working on the task. Use your tools as needed. "
                    "If you have gathered enough information, respond with your final answer."
                )
            prompt = "\n\n".join(prompt_parts)

            if verbose:
                print("    [loop] Calling LLM...")

        if step_dir is not None:
            input_messages = llm.build_messages(
                prompt=prompt,
                system_prompt=system_prompt,
                history=context_history,
            ).result(timeout=10.0)
            _save_json(
                step_dir / "input.json",
                {
                    "step": step,
                    "num_messages": len(input_messages),
                    "messages": input_messages,
                },
            )

        step_messages = llm.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            history=context_history,
            tools=tools,
            tool_choice="auto",
        ).result(timeout=120.0)

        if step == 1:
            all_messages = step_messages
        else:
            assistant_response = step_messages[-1]
            all_messages.append({"role": "user", "content": prompt})
            all_messages.append(assistant_response)

        step_elapsed = time.time() - step_start

        assistant_message = all_messages[-1]
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
                    print(f"    [llm]  tool call: {function_name}({arguments_preview})")

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

                if verbose:
                    print(f"    [loop] Executed {function_name} → {len(tool_result)} chars returned")

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

                all_messages.append(
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
                    print(f"    [llm]  {assistant_content}")

                print(
                    f"    [cost] step ${step_cost:.6f} | "
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

                print("    [llm]  DONE — final text response:")
                print(f"    [llm]  {final_text}")
                print(
                    f"    [cost] step ${step_cost:.6f} | "
                    f"total ${current_cost:.6f} | "
                    f"{step} LLM calls | {total_tool_calls} tool calls"
                )

            return {
                "result": final_text,
                "steps": step,
                "tool_calls": total_tool_calls,
                "messages": all_messages,
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
    if len(all_messages) > 0:
        final_text = all_messages[-1].get("content", "") or ""
    return {
        "result": f"[Agent reached max_steps={max_steps}] {final_text}",
        "steps": max_steps,
        "tool_calls": total_tool_calls,
        "messages": all_messages,
    }
