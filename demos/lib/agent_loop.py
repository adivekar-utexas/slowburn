"""
ReAct agent loop with SlowBurn cost tracking and step-level logging.

Implements the standard tool-calling cycle:
1. Send messages + tool schemas to LLM
2. If LLM returns tool_calls -> execute each tool -> append results -> goto 1
3. If LLM returns text content -> return it (agent is done)

Every LLM call goes through a Concurry LimitSet with CostLimit, providing
budget-aware backpressure. When the budget is exhausted, the next LLM call
blocks until the window rolls over.

Each step is logged to disk at ``<log_dir>/step_<N>/`` with:
- ``input.json``: messages sent to the LLM
- ``output.json``: raw LLM response (content or tool_calls)
- ``tool_<M>_call.json``: tool call arguments
- ``tool_<M>_result.json``: tool execution result
- ``cost.json``: cost data for this step
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import litellm

from slowburn.limits import DEFAULT_COST_LIMIT_KEY, microdollars_to_dollars
from slowburn.pricing import PricingCache
from slowburn.reporter import CostReporter

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.set_verbose = False


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))


async def run_agent(
    *,
    model: str,
    task: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, Dict[str, Any]], str],
    limit_set: Any,
    reporter: CostReporter,
    api_key: str = "",
    system_prompt: str = "You are a helpful agent. Use your tools to complete the task.",
    max_steps: int = 20,
    max_tokens: int = 1000,
    temperature: float = 0.3,
    verbose: bool = True,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a ReAct agent loop with cost-tracked tool calling.

    Args:
        model: litellm model identifier.
        task: The user's task description.
        tools: Tool schemas in OpenAI format (list of dicts).
        tool_executor: Callable(tool_name, tool_args) -> str that executes tools.
        limit_set: Concurry LimitSet with CostLimit for budget tracking.
        reporter: CostReporter to log per-call costs.
        api_key: API key for the LLM provider.
        system_prompt: System prompt for the agent.
        max_steps: Maximum number of LLM calls before forcing termination.
        max_tokens: Maximum output tokens per LLM call.
        temperature: Sampling temperature.
        verbose: Print step-by-step progress.
        log_dir: Directory to save step-level logs. If None, no logs are saved.

    Returns:
        Dict with keys: "result" (final text), "steps" (int), "tool_calls" (int),
        "messages" (full conversation), "cost_usd" (total cost for this run).
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    total_tool_calls = 0
    run_start_cost = reporter.total_cost()

    for step in range(1, max_steps + 1):
        step_dir = None
        if log_dir is not None:
            step_dir = log_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)

        step_start = time.time()

        estimated_input = int(
            max(sum(len(str(m.get("content", ""))) for m in messages) // 3, 1) * 5.0
        ) + 50
        estimated_output = max_tokens
        estimated_cost = PricingCache.estimate_cost_microdollars(
            model, estimated_input, estimated_output,
        )

        if step_dir:
            _save_json(step_dir / "input.json", {
                "step": step,
                "model": model,
                "estimated_input_tokens": estimated_input,
                "estimated_output_tokens": estimated_output,
                "estimated_cost_microdollars": estimated_cost,
                "num_messages": len(messages),
                "messages": messages,
            })

        from slowburn.backpressure import timed_acquire

        requested = {DEFAULT_COST_LIMIT_KEY: max(estimated_cost, 1)}
        with timed_acquire(
            limit_set, requested, context=f"step {step}, model={model}"
        ) as acq:
            try:
                litellm.drop_params = True
                last_error = None
                for attempt in range(3):
                    try:
                        response = await asyncio.wait_for(
                            litellm.acompletion(
                                model=model,
                                messages=messages,
                                tools=tools,
                                tool_choice="auto",
                                api_key=api_key if api_key else None,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            ),
                            timeout=60.0,
                        )
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < 2:
                            if verbose:
                                print(f"    [{step:2d}] Retry {attempt + 1}/3: {type(e).__name__}")
                            await asyncio.sleep(1.0)
                if last_error is not None:
                    raise last_error

                actual_input = response.usage.prompt_tokens
                actual_output = response.usage.completion_tokens
                actual_cost = PricingCache.actual_cost_microdollars(
                    response, model=model,
                )

                acq.update(usage={DEFAULT_COST_LIMIT_KEY: actual_cost})

                reporter.log_call(
                    model=model,
                    cost_usd=microdollars_to_dollars(actual_cost),
                    input_tokens=actual_input,
                    output_tokens=actual_output,
                    metadata={"step": step, "type": "agent_loop"},
                )
            except Exception:
                acq.update(usage={DEFAULT_COST_LIMIT_KEY: estimated_cost})
                if step_dir:
                    _save_json(step_dir / "error.json", {
                        "step": step,
                        "error": str(last_error) if last_error else "unknown",
                    })
                raise

        msg = response.choices[0].message
        step_elapsed = time.time() - step_start
        cost_this_step = microdollars_to_dollars(actual_cost)

        if msg.tool_calls is not None and len(msg.tool_calls) > 0:
            messages.append(msg.model_dump())

            tool_call_log = []
            for tc_idx, tc in enumerate(msg.tool_calls, 1):
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                if verbose:
                    args_preview = json.dumps(fn_args, ensure_ascii=False)
                    if len(args_preview) > 100:
                        args_preview = args_preview[:97] + "..."
                    print(f"    [{step:2d}] tool: {fn_name}({args_preview})")

                if step_dir:
                    _save_json(step_dir / f"tool_{tc_idx:02d}_call.json", {
                        "tool_call_id": tc.id,
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
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            if step_dir:
                _save_json(step_dir / "output.json", {
                    "step": step,
                    "type": "tool_calls",
                    "tool_calls": tool_call_log,
                    "actual_input_tokens": actual_input,
                    "actual_output_tokens": actual_output,
                    "cost_usd": cost_this_step,
                    "elapsed_seconds": step_elapsed,
                })

            if verbose:
                cost_so_far = reporter.total_cost() - run_start_cost
                print(
                    f"         ${cost_so_far:.6f} | "
                    f"{reporter.num_calls} LLM calls | "
                    f"{total_tool_calls} tool calls"
                )
        else:
            final_text = msg.content or ""

            if step_dir:
                _save_json(step_dir / "output.json", {
                    "step": step,
                    "type": "final_answer",
                    "content": final_text,
                    "actual_input_tokens": actual_input,
                    "actual_output_tokens": actual_output,
                    "cost_usd": cost_this_step,
                    "elapsed_seconds": step_elapsed,
                })

            if verbose:
                cost_so_far = reporter.total_cost() - run_start_cost
                print(
                    f"    [{step:2d}] DONE (${cost_so_far:.6f}, "
                    f"{step} LLM calls, {total_tool_calls} tool calls)"
                )
                preview = final_text[:100].replace("\n", " ")
                print(f"         {preview}...")

            return {
                "result": final_text,
                "steps": step,
                "tool_calls": total_tool_calls,
                "messages": messages,
                "cost_usd": reporter.total_cost() - run_start_cost,
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
        "cost_usd": reporter.total_cost() - run_start_cost,
    }
