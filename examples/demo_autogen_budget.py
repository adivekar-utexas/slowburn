"""
Demo: AutoGen (AG2) multi-agent chat with SlowBurn cost control.

Shows how to use SlowBurnModelClient to add dollar-budget backpressure
to any AutoGen agent. The ModelClient protocol gives us full access to
the litellm response object, providing exact cost tracking.

Usage:
    pip install slowburn[autogen]
    export OPENAI_API_KEY="sk-..."
    python examples/demo_autogen_budget.py
"""

import os


def main():
    try:
        from autogen import AssistantAgent, UserProxyAgent, gather_usage_summary
    except ImportError:
        print("AutoGen (AG2) not installed. Run: pip install slowburn[autogen]")
        return

    from concurry import LimitSet

    from slowburn.integrations.autogen import SlowBurnModelClient
    from slowburn.limits import CostLimit
    from slowburn.reporter import CostReporter

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY to run this demo.")
        print("Showing the setup pattern instead.\n")

    # === 1. Create shared budget (all agents share this) ===
    limit_set = LimitSet(
        limits=[CostLimit(budget_usd=3.0, window_seconds=3600)],  # $3/hour
        mode="thread",
        shared=True,
    )
    reporter = CostReporter()

    # === 2. Create AG2 agents with SlowBurn model client ===
    config_list = [{"model": "slowburn/gpt-4o-mini", "api_key": api_key}]

    assistant = AssistantAgent(
        "assistant",
        llm_config={"config_list": config_list},
        system_message="You are a helpful AI assistant. Be concise.",
    )
    assistant.register_model_client(
        model_client_cls=SlowBurnModelClient,
        limit_set=limit_set,
        reporter=reporter,
    )

    user_proxy = UserProxyAgent(
        "user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )

    # === 3. Run conversation ===
    if api_key:
        user_proxy.initiate_chat(
            assistant,
            message="Explain in 3 bullet points why cost control matters for LLM agents.",
        )
    else:
        print("[dry run] Would run multi-turn conversation with budget-controlled LLM calls")

    # === 4. Report costs ===
    print(f"\n{'=' * 60}")
    print("SLOWBURN COST REPORT")
    print(f"{'=' * 60}")
    print(reporter.to_markdown())
    print(f"\nTotal cost: ${reporter.total_cost():.6f}")

    # AG2's built-in tracking also works:
    if api_key:
        print(f"\n{'=' * 60}")
        print("AG2 USAGE SUMMARY")
        print(f"{'=' * 60}")
        usage = gather_usage_summary([assistant, user_proxy])
        print(usage)

    print("\nDone.")


if __name__ == "__main__":
    main()
