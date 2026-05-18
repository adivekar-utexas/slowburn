"""Shared type aliases and constants for SlowBurn."""

from typing import Literal

ImageDetailLevel = Literal["low", "high", "auto"]
ToolChoiceOption = Literal["auto", "required", "none"]
BackpressureNotify = Literal["ignore", "warn"]
BudgetOverflowAction = Literal["warn", "error", "ignore"]
PricingUnavailableAction = Literal["error", "warn", "ignore"]
WindowAlias = Literal["weekly", "daily", "hourly", "minutely"]
ExecutionBackend = Literal["Asyncio", "Ray"]

WINDOW_ALIAS_SECONDS = {
    "weekly": 7 * 24 * 3600,
    "daily": 24 * 3600,
    "hourly": 3600,
    "minutely": 60,
}
