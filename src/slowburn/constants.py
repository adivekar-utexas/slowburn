"""Shared type aliases and constants for SlowBurn."""

from typing import Literal

ImageDetailLevel = Literal["low", "high", "auto"]
ToolChoiceOption = Literal["auto", "required", "none"]
BackpressureNotify = Literal["ignore", "warn"]
BudgetOverflowAction = Literal["warn", "error", "ignore"]
PricingUnavailableAction = Literal["error", "warn", "ignore"]
ExecutionBackend = Literal["Asyncio", "Ray"]
