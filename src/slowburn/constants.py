"""Shared type aliases and constants for SlowBurn."""

from typing import Literal

ImageDetailLevel = Literal["low", "high", "auto"]
ToolChoiceOption = Literal["auto", "required", "none"]
WindowAlias = Literal["daily", "hourly", "minutely"]
ExecutionBackend = Literal["asyncio", "ray"]

WINDOW_ALIAS_SECONDS = {
    "daily": 86400,
    "hourly": 3600,
    "minutely": 60,
}
