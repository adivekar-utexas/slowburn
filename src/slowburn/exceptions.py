"""SlowBurn-specific exception hierarchy."""

from abc import ABC


class SlowBurnNonRetryableError(RuntimeError, ABC):
    """Abstract base class for deterministic non-retryable SlowBurn errors."""


class PricingUnavailableError(SlowBurnNonRetryableError):
    """Raised when pricing is unavailable and policy is set to error."""


class BudgetOverflowError(SlowBurnNonRetryableError):
    """Raised when a single call cannot fit within configured budget capacity."""


class ToolCallContractError(SlowBurnNonRetryableError):
    """Raised when tool-call responses violate the call_llm return contract."""


class InvalidConfigValueError(SlowBurnNonRetryableError):
    """Raised when a SlowBurn config/action field has an invalid value."""


class BatchInputMismatchError(SlowBurnNonRetryableError):
    """Raised when batch input arrays have inconsistent lengths."""
