"""Shared pytest fixtures and configuration for slowburn tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def _have_api_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))


def _get_model_and_key():
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter/google/gemini-2.0-flash-001", os.getenv("OPENROUTER_API_KEY")
    return "gpt-4o-mini", os.getenv("OPENAI_API_KEY", "")


skip_no_api_key = pytest.mark.skipif(
    not _have_api_key(),
    reason="No OPENROUTER_API_KEY or OPENAI_API_KEY in .env",
)


@pytest.fixture(scope="session")
def llm_model_and_key():
    """Return (model, api_key) resolved from .env for real LLM tests."""
    return _get_model_and_key()


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("timeout", default=None) is None:
        config.option.timeout = 60
        config.option.timeout_method = "thread"
