"""Shared pytest fixtures and configuration for slowburn tests."""

import os
import sys
from pathlib import Path

# Use local concurry source if available (development convenience)
_local_concurry = Path(__file__).parent.parent.parent / "concurry" / "src"
if _local_concurry.is_dir():
    sys.path.insert(0, str(_local_concurry))

import pytest
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Centralized test constants — change HERE when models are deprecated
# ---------------------------------------------------------------------------
MOCK_MODEL_NAME = "gpt-4o-mini"
"""Model name used in mocked tests. Not a real call; just a label in mock
responses and worker constructors. Safe to change without cost implications."""

E2E_OPENROUTER_MODEL = "openrouter/google/gemini-2.0-flash-001"
"""Model used for real e2e tests when OPENROUTER_API_KEY is available."""

E2E_OPENAI_MODEL = "gpt-4o-mini"
"""Model used for real e2e tests when only OPENAI_API_KEY is available."""


def _have_api_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))


def _get_model_and_key():
    if os.getenv("OPENROUTER_API_KEY"):
        return E2E_OPENROUTER_MODEL, os.getenv("OPENROUTER_API_KEY")
    return E2E_OPENAI_MODEL, os.getenv("OPENAI_API_KEY", "")


skip_no_api_key = pytest.mark.skipif(
    not _have_api_key(),
    reason="No OPENROUTER_API_KEY or OPENAI_API_KEY set (env var or .env file)",
)


@pytest.fixture(scope="session")
def llm_model_and_key():
    """Return (model, api_key) resolved from environment for real LLM tests."""
    return _get_model_and_key()


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("timeout", default=None) is None:
        config.option.timeout = 60
        config.option.timeout_method = "thread"
