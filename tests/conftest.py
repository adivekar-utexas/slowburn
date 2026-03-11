"""Shared pytest fixtures and configuration for slowburn tests."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("timeout", default=None) is None:
        config.option.timeout = 60
        config.option.timeout_method = "thread"
