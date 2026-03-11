# SlowBurn

[![PyPI version](https://img.shields.io/pypi/v/slowburn.svg)](https://pypi.org/project/slowburn/)
[![Tests](https://github.com/adivekar-utexas/slowburn/actions/workflows/tests.yml/badge.svg)](https://github.com/adivekar-utexas/slowburn/actions/workflows/tests.yml)
[![Linting](https://github.com/adivekar-utexas/slowburn/actions/workflows/linting.yml/badge.svg)](https://github.com/adivekar-utexas/slowburn/actions/workflows/linting.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SlowBurn: Cost-Sustainable Concurrent Execution for Long-Horizon LLM Agents**

SlowBurn is a Python library that lets LLM agent workflows run within a dollar budget by automatically slowing down — not crashing — when the budget is tight. It combines concurrent execution with dollar-denominated rate limiting and per-call cost tracking, so researchers can launch overnight batch experiments on a $5/day budget and wake up to completed results instead of a crashed script.

SlowBurn provides a native asyncio LLM worker for direct use, and drop-in integrations for [AutoGen (AG2)](https://github.com/ag2ai/ag2) and [CrewAI](https://github.com/crewAIInc/crewAI) that add budget control to existing multi-agent workflows without code changes.

## Installation

```bash
pip install slowburn
```

## Development

```bash
git clone https://github.com/adivekar-utexas/slowburn.git
cd slowburn
pip install -e ".[dev]"
pytest
```

## License

MIT
