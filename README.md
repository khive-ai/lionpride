# lionpride

[![PyPI version](https://img.shields.io/pypi/v/lionpride.svg)](https://pypi.org/project/lionpride/)
[![Python](https://img.shields.io/pypi/pyversions/lionpride.svg)](https://pypi.org/project/lionpride/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/khive-ai/lionpride/blob/main/LICENSE)
[![CI](https://github.com/khive-ai/lionpride/actions/workflows/ci.yml/badge.svg)](https://github.com/khive-ai/lionpride/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/khive-ai/lionpride/graph/badge.svg?token=FAE47FY26T)](https://app.codecov.io/github/khive-ai/lionpride)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-ready multi-agent workflow orchestration framework.**

> **Alpha Release** - API may change. Originated from [lionagi](https://github.com/khive-ai/lionagi) v0, elevated and hardened for production use.

## Features

- **Model Agnostic** - Built-in providers for OpenAI-compatible APIs, Anthropic, Gemini, Claude Code
- **LNDL** - Domain-specific language for LLM structured output and enhanced reasoning (JSON fallback supported)
- **Async Native** - Operation graph building, dependency-aware execution, auto-extensions
- **Modular Architecture** - Protocol-based composition, zero framework lock-in

## Installation

```bash
pip install lionpride
```

## Quick Start

```python
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

# Create session with model
session = Session()
model = iModel(provider="openai", model="gpt-4o-mini")
session.services.register(model)

# Create branch and communicate
branch = session.create_branch(name="main")
result = await communicate(
    session=session,
    branch=branch,
    parameters={
        "instruction": "Analyze this data",
        "imodel": model.name,
    }
)
```

## Roadmap

- Formal mathematical framework
- Rust core for performance-critical paths

## License

Apache-2.0
