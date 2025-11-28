# lionpride

[![PyPI version](https://img.shields.io/pypi/v/lionpride.svg)](https://pypi.org/project/lionpride/)
[![Python](https://img.shields.io/pypi/pyversions/lionpride.svg)](https://pypi.org/project/lionpride/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/khive-ai/lionpride/blob/main/LICENSE)
[![CI](https://github.com/khive-ai/lionpride/actions/workflows/ci.yml/badge.svg)](https://github.com/khive-ai/lionpride/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/khive-ai/lionpride/graph/badge.svg?token=FAE47FY26T)](https://app.codecov.io/github/khive-ai/lionpride)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-ready primitives for multi-agent workflow orchestration.**

> **Alpha Release** - API stabilizing. Originated from [lionagi](https://github.com/khive-ai/lionagi) v0, elevated and hardened for production use.

## Features

- **Model Agnostic** - Built-in providers for OpenAI-compatible APIs, Anthropic, Gemini
- **LNDL** - Domain-specific language for LLM structured output and enhanced reasoning
- **Declarative Workflows** - Report/Form system for multi-step agent pipelines
- **Async Native** - Operation graph building, dependency-aware execution
- **Modular Architecture** - Protocol-based composition, zero framework lock-in
- **99%+ Test Coverage** - Production-hardened with comprehensive test suites

## Installation

```bash
pip install lionpride
```

## Quick Start

```python
import asyncio
from lionpride import Session, iModel

# Create model and session
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)

# Create branch with model access
branch = session.create_branch(name="main")

# Conduct an operation
async def main():
    result = await session.conduct(
        "operate",
        branch,
        instruction="What is 2 + 2?",
    )
    print(result)

asyncio.run(main())
```

## Core Concepts

### Session & Branch

`Session` orchestrates messages, services, and operations. `Branch` is a named conversation thread with access control.

```python
from lionpride import Session, iModel

# Session with default model
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)

# Branch inherits access to default model
branch = session.create_branch(
    name="analysis",
    capabilities={"MyOutputSchema"},  # Allowed output types
)
```

### Operations

Operations are composable building blocks for agent workflows:

```python
from lionpride.operations.operate import operate, OperateParams, CommunicateParams, GenerateParams

# Structured output with validation
params = OperateParams(
    communicate=CommunicateParams(
        generate=GenerateParams(
            instruction="Analyze this data and return insights",
            request_model=MyInsightsModel,  # Pydantic model for validation
        )
    )
)

result = await operate(session, branch, params)
```

### Services

`ServiceRegistry` manages models and tools with O(1) name lookup:

```python
from lionpride import Session, iModel, ServiceRegistry

# Register multiple models
registry = ServiceRegistry()
registry.register(iModel(provider="openai", model="gpt-4o", name="gpt4"))
registry.register(iModel(provider="anthropic", model="claude-3-5-sonnet", name="claude"))

session = Session(services=registry)
branch = session.create_branch(resources={"gpt4", "claude"})  # Access to both
```

### Declarative Workflows

`Report` and `Form` enable multi-step agent pipelines with automatic dependency resolution:

```python
from pydantic import BaseModel
from lionpride.work import Report, flow_report

class Analysis(BaseModel):
    summary: str
    score: float

class MyReport(Report):
    analysis: Analysis | None = None  # Schema attribute

    assignment: str = "topic -> analysis"
    form_assignments: list[str] = ["topic -> analysis"]

report = MyReport()
report.initialize(topic="AI coding assistants")
result = await flow_report(session, branch, report)
```

## Architecture

```text
lionpride/
├── core/           # Primitives: Element, Pile, Flow, Graph, Event
├── session/        # Session, Branch, Message management
├── services/       # iModel, Tool, ServiceRegistry, MCP integration
├── operations/     # operate, react, communicate, generate, parse
├── work/           # Declarative workflows: Report, Form, flow_report
├── rules/          # Validation rules and auto-correction
├── types/          # Spec, Operable, type system
├── lndl/           # LNDL parser and resolver
└── ln/             # Utility functions
```

See [CLAUDE.md](CLAUDE.md) for detailed codebase navigation.

## Documentation

- [CLAUDE.md](CLAUDE.md) - AI agent codebase guide
- [AGENTS.md](AGENTS.md) - Quick reference for AI agents
- [notebooks/](notebooks/) - Example notebooks

## Roadmap

- Formal mathematical framework for agent composition
- Rust core for performance-critical paths
- Enhanced MCP (Model Context Protocol) support

## License

Apache-2.0
