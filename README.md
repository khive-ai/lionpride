# lionpride

[![PyPI version](https://img.shields.io/pypi/v/lionpride.svg)](https://pypi.org/project/lionpride/)
[![Python](https://img.shields.io/pypi/pyversions/lionpride.svg)](https://pypi.org/project/lionpride/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/khive-ai/lionpride/blob/main/LICENSE)
[![CI](https://github.com/khive-ai/lionpride/actions/workflows/ci.yml/badge.svg)](https://github.com/khive-ai/lionpride/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/khive-ai/lionpride/graph/badge.svg?token=FAE47FY26T)](https://app.codecov.io/github/khive-ai/lionpride)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-ready primitives for multi-agent workflow orchestration.**

> ⚠️ **Alpha/Experimental** - API unstable. For research and development use. Originated from [lionagi](https://github.com/khive-ai/lionagi) v0.

## Features

- **Model Agnostic** - Built-in providers for OpenAI-compatible APIs, Anthropic, Gemini
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
from lionpride.operations.operate import generate, GenerateParams

# Create model and session
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch(name="main", resources={model.name})

async def main():
    # Simple text generation
    result = await generate(
        session, branch,
        params=GenerateParams(instruction="What is 2 + 2?", return_as="text"),
    )
    print(result)  # "4" or similar

asyncio.run(main())
```

## Core Concepts

### Session & Branch

`Session` orchestrates messages, services, and operations. `Branch` is a named conversation thread with capability-based access control.

```python
from lionpride import Session, iModel

# Session with default model
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)

# Branch with capability and resource restrictions
branch = session.create_branch(
    name="restricted",
    capabilities={"AnalysisModel"},  # Only these output types allowed
    resources={"gpt-4o-mini"},        # Only these services allowed
)
```

### Operations

Operations are composable building blocks for agent workflows:

```python
from pydantic import BaseModel
from lionpride.operations.operate import operate, OperateParams, GenerateParams

class Insights(BaseModel):
    summary: str
    score: float

# Branch must have capability for output field
branch = session.create_branch(capabilities={"insights"}, resources={model.name})

# Structured output with validation
params = OperateParams(
    generate=GenerateParams(
        instruction="Analyze this data",
        request_model=Insights,
    ),
    capabilities={"insights"},  # Explicit capability declaration
)

result = await operate(session, branch, params)
print(result.insights)  # Insights(summary="...", score=0.85)
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

`Report` and `Form` enable multi-step agent pipelines with automatic dependency resolution. Model docstrings serve as agent instructions:

```python
from pydantic import BaseModel
from lionpride.work import Report, flow_report

class Analysis(BaseModel):
    '''Analyze the topic and provide insights.
    Focus on actionable recommendations.'''
    summary: str
    recommendations: list[str]

class MyReport(Report):
    analysis: Analysis | None = None  # Schema attribute (docstring = instruction)

    assignment: str = "topic -> analysis"
    form_assignments: list[str] = ["topic -> analysis"]

report = MyReport()
report.initialize(topic="AI coding assistants")
result = await flow_report(session, report, branch=branch)
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
└── ln/             # Utility functions
```

## Documentation

- [API Reference](docs/api/) - Comprehensive API docs
  - [Session](docs/api/session/) - Session, Branch, Message
  - [Services](docs/api/services/) - iModel, ServiceRegistry, Tool
  - [Operations](docs/api/operations/) - operate, react, communicate
  - [Rules](docs/api/rules/) - Validation and auto-correction
  - [Work](docs/api/work/) - Report, Form, flow_report
- [notebooks/](notebooks/) - Example notebooks

## Roadmap

- Formal mathematical framework for agent composition
- Rust core for performance-critical paths
- Enhanced MCP (Model Context Protocol) support

## License

Apache-2.0
