# Operations

> Composable LLM operations from low-level generation to multi-turn reasoning

## Overview

The `operations` module provides a layered system of composable operations for LLM interactions. Each operation builds on lower-level ones, enabling both fine-grained control and high-level convenience.

**Operation Hierarchy:**

```text
generate ──────────────┐
parse ─────────────────┼─► communicate
                       │
communicate ───────────┤
act ───────────────────┼─► operate
                       │
operate ───────────────┴─► react
```

**Key Capabilities:**

- **generate**: Lowest-level LLM call (no message persistence)
- **parse**: Extract structured data from text
- **communicate**: Generate + parse in one operation
- **operate**: Full structured output with validation and auto-correction
- **react**: Multi-turn operation loop (ReAct pattern)
- **flow**: Dependency-aware execution of operation graphs

The module follows lionagi v0's proven patterns for structured output extraction, validation, and tool use orchestration. Each operation is parameterized via a dedicated `*Params` class for type safety and IDE support.

## Functions

### Core Operations

| Function | Description |
|----------|-------------|
| `generate` | Low-level LLM call without message persistence |
| `parse` | Extract structured data from text using Pydantic models |
| `communicate` | Generate + parse in one operation |
| `operate` | Full structured output with validation and tool use |
| `react` | Multi-turn reasoning loop (ReAct pattern) |
| `interpret` | Interpret unmatched content against expected schema |

### Flow Execution

| Function | Description |
|----------|-------------|
| `flow` | Execute operation graph with dependency resolution |
| `flow_stream` | Streaming version of flow execution |

## Parameter Classes

| Class | Used By | Description |
|-------|---------|-------------|
| `GenerateParams` | `generate`, `communicate`, `operate` | LLM generation parameters |
| `ParseParams` | `parse`, `communicate`, `operate` | Structured output parsing parameters |
| `CommunicateParams` | `communicate`, `operate` | Combined generate + parse parameters |
| `ActParams` | `operate` | Tool use parameters |
| `OperateParams` | `operate`, `react` | Full operation parameters |
| `ReactParams` | `react` | Multi-turn loop parameters |
| `InterpretParams` | `interpret` | Unmatched content interpretation parameters |

## Result Classes

| Class | Description |
|-------|-------------|
| `ReactResult` | Final result of react loop |
| `ReactStep` | Single step in react loop |
| `ReactStepResponse` | Response from a react step |

## Builder Classes

| Class | Description |
|-------|-------------|
| `Operation` | Node in operation graph |
| `OperationType` | Enum of operation types |
| `Builder` | Fluent builder for operation graphs |
| `OperationGraphBuilder` | Low-level graph construction |
| `OperationRegistry` | Registry of available operations |
| `DependencyAwareExecutor` | Executes graphs respecting dependencies |

## Quick Start

```python
from lionpride import Session, iModel
from lionpride.types import Operable
from lionpride.operations import (
    generate, communicate, operate,
    GenerateParams, CommunicateParams, OperateParams,
)
from pydantic import BaseModel

# Setup
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)

# Low-level: generate (text output, no capabilities needed)
branch = session.create_branch(resources={model.name})
params = GenerateParams(instruction="Translate to French: Hello", return_as="text")
result = await generate(session, branch, params)  # Returns str

# Mid-level: communicate (structured output)
class Translation(BaseModel):
    text: str
    language: str

branch = session.create_branch(capabilities={"translation"}, resources={model.name})
params = CommunicateParams(
    generate=GenerateParams(instruction="Translate 'Hello' to French"),
    operable=Operable.from_model(Translation),
    capabilities={"translation"},
)
result = await communicate(session, branch, params)  # Returns Translation

# High-level: operate (structured + tools)
class SearchResult(BaseModel):
    query: str
    results: list[str]

branch = session.create_branch(
    capabilities={"searchresult", "action_requests", "action_responses"},
    resources={model.name},
)
params = OperateParams(
    generate=GenerateParams(instruction="Search for Python tutorials", request_model=SearchResult),
    capabilities={"searchresult"},
    actions=True,
)
result = await operate(session, branch, params)
```

## See Also

- [Session](../session/index.md) - Session.conduct() dispatches to operations
- [Services](../services/index.md) - iModel used for LLM calls
- [Rules](../rules/index.md) - Validation in operate
- [Work](../work/index.md) - Higher-level workflow orchestration
- [User Guide: Operations](../../user_guide/operations.md) - Tutorial and patterns
