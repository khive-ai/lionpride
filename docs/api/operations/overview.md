# operations Module

> High-level LLM operations with validation, tool calling, and graph execution

## Overview

The `operations` module provides the primary interface for LLM interactions in lionpride, offering a progression from simple chat to complex multi-step workflows. Built on top of the `iModel` service abstraction, operations handle conversation state, structured outputs, validation strategies, and dependency-aware execution.

**Key Capabilities:**

- **Stateful Chat**: Conversation management with automatic message persistence
- **Structured Outputs**: JSON and LNDL validation with retry logic
- **Tool Calling**: Function execution with action request/response models
- **ReAct Pattern**: Multi-step reasoning with tool use
- **Graph Execution**: DAG-based operation workflows with dependency management
- **Two-Tier Validation**: Strict then fuzzy parsing for robust LLM output handling

**When to Use This Module:**

- Building conversational agents with memory
- Extracting structured data from LLM responses
- Creating tool-using agents (function calling)
- Orchestrating multi-step LLM workflows
- Implementing ReAct-style reasoning loops

## Operation Hierarchy

The module follows a progressive complexity model:

```text
generate()      → Stateless generation (no conversation state)
    ↓
communicate()   → Stateful chat with optional validation
    ↓
operate()       → Structured outputs with action support
    ↓
react()         → Multi-step reasoning with tools
```

Each level builds on the previous:

- `generate()` is the foundation (stateless text generation)
- `communicate()` adds conversation state and validation
- `operate()` adds structured outputs and action requests
- `react()` adds multi-step reasoning loops

## Module Exports

```python
from lionpride.operations import (
    # Core operations
    generate,
    communicate,
    operate,
    react,

    # Operative (validator)
    Operative,
    create_operative_from_model,
    create_action_operative,

    # Graph building
    OperationGraphBuilder,
    Builder,  # Alias

    # Graph execution
    flow,
    flow_stream,
    DependencyAwareExecutor,
    OperationResult,

    # Models
    ActionRequestModel,
    ActionResponseModel,
    ReactResult,
    ReactStep,
    Reason,

    # Parameter types
    GenerateParam,
    CommunicateParam,
    OperateParam,
    ReactParam,

    # Core types
    Operation,
    OperationType,
    OperationRegistry,
)
```

## Quick Reference

| Component | Category | Purpose |
|-----------|----------|---------|
| [`generate`](generate.md) | Operation | Stateless text generation (no state) |
| [`communicate`](communicate.md) | Operation | Stateful chat with validation |
| [`operate`](operate.md) | Operation | Structured outputs + actions |
| [`react`](react.md) | Operation | ReAct reasoning loop |
| [`Operative`](operate.md#operative-class) | Validation | Two-tier validation strategy |
| [`OperationGraphBuilder`](builder.md) | Graph | Build operation DAGs |
| [`flow`](flow.md) | Execution | Execute operation graphs |
| [`DependencyAwareExecutor`](flow.md#dependencyawareexecutor) | Execution | Graph executor with IPU |

## Core Operations

### `generate()`

Stateless text generation - does not persist messages to conversation.

```python
async def generate(
    session: Session,
    branch: Branch | str,
    parameters: GenerateParam | dict,
) -> str | dict | Message | Any: ...
```

**Use when**: You need one-off generation without conversation context.

**See**: [generate.md](generate.md) for detailed API reference

### `communicate()`

Stateful chat with optional structured output and retry logic.

```python
async def communicate(
    session: Session,
    branch: Branch | str,
    parameters: CommunicateParam | dict,
) -> str | dict | Message | BaseModel: ...
```

**Use when**: You need conversational context with optional validation.

**See**: [communicate.md](communicate.md) for detailed API reference

### `operate()`

Structured outputs with optional action requests and reasoning.

```python
async def operate(
    session: Session,
    branch: Branch | str,
    parameters: OperateParam | dict,
) -> Any: ...
```

**Use when**: You need validated structured outputs or tool calling.

**See**: [operate.md](operate.md) for detailed API reference

### `react()`

Multi-step ReAct (Reasoning + Acting) loop with tool execution.

```python
async def react(
    session: Session,
    branch: Branch | str,
    parameters: ReactParam | dict,
) -> ReactResult: ...
```

**Use when**: You need multi-step reasoning with tool use.

**See**: [react.md](react.md) for detailed API reference

## Validation Strategy

### Two-Tier Validation

The `Operative` class implements a two-tier validation strategy:

1. **Strict Validation**: Attempts exact schema matching first
2. **Fuzzy Fallback**: Falls back to fuzzy parsing if strict fails

This provides robust handling of LLM outputs that may not perfectly match schemas.

### Validation Modes

**LNDL Mode** (via `operable`):

- Uses LNDL (Lion Natural Data Language) fuzzy parsing
- More forgiving of LLM formatting variations
- Suitable for complex nested structures

**JSON Mode** (via `response_model`):

- Uses Pydantic strict validation
- Requires exact JSON schema matching
- Better for simple, well-defined structures

**Example**:

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# JSON mode (strict)
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract user info: John is 30 years old",
        imodel="gpt-4o",
        response_model=UserInfo,
        return_as="model",
    )
)

# LNDL mode (fuzzy)
from lionpride.operations import create_operative_from_model

operative = create_operative_from_model(UserInfo)
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract user info: John is 30 years old",
        imodel="gpt-4o",
        operable=operative.operable,
        return_as="model",
    )
)
```

## Graph Execution

### Building Graphs

Use `OperationGraphBuilder` for complex workflows:

```python
from lionpride.operations import Builder

builder = Builder()
builder.add("extract", "communicate", instruction="Extract data", imodel="gpt-4o")
builder.add("summarize", "communicate",
    instruction="Summarize results",
    imodel="gpt-4o",
    depends_on=["extract"]
)

graph = builder.build()
```

**See**: [builder.md](builder.md) for detailed API reference

### Executing Graphs

Use `flow()` for dependency-aware execution:

```python
from lionpride.operations import flow

results = await flow(
    session,
    "main",
    graph,
    ipu,
    max_concurrent=3,
    verbose=True,
)
```

**See**: [flow.md](flow.md) for detailed API reference

## Action Models

### ActionRequestModel

Represents a tool call request from the LLM:

```python
class ActionRequestModel(BaseModel):
    function: str  # Tool name
    arguments: dict[str, Any]  # Tool arguments
```

### ActionResponseModel

Represents the result of tool execution:

```python
class ActionResponseModel(BaseModel):
    function: str  # Tool name
    output: Any  # Tool result
    error: str | None  # Error message if failed
```

### Example

```python
# LLM requests an action
action_request = ActionRequestModel(
    function="search_web",
    arguments={"query": "Python typing"}
)

# Tool executes and returns response
action_response = ActionResponseModel(
    function="search_web",
    output={"results": [...]},
    error=None
)
```

## Usage Patterns

### Basic Chat

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="What is the capital of France?",
        imodel="gpt-4o",
    )
)
print(result)  # "The capital of France is Paris."
```

### Structured Extraction

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract info: Paris is in France with 2.2M people",
        imodel="gpt-4o",
        response_model=CityInfo,
        return_as="model",
    )
)
# result = CityInfo(name="Paris", country="France", population=2200000)
```

### Tool Calling

```python
result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Search for Python typing best practices",
        imodel="gpt-4o",
        response_model=SearchResult,
        actions=True,
        tools=True,  # Enable all registered tools
    )
)
# Actions are automatically executed and results added to response
```

### Multi-Step Reasoning

```python
from lionpride.services.types import Tool

class WebSearchTool(Tool):
    # Tool implementation
    ...

result = await react(
    session,
    "main",
    ReactParam(
        instruction="Find the current weather in Tokyo",
        imodel="gpt-4o",
        tools=[WebSearchTool],
        model_name="gpt-4o",
        max_steps=5,
        verbose=True,
    )
)

print(result.total_steps)  # Number of steps taken
print(result.final_response)  # Final answer
```

## Common Pitfalls

- **Missing imodel**: All operations require `imodel` parameter (model name or instance)
  - **Solution**: Always provide `imodel="gpt-4o"` or similar

- **Validation mode confusion**: Using both `response_model` and `operable`
  - **Solution**: Use one validation mode - JSON (`response_model`) OR LNDL (`operable`)

- **State vs stateless**: Using `generate()` when you need conversation history
  - **Solution**: Use `communicate()` for stateful chat, `generate()` for one-offs

- **Tool registration**: Tools not registered in session before `react()`
  - **Solution**: `react()` auto-registers tools, but ensure they're valid Tool instances

- **Graph cycles**: Creating circular dependencies in operation graphs
  - **Solution**: `OperationGraphBuilder.build()` validates DAG, raises on cycles

## Design Rationale

### Why Four Operation Levels?

The progressive hierarchy mirrors real-world LLM application patterns:

1. **generate()**: Simple templates, one-off generations
2. **communicate()**: Chatbots, conversational interfaces
3. **operate()**: Structured data extraction, API integration
4. **react()**: Complex agents, research assistants

Each level adds complexity only when needed, keeping simple use cases simple.

### Why Two-Tier Validation?

LLMs are probabilistic and rarely produce perfect JSON on first attempt. Strict validation catches well-formed outputs quickly, while fuzzy fallback handles edge cases. This maximizes both performance (strict is fast) and robustness (fuzzy recovers errors).

### Why Separate Builder and Executor?

Separating graph building (`OperationGraphBuilder`) from execution (`DependencyAwareExecutor`) allows:

- **Reusability**: Same graph with different sessions/branches
- **Serialization**: Store graph definitions
- **Testing**: Build graphs without execution
- **Optimization**: Analyze graph before execution

## See Also

- **Related Modules**:
  - [session](../session/overview.md): Session and conversation management
  - [services](../services/overview.md): iModel and Tool abstractions
  - [ipu](../ipu/overview.md): Input Processing Unit for validated execution
  - [types](../types/overview.md): Operable and Spec types

- **User Guide**:
  - Quickstart: Building your first agent (documentation pending)
  - Validation strategies: JSON vs LNDL (documentation pending)
  - Tool calling patterns (documentation pending)

## Examples

See individual operation pages for comprehensive examples:

- [generate.md](generate.md): Stateless generation examples
- [communicate.md](communicate.md): Chat and validation examples
- [operate.md](operate.md): Structured output and action examples
- [react.md](react.md): Multi-step reasoning examples
- [builder.md](builder.md): Graph building patterns
- [flow.md](flow.md): Graph execution patterns
