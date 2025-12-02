# Session

> Central orchestrator for messages, branches, services, and operations

## Overview

`Session` is the **primary entry point** for lionpride's conversation management. It combines message storage (`Flow[Message, Branch]`), branch management with access control, service registration, and operation execution (`conduct()`).

**Use Session for**: Multi-turn conversations, multi-branch exploration, multi-model workflows, custom operations.

**Skip Session when**: Single stateless LLM call (use `iModel.invoke()` directly).

See [Branch](branch.md) and [Message](message.md) for related components.

## Class Signature

```python
from lionpride import Session, Branch, iModel

class Session(Element):
    """Central orchestrator for messages, branches, services, and operations."""

    # Constructor signature
    def __init__(
        self,
        user: str | UUID | None = None,
        conversations: Flow[Message, Branch] | None = None,
        services: ServiceRegistry | None = None,
        *,
        default_branch: Branch | UUID | str | None = None,
        default_generate_model: iModel | str | None = None,
        default_parse_model: iModel | str | None = None,
        default_capabilities: set[str] | None = None,
        default_system: Message | None = None,
        **data,
    ) -> None: ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user` | `str \| UUID` | `None` | User identifier for tracking |
| `conversations` | `Flow[Message, Branch]` | Auto-created | Pre-built Flow (rarely used directly) |
| `services` | `ServiceRegistry` | Auto-created | Pre-built registry (rarely used directly) |
| `default_branch` | `Branch \| UUID \| str` | `None` | Default branch for operations |
| `default_generate_model` | `iModel \| str` | `None` | Default model for generate operations |
| `default_parse_model` | `iModel \| str` | `None` | Default model for parse (often smaller) |
| `default_capabilities` | `set[str]` | `None` | Capabilities for default branch |
| `default_system` | `Message` | `None` | System message for default branch |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Unique identifier (inherited from Element, frozen) |
| `created_at` | `datetime` | UTC timestamp (inherited from Element, frozen) |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (inherited from Element) |
| `user` | `str \| None` | User identifier for this session |
| `conversations` | `Flow[Message, Branch]` | Message storage (items) and branch progressions |
| `services` | `ServiceRegistry` | Registered models and tools |
| `operations` | `OperationRegistry` | Registered operation factories |

### Computed Properties

| Property | Type | Description |
|----------|------|-------------|
| `messages` | `Pile[Message]` | All messages in session (via `conversations.items`) |
| `branches` | `Pile[Branch]` | All branches in session (via `conversations.progressions`) |
| `default_branch` | `Branch \| None` | Default branch for operations |
| `default_generate_model` | `iModel \| None` | Default model for generate operations |
| `default_parse_model` | `iModel \| None` | Default model for parse operations |

## Methods

### Branch Management

#### `create_branch()`

Create a new branch with optional initial configuration.

```python
def create_branch(
    self, *, name: str | None = None, system: Message | UUID | None = None,
    capabilities: set[str] | None = None, resources: set[str] | None = None,
    messages: Iterable[UUID | Message] | None = None,
) -> Branch
```

- `name`: Branch name (auto-generated as `"branch_{n}"` if None)
- `system`: System message at order[0]
- `capabilities`/`resources`: Access control sets

```python
>>> branch = session.create_branch(
...     name="assistant", system=Message(content={"system_message": "You are helpful."}),
...     capabilities={"AnalysisModel"}, resources={"gpt-4o-mini"},
... )
```

#### `get_branch()`

Get branch by UUID, name, or instance. Raises `NotFoundError` if not found (unless `default` provided).

```python
def get_branch(self, branch: UUID | str | Branch, default=Unset, /) -> Branch
```

```python
>>> session.get_branch("analysis")  # by name
>>> session.get_branch(branch.id)   # by UUID
>>> session.get_branch("missing", default=None)  # with default
```

#### `set_default_branch()`

Set the default branch for operations. Branch must exist in session.

```python
def set_default_branch(self, branch: Branch | UUID | str) -> None
```

#### `fork()`

Fork branch for divergent exploration. Use `True` to copy permissions from source.

```python
def fork(
    self, branch: Branch | UUID | str, *, name: str | None = None,
    capabilities: set[str] | Literal[True] | None = None,
    resources: set[str] | Literal[True] | None = None,
    system: UUID | Message | Literal[True] | None = None,
) -> Branch
```

```python
>>> fork = session.fork(main, name="experiment", resources=True)  # copy resources
>>> fork.metadata["forked_from"]["branch_name"]  # lineage tracking
'main'
```

### Message Management

#### `add_message()`

Add message to session and optionally to branches. System messages auto-set at order[0].

```python
def add_message(self, message: Message, branches: list | Branch | UUID | str | None = None) -> None
```

```python
>>> session.add_message(msg, branches=branch)
>>> session.add_message(msg2, branches=[branch1, branch2])  # multiple branches
```

#### `get_branch_system()` / `set_branch_system()`

Get or set the system message for a branch.

```python
def get_branch_system(self, branch: Branch | UUID | str) -> Message | None
def set_branch_system(self, branch: Branch | UUID | str, system: Message | UUID) -> None
```

### Model Management

#### `set_default_model()`

Set default model for generate or parse operations.

```python
def set_default_model(self, model: iModel | str, operation: Literal["generate", "parse"] = "generate") -> None
```

### Operations

#### `conduct()` (async)

Execute an operation on a branch. Thread-safe.

```python
async def conduct(self, operation_type: str, branch: Branch | UUID | str | None = None, params: Any | None = None) -> Operation
```

- `operation_type`: 'operate', 'react', 'communicate', 'generate', 'parse'
- Returns `Operation` with result in `op.response`
- Raises `RuntimeError` if no branch and no default set

```python
>>> params = OperateParams(generate=GenerateParams(instruction="Say hello"))
>>> op = await session.conduct("operate", branch, params=params)
>>> op.response
```

#### `flow()` (async)

Execute operation DAG with dependency scheduling.

```python
async def flow(self, graph: Graph, branch: Branch | UUID | str | None = None, *, max_concurrent: int | None = None, stop_on_error: bool = True) -> dict
```

Returns mapping of operation names to results. No automatic context injection - use `flow_report` for context passing.

#### `request()` (async)

Direct service request (LLM or tool).

```python
async def request(self, service_name: str, *, poll_timeout: float | None = None, **kwargs) -> Calling
```

#### `register_operation()`

Register custom operation factory `async (session, branch, params) -> result`.

```python
def register_operation(self, name: str, factory, *, override: bool = False) -> None
```

## Protocol Implementations

Inherits from Element: **Observable** (`id`), **Serializable** (`to_dict()`, `to_json()`), **Deserializable** (`from_dict()`, `from_json()`), **Hashable** (`__hash__()` by ID).

## Usage Patterns

### Basic Chat Session

```python
from lionpride import Session, Message, iModel
from lionpride.operations.operate import OperateParams, GenerateParams

model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model, default_branch="chat")

# Add system message and conduct operation
system = Message(content={"system_message": "You are a helpful assistant."})
session.add_message(system, branches=session.default_branch)

params = OperateParams(generate=GenerateParams(instruction="What is Python?"))
op = await session.conduct("operate", params=params)
```

### Multi-Model with Branch Resources

```python
from lionpride import Session, iModel

session = Session()
session.services.register(iModel(provider="openai", model="gpt-4o", name="gpt4"))
session.services.register(iModel(provider="anthropic", model="claude-3-5-sonnet", name="claude"))

# Branches with different model access
gpt_branch = session.create_branch(name="gpt_chat", resources={"gpt4"})
claude_branch = session.create_branch(name="claude_chat", resources={"claude"})
```

### Structured Output with Capabilities

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    confidence: float

branch = session.create_branch(name="analysis", capabilities={"Analysis"}, resources={"gpt4"})

params = OperateParams(generate=GenerateParams(
    instruction="Analyze AI impact on healthcare", request_model=Analysis,
))
op = await session.conduct("operate", branch, params=params)
result: Analysis = op.response
```

## Common Pitfalls

### 1. No Default Branch Set

`conduct()` without branch argument raises `RuntimeError` if no default set.

```python
# Fix: Use default_branch in constructor or pass branch explicitly
session = Session(default_generate_model=model, default_branch="main")
```

### 2. Fork Without Copying Resources

Forked branches default to empty permissions.

```python
# WRONG: fork.resources = set() (empty)
fork = session.fork(main, name="experiment")

# FIX: Use resources=True to copy
fork = session.fork(main, name="experiment", resources=True)
```

### 3. Mutating Messages After Adding

Messages are immutable by design. Create new messages instead of mutating.

## Design Rationale

**Flow[Message, Branch]**: O(1) UUID lookup via Pile-backed items. Separates storage (items) from organization (progressions) for efficient multi-branch conversations without duplication.

**Branch Resources/Capabilities**: Branch-level access control for model routing, cost control, and security in multi-tenant scenarios.

**Separate Generate/Parse Models**: Cost optimization (smaller models for parsing), latency reduction, specialization per operation type.

## See Also

- [Branch](branch.md), [Message](message.md), [ServiceRegistry](../services/registry.md), [iModel](../services/imodel.md)
- [Operations](../operations/index.md), [Flow](../base/flow.md), [Element](../base/element.md)

## Examples

### Complete Chat Application

```python
from lionpride import Session, Message, iModel
from lionpride.operations.operate import OperateParams, GenerateParams

model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(user="user_123", default_generate_model=model, default_branch="main")

system = Message(content={"system_message": "You are a Python expert."})
session.add_message(system, branches=session.default_branch)

async def chat(user_input: str) -> str:
    params = OperateParams(generate=GenerateParams(instruction=user_input))
    op = await session.conduct("operate", params=params)
    return op.response
```

### Tool Integration

```python
from lionpride import Session, iModel, Tool

def calculate(expression: str) -> str:
    return str(eval(expression))

tool = Tool(func_callable=calculate)
session = Session(default_generate_model=model, default_branch="main")
session.services.register(tool)
session.default_branch.resources.add(tool.name)

params = OperateParams(
    generate=GenerateParams(instruction="What is 15 * 23 + 7?"),
    actions=True,
)
op = await session.conduct("operate", params=params)
```

### Serialization Roundtrip

```python
session_dict = session.to_dict(mode="json")
restored = Session.from_dict(session_dict)
```
