# Branch

> Named conversation thread with capability-based access control

## Overview

`Branch` extends `Progression` with session binding and access control. It represents a conversation thread, tracking message UUIDs while enforcing permissions via `capabilities` (output schemas) and `resources` (services).

**Use Branch for**: Conversation threads with access control, multi-model workflows, structured output validation, parallel exploration (forking).

**Note**: Operations are invoked via `Session.conduct()`, not Branch methods. Branch = state, Session = operations.

See [Progression](../base/progression.md) and [Session](session.md) for related components.

## Capability-Based Security

| Attribute | Controls | Default | Note |
|-----------|----------|---------|------|
| `capabilities` | Output schema names | `set()` (empty = no access) | Checked during operate/communicate |
| `resources` | Service names (models, tools) | `set()` (empty = no access) | Checked during service resolution |

**Important**: Empty sets deny all access. You must explicitly grant permissions for operations to succeed.

```python
branch = session.create_branch(
    name="analysis",
    capabilities={"AnalysisModel"},  # Only this schema allowed
    resources={"gpt-4o-mini"},        # Only this model allowed
)
```

## Class Signature

```python
from lionpride import Branch

class Branch(Progression):
    """Named progression of messages with access control.

    Extends Progression (ordered UUIDs) with session context and permissions.
    Operations are invoked via Session.conduct(), not Branch methods.
    """

    # Constructor signature
    def __init__(
        self,
        session_id: UUID,                    # Required, frozen
        name: str | None = None,             # Inherited from Progression
        order: list[UUID] | None = None,     # Inherited from Progression
        system_message: UUID | None = None,  # System message UUID
        capabilities: set[str] | None = None,  # Allowed output schemas
        resources: set[str] | None = None,     # Allowed services
        # Inherited from Element (keyword-only):
        **data: Any,  # id, created_at, metadata
    ) -> None: ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `UUID` (required, frozen) | - | Parent session UUID (immutable) |
| `name` | `str` | `None` (auto: `branch_{n}`) | Human-readable name |
| `order` | `list[UUID]` | `[]` | Message UUIDs (system at index 0) |
| `system_message` | `UUID` | `None` | System message UUID at order[0] |
| `capabilities` | `set[str]` | `set()` | Allowed output schemas (empty = unrestricted) |
| `resources` | `set[str]` | `set()` | Allowed services (empty = unrestricted) |
| `id`, `created_at`, `metadata` | - | - | Inherited from Element |

## Attributes

| Attribute | Type | Mutable | Inherited | Description |
|-----------|------|---------|-----------|-------------|
| `session_id` | `UUID` | No | No | Parent session UUID (frozen) |
| `system_message` | `UUID \| None` | Yes | No | System message UUID at order[0] |
| `capabilities` | `set[str]` | Yes | No | Allowed output schema names |
| `resources` | `set[str]` | Yes | No | Allowed backend service names |
| `name` | `str \| None` | Yes | Yes | Branch name for identification |
| `order` | `list[UUID]` | Yes | Yes | Ordered message UUIDs |
| `id` | `UUID` | No | Yes | Unique identifier (frozen) |
| `created_at` | `datetime` | No | Yes | Creation timestamp (frozen) |
| `metadata` | `dict[str, Any]` | Yes | Yes | Additional metadata |

## Methods

### `set_system_message()`

Set or replace the system message at `order[0]`.

```python
def set_system_message(self, message_id: UUID | Message) -> None
```

**Note**: Message must exist in parent Session first.

### Inherited from Progression

**List ops**: `append()`, `extend()`, `insert()`, `remove()`, `pop()`, `popleft()`, `clear()`

**Workflow ops**: `move()`, `swap()`, `reverse()`

**Idempotent ops**: `include()`, `exclude()`

**Query ops**: `len()`, `bool()`, `in`, `[]`, slicing, iteration

### Inherited from Element

`to_dict()`, `from_dict()`, `to_json()`, `from_json()`

## Protocol Implementations

Inherits: **Observable**, **Serializable**, **Deserializable**, **Hashable**, **Containable**

## Design Rationale

**Capability-Based Security**: Fine-grained permission control at conversation level. Empty sets default to unrestricted for development convenience.

**Frozen session_id**: Prevents accidental cross-session corruption. Branch messages are stored in Session's Flow.

**System Message at order[0]**: LLM APIs expect system first. `set_system_message()` maintains this invariant.

**Operations via Session.conduct()**: Branch = state, Session = operations. Access control validated at Session level.

## Usage Patterns

### Basic Usage with Permissions

```python
from lionpride import Session, iModel

session = Session(default_generate_model=iModel(provider="openai", model="gpt-4o-mini"))
branch = session.create_branch(
    name="analysis",
    capabilities={"AnalysisResult"},  # Restricted output schema
    resources={"gpt-4o-mini"},         # Restricted model access
)
```

### System Message Setup

```python
from lionpride import Message
from lionpride.session.messages import SystemContent

system = Message(content=SystemContent(system_info="You are a helpful assistant."))
session.add_message(system, branches=[branch])
# System message is now at branch[0]
```

### Forking for Exploration

```python
fork = session.fork(main, name="experiment", capabilities=True, resources=True)
print(fork.metadata["forked_from"]["branch_name"])  # lineage tracking
```

## Common Pitfalls

### 1. Empty Permissions = Unrestricted

`capabilities={}` and `resources={}` means **all allowed**, not "no access".

```python
# Production: explicitly set permissions
branch = session.create_branch(capabilities={"SafeOutput"}, resources={"approved_model"})
```

### 2. Branch Stores UUIDs, Not Messages

```python
# Access messages through Session
messages = [session.messages[uid] for uid in branch]
```

### 3. Always Create via Session

Never instantiate Branch directly. Use `session.create_branch()`.

## See Also

- [Progression](../base/progression.md), [Session](session.md), [Message](message.md), [Element](../base/element.md)

## Examples

### Capability-Restricted Analysis

```python
from pydantic import BaseModel
from lionpride.operations.operate import OperateParams, GenerateParams

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float

branch = session.create_branch(
    name="sentiment", capabilities={"SentimentResult"}, resources={"gpt-4o-mini"}
)

params = OperateParams(generate=GenerateParams(
    instruction="Analyze: 'I love this product!'", request_model=SentimentResult,
))
result = await session.conduct("operate", branch, params=params)
```

### Multi-Model Cost Boundaries

```python
# Dev: unrestricted, Prod: cost-controlled
dev_branch = session.create_branch(name="dev")
prod_branch = session.create_branch(name="production", resources={"cheap"})
```
