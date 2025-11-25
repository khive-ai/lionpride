# Session

> Top-level conversation manager with multi-branch support and service integration

## Overview

`Session` is the central orchestrator for conversational workflows, managing messages, branches, services, and operations within a unified context. It extends [`Element`](../base/element.md) and uses [`Flow[Message, Branch]`](../base/flow.md) to enable efficient message storage with multi-branch conversation tracking.

**Key Capabilities:**

- **Message Storage**: Centralized Pile[Message] with O(1) UUID lookup
- **Branch Management**: Multiple conversation threads (Pile[Branch])
- **Service Registry**: Unified access to models, tools, and resources
- **Operation Registry**: Built-in operations (generate, communicate, operate, react)
- **Access Control**: Branch-level validation for resources and capabilities
- **Message Sharing**: Messages stored once, referenced by multiple branches
- **Async Execution**: Service requests and operation execution via IPU pattern

**When to Use Session:**

- Managing complex multi-turn conversations with branching logic
- Coordinating multiple LLM services within a single context
- Implementing access control for different conversation modes
- Building conversational agents with structured outputs and tool use
- A/B testing prompts with forked branches

## Class Signature

```python
from lionpride.session import Session
from lionpride.core import Element

class Session(Element):
    """Top-level conversation manager with service and operation registries."""

    # Constructor signature
    def __init__(
        self,
        *,
        user: str | UUID = "user",
        conversations: Flow[Message, Branch] | None = None,
        services: ServiceRegistry | None = None,
        operations: OperationRegistry | None = None,
        # Element fields
        id: UUID | str | None = None,
        created_at: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**user** : str or UUID, default "user"

Default entity identifier for branches created without explicit user.

**conversations** : Flow[Message, Branch], optional

Flow instance managing messages and branches. If None, auto-created as `Flow(item_type=Message, progressions=Pile(item_type=Branch))`.

**services** : ServiceRegistry, optional

Service registry for models and tools. If None, auto-created as empty `ServiceRegistry()`.

**operations** : OperationRegistry, optional

Operation registry. If None, auto-created with built-in operations (generate, communicate, operate, react).

**Element parameters** : See [Element](../base/element.md) for `id`, `created_at`, `metadata`.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Session identifier (inherited from Element, frozen) |
| `created_at` | `datetime` | Creation timestamp (inherited from Element, frozen) |
| `metadata` | `dict[str, Any]` | Session metadata (inherited from Element, mutable) |
| `user` | `str \| UUID` | Default user identifier for branches |
| `conversations` | `Flow[Message, Branch]` | Message storage and branch management |
| `services` | `ServiceRegistry` | Available services (models, tools) |
| `operations` | `OperationRegistry` | Registered operations |
| `messages` | `Pile[Message]` | Read-only view of conversations.items |
| `branches` | `Pile[Branch]` | Read-only view of conversations.progressions |

## Methods

### Branch Management

#### `create_branch()`

Create new conversation branch with optional system message and access control.

**Signature:**

```python
def create_branch(
    self,
    *,
    name: str | None = None,
    system: Message | UUID | None = None,
    capabilities: set[str] | None = None,
    resources: set[str] | None = None,
    messages: Iterable[UUID | Message] | None = None,
) -> Branch: ...
```

**Parameters:**

- `name` (str, optional): Branch name. If None, auto-generates `"branch_N"` where N is current branch count.
- `system` (Message or UUID, optional): System message for this branch. If Message instance not in session, automatically added to session.messages.
- `capabilities` (set[str], optional): Allowed structured output schema names. Empty set allows no structured outputs.
- `resources` (set[str], optional): Allowed service names (from `session.services`). Empty set allows no service access.
- `messages` (iterable of UUID or Message, optional): Initial messages to include in branch order.

**Returns:**

- Branch: Created branch (automatically added to `session.branches`)

**Raises:**

- ValueError: If `system` is UUID not found in session.messages, or if `system` is not a Message instance

**Examples:**

```python
>>> from lionpride.session import Session, Message, SystemContent
>>> session = Session()

# Basic branch
>>> branch = session.create_branch(name="main")
>>> branch.name
'main'

# Branch with system message and access control
>>> system_msg = Message(content=SystemContent(system_message="You are helpful"))
>>> branch = session.create_branch(
...     name="production",
...     system=system_msg,
...     capabilities={"Analysis", "Report"},
...     resources={"gpt4", "search_tool"},
... )
>>> branch.capabilities
{'Analysis', 'Report'}
>>> branch.resources
{'gpt4', 'search_tool'}
>>> session.messages[branch.system].content.system_message
'You are helpful'
```

**See Also:**

- `fork()`: Create branch by forking existing branch
- `get_branch()`: Retrieve existing branch

#### `get_branch()`

Retrieve branch by UUID, name, or instance.

**Signature:**

```python
def get_branch(self, branch: UUID | str | Branch) -> Branch: ...
```

**Parameters:**

- `branch` (UUID or str or Branch): Branch identifier (UUID), name, or instance

**Returns:**

- Branch: Retrieved branch

**Raises:**

- NotFoundError: If branch not found in session.branches

**Examples:**

```python
>>> from lionpride.session import Session
>>> session = Session()
>>> branch = session.create_branch(name="main")

# Retrieve by name
>>> retrieved = session.get_branch("main")
>>> retrieved.id == branch.id
True

# Retrieve by UUID
>>> retrieved = session.get_branch(branch.id)
>>> retrieved.id == branch.id
True

# Pass through if already Branch instance
>>> retrieved = session.get_branch(branch)
>>> retrieved is branch
True
```

**See Also:**

- `create_branch()`: Create new branch
- `get_branch_system()`: Get system message for branch

#### `get_branch_system()`

Get system message for a branch.

**Signature:**

```python
def get_branch_system(self, branch: Branch | UUID | str) -> Message | None: ...
```

**Parameters:**

- `branch` (Branch or UUID or str): Branch instance, UUID, or name

**Returns:**

- Message or None: System message if set, None otherwise

**Examples:**

```python
>>> from lionpride.session import Session, Message, SystemContent
>>> session = Session()
>>> system_msg = Message(content=SystemContent(system_message="Be helpful"))
>>> branch = session.create_branch(name="main", system=system_msg)

>>> retrieved_msg = session.get_branch_system(branch)
>>> retrieved_msg.content.system_message
'Be helpful'

# Returns None if no system message
>>> branch2 = session.create_branch(name="no_system")
>>> session.get_branch_system(branch2) is None
True
```

**See Also:**

- `create_branch()`: Create branch with system message
- `Branch.set_system_message()`: Update branch system message

#### `fork()`

Create new branch by forking existing branch with shared message history.

**Signature:**

```python
def fork(
    self,
    branch: Branch | UUID | str,
    *,
    name: str | None = None,
    capabilities: set[str] | Literal[True] | None = None,
    resources: set[str] | Literal[True] | None = None,
    system: UUID | Message | Literal[True] | None = None,
) -> Branch: ...
```

**Parameters:**

- `branch` (Branch or UUID or str): Source branch to fork
- `name` (str, optional): New branch name. If None, auto-generates `"{source_name}_fork"`.
- `capabilities` (set[str] or True, optional): Capabilities for forked branch. If True, copies from source branch. If None, empty set.
- `resources` (set[str] or True, optional): Resources for forked branch. If True, copies from source branch. If None, empty set.
- `system` (UUID or Message or True, optional): System message for forked branch. If True, copies from source branch.

**Returns:**

- Branch: Forked branch with copied message order and optional capability/resource inheritance

**Examples:**

```python
>>> from lionpride.session import Session, Message, InstructionContent
>>> session = Session()
>>> main = session.create_branch(
...     name="main",
...     capabilities={"Analysis"},
...     resources={"gpt4"},
... )

# Add messages to main
>>> msg1 = Message(content=InstructionContent(instruction="Hello"))
>>> msg2 = Message(content=InstructionContent(instruction="Analyze data"))
>>> session.add_message(msg1, branches=main)
>>> session.add_message(msg2, branches=main)

# Fork with inheritance
>>> experimental = session.fork(
...     main,
...     name="experimental",
...     capabilities=True,  # Copy capabilities
...     resources=True,     # Copy resources
...     system=True,        # Copy system message
... )

# Forked branch has same message history
>>> len(experimental) == len(main)
True

# But is independent (adding to experimental doesn't affect main)
>>> new_msg = Message(content=InstructionContent(instruction="Try new approach"))
>>> session.add_message(new_msg, branches=experimental)
>>> len(experimental) > len(main)
True
```

**Notes:**

The forked branch's metadata includes tracking info:

```python
>>> experimental.metadata["forked_from"]
{
    "branch_id": "...",
    "branch_name": "main",
    "created_at": "2025-11-24T...",
    "message_count": 2
}
```

**See Also:**

- `create_branch()`: Create branch from scratch
- `add_message()`: Add messages to branches

### Message Management

#### `add_message()`

Add message to session storage and optionally to one or more branches.

**Signature:**

```python
def add_message(
    self,
    message: Message,
    branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
) -> None: ...
```

**Parameters:**

- `message` (Message): Message to add
- `branches` (list of Branch/UUID/str, or Branch/UUID/str, optional): Branches to add message to. If None, message added to storage only (not in any branch).

**Returns:**

- None (mutates session.messages and specified branches)

**Special Behavior for SystemContent:**

When adding a message with `SystemContent`, the message is added to storage and set as the system message for specified branches (via `branch.set_system_message()`), which places it at position 0.

**Examples:**

```python
>>> from lionpride.session import Session, Message, InstructionContent, SystemContent
>>> session = Session()
>>> branch1 = session.create_branch(name="branch1")
>>> branch2 = session.create_branch(name="branch2")

# Add to single branch
>>> msg1 = Message(content=InstructionContent(instruction="Hello"))
>>> session.add_message(msg1, branches=branch1)
>>> msg1.id in branch1
True

# Add to multiple branches (message stored once)
>>> msg2 = Message(content=InstructionContent(instruction="Shared message"))
>>> session.add_message(msg2, branches=[branch1, branch2])
>>> msg2.id in branch1 and msg2.id in branch2
True
>>> len(session.messages)  # Message stored once
2

# Add to storage only (no branches)
>>> msg3 = Message(content=InstructionContent(instruction="Orphaned"))
>>> session.add_message(msg3)
>>> msg3.id in session.messages
True
>>> msg3.id in branch1 or msg3.id in branch2
False

# SystemContent special handling
>>> system_msg = Message(content=SystemContent(system_message="Be helpful"))
>>> session.add_message(system_msg, branches=branch1)
>>> branch1.system == system_msg.id
True
>>> branch1[0] == system_msg.id  # Placed at position 0
True
```

**See Also:**

- `create_branch()`: Create branches to add messages to
- `messages`: Property for accessing stored messages

### Service and Operation Execution

#### `request()`

Execute service request directly (low-level API, bypasses access control).

**Signature:**

```python
async def request(
    self,
    service_name: str,
    *,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
    **kwargs,
) -> Calling: ...
```

**Parameters:**

- `service_name` (str): Name of registered service (from `session.services`)
- `poll_timeout` (float, optional): Timeout for polling service response
- `poll_interval` (float, optional): Interval between poll attempts
- `**kwargs`: Parameters forwarded to service's `invoke()` method

**Returns:**

- Calling: Service response object

**Examples:**

```python
>>> from lionpride.session import Session
>>> session = Session()
>>> session.services.register(my_llm_model, name="gpt4")

# Direct service request
>>> response = await session.request("gpt4", messages=[...])
>>> response.content
'Generated response from GPT-4'
```

**Notes:**

This is a low-level API that bypasses branch access control. For access-controlled execution, use `conduct()`.

**See Also:**

- `conduct()`: High-level API with access control
- `services`: ServiceRegistry property

#### `conduct()`

Create and queue operation for execution with branch access control validation.

**Signature:**

```python
async def conduct(
    self,
    branch: Branch | UUID | str,
    operation: str,
    ipu: Any,  # IPU instance
    params: Any = None,  # Typed Param instance
    **kwargs,
) -> Operation: ...
```

**Parameters:**

- `branch` (Branch or UUID or str): Branch to execute in (used for access control)
- `operation` (str): Operation type name (must be registered in `session.operations`)
- `ipu` (IPU): IPU instance for operation execution
- `params` (Param, optional): Typed parameter instance (GenerateParam, CommunicateParam, etc.). If None, `**kwargs` converted to appropriate Param type.
- `**kwargs`: Operation parameters (converted to Param if `params=None`)

**Returns:**

- Operation: Queued operation (call `await operation.invoke()` to get result)

**Raises:**

- NotFoundError: If branch or operation not found
- PermissionError: If branch lacks required resources or capabilities
- ValueError: If neither params nor kwargs provided

**Access Control Flow:**

1. Resolve branch by UUID/name/instance
2. Check operation registered in `session.operations`
3. Validate `operation.required_resources ⊆ branch.resources`
4. Validate `operation.required_capabilities ⊆ branch.capabilities`
5. Convert `kwargs` to typed Param if needed
6. Create Operation with Param
7. Queue to IPU for execution
8. Return Operation instance

**Examples:**

```python
>>> from lionpride.session import Session
>>> from lionpride.ipu import IPU
>>> session = Session()
>>> session.services.register(gpt4_model, name="gpt4")

>>> branch = session.create_branch(
...     name="production",
...     resources={"gpt4"},
...     capabilities={"Analysis"},
... )

>>> ipu = IPU()

# Flexible kwargs interface
>>> operation = await session.conduct(
...     branch=branch,
...     operation="generate",
...     ipu=ipu,
...     imodel="gpt4",
...     messages=[...],
... )
>>> result = await operation.invoke()

# Direct Param interface (for workflows)
>>> from lionpride.operations import GenerateParam
>>> param = GenerateParam(imodel="gpt4", messages=[...])
>>> operation = await session.conduct(
...     branch=branch,
...     operation="generate",
...     ipu=ipu,
...     params=param,
... )

# Access control validation
>>> try:
...     # Fails: "claude" not in branch.resources
...     operation = await session.conduct(
...         branch=branch,
...         operation="generate",
...         ipu=ipu,
...         imodel="claude",
...         messages=[...],
...     )
... except PermissionError as e:
...     print(f"Access denied: {e}")
```

**Notes:**

Built-in operations registered automatically:

- `"generate"`: Basic LLM generation
- `"communicate"`: Multi-turn conversation
- `"operate"`: Tool-augmented operations
- `"react"`: ReAct agent pattern

**See Also:**

- `request()`: Low-level service request (no access control)
- `operations`: OperationRegistry property

## Properties

#### `messages`

Read-only view of message storage (`conversations.items`).

**Signature:**

```python
@property
def messages(self) -> Pile[Message]: ...
```

**Returns:**

- Pile[Message]: All messages in session (O(1) UUID lookup)

**Examples:**

```python
>>> from lionpride.session import Session, Message, InstructionContent
>>> session = Session()
>>> msg = Message(content=InstructionContent(instruction="Hello"))
>>> session.add_message(msg)

>>> session.messages[msg.id]
Message(id=...)
>>> len(session.messages)
1
```

**Notes:**

This is a read-only reference to `session.conversations.items`. Use `add_message()` to add messages.

**See Also:**

- `add_message()`: Add messages to session
- `branches`: Property for accessing branches

#### `branches`

Read-only view of branch storage (`conversations.progressions`).

**Signature:**

```python
@property
def branches(self) -> Pile[Branch]: ...
```

**Returns:**

- Pile[Branch]: All branches in session (O(1) UUID lookup)

**Examples:**

```python
>>> from lionpride.session import Session
>>> session = Session()
>>> branch = session.create_branch(name="main")

>>> session.branches[branch.id]
Branch(messages=0, session=..., name='main')
>>> len(session.branches)
1
```

**Notes:**

This is a read-only reference to `session.conversations.progressions`. Use `create_branch()` or `fork()` to add branches.

**See Also:**

- `create_branch()`: Create new branch
- `messages`: Property for accessing messages

## Protocol Implementations

Session inherits protocol implementations from Element:

- **Observable**: UUID identifier via `id` property
- **Serializable**: `to_dict(mode='python'|'json'|'db')`, `to_json()`
- **Deserializable**: `from_dict()`, `from_json()` with polymorphic reconstruction
- **Hashable**: ID-based hashing via `__hash__()`

See [Element Protocol Implementations](../base/element.md#protocol-implementations) for details.

## Usage Patterns

### Basic Session with Single Branch

```python
from lionpride.session import Session, Message, InstructionContent

# Create session
session = Session()

# Create branch
branch = session.create_branch(name="main")

# Add messages
msg1 = Message(content=InstructionContent(instruction="Hello"))
msg2 = Message(content=InstructionContent(instruction="How are you?"))

session.add_message(msg1, branches=branch)
session.add_message(msg2, branches=branch)

# Access messages
for msg_id in branch:
    msg = session.messages[msg_id]
    print(f"{msg.role.value}: {msg.rendered}")
```

### Multi-Branch A/B Testing

```python
from lionpride.session import Session, Message, InstructionContent, SystemContent

session = Session()

# Shared system message
system_msg = Message(content=SystemContent(system_message="You are a helpful assistant"))

# Variant A: Friendly tone
variant_a = session.create_branch(name="friendly", system=system_msg)
a_msg = Message(content=InstructionContent(
    instruction="Explain quantum computing in a friendly, approachable way"
))
session.add_message(a_msg, branches=variant_a)

# Variant B: Technical tone
variant_b = session.create_branch(name="technical", system=system_msg)
b_msg = Message(content=InstructionContent(
    instruction="Explain quantum computing with technical precision"
))
session.add_message(b_msg, branches=variant_b)

# Execute both variants and compare results
```

### Service Registry and Access Control

```python
from lionpride.session import Session

session = Session()

# Register services
session.services.register(gpt4_model, name="gpt4")
session.services.register(claude_model, name="claude")
session.services.register(search_tool, name="search")

# Production branch: GPT-4 only
prod = session.create_branch(
    name="production",
    resources={"gpt4"},
    capabilities={"ApprovedSchema"},
)

# Experimental branch: All services
exp = session.create_branch(
    name="experimental",
    resources={"gpt4", "claude", "search"},
    capabilities={"ApprovedSchema", "ExperimentalSchema"},
)

# Access control enforced at execution
from lionpride.ipu import IPU
ipu = IPU()

# This works
op1 = await session.conduct(prod, "generate", ipu, imodel="gpt4", messages=[...])

# This fails (claude not in prod.resources)
try:
    op2 = await session.conduct(prod, "generate", ipu, imodel="claude", messages=[...])
except PermissionError:
    print("Access denied")
```

### Forking for Rollback Scenarios

```python
from lionpride.session import Session, Message, InstructionContent

session = Session()
main = session.create_branch(name="main")

# Build conversation
for i in range(5):
    msg = Message(content=InstructionContent(instruction=f"Step {i+1}"))
    session.add_message(msg, branches=main)

# Checkpoint: fork before risky operation
checkpoint = session.fork(main, name="checkpoint", system=True, resources=True)

# Try risky operation in main
risky_msg = Message(content=InstructionContent(instruction="Risky operation"))
session.add_message(risky_msg, branches=main)

# If operation fails, rollback to checkpoint
if operation_failed:
    # Discard main, continue from checkpoint
    main = checkpoint
```

## Common Pitfalls

### Pitfall 1: Forgetting Access Control Validation

**Issue**: Using `request()` bypasses access control.

```python
# Bypasses access control (low-level)
response = await session.request("restricted_service")

# Validates access control (high-level)
operation = await session.conduct(
    branch=branch,
    operation="generate",
    ipu=ipu,
    imodel="restricted_service"  # Checked against branch.resources
)
```

**Solution**: Use `conduct()` for access-controlled execution. Use `request()` only when bypass is intentional.

### Pitfall 2: Adding System Messages Incorrectly

**Issue**: Adding SystemContent to middle of branch.

```python
# DON'T: System message added to middle
session.add_message(user_msg, branches=branch)
session.add_message(system_msg, branches=branch)  # Should be first

# DO: Set system message during branch creation
branch = session.create_branch(name="main", system=system_msg)
```

**Solution**: Use `create_branch(system=...)` or `Branch.set_system_message()` to ensure system message is at position 0.

### Pitfall 3: Mutating Messages After Adding

**Issue**: Expecting message mutations to affect session.

```python
msg = Message(content=InstructionContent(instruction="Original"))
session.add_message(msg, branches=branch)

# DON'T: Can't mutate MessageContent (frozen dataclass)
# msg.content.instruction = "Modified"  # Error

# DO: Create new message
new_msg = Message(content=InstructionContent(instruction="Modified"))
session.add_message(new_msg, branches=branch)
```

**Solution**: MessageContent is immutable. Create new messages instead of mutating.

## Design Rationale

### Why Flow[Message, Branch]?

Flow provides:

1. **Efficient Storage**: Messages stored once in Pile[Message], referenced by UUID
2. **O(1) Lookup**: Pile provides constant-time access by UUID
3. **Referential Integrity**: Progressions (Branches) reference existing messages
4. **Memory Efficiency**: Shared messages don't duplicate storage

### Why Separate messages and branches Properties?

Direct access to `session.messages` and `session.branches` is more ergonomic than `session.conversations.items` and `session.conversations.progressions`:

```python
# Clear and concise
msg = session.messages[msg_id]
branch = session.branches[branch_id]

# Verbose
msg = session.conversations.items[msg_id]
branch = session.conversations.progressions[branch_id]
```

### Why Auto-Register Built-In Operations?

The four built-in operations (generate, communicate, operate, react) cover common workflow patterns. Auto-registration eliminates boilerplate while allowing custom operations via `session.operations.register()`.

### Why Branch-Level Access Control?

Different conversation contexts have different security requirements:

- **Production**: Only approved models and schemas
- **Development**: Experimental models and schemas
- **Testing**: Mock services for testing

Branch-level access control enforces these boundaries at execution time.

## See Also

- **Related Classes**:
  - [Branch](branch.md): Conversation branch (extends Progression)
  - [Message](message.md): Message container (extends Node)
  - [Flow](../base/flow.md): Composition pattern for items + progressions
  - [Element](../base/element.md): Base class with identity

- **Module Overview**:
  - [session Overview](overview.md): Module-level documentation

## Examples

See [overview.md Examples](overview.md#examples) for comprehensive usage patterns including:

- Basic conversations
- A/B testing with forks
- Structured output workflows
- Tool calling workflows
- Multi-service coordination
