# Branch

> Conversation thread with access control and system message support (extends Progression)

## Overview

`Branch` represents an ordered conversation thread within a [`Session`](session.md), tracking message UUIDs in sequence while enforcing resource and capability access control. It extends [`Progression`](../base/progression.md) with session-specific features like system messages, allowed services, and structured output schemas.

**Key Capabilities:**

- **Message Ordering**: Maintains ordered list of message UUIDs (via Progression)
- **System Message**: Optional system message at position 0
- **Access Control**: Restricts which services (resources) and schemas (capabilities) can be used
- **Session Binding**: References parent session for message resolution
- **Metadata Tracking**: Inherits Progression's metadata support

**When to Use Branch:**

- Organizing messages into conversation threads within a Session
- Implementing access control for different conversation modes
- Managing system instructions per conversation context
- Forking conversations for A/B testing or rollback scenarios
- Tracking conversation lineage (forked_from metadata)

## Class Signature

```python
from lionpride.session import Branch
from lionpride.core import Progression

class Branch(Progression):
    """Conversation branch with system message and access control."""

    # Constructor signature
    def __init__(
        self,
        *,
        user: str | UUID,
        session_id: UUID,
        system: UUID | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
        # Progression fields
        name: str | None = None,
        order: list[UUID] | None = None,
        # Element fields
        id: UUID | str | None = None,
        created_at: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**user** : str or UUID, required

Entity identifier giving commands to this branch (name or ID). Defaults to session_id if not specified.

**session_id** : UUID, required

Parent Session UUID. Frozen after creation.

**system** : UUID, optional

System message UUID in parent Session's messages. If set, automatically placed at position 0 in order.

**capabilities** : set[str], optional

Allowed structured output schema names during operations. Empty set (default) prohibits all structured outputs.

**resources** : set[str], optional

Allowed backend service names (from parent Session's ServiceRegistry). Empty set (default) prohibits all service access.

**Progression parameters** : See [Progression](../base/progression.md) for `name`, `order`.

**Element parameters** : See [Element](../base/element.md) for `id`, `created_at`, `metadata`.

## Attributes

| Attribute | Type | Frozen | Description |
|-----------|------|--------|-------------|
| `id` | `UUID` | Yes | Branch identifier (inherited from Element) |
| `created_at` | `datetime` | Yes | Creation timestamp (inherited from Element) |
| `metadata` | `dict[str, Any]` | No | Branch metadata (inherited from Element) |
| `name` | `str \| None` | No | Branch name (inherited from Progression) |
| `order` | `list[UUID]` | No | Message UUIDs in conversation order (inherited from Progression) |
| `user` | `str \| UUID` | Yes | Entity giving commands to this branch |
| `session_id` | `UUID` | Yes | Parent Session UUID |
| `system` | `UUID \| None` | No | System message UUID (if set) |
| `capabilities` | `set[str]` | No | Allowed structured output schema names |
| `resources` | `set[str]` | No | Allowed service names |

## Methods

### System Message Management

#### `set_system_message()`

Set or update system message for this branch.

**Signature:**

```python
def set_system_message(self, message: UUID | Message) -> None: ...
```

**Parameters:**

- `message` (UUID or Message): System message UUID or instance

**Returns:**

- None (mutates `self.system` and `self.order`)

**Behavior:**

1. Sets `self.system` to message UUID
2. If branch has messages (`len(self) > 0`):
   - Replaces `self[0]` with new system message UUID
3. Otherwise:
   - Inserts system message UUID at position 0

**Examples:**

```python
>>> from lionpride.session import Session, Message, SystemContent
>>> session = Session()
>>> branch = session.create_branch(name="main")

# Set system message
>>> system_msg = Message(content=SystemContent(system_message="Be helpful"))
>>> session.add_message(system_msg)
>>> branch.set_system_message(system_msg)

>>> branch.system == system_msg.id
True
>>> branch[0] == system_msg.id  # System message at position 0
True

# Update system message (replaces position 0)
>>> new_system = Message(content=SystemContent(system_message="Be concise"))
>>> session.add_message(new_system)
>>> branch.set_system_message(new_system)

>>> branch.system == new_system.id
True
>>> branch[0] == new_system.id  # Position 0 updated
True
```

**Notes:**

System messages are conventionally placed at position 0 in the conversation. This method ensures that invariant is maintained.

**See Also:**

- `Session.create_branch(system=...)`: Set system message during branch creation
- `Session.get_branch_system()`: Retrieve system message for branch

### Special Methods

#### `__repr__()`

String representation for debugging.

**Signature:**

```python
def __repr__(self) -> str: ...
```

**Returns:**

- str: Representation showing message count, session ID, name, and bind status

**Examples:**

```python
>>> from lionpride.session import Session
>>> session = Session()
>>> branch = session.create_branch(name="main")

>>> repr(branch)
'Branch(messages=0, session=123e4567-..., name='main', bound)'

# Bound indicates branch has reference to session
# (set via internal _session attribute)
```

## Inherited Methods

Branch inherits all Progression methods:

### List Operations (from Progression)

- `append(uuid)`: Add message UUID to end
- `insert(index, uuid)`: Insert message UUID at index
- `remove(uuid)`: Remove message UUID
- `pop(index=-1)`: Remove and return message UUID at index
- `clear()`: Remove all message UUIDs
- `__len__()`: Count of messages
- `__getitem__(index)`: Access message UUID by index
- `__setitem__(index, uuid)`: Replace message UUID at index
- `__delitem__(index)`: Delete message UUID at index
- `__contains__(uuid)`: Check if message UUID in order
- `__iter__()`: Iterate over message UUIDs

### Set Operations (from Progression)

- `include(uuid)`: Add UUID if not present (idempotent)
- `exclude(uuid)`: Remove UUID if present (idempotent)

### Workflow Operations (from Progression)

- `move(from_idx, to_idx)`: Reorder messages

See [Progression](../base/progression.md) for detailed method documentation.

## Protocol Implementations

Branch inherits protocol implementations from Progression and Element:

- **Observable**: UUID identifier via `id` property
- **Serializable**: `to_dict(mode='python'|'json'|'db')`, `to_json()`
- **Deserializable**: `from_dict()`, `from_json()` with polymorphic reconstruction
- **Hashable**: ID-based hashing via `__hash__()`
- **Allowable**: `allowed()` returns field names for validation

See [Element Protocol Implementations](../base/element.md#protocol-implementations) for details.

## Usage Patterns

### Basic Branch with System Message

```python
from lionpride.session import Session, Message, SystemContent, InstructionContent

session = Session()

# Create system message
system_msg = Message(content=SystemContent(system_message="You are a helpful coding assistant"))

# Create branch with system message
branch = session.create_branch(
    name="coding",
    system=system_msg,
)

# Add user messages
msg1 = Message(content=InstructionContent(instruction="Write a Python function"))
msg2 = Message(content=InstructionContent(instruction="Add type hints"))

session.add_message(msg1, branches=branch)
session.add_message(msg2, branches=branch)

# System message is always at position 0
print(session.messages[branch[0]].content.system_message)
# Output: "You are a helpful coding assistant"
```

### Access Control with Capabilities and Resources

```python
from lionpride.session import Session
from pydantic import BaseModel

class CodeAnalysis(BaseModel):
    summary: str
    issues: list[str]

session = Session()

# Register services
session.services.register(gpt4_model, name="gpt4")
session.services.register(code_analyzer_tool, name="analyzer")

# Branch with restricted access
branch = session.create_branch(
    name="production",
    capabilities={"CodeAnalysis"},  # Only this schema allowed
    resources={"gpt4"},              # Only GPT-4 allowed
)

# Attempting to use "analyzer" will fail at execution
# Attempting to use schema other than CodeAnalysis will fail
```

### Forking Branches

```python
from lionpride.session import Session, Message, InstructionContent

session = Session()
main = session.create_branch(
    name="main",
    capabilities={"BaseSchema"},
    resources={"gpt4"},
)

# Add shared conversation history
for i in range(3):
    msg = Message(content=InstructionContent(instruction=f"Step {i+1}"))
    session.add_message(msg, branches=main)

# Fork with inheritance
experimental = session.fork(
    main,
    name="experimental",
    capabilities=True,  # Copy from main
    resources={"gpt4", "claude"},  # Override (add Claude)
    system=True,  # Copy system message from main
)

# Check fork metadata
print(experimental.metadata["forked_from"])
# Output:
# {
#     "branch_id": "...",
#     "branch_name": "main",
#     "created_at": "2025-11-24T...",
#     "message_count": 3
# }

# Branches are independent after fork
new_msg = Message(content=InstructionContent(instruction="New approach"))
session.add_message(new_msg, branches=experimental)

print(len(main))          # 3 (original messages)
print(len(experimental))  # 4 (original + new)
```

### Dynamic System Message Updates

```python
from lionpride.session import Session, Message, SystemContent

session = Session()
branch = session.create_branch(name="adaptive")

# Initial system message
initial_system = Message(content=SystemContent(system_message="Be brief"))
session.add_message(initial_system)
branch.set_system_message(initial_system)

# Add conversation
msg1 = Message(content=InstructionContent(instruction="Explain AI"))
session.add_message(msg1, branches=branch)

# Update system message based on context (e.g., user requests detail)
detailed_system = Message(content=SystemContent(system_message="Be detailed and comprehensive"))
session.add_message(detailed_system)
branch.set_system_message(detailed_system)

# Position 0 now has updated system message
print(session.messages[branch[0]].content.system_message)
# Output: "Be detailed and comprehensive"
```

### Iterating Over Branch Messages

```python
from lionpride.session import Session, Message, InstructionContent

session = Session()
branch = session.create_branch(name="main")

# Add messages
messages = [
    Message(content=InstructionContent(instruction=f"Message {i}"))
    for i in range(5)
]
for msg in messages:
    session.add_message(msg, branches=branch)

# Iterate via UUIDs
for msg_uuid in branch:
    msg = session.messages[msg_uuid]
    print(f"{msg.role.value}: {msg.rendered}")

# List comprehension
branch_messages = [session.messages[uid] for uid in branch]
```

## Common Pitfalls

### Pitfall 1: Forgetting to Add Message to Session First

**Issue**: Setting system message with UUID not in session.

```python
session = Session()
branch = session.create_branch(name="main")

# DON'T: Message not in session
system_msg = Message(content=SystemContent(system_message="Help"))
# branch.set_system_message(system_msg)  # Works (extracts UUID)

# But better to add to session explicitly
session.add_message(system_msg)
branch.set_system_message(system_msg)
```

**Solution**: Always add messages to session before setting as system message.

### Pitfall 2: Mutating capabilities or resources After Branch Creation

**Issue**: Expecting branch access control to update automatically.

```python
branch = session.create_branch(
    name="main",
    resources={"gpt4"},
)

# Later: add resource dynamically
branch.resources.add("claude")  # Works (set is mutable)

# But operations already cached metadata won't update
# Solution: Re-register operations or use immutable access control
```

**Solution**: Set capabilities and resources during branch creation. If dynamic changes needed, consider creating new branch or re-validating access control.

### Pitfall 3: Assuming system Message is Automatically Added

**Issue**: Creating branch with system message UUID not in session.

```python
session = Session()
system_uuid = UUID("...")  # Random UUID

# Raises ValueError: system message not found
# branch = session.create_branch(name="main", system=system_uuid)

# DO: Add message first, or pass Message instance
system_msg = Message(content=SystemContent(system_message="Help"))
branch = session.create_branch(name="main", system=system_msg)
# Session automatically adds system_msg to session.messages
```

**Solution**: Pass Message instance to `create_branch()`, or add message to session before referencing by UUID.

## Design Rationale

### Why Extend Progression Instead of Composing?

Branch *is* a Progression (ordered UUIDs) with additional session-specific metadata. Extending Progression:

1. Inherits list operations (append, insert, remove, etc.)
2. Inherits set operations (include, exclude)
3. Adds session-specific fields (user, system, capabilities, resources)
4. Preserves Progression's identity (ID, created_at)

Composition would require forwarding all Progression methods.

### Why Frozen user and session_id?

Freezing `user` and `session_id` ensures:

1. Branch ownership doesn't change after creation
2. Session binding is immutable (branch always belongs to same session)
3. Access control decisions remain stable (based on session context)

### Why System Message at Position 0?

Chat APIs expect system messages at the beginning of conversation. Enforcing position 0 ensures:

1. Consistent rendering for chat APIs
2. Clear separation between system instructions and user messages
3. Predictable behavior when forking or preparing messages

### Why Separate capabilities and resources?

Different access control dimensions:

- **capabilities**: What structured outputs are allowed (schema names)
- **resources**: What services are allowed (model/tool names)

Separating these enables fine-grained control:

```python
# Branch can use GPT-4 but only with approved schemas
branch = session.create_branch(
    resources={"gpt4"},
    capabilities={"ApprovedSchema"},  # No experimental schemas
)
```

## See Also

- **Related Classes**:
  - [Session](session.md): Parent container for branches and messages
  - [Progression](../base/progression.md): Base class for ordered UUIDs
  - [Message](message.md): Message container referenced by branch

- **Module Overview**:
  - [session Overview](overview.md): Module-level documentation

## Examples

See [overview.md Examples](overview.md#examples) for comprehensive usage patterns including:

- Multi-branch workflows
- Forking for A/B testing
- Access control enforcement
- System message management
