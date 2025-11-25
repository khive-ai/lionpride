# Message

> Universal message container with auto-derived role and polymorphic content (extends Node)

## Overview

`Message` is the universal container for conversational messages, wrapping typed [`MessageContent`](message_content.md) with automatic role derivation and sender/recipient tracking. It extends [`Node`](../base/node.md) to inherit Element's identity and polymorphic serialization while adding message-specific properties.

**Key Capabilities:**

- **Auto-Derived Role**: Message role automatically determined by content type
- **Discriminated Content**: Type-safe MessageContent variants (SystemContent, InstructionContent, etc.)
- **Sender/Recipient Tracking**: Optional sender and recipient identifiers
- **Chat API Format**: One-line conversion to chat API format via `chat_msg`
- **Content Rendering**: Delegates to content-specific rendering logic
- **Lineage Tracking**: Clone messages with metadata tracking

**When to Use Message:**

- Building conversational workflows with typed message content
- Tracking message flow between entities (user, assistant, tools)
- Preparing messages for chat APIs
- Managing message history with identity-based deduplication
- Cloning messages for forked conversations

## Class Signature

```python
from lionpride.session import Message
from lionpride.core import Node

class Message(Node):
    """Message container with auto-derived role from content type."""

    # Constructor signature
    def __init__(
        self,
        *,
        content: MessageContent | dict[str, Any],
        sender: SenderRecipient | None = None,
        recipient: SenderRecipient | None = None,
        # Node fields
        embedding: list[float] | None = None,
        # Element fields
        id: UUID | str | None = None,
        created_at: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**content** : MessageContent or dict[str, Any], required

Message content. Can be:

- **MessageContent instance**: SystemContent, InstructionContent, AssistantResponseContent, ActionRequestContent, ActionResponseContent
- **dict**: Auto-inferred from keys (see content inference logic below)

**sender** : SenderRecipient, optional

Message sender identifier. Can be:

- **MessageRole**: SYSTEM, USER, ASSISTANT, TOOL, UNSET
- **str**: Entity name or UUID string
- **UUID**: Entity identifier

**recipient** : SenderRecipient, optional

Message recipient identifier (same types as sender).

**Node parameters** : See [Node](../base/node.md) for `embedding`.

**Element parameters** : See [Element](../base/element.md) for `id`, `created_at`, `metadata`.

### Content Inference Logic

When `content` is a dict, Message infers the content type from dict keys:

| Dict Keys | Inferred Type |
|-----------|---------------|
| `instruction`, `context`, `response_model`, `tool_schemas`, `images` | `InstructionContent` |
| `assistant_response` | `AssistantResponseContent` |
| `result` or `error` | `ActionResponseContent` |
| `function` or `arguments` | `ActionRequestContent` |
| `system_message` or `system_datetime` | `SystemContent` |
| Empty dict `{}` | `InstructionContent` (default) |

**Examples:**

```python
# Explicit content type
>>> msg = Message(content=InstructionContent(instruction="Hello"))

# Dict inference (instruction → InstructionContent)
>>> msg = Message(content={"instruction": "Hello"})
>>> type(msg.content).__name__
'InstructionContent'

# Dict inference (assistant_response → AssistantResponseContent)
>>> msg = Message(content={"assistant_response": "Hi there"})
>>> type(msg.content).__name__
'AssistantResponseContent'

# Dict inference (system_message → SystemContent)
>>> msg = Message(content={"system_message": "You are helpful"})
>>> type(msg.content).__name__
'SystemContent'

# Dict inference (result → ActionResponseContent)
>>> msg = Message(content={"result": {"data": 42}})
>>> type(msg.content).__name__
'ActionResponseContent'
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `UUID` | Message identifier (inherited from Element, frozen) |
| `created_at` | `datetime` | Creation timestamp (inherited from Element, frozen) |
| `metadata` | `dict[str, Any]` | Message metadata (inherited from Element, mutable) |
| `content` | `MessageContent` | Message content (discriminated union) |
| `sender` | `SenderRecipient \| None` | Message sender identifier |
| `recipient` | `SenderRecipient \| None` | Message recipient identifier |
| `role` | `MessageRole` | Auto-derived from content.role (read-only property) |
| `chat_msg` | `dict[str, Any] \| None` | Chat API format (read-only property) |
| `rendered` | `str \| list[dict]` | Rendered content (read-only property) |
| `embedding` | `list[float] \| None` | Optional embedding vector (inherited from Node) |

## Properties

### `role`

Auto-derive message role from content type.

**Signature:**

```python
@property
def role(self) -> MessageRole: ...
```

**Returns:**

- MessageRole: Role determined by `content.role` (ClassVar on each content type)

**Examples:**

```python
>>> from lionpride.session import Message, InstructionContent, SystemContent

>>> msg = Message(content=InstructionContent(instruction="Hello"))
>>> msg.role
MessageRole.USER

>>> msg = Message(content=SystemContent(system_message="Be helpful"))
>>> msg.role
MessageRole.SYSTEM

>>> from lionpride.session import AssistantResponseContent
>>> msg = Message(content=AssistantResponseContent(assistant_response="Hi"))
>>> msg.role
MessageRole.ASSISTANT
```

**Notes:**

Role is read-only and automatically derived. You cannot set it manually. Use the appropriate content type to control role.

**Role Mapping:**

| Content Type | Role |
|--------------|------|
| `SystemContent` | `MessageRole.SYSTEM` |
| `InstructionContent` | `MessageRole.USER` |
| `AssistantResponseContent` | `MessageRole.ASSISTANT` |
| `ActionRequestContent` | `MessageRole.ASSISTANT` |
| `ActionResponseContent` | `MessageRole.TOOL` |

**See Also:**

- [MessageRole](#messagerole): Role enumeration
- [MessageContent](message_content.md): Content base class

### `chat_msg`

Format message for chat API: `{"role": "...", "content": "..."}`.

**Signature:**

```python
@property
def chat_msg(self) -> dict[str, Any] | None: ...
```

**Returns:**

- dict[str, Any] or None: Chat API format with role and rendered content, or None if rendering fails

**Examples:**

```python
>>> from lionpride.session import Message, InstructionContent

>>> msg = Message(content=InstructionContent(instruction="Explain AI"))
>>> msg.chat_msg
{'role': 'user', 'content': 'Instruction: Explain AI'}

>>> from lionpride.session import AssistantResponseContent
>>> msg = Message(content=AssistantResponseContent(assistant_response="AI is..."))
>>> msg.chat_msg
{'role': 'assistant', 'content': 'AI is...'}

# InstructionContent with images returns list of content blocks
>>> msg = Message(content=InstructionContent(
...     instruction="Describe this",
...     images=["https://example.com/image.jpg"]
... ))
>>> msg.chat_msg
{
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Instruction: Describe this'},
        {'type': 'image_url', 'image_url': {'url': 'https://...', 'detail': 'auto'}}
    ]
}
```

**Notes:**

Delegates to `content.chat_msg`, which calls `content.rendered` and formats with role.

**See Also:**

- `rendered`: Direct access to rendered content
- [prepare_messages_for_chat()](#prepare_messages_for_chat): Batch message preparation

### `rendered`

Rendered content (delegates to content-specific rendering).

**Signature:**

```python
@property
def rendered(self) -> str | list[dict[str, Any]]: ...
```

**Returns:**

- str or list[dict]: Rendered content (format depends on content type)

**Examples:**

```python
>>> from lionpride.session import Message, InstructionContent, SystemContent

# SystemContent renders to string
>>> msg = Message(content=SystemContent(
...     system_message="You are helpful",
...     system_datetime=True,
... ))
>>> msg.rendered
'System Time: 2025-11-24T...\n\nYou are helpful'

# InstructionContent renders to string (or list if images present)
>>> msg = Message(content=InstructionContent(instruction="Hello"))
>>> msg.rendered
'Instruction: Hello'

# AssistantResponseContent renders to string
>>> from lionpride.session import AssistantResponseContent
>>> msg = Message(content=AssistantResponseContent(assistant_response="Hi"))
>>> msg.rendered
'Hi'

# ActionResponseContent renders to YAML string
>>> from lionpride.session import ActionResponseContent
>>> msg = Message(content=ActionResponseContent(result={"data": 42}))
>>> msg.rendered
'success: true\nresult:\n  data: 42'
```

**See Also:**

- [MessageContent.rendered](message_content.md): Content-specific rendering logic
- `chat_msg`: Chat API formatted output

## Methods

### `clone()`

Create copy with new ID and lineage tracking in metadata.

**Signature:**

```python
def clone(self, *, sender: SenderRecipient | None = None) -> Message: ...
```

**Parameters:**

- `sender` (SenderRecipient, optional): Override sender for cloned message

**Returns:**

- Message: New message instance with new ID, original lineage in metadata

**Lineage Metadata:**

Cloned message's metadata includes:

- `clone_from`: Original message UUID (str)
- `original_created_at`: Original creation timestamp (ISO format)

**Examples:**

```python
>>> from lionpride.session import Message, InstructionContent

>>> original = Message(
...     content=InstructionContent(instruction="Hello"),
...     sender="user",
... )

# Clone with same sender
>>> cloned = original.clone()
>>> cloned.id != original.id
True
>>> cloned.content.instruction == original.content.instruction
True
>>> cloned.metadata["clone_from"] == str(original.id)
True

# Clone with different sender
>>> cloned = original.clone(sender="assistant")
>>> cloned.sender
'assistant'
>>> original.sender
'user'
```

**Use Cases:**

- Forking conversations with modified context
- Creating message templates
- Tracking message lineage across branches

**See Also:**

- `Message.from_dict()`: Deserialization (underlying implementation)

## Types

### MessageRole

Enumeration of message roles in chat interactions.

**Values:**

- `SYSTEM`: System/Developer instructions defining model behavior
- `USER`: Direct message from user to assistant
- `ASSISTANT`: Assistant response (model-generated)
- `TOOL`: Tool result returned after tool_call execution
- `UNSET`: No role specified (fallback/unknown)

**Examples:**

```python
>>> from lionpride.session import MessageRole

>>> MessageRole.USER
MessageRole.USER
>>> MessageRole.USER.value
'user'

>>> MessageRole.allowed()
{'system', 'user', 'assistant', 'tool', 'unset'}
```

**See Also:**

- `Message.role`: Auto-derived role property

### SenderRecipient

Type alias for sender/recipient identifiers.

**Definition:**

```python
SenderRecipient: TypeAlias = MessageRole | str | UUID
```

**Examples:**

```python
>>> from lionpride.session import Message, MessageRole, InstructionContent
>>> from uuid import UUID

# MessageRole
>>> msg = Message(
...     content=InstructionContent(instruction="Hello"),
...     sender=MessageRole.USER,
...     recipient=MessageRole.ASSISTANT,
... )

# String (entity name)
>>> msg = Message(
...     content=InstructionContent(instruction="Hello"),
...     sender="user_123",
...     recipient="assistant_456",
... )

# UUID (entity identifier)
>>> user_id = UUID("12345678-1234-1234-1234-123456789012")
>>> msg = Message(
...     content=InstructionContent(instruction="Hello"),
...     sender=user_id,
... )
```

**Validation:**

Values are validated and normalized:

- `MessageRole` → kept as-is
- `str` in `MessageRole.allowed()` → converted to `MessageRole`
- Valid UUID string → converted to `UUID`
- Other `str` → kept as-is
- `None` → `MessageRole.UNSET`

**See Also:**

- `validate_sender_recipient()`: Validation function
- `serialize_sender_recipient()`: Serialization function

## Utility Functions

### `prepare_messages_for_chat()`

Prepare messages for chat API with intelligent content organization.

**Signature:**

```python
def prepare_messages_for_chat(
    messages: Pile[Message],
    progression: Progression | None = None,
    new_instruction: Message | None = None,
    to_chat: bool = False,
) -> list[MessageContent] | list[dict[str, Any]]: ...
```

**Parameters:**

- `messages` (Pile[Message]): Message storage (typically `session.messages`)
- `progression` (Progression, optional): Progression or list of UUIDs. If None, uses all messages.
- `new_instruction` (Message, optional): New instruction to append
- `to_chat` (bool, default False): If True, return list[dict] in chat format; otherwise list[MessageContent]

**Returns:**

- list[MessageContent] or list[dict]: Prepared messages ready for chat API

**Algorithm:**

1. **Auto-detect system message** from first message (if SystemContent)
2. **Collect ActionResponseContent** and embed into following instruction's context
3. **Merge consecutive AssistantResponses**
4. **Embed system into first instruction**
5. **Remove tool_schemas and response_model** from historical instructions
6. **Append new_instruction** with any remaining action outputs

**Examples:**

```python
>>> from lionpride.session import Session, Message, prepare_messages_for_chat
>>> from lionpride.session import (
...     SystemContent, InstructionContent, AssistantResponseContent, ActionResponseContent
... )

>>> session = Session()
>>> branch = session.create_branch(name="main")

# Add various message types
>>> session.add_message(
...     Message(content=SystemContent(system_message="Be helpful")),
...     branches=branch
... )
>>> session.add_message(
...     Message(content=InstructionContent(instruction="Hello")),
...     branches=branch
... )
>>> session.add_message(
...     Message(content=AssistantResponseContent(assistant_response="Hi there")),
...     branches=branch
... )
>>> session.add_message(
...     Message(content=ActionResponseContent(result={"data": 42})),
...     branches=branch
... )
>>> session.add_message(
...     Message(content=InstructionContent(instruction="Analyze result")),
...     branches=branch
... )

# Prepare for chat API
>>> chat_messages = prepare_messages_for_chat(
...     messages=session.messages,
...     progression=branch,
...     to_chat=True,
... )

>>> len(chat_messages)
3  # Consolidated: system+instruction, assistant, instruction+context

>>> chat_messages[0]
{
    'role': 'user',
    'content': 'Be helpful\n\nHello'  # System embedded into first instruction
}

>>> chat_messages[1]
{
    'role': 'assistant',
    'content': 'Hi there'
}

>>> chat_messages[2]
{
    'role': 'user',
    'content': 'Instruction: Analyze result\nContext:\n  - success: true\n    result:\n      data: 42'
    # ActionResponseContent embedded into context
}
```

**Consolidation Details:**

**System Message Embedding:**

```python
# Before: Separate system message
[
    {"role": "system", "content": "Be helpful"},
    {"role": "user", "content": "Hello"}
]

# After: System embedded into first user message
[
    {"role": "user", "content": "Be helpful\n\nHello"}
]
```

**Action Output Embedding:**

```python
# Before: Separate tool result
[
    {"role": "user", "content": "Call search tool"},
    {"role": "assistant", "content": "Calling search..."},
    {"role": "tool", "content": '{"results": [...]}'},
    {"role": "user", "content": "Analyze results"}
]

# After: Tool result embedded into next instruction's context
[
    {"role": "user", "content": "Call search tool"},
    {"role": "assistant", "content": "Calling search..."},
    {"role": "user", "content": "Instruction: Analyze results\nContext:\n  - {\"results\": [...]}"}
]
```

**Consecutive Assistant Merging:**

```python
# Before: Multiple assistant messages
[
    {"role": "assistant", "content": "First response"},
    {"role": "assistant", "content": "Second response"}
]

# After: Merged
[
    {"role": "assistant", "content": "First response\n\nSecond response"}
]
```

**Notes:**

This function implements the lionagi pattern for preparing messages, optimizing for chat API requirements:

- Reduces message count (fewer API calls)
- Preserves information (nothing lost)
- Maintains conversation flow
- Removes historical schemas (not needed for past messages)

**Use Cases:**

- Preparing session messages for LLM API calls
- Optimizing conversation history before sending to model
- Consolidating multi-turn interactions

**See Also:**

- `Message.chat_msg`: Single message chat format
- [Session.conduct()](session.md#conduct): Operation execution with message preparation

## Protocol Implementations

Message inherits protocol implementations from Node and Element:

- **Observable**: UUID identifier via `id` property
- **Serializable**: `to_dict(mode='python'|'json'|'db')`, `to_json()`
- **Deserializable**: `from_dict()`, `from_json()` with polymorphic reconstruction
- **Hashable**: ID-based hashing via `__hash__()`
- **Adaptable**: `adapt_to()`, `adapt_from()` for format conversion (Node adds adapters)

See [Element Protocol Implementations](../base/element.md#protocol-implementations) and [Node](../base/node.md) for details.

## Usage Patterns

### Creating Messages with Different Content Types

```python
from lionpride.session import Message
from lionpride.session import (
    SystemContent, InstructionContent, AssistantResponseContent,
    ActionRequestContent, ActionResponseContent
)

# System message
system_msg = Message(content=SystemContent(
    system_message="You are a helpful assistant",
    system_datetime=True,
))

# User instruction
user_msg = Message(content=InstructionContent(
    instruction="Explain quantum computing",
    context=["For beginners", "Use analogies"],
))

# Assistant response
assistant_msg = Message(content=AssistantResponseContent(
    assistant_response="Quantum computing uses quantum bits..."
))

# Tool call request
tool_request = Message(content=ActionRequestContent(
    function="search",
    arguments={"query": "quantum computing", "max_results": 5},
))

# Tool result
tool_result = Message(content=ActionResponseContent(
    result=["Result 1", "Result 2", "Result 3"],
))
```

### Tracking Sender and Recipient

```python
from lionpride.session import Message, MessageRole, InstructionContent
from uuid import uuid4

# Using MessageRole
msg = Message(
    content=InstructionContent(instruction="Hello"),
    sender=MessageRole.USER,
    recipient=MessageRole.ASSISTANT,
)

# Using entity identifiers
user_id = uuid4()
assistant_id = uuid4()

msg = Message(
    content=InstructionContent(instruction="Hello"),
    sender=user_id,
    recipient=assistant_id,
)

# Using string names
msg = Message(
    content=InstructionContent(instruction="Hello"),
    sender="user_123",
    recipient="assistant_gpt4",
)
```

### Cloning Messages for Forked Conversations

```python
from lionpride.session import Session, Message, InstructionContent

session = Session()
main = session.create_branch(name="main")

# Original message
original = Message(
    content=InstructionContent(instruction="Explain approach A"),
    sender="user",
)
session.add_message(original, branches=main)

# Fork and modify
experimental = session.fork(main, name="experimental")

# Clone with modified content for experimental branch
modified = Message(
    content=InstructionContent(instruction="Explain approach B"),
    sender="user",
    metadata={
        "clone_from": str(original.id),
        "modification": "Changed approach A to B",
    }
)
session.add_message(modified, branches=experimental)

# Track lineage
print(modified.metadata["clone_from"])  # Original message UUID
```

### Converting to Chat API Format

```python
from lionpride.session import Session, Message, prepare_messages_for_chat
from lionpride.session import InstructionContent, AssistantResponseContent

session = Session()
branch = session.create_branch(name="chat")

# Build conversation
messages_data = [
    {"instruction": "What is Python?"},
    {"assistant_response": "Python is a programming language..."},
    {"instruction": "Show me an example"},
    {"assistant_response": "Here's a simple example:\n```python\nprint('Hello')\n```"},
]

for data in messages_data:
    msg = Message(content=data)  # Auto-infers content type
    session.add_message(msg, branches=branch)

# Prepare for chat API
chat_messages = prepare_messages_for_chat(
    messages=session.messages,
    progression=branch,
    to_chat=True,
)

# Send to LLM API
# response = llm_api.chat(messages=chat_messages)
```

## Common Pitfalls

### Pitfall 1: Manually Setting Role

**Issue**: Trying to set message role directly.

```python
# DON'T: role is not a field
# msg = Message(
#     content=InstructionContent(instruction="Hello"),
#     role=MessageRole.USER  # Error: unexpected keyword argument
# )

# DO: Role is auto-derived from content type
msg = Message(content=InstructionContent(instruction="Hello"))
print(msg.role)  # MessageRole.USER (automatic)
```

**Solution**: Use the correct content type. Role is read-only and auto-derived.

### Pitfall 2: Mutating Content After Creation

**Issue**: Expecting content mutations to persist.

```python
msg = Message(content=InstructionContent(instruction="Original"))

# DON'T: Content is immutable (dataclass with slots=True)
# msg.content.instruction = "Modified"  # Error: can't set attribute

# DO: Create new message with modified content
new_msg = Message(content=InstructionContent(instruction="Modified"))
```

**Solution**: MessageContent is immutable. Create new messages instead of mutating.

### Pitfall 3: Incorrect Content Inference

**Issue**: Dict keys don't match any content type pattern.

```python
# Ambiguous dict (defaults to InstructionContent)
msg = Message(content={"unknown_key": "value"})
print(type(msg.content).__name__)  # InstructionContent (default)

# DO: Be explicit with content type
msg = Message(content=AssistantResponseContent(assistant_response="value"))
```

**Solution**: Use explicit content types for clarity, or ensure dict keys match inference patterns.

### Pitfall 4: Forgetting to Add Messages to Session

**Issue**: Creating messages without adding to session storage.

```python
session = Session()
branch = session.create_branch(name="main")

msg = Message(content=InstructionContent(instruction="Hello"))
# Forgot to add to session!

# Later: branch references msg.id, but msg not in session.messages
# This causes lookup errors
```

**Solution**: Always add messages to session via `session.add_message()`.

## Design Rationale

### Why Auto-Derived Role?

Manual role assignment is error-prone:

```python
# Manual (error-prone)
msg = Message(
    content=InstructionContent(...),
    role=MessageRole.ASSISTANT  # BUG: should be USER
)

# Auto-derived (correct by construction)
msg = Message(content=InstructionContent(...))
# role = MessageRole.USER (automatic)
```

Auto-derivation eliminates entire class of bugs.

### Why Discriminated Union for Content?

Type safety + flexibility:

1. **Type Safety**: Pydantic validates fields per content type
2. **Custom Rendering**: Each type renders differently
3. **Auto-Inference**: Dict keys determine content type
4. **Extensibility**: Easy to add new content types

### Why Extend Node Instead of Element?

Node provides:

1. **Content Field**: Polymorphic content storage (MessageContent)
2. **Embedding Support**: Optional vector embeddings for semantic search
3. **Adapter Registry**: Custom format conversions (TOML, YAML, etc.)

Message inherits these capabilities while adding message-specific properties (sender, recipient, role).

### Why sender and recipient Instead of from and to?

Naming clarity:

- `sender`/`recipient` are unambiguous (entity that sent/received message)
- `from`/`to` are Python keywords (would require escaping)
- Consistent with email/messaging conventions

## See Also

- **Related Classes**:
  - [MessageContent](message_content.md): Content variants
  - [Node](../base/node.md): Base class with content support
  - [Session](session.md): Container for messages and branches

- **Module Overview**:
  - [session Overview](overview.md): Module-level documentation

## Examples

See [overview.md Examples](overview.md#examples) for comprehensive usage patterns including:

- Basic conversations
- Multi-branch workflows
- Structured output workflows
- Tool calling workflows
- Message preparation for chat APIs
