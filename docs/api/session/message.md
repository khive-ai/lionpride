# Message

> Universal message container with auto-derived role from content type

## Overview

`Message` extends `Node` with **automatic role derivation** from content type via discriminated union pattern. Dict keys determine which `MessageContent` subclass is instantiated.

**Role Inference:**

| Dict Keys | Content Type | Role |
|-----------|--------------|------|
| `instruction`, `context`, `request_model`, `images` | `InstructionContent` | `USER` |
| `assistant_response` | `AssistantResponseContent` | `ASSISTANT` |
| `function`, `arguments` | `ActionRequestContent` | `ASSISTANT` |
| `result`, `error` | `ActionResponseContent` | `TOOL` |
| `system_message` | `SystemContent` | `SYSTEM` |

## Class Signature

```python
from lionpride import Message

class Message(Node):
    """Message container with auto-derived role from content type."""

    # Constructor signature
    def __init__(
        self,
        *,
        content: MessageContent | dict[str, Any],
        sender: MessageRole | str | UUID | None = None,
        recipient: MessageRole | str | UUID | None = None,
        # Inherited from Node/Element
        id: UUID | str | None = None,
        created_at: datetime | str | int | float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | `MessageContent \| dict` | Required. Dict auto-converts to appropriate subclass |
| `sender` | `MessageRole \| str \| UUID` | Optional sender identifier |
| `recipient` | `MessageRole \| str \| UUID` | Optional recipient identifier |
| `id`, `created_at`, `metadata` | - | Inherited from Element |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | `MessageContent` | Auto-converted from dict |
| `sender` / `recipient` | `SenderRecipient \| None` | Optional identifiers |
| `role` | `MessageRole` | **Read-only**, auto-derived from content type |
| `embedding` | `list[float] \| None` | Optional (from Node) |
| `id`, `created_at`, `metadata` | - | From Element |

```python
>>> msg = Message(content={"instruction": "Hello"})
>>> msg.role  # MessageRole.USER (auto-derived)
```

## Methods

### `clone()`

Create copy with new ID and lineage tracking (`metadata["clone_from"]`).

```python
def clone(self, *, sender: SenderRecipient | None = None) -> Message
```

```python
>>> cloned = original.clone(sender="agent_1")
>>> cloned.metadata["clone_from"]  # original.id
```

### Inherited

`to_dict()`, `to_json()`, `from_dict()`, `from_json()`, `adapt_to()`, `adapt_from()`

## MessageRole Enum

`SYSTEM`, `USER`, `ASSISTANT`, `TOOL`, `UNSET`

## MessageContent Types

### SystemContent

`system_message`, `system_datetime` (True = auto-timestamp)

```python
>>> Message(content={"system_message": "You are helpful.", "system_datetime": True})
```

### InstructionContent

`instruction`, `context`, `request_model`, `tool_schemas`, `images`, `image_detail`

```python
>>> Message(content={"instruction": "Analyze this", "request_model": Analysis})
```

### AssistantResponseContent

`assistant_response`

```python
>>> Message(content={"assistant_response": "Here's my analysis..."})
```

### ActionRequestContent

`function`, `arguments`

```python
>>> Message(content={"function": "search", "arguments": {"query": "docs"}})
```

### ActionResponseContent

`result`, `error`, `request_id`. Property: `success` (bool)

```python
>>> Message(content={"result": ["a", "b"], "request_id": "req_123"})
```

## Chat API Format

All content types provide `to_chat()` -> `{"role": "...", "content": "..."}` for LLM API compatibility.

## Protocol Implementations

Inherits: **Observable**, **Serializable**, **Deserializable**, **Hashable**, **Adaptable**

## Usage Patterns

### Basic Usage

```python
from lionpride import Message

user_msg = Message(content={"instruction": "What is the capital of France?"})
assistant_msg = Message(content={"assistant_response": "Paris."})

# Chat format for LLM API
assistant_msg.content.to_chat()  # {'role': 'assistant', 'content': 'Paris.'}
```

### Tool Call Workflow

```python
# 1. User request -> 2. Tool call -> 3. Tool result -> 4. Assistant response
tool_request = Message(content={"function": "search", "arguments": {"query": "docs"}})
tool_response = Message(content={"result": ["a", "b"], "request_id": str(tool_request.id)})
```

### Message Forwarding

```python
forwarded = original.clone(sender="branch_a")
# Lineage in forwarded.metadata["clone_from"]
```

## Common Pitfalls

### 1. Role is Read-Only

Role is auto-derived from content. Use appropriate content keys, not `role=`.

### 2. Don't Mix Content Keys

Use keys for one content type per message.

### 3. Clone When Forwarding

Use `msg.clone()` when adding same message to multiple branches.

## Design Rationale

**Auto-Derived Role**: Ensures consistency - role always matches content type. Users don't need to remember role constants.

**Discriminated Union**: Dict key detection enables flexible input without knowing class names, JSON-compatible serialization.

**Separate Content from Message**: Enables `content.render()` customization, `content.to_chat()` for LLM APIs, content-specific validation.

## See Also

- [Node](../base/node.md), [Element](../base/element.md), [Session](session.md), [Branch](branch.md)

## Examples

### Complete Conversation

```python
from lionpride import Message

messages = [
    Message(content={"system_message": "You are a coding assistant.", "system_datetime": True}),
    Message(content={"instruction": "How do I read a file in Python?"}),
    Message(content={"assistant_response": "Use `with open('file.txt') as f: ...`"}),
]

# Convert to LLM API format
chat_messages = [msg.content.to_chat() for msg in messages]
```

### Multimodal Message

```python
msg = Message(content={
    "instruction": "Describe this image",
    "images": ["https://example.com/image.jpg"],
    "image_detail": "high"
})
```

### Serialization Roundtrip

```python
json_str = msg.to_json()
restored = Message.from_json(json_str)
assert restored.role == msg.role
```
