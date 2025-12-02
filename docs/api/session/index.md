# Session

> Conversation state management with messages, branches, and logging

## Overview

The `session` module provides the core conversation and state management infrastructure for lionpride. It orchestrates how messages flow through the system, how conversations branch and merge, and how interaction history is persisted.

**Key Capabilities:**

- **Session**: Central orchestrator managing branches, services, and operations
- **Branch**: Named progression of messages with access control (capabilities and resources)
- **Message**: Polymorphic container with auto-derived role from content type
- **Log System**: Persistent logging with adapters (SQLite, PostgreSQL) and broadcasters (S3, webhooks)

The session module follows a **discriminated union** pattern for message content, where dict keys determine which `MessageContent` subclass is instantiated. This enables flexible input while maintaining type safety and automatic role derivation.

## Classes

| Class | Description |
|-------|-------------|
| [Session](session.md) | Central orchestrator for branches, services, and operations |
| [Branch](branch.md) | Named progression of messages with capabilities and resources |
| [Message](message.md) | Universal message container with auto-derived role |

### Message Content Types

| Class | Description |
|-------|-------------|
| `SystemContent` | System/developer instructions (role: SYSTEM) |
| `InstructionContent` | User instructions with structured output support (role: USER) |
| `AssistantResponseContent` | Assistant text responses (role: ASSISTANT) |
| `ActionRequestContent` | Tool/function call requests (role: ASSISTANT) |
| `ActionResponseContent` | Tool execution results (role: TOOL) |

### Log System

| Class | Description |
|-------|-------------|
| `Log` | Individual log entry with type and content |
| `LogStore` | In-memory log storage with optional persistence |
| `LogAdapter` | Base class for log persistence adapters |
| `LogBroadcaster` | Distributes logs to multiple subscribers |
| `SQLiteWALLogAdapter` | SQLite adapter with WAL mode |
| `PostgresLogAdapter` | PostgreSQL adapter |
| `S3LogSubscriber` | S3 log subscriber |
| `WebhookLogSubscriber` | Webhook log subscriber |

## Quick Start

```python
from lionpride import Session, Branch, Message, iModel

# Create session with default model
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)

# Create a branch for conversation
branch = session.create_branch(
    name="main",
    capabilities={"Analysis", "Summary"},  # Allowed output types
    resources={"gpt-4o-mini"},             # Allowed services
)

# Messages auto-derive role from content
user_msg = Message(content={"instruction": "Explain quantum computing"})
user_msg.role  # MessageRole.USER

assistant_msg = Message(content={"assistant_response": "Quantum computing uses..."})
assistant_msg.role  # MessageRole.ASSISTANT

# Conduct operations through session
from lionpride.operations.operate import OperateParams, GenerateParams

params = OperateParams(generate=GenerateParams(instruction="Hello!"))
result = await session.conduct("operate", branch, params=params)
```

## See Also

- [Services](../services/index.md) - iModel and ServiceRegistry
- [Operations](../operations/index.md) - generate, operate, react functions
- [Base Classes](../base/element.md) - Element, Node inheritance
- [Protocols](../../user_guide/protocols.md) - Protocol implementations guide
