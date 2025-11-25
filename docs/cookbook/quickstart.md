# Quick Start: Your First lionpride Chat

## Overview

Get started with lionpride in 5 minutes. This recipe shows the absolute minimum code needed to have a conversation with an LLM.

## Prerequisites

```bash
pip install lionpride
```

Set your API key as an environment variable:

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# For Gemini
export GEMINI_API_KEY="..."
```

## The Code

### Example 1: Single Turn Chat (OpenAI)

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel

async def main():
    # Create session and model
    session = Session()
    model = iModel(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7
    )
    session.services.register(model)

    # Create a conversation branch
    branch = session.create_branch(name="main")

    # Send a message
    result = await session.invoke(
        operation="communicate",
        branch=branch,
        parameters={
            "instruction": "Write a haiku about Python programming",
            "imodel": model.name,
        }
    )

    print(result)

asyncio.run(main())
```

### Example 2: Same with Anthropic

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel

async def main():
    session = Session()

    # Just change the provider - API is identical
    model = iModel(
        provider="anthropic",
        endpoint="messages",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7
    )
    session.services.register(model)

    branch = session.create_branch(name="main")

    result = await session.invoke(
        operation="communicate",
        branch=branch,
        parameters={
            "instruction": "Explain what makes Python a great language for beginners",
            "imodel": model.name,
        }
    )

    print(result)

asyncio.run(main())
```

### Example 3: Direct Operation Call

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def main():
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini")
    session.services.register(model)

    branch = session.create_branch(name="main")

    # Call communicate() directly
    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "What are the key features of async Python?",
            "imodel": model.name,
        }
    )

    print(result)

asyncio.run(main())
```

## Expected Output

### Example 1 (Haiku)

```
Code flows like water,
Indentation shapes the mind,
Snake syntax is clean.
```

### Example 2 (Explanation)

```
Python is considered an excellent language for beginners for several reasons:

1. Readable Syntax: Python's syntax resembles plain English, making it
   intuitive to read and write code...

[Full response will continue with detailed explanation]
```

### Example 3 (Async Features)

```
Key features of async Python include:

1. async/await syntax: Clean, readable way to write asynchronous code
2. Coroutines: Functions that can pause and resume execution
3. Event loop: Manages execution of async tasks...

[Full response continues]
```

## What Just Happened?

1. **Session**: Container for conversations, services, and operations
2. **iModel**: Unified interface wrapping any LLM provider (OpenAI, Anthropic, Gemini, etc.)
3. **Branch**: Represents a single conversation thread (like a chat history)
4. **communicate**: Stateful operation that persists messages to the branch
5. **invoke**: Generic method to call any registered operation

## Key Concepts

### Session vs Branch vs Message

```python
Session
├── services (ServiceRegistry)
│   └── iModel instances
├── conversations (Flow[Message, Branch])
│   ├── messages (all Message instances)
│   └── branches (conversation threads)
└── operations (OperationRegistry)
    └── communicate, generate, react, etc.

Branch
├── order: list[UUID]  # Message IDs in order
└── system: UUID       # Optional system message
```

### Provider Flexibility

lionpride uses a unified interface - switching providers is just changing the `provider` parameter:

```python
# All equivalent API
model = iModel(provider="openai", model="gpt-4o-mini")
model = iModel(provider="anthropic", endpoint="messages", model="claude-3-5-sonnet-20241022")
model = iModel(provider="gemini", model="gemini-2.0-flash-exp")
model = iModel(provider="groq", model="llama-3.1-8b-instant")
```

## Variations

### Stateless Generation (No Persistence)

Use `generate` when you don't need conversation history:

```python
from lionpride.operations import generate

result = await generate(
    session=session,
    branch=branch,  # Branch required but messages not persisted
    parameters={
        "imodel": model.name,
        "messages": [{"role": "user", "content": "Hello!"}],
    }
)
```

### Access Conversation History

```python
# Get all messages in a branch
for msg_id in branch.order:
    message = session.messages[msg_id]
    print(f"{message.role}: {message.content}")
```

### System Messages

```python
from lionpride import Message
from lionpride.session import SystemContent

system_msg = Message(
    content=SystemContent(
        system="You are a helpful Python tutor. Always provide code examples."
    )
)

branch = session.create_branch(
    name="tutoring",
    system=system_msg
)
```

## Common Pitfalls

1. **Forgetting async/await**: All operations are async

   ```python
   # ❌ Wrong
   result = session.invoke(...)

   # ✅ Right
   result = await session.invoke(...)
   ```

2. **Missing API key**: Set environment variable before running

   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Wrong endpoint for provider**: Anthropic uses `"messages"`, OpenAI uses `"chat/completions"` (default)

   ```python
   # ✅ Anthropic
   iModel(provider="anthropic", endpoint="messages", ...)

   # ✅ OpenAI (endpoint optional, defaults to chat/completions)
   iModel(provider="openai", ...)
   ```

## Next Steps

- **Multi-turn conversations**: See [Multi-Turn Chat](multi_turn.md)
- **Structured outputs**: See [Structured Outputs](structured_outputs.md)
- **Tool calling**: See [Tool Calling](tool_calling.md)
- **Different providers**: See [Chat Examples](chat.md)

## See Also

- [API Reference: Session](../api/session.md)
- [API Reference: iModel](../api/services.md)
- [API Reference: Operations](../api/operations.md)
