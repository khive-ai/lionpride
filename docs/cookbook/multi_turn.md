# Multi-Turn Conversations with Branch

## Overview

This recipe demonstrates multi-turn conversations using lionpride's Branch system. A Branch represents a conversation thread with message history, system prompts, and context management.

## Prerequisites

```bash
pip install lionpride
```

## The Code

### Example 1: Simple Multi-Turn Chat

```python
import asyncio
from lionpride import Session, Message
from lionpride.services import iModel
from lionpride.operations import communicate
from lionpride.session import SystemContent

async def multi_turn_chat():
    """Multi-turn conversation with message history"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    session.services.register(model)

    # Create branch with system message
    system_msg = Message(
        content=SystemContent(
            system="You are a helpful Python tutor. Provide clear, "
                   "beginner-friendly explanations with code examples."
        )
    )

    branch = session.create_branch(
        name="tutoring",
        system=system_msg
    )

    # Turn 1: Ask about lists
    response1 = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "What are Python lists?",
            "imodel": model.name,
        }
    )
    print(f"Turn 1:\nUser: What are Python lists?")
    print(f"Assistant: {response1}\n")

    # Turn 2: Follow-up question (has context from turn 1)
    response2 = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "How do I add items to them?",
            "imodel": model.name,
        }
    )
    print(f"Turn 2:\nUser: How do I add items to them?")
    print(f"Assistant: {response2}\n")

    # Turn 3: Another follow-up
    response3 = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Show me an example with numbers",
            "imodel": model.name,
        }
    )
    print(f"Turn 3:\nUser: Show me an example with numbers")
    print(f"Assistant: {response3}\n")

    # View conversation history
    print(f"\nConversation has {len(branch)} messages")
    return session, branch

asyncio.run(multi_turn_chat())
```

### Example 2: Inspecting Message History

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def inspect_conversation():
    """Examine conversation history in detail"""
    session = Session()
    model = iModel(provider="anthropic", endpoint="messages",
                   model="claude-3-5-sonnet-20241022", temperature=0.7)
    session.services.register(model)

    branch = session.create_branch(name="inspection")

    # Have a short conversation
    await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "What is recursion?",
            "imodel": model.name,
        }
    )

    await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Give me a simple example",
            "imodel": model.name,
        }
    )

    # Inspect messages
    print("Conversation History:\n")
    for i, msg_id in enumerate(branch.order, 1):
        message = session.messages[msg_id]
        role = message.role.value

        # Extract content based on type
        if hasattr(message.content, 'instruction'):
            content = message.content.instruction
        elif hasattr(message.content, 'assistant_response'):
            content = message.content.assistant_response
        else:
            content = str(message.content)

        print(f"Message {i} [{role}]:")
        print(f"{content[:100]}...")  # First 100 chars
        print(f"Created: {message.created_at}")
        print(f"ID: {message.id}\n")

    return session, branch

asyncio.run(inspect_conversation())
```

### Example 3: Multiple Branches in One Session

```python
import asyncio
from lionpride import Session, Message
from lionpride.services import iModel
from lionpride.operations import communicate
from lionpride.session import SystemContent

async def multiple_branches():
    """Manage multiple conversation threads in one session"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    session.services.register(model)

    # Branch 1: Technical discussion
    tech_system = Message(
        content=SystemContent(
            system="You are a senior software engineer discussing architecture."
        )
    )
    tech_branch = session.create_branch(name="technical", system=tech_system)

    # Branch 2: Creative writing
    creative_system = Message(
        content=SystemContent(
            system="You are a creative writing coach helping with storytelling."
        )
    )
    creative_branch = session.create_branch(name="creative", system=creative_system)

    # Parallel conversations in different branches
    tech_response = await communicate(
        session=session,
        branch=tech_branch,
        parameters={
            "instruction": "Explain microservices architecture",
            "imodel": model.name,
        }
    )

    creative_response = await communicate(
        session=session,
        branch=creative_branch,
        parameters={
            "instruction": "Help me write an opening for a sci-fi story",
            "imodel": model.name,
        }
    )

    print("Technical Branch:")
    print(f"{tech_response}\n")

    print("Creative Branch:")
    print(f"{creative_response}\n")

    # Continue each conversation independently
    tech_response2 = await communicate(
        session=session,
        branch=tech_branch,
        parameters={
            "instruction": "What are the trade-offs?",
            "imodel": model.name,
        }
    )

    print(f"\nTechnical branch (turn 2):\n{tech_response2}")

    # Session tracks all branches
    print(f"\nSession has {len(session.branches)} branches")
    print(f"Total messages: {len(session.messages)}")

    return session

asyncio.run(multiple_branches())
```

### Example 4: Context Management

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

async def context_management():
    """Manage conversation context explicitly"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    session.services.register(model)

    branch = session.create_branch(name="context-demo")

    # Provide explicit context in each turn
    response1 = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Analyze this code for bugs",
            "context": {
                "code": """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
                """,
                "language": "Python"
            },
            "imodel": model.name,
        }
    )
    print(f"Analysis:\n{response1}\n")

    # Follow-up with additional context
    response2 = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "How would you fix it?",
            "context": {
                "note": "The function should handle empty lists gracefully"
            },
            "imodel": model.name,
        }
    )
    print(f"Fix:\n{response2}")

    return session, branch

asyncio.run(context_management())
```

### Example 5: Conversation Export and Resume

```python
import asyncio
import json
from lionpride import Session, Message
from lionpride.services import iModel
from lionpride.operations import communicate

async def save_and_resume():
    """Save conversation and resume later"""
    # === Part 1: Create and save conversation ===
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    session.services.register(model)

    branch = session.create_branch(name="persistent")

    await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": "Tell me about async Python",
            "imodel": model.name,
        }
    )

    # Export conversation
    conversation_data = {
        "branch_id": str(branch.id),
        "branch_name": branch.name,
        "messages": [
            {
                "id": str(msg_id),
                "message": session.messages[msg_id].to_dict()
            }
            for msg_id in branch.order
        ]
    }

    # Save to file
    with open("conversation.json", "w") as f:
        json.dump(conversation_data, f, default=str, indent=2)

    print("Conversation saved to conversation.json")

    # === Part 2: Resume conversation ===
    new_session = Session()
    new_model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    new_session.services.register(new_model)

    # Load conversation
    with open("conversation.json", "r") as f:
        loaded_data = json.load(f)

    # Reconstruct messages
    messages = []
    for msg_data in loaded_data["messages"]:
        msg = Message.from_dict(msg_data["message"])
        messages.append(msg)
        new_session.conversations.add_item(msg)

    # Recreate branch
    new_branch = new_session.create_branch(
        name=loaded_data["branch_name"],
        messages=[msg.id for msg in messages]
    )

    print(f"\nResumed conversation with {len(new_branch)} messages")

    # Continue conversation
    response = await communicate(
        session=new_session,
        branch=new_branch,
        parameters={
            "instruction": "Can you give me a practical example?",
            "imodel": new_model.name,
        }
    )

    print(f"\nContinued conversation:\n{response}")

    return new_session, new_branch

asyncio.run(save_and_resume())
```

## Expected Output

### Example 1 (Multi-Turn)

```
Turn 1:
User: What are Python lists?
Assistant: Python lists are ordered, mutable collections that can hold items
of any type. They're defined using square brackets: [1, 2, 3]...

Turn 2:
User: How do I add items to them?
Assistant: You can add items to lists using .append() for single items or
.extend() for multiple items. For example: my_list.append(4)...

Turn 3:
User: Show me an example with numbers
Assistant: Here's a complete example:
numbers = [1, 2, 3]
numbers.append(4)      # [1, 2, 3, 4]
numbers.extend([5, 6]) # [1, 2, 3, 4, 5, 6]

Conversation has 6 messages
```

### Example 2 (History Inspection)

```
Conversation History:

Message 1 [user]:
What is recursion?...
Created: 2025-01-15 10:30:00.123456+00:00
ID: 550e8400-e29b-41d4-a716-446655440000

Message 2 [assistant]:
Recursion is a programming technique where a function calls itself...
Created: 2025-01-15 10:30:02.789012+00:00
ID: 660e8400-e29b-41d4-a716-446655440001

[Additional messages...]
```

### Example 3 (Multiple Branches)

```
Technical Branch:
Microservices architecture is a design pattern where an application is
composed of small, independent services...

Creative Branch:
The stars had gone silent. Not dimmed, not distant—silent. Captain Sarah Chen
stared at the viewport, her reflection ghosting over the impossible darkness...

Technical branch (turn 2):
The main trade-offs of microservices include increased operational complexity,
network latency, data consistency challenges...

Session has 2 branches
Total messages: 6
```

## Key Concepts

### Branch Structure

```python
Branch
├── order: list[UUID]        # Message IDs in chronological order
├── system: UUID | None      # System message (first in order)
├── capabilities: set[str]   # Allowed structured outputs
├── resources: set[str]      # Allowed service resources
└── session_id: UUID         # Parent session

# Access messages via session
for msg_id in branch.order:
    message = session.messages[msg_id]
```

### Message Flow

```
communicate() → Creates Message → Adds to session.messages → Appends ID to branch.order
```

### Session Organization

```
Session
├── conversations: Flow[Message, Branch]
│   ├── messages (Pile[Message])      # All messages across all branches
│   └── branches (Pile[Branch])       # All conversation threads
└── services: ServiceRegistry
    └── models, tools
```

## Variations

### Clear Branch History

```python
# Option 1: Create new branch
new_branch = session.create_branch(name="fresh-start")

# Option 2: Manually clear (advanced)
branch.order.clear()  # Remove all message references
```

### Branch with Capabilities

```python
# Restrict structured outputs to specific formats
branch = session.create_branch(
    name="restricted",
    capabilities={"json", "lndl"}  # Only allow these output types
)
```

### Get Recent Messages

```python
# Last N messages
def get_recent_messages(session, branch, n=5):
    recent_ids = branch.order[-n:]
    return [session.messages[msg_id] for msg_id in recent_ids]

recent = get_recent_messages(session, branch, n=3)
```

### Branching from Existing Conversation

```python
# Fork conversation at specific point
original_branch = session.branches["main"]
fork_point = 5  # Message index

# Create new branch with history up to fork point
forked_branch = session.create_branch(
    name="alternative",
    messages=original_branch.order[:fork_point]
)

# Now conversations diverge
```

## Common Pitfalls

1. **Modifying branch.order directly**

   ```python
   # ❌ Wrong - breaks consistency
   branch.order.append(some_uuid)

   # ✅ Right - use communicate() to add messages
   await communicate(session=session, branch=branch, ...)
   ```

2. **Losing message references**

   ```python
   # ❌ Wrong - message not in session
   msg = Message(content=...)
   branch.order.append(msg.id)  # Broken reference!

   # ✅ Right - add to session first
   session.conversations.add_item(msg)
   branch.order.append(msg.id)
   ```

3. **Forgetting system message position**

   ```python
   # System message is always first (index 0)
   system_msg_id = branch.order[0] if branch.system else None
   ```

4. **Branch name conflicts**

   ```python
   # ❌ Wrong - duplicate names (allowed but confusing)
   branch1 = session.create_branch(name="chat")
   branch2 = session.create_branch(name="chat")  # Same name!

   # ✅ Right - unique names
   branch1 = session.create_branch(name="chat-1")
   branch2 = session.create_branch(name="chat-2")
   ```

## Next Steps

- **Tool calling with context**: See [Tool Calling](tool_calling.md)
- **Multi-agent conversations**: See [Multi-Agent](multi_agent.md)
- **Streaming responses**: See [Streaming](streaming.md)

## See Also

- [API Reference: Branch](../api/session.md#branch)
- [API Reference: Message](../api/session.md#message)
- [Conversation Patterns](../patterns/conversation.md)
