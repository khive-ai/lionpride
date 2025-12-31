# Mail

> Multi-entity communication primitives for groupchat and gossip patterns

## Overview

The `mail` module provides infrastructure for **entity-to-entity messaging** in
multi-agent systems. It enables patterns like groupchat, gossip protocols, and broadcast
communication through two core classes:

- **Mail**: A message envelope carrying content from sender to recipient
- **Exchange**: A router that manages entity mailboxes and synchronizes message delivery

**Use Mail/Exchange for**: Multi-agent coordination, groupchat patterns, broadcast
messaging, asynchronous entity communication, gossip protocols.

**Skip Mail/Exchange when**: Single-entity conversations (use `Session` directly),
synchronous request-response (use `Session.conduct()`).

See [Session](session.md) and [Branch](branch.md) for single-entity conversation
management.

## Architecture

```text
Exchange
  |
  +-- flows: Pile[Flow[Mail, Progression]]
        |
        +-- Flow (entity mailbox)
              |
              +-- items: Pile[Mail]           # All mail for this entity
              +-- progressions:
                    +-- "outbox"              # Pending outbound mail
                    +-- "inbox_{sender_id}"   # Inbound mail per sender
```

Each registered entity gets a `Flow[Mail, Progression]` as their mailbox. The Exchange
routes mail from outboxes to recipient inboxes during `sync()`.

---

## Mail

> Simple message envelope for entity-to-entity communication

### Class Signature

```python
from lionpride.session.mail import Mail

class Mail(Element):
    """A message that can be sent between communicatable entities."""

    def __init__(
        self,
        sender: UUID,                      # Required: sending entity
        recipient: UUID | None = None,     # None = broadcast
        content: Any = None,               # Message payload
        channel: str | None = None,        # Optional namespace
        **data: Any,                       # id, created_at, metadata
    ) -> None: ...
```

### Parameters

| Parameter   | Type              | Default | Description                                       |
| ----------- | ----------------- | ------- | ------------------------------------------------- |
| `sender`    | `UUID` (required) | -       | UUID of the sending entity                        |
| `recipient` | `UUID \| None`    | `None`  | Receiving entity UUID (`None` = broadcast to all) |
| `content`   | `Any`             | `None`  | Message payload (any serializable value)          |
| `channel`   | `str \| None`     | `None`  | Optional namespace for message grouping/filtering |

### Attributes

| Attribute    | Type             | Description                                        |
| ------------ | ---------------- | -------------------------------------------------- |
| `id`         | `UUID`           | Unique identifier (inherited from Element, frozen) |
| `created_at` | `datetime`       | UTC timestamp (inherited from Element, frozen)     |
| `metadata`   | `dict[str, Any]` | Arbitrary metadata (inherited from Element)        |
| `sender`     | `UUID`           | Sending entity's UUID                              |
| `recipient`  | `UUID \| None`   | Receiving entity's UUID (None for broadcast)       |
| `content`    | `Any`            | Message payload                                    |
| `channel`    | `str \| None`    | Message namespace                                  |

### Properties

| Property       | Type   | Description                       |
| -------------- | ------ | --------------------------------- |
| `is_broadcast` | `bool` | `True` if `recipient is None`     |
| `is_direct`    | `bool` | `True` if `recipient is not None` |

### Protocol Implementations

Inherits from Element: **Observable** (`id`), **Serializable** (`to_dict()`,
`to_json()`), **Deserializable** (`from_dict()`, `from_json()`), **Hashable**
(`__hash__()` by ID).

### Usage

```python
from lionpride.session.mail import Mail
from uuid import uuid4

alice, bob = uuid4(), uuid4()

# Direct message
dm = Mail(sender=alice, recipient=bob, content="Hello Bob!")
dm.is_direct    # True
dm.is_broadcast # False

# Broadcast message
broadcast = Mail(sender=alice, content="Hello everyone!")
broadcast.is_broadcast  # True

# With channel namespace
task_msg = Mail(sender=alice, recipient=bob, content={"task": "analyze"}, channel="tasks")
```

---

## Exchange

> Routes mail between entity mailboxes with async-safe synchronization

### Class Signature

```python
from lionpride.session.mail import Exchange, OUTBOX

class Exchange(Element):
    """Routes mail between entity Flows."""

    def __init__(
        self,
        flows: Pile[Flow[Mail, Progression]] | None = None,  # Auto-created
        **data: Any,  # id, created_at, metadata
    ) -> None: ...
```

### Parameters

| Parameter | Type         | Default      | Description                            |
| --------- | ------------ | ------------ | -------------------------------------- |
| `flows`   | `Pile[Flow]` | Auto-created | Pre-built flows (rarely used directly) |

### Attributes

| Attribute    | Type                            | Description                                        |
| ------------ | ------------------------------- | -------------------------------------------------- |
| `id`         | `UUID`                          | Unique identifier (inherited from Element, frozen) |
| `created_at` | `datetime`                      | UTC timestamp (inherited from Element, frozen)     |
| `metadata`   | `dict[str, Any]`                | Arbitrary metadata (inherited from Element)        |
| `flows`      | `Pile[Flow[Mail, Progression]]` | Entity mailboxes                                   |
| `owner_ids`  | `list[UUID]`                    | List of registered entity UUIDs (property)         |

### Methods

#### Entity Registration

##### `register()`

Create and register a mailbox Flow for an entity.

```python
def register(self, owner_id: UUID) -> Flow[Mail, Progression]
```

**Parameters:**

- `owner_id` (UUID): Entity's UUID

**Returns:**

- `Flow[Mail, Progression]`: The created mailbox Flow

**Raises:**

- `ValueError`: If owner already registered

```python
>>> exchange = Exchange()
>>> alice_flow = exchange.register(alice_id)
>>> alice_flow.name == str(alice_id)
True
```

##### `unregister()`

Remove an entity's mailbox.

```python
def unregister(self, owner_id: UUID) -> Flow[Mail, Progression] | None
```

**Returns:** The removed Flow, or `None` if not found.

##### `get()`

Get an entity's mailbox by owner ID.

```python
def get(self, owner_id: UUID) -> Flow[Mail, Progression] | None
```

##### `has()`

Check if an entity is registered.

```python
def has(self, owner_id: UUID) -> bool
```

#### Message Operations

##### `send()`

Create and queue mail from sender to recipient.

```python
def send(
    self,
    sender: UUID,
    recipient: UUID | None,
    content: Any,
    channel: str | None = None,
) -> Mail
```

**Parameters:**

- `sender` (UUID): Sending entity's UUID
- `recipient` (UUID | None): Receiving entity's UUID (None for broadcast)
- `content` (Any): Mail content
- `channel` (str | None): Optional channel/topic

**Returns:**

- `Mail`: The created Mail object

**Raises:**

- `ValueError`: If sender not registered

```python
>>> exchange.send(alice, bob, content="Hello!")
Mail(abc123... -> def456...)
```

##### `receive()`

Get pending inbound mail for an entity (does not remove them).

```python
def receive(
    self,
    owner_id: UUID,
    sender: UUID | None = None,
) -> list[Mail]
```

**Parameters:**

- `owner_id` (UUID): Entity to get mail for
- `sender` (UUID | None): If provided, only return mail from this sender

**Returns:**

- `list[Mail]`: Pending mail (non-destructive read)

```python
>>> messages = exchange.receive(bob, sender=alice)
>>> messages[0].content
"Hello!"
```

##### `pop_mail()`

Pop next mail from a specific sender's inbox (FIFO, destructive).

```python
def pop_mail(self, owner_id: UUID, sender: UUID) -> Mail | None
```

**Returns:** Next mail or `None` if inbox empty.

#### Synchronization (async)

##### `sync()` (async)

Synchronize all mailboxes - collect and route all pending mail.

```python
async def sync(self) -> int
```

**Returns:** Number of mail items routed.

**Concurrency**: Thread-safe, uses internal locking on flows.

```python
>>> await exchange.sync()
3  # 3 messages routed
```

##### `collect()` (async)

Collect outbound mail from a specific entity and route to recipients.

```python
async def collect(self, owner_id: UUID) -> int
```

**Parameters:**

- `owner_id` (UUID): Entity whose outbox to collect from

**Returns:** Number of mail items collected and routed.

**Raises:**

- `ValueError`: If owner not registered

##### `collect_all()` (async)

Collect and route mail from all entities.

```python
async def collect_all(self) -> int
```

**Returns:** Total number of mail items routed.

##### `run()` (async)

Run continuous sync loop at specified interval.

```python
async def run(self, interval: float = 1.0) -> None
```

**Parameters:**

- `interval` (float): Seconds between sync cycles

##### `stop()`

Stop the continuous run loop.

```python
def stop(self) -> None
```

### Protocol Implementations

Inherits from Element: **Observable** (`id`), **Serializable** (`to_dict()`,
`to_json()`), **Deserializable** (`from_dict()`, `from_json()`), **Hashable**
(`__hash__()` by ID).

**Container Protocol**: `len(exchange)` returns entity count, `owner_id in exchange`
checks registration.

---

## OUTBOX Constant

```python
from lionpride.session.mail import OUTBOX

OUTBOX = "outbox"  # Progression name for outbound mail
```

Used internally by Exchange for the outbox progression name. Each entity's Flow has an
"outbox" progression for pending outbound mail.

---

## Usage Patterns

### Basic: Two-Entity Communication

```python
import asyncio
from uuid import uuid4
from lionpride.session.mail import Mail, Exchange

async def main():
    alice_id, bob_id = uuid4(), uuid4()

    # Create exchange and register entities
    exchange = Exchange()
    exchange.register(alice_id)
    exchange.register(bob_id)

    # Alice sends to Bob
    exchange.send(alice_id, bob_id, content="Hello Bob!")

    # Sync routes the mail
    await exchange.sync()

    # Bob receives
    messages = exchange.receive(bob_id, sender=alice_id)
    print(messages[0].content)  # "Hello Bob!"

asyncio.run(main())
```

### Broadcast Messaging

```python
async def broadcast_example():
    alice, bob, charlie = uuid4(), uuid4(), uuid4()

    exchange = Exchange()
    for entity in [alice, bob, charlie]:
        exchange.register(entity)

    # Alice broadcasts to all
    exchange.send(alice, recipient=None, content="Announcement!")

    await exchange.sync()

    # Both Bob and Charlie receive the broadcast
    bob_mail = exchange.receive(bob, sender=alice)
    charlie_mail = exchange.receive(charlie, sender=alice)

    assert len(bob_mail) == 1
    assert len(charlie_mail) == 1
```

### Channel-Based Filtering

```python
async def channel_example():
    alice, bob = uuid4(), uuid4()

    exchange = Exchange()
    exchange.register(alice)
    exchange.register(bob)

    # Send on different channels
    exchange.send(alice, bob, content="General message", channel=None)
    exchange.send(alice, bob, content="Task assignment", channel="tasks")
    exchange.send(alice, bob, content="Debug info", channel="debug")

    await exchange.sync()

    # Filter by channel in application logic
    all_mail = exchange.receive(bob, sender=alice)
    task_mail = [m for m in all_mail if m.channel == "tasks"]
```

### Continuous Sync Loop

```python
import asyncio

async def continuous_sync():
    exchange = Exchange()
    # ... register entities ...

    # Start sync loop in background
    sync_task = asyncio.create_task(exchange.run(interval=0.5))

    try:
        # Your application logic
        await some_agent_workflow()
    finally:
        exchange.stop()
        await sync_task
```

### Integration with Session/Branch

```python
from lionpride import Session, iModel
from lionpride.session.mail import Exchange
from uuid import uuid4

async def multi_agent_session():
    # Each agent has a Session for LLM operations
    # and an Exchange ID for inter-agent communication
    agent_a_id = uuid4()
    agent_b_id = uuid4()

    exchange = Exchange()
    exchange.register(agent_a_id)
    exchange.register(agent_b_id)

    # Agent A conducts LLM operation, then sends result to Agent B
    session_a = Session(default_generate_model=iModel(provider="openai", model="gpt-4o-mini"))
    branch_a = session_a.create_branch(name="main")

    # ... conduct operation ...
    result = "Analysis complete"

    # Send to Agent B via Exchange
    exchange.send(agent_a_id, agent_b_id, content={"result": result, "task": "review"})
    await exchange.sync()

    # Agent B receives and processes
    incoming = exchange.pop_mail(agent_b_id, sender=agent_a_id)
    if incoming:
        task = incoming.content["task"]
        # ... process task ...
```

## Common Pitfalls

### 1. Forgetting to Call `sync()`

Mail is queued in outboxes until `sync()` routes it to recipients.

```python
# WRONG: Mail never delivered
exchange.send(alice, bob, content="Hello")
messages = exchange.receive(bob)  # Empty!

# FIX: Always sync after sending
exchange.send(alice, bob, content="Hello")
await exchange.sync()
messages = exchange.receive(bob)  # Has message
```

### 2. Unregistered Entities

Sending from or to unregistered entities fails silently or raises errors.

```python
# WRONG: Bob not registered, mail is dropped
exchange.register(alice)
exchange.send(alice, bob, content="Hello")  # bob not registered
await exchange.sync()  # Mail dropped (no recipient flow)

# FIX: Register all entities first
exchange.register(alice)
exchange.register(bob)
exchange.send(alice, bob, content="Hello")
```

### 3. Broadcast Content Mutation

Broadcast copies mail to each recipient. For mutable content, mutations affect all
copies if deep copy fails.

```python
# SAFER: Use immutable content for broadcasts
exchange.send(alice, recipient=None, content={"msg": "Hello"})  # dict is copied
```

### 4. Using `receive()` Instead of `pop_mail()` in Loops

`receive()` is non-destructive - calling it repeatedly returns the same messages.

```python
# WRONG: Infinite loop
while messages := exchange.receive(bob, sender=alice):
    process(messages[0])  # Same message forever

# FIX: Use pop_mail() for consumption
while mail := exchange.pop_mail(bob, sender=alice):
    process(mail)  # Removes from inbox
```

## Design Rationale

**Flow-Based Mailboxes**: Each entity's mailbox is a `Flow[Mail, Progression]`, enabling
O(1) lookup via Pile and ordered message access via progressions. This reuses
lionpride's core primitives rather than introducing new collections.

**Separate Inboxes Per Sender**: Messages are organized into `inbox_{sender_id}`
progressions, allowing efficient filtering by sender and preventing cross-sender
interference.

**Async-Safe Synchronization**: `sync()` uses internal locking to ensure thread-safe
operation. Phase 1 (collection) holds the lock, Phase 2 (delivery) releases it for
concurrent delivery to different recipients.

**Best-Effort Delivery**: If a recipient unregisters during delivery, mail is silently
dropped rather than raising errors. This follows the "fail gracefully" pattern for
distributed systems.

**Content Agnostic**: Mail doesn't impose structure on `content` - the receiver decides
interpretation. This enables flexible protocols without schema coupling.

## See Also

- [Session](session.md) - Central orchestrator for LLM operations
- [Branch](branch.md) - Conversation thread management
- [Message](message.md) - LLM message container
- [Flow](../core/flow.md) - Items + progressions collection
- [Element](../core/element.md) - Base class with UUID identity
