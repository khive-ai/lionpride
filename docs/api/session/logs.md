# Logs

> Structured logging system for API calls, operations, and conversation events

## Overview

The lionpride logging system provides **structured logging** for tracking API calls,
conversation events, operations, and errors. Built on the Pile-based storage pattern, it
offers O(1) UUID lookup and async-safe operations.

**Use Logs for**: API call tracking (model, tokens, duration), operation events, error
tracking, audit trails, multi-destination persistence.

**Key features**:

- Immutable log entries extending Element (UUID identity)
- Pile-backed storage with O(1) UUID lookup
- Async context manager for safe concurrent access
- Extensible via LogAdapter (SQLite WAL, PostgreSQL) and LogBroadcaster (S3, webhooks)

See [Session](session.md) for conversation orchestration and [Branch](branch.md) for
message management.

## LogType

Enumeration of log entry types.

```python
from lionpride.session.logs import LogType

class LogType(str, Enum):
    API_CALL = "api_call"      # LLM API call with timing/tokens
    MESSAGE = "message"        # Conversation message event
    OPERATION = "operation"    # Operation execution event
    ERROR = "error"            # Error occurrence
    WARNING = "warning"        # Warning event
    INFO = "info"              # Informational message
```

## Log

> Immutable log entry extending Element

### Class Signature

```python
from lionpride.session.logs import Log, LogType

class Log(Element):
    """Immutable log entry that extends Element."""

    def __init__(
        self,
        log_type: LogType,
        source: str = "",
        *,
        # API call fields
        model: str | None = None,
        provider: str | None = None,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        # General fields
        message: str | None = None,
        data: dict[str, Any] | None = None,
        error: str | None = None,
        **kwargs,
    ) -> None: ...
```

### Parameters

| Parameter       | Type      | Default  | Description                                     |
| --------------- | --------- | -------- | ----------------------------------------------- |
| `log_type`      | `LogType` | Required | Type of log entry (API_CALL, MESSAGE, etc.)     |
| `source`        | `str`     | `""`     | Source identifier (branch ID, operation name)   |
| `model`         | `str`     | `None`   | Model name for API calls                        |
| `provider`      | `str`     | `None`   | Provider name for API calls (openai, anthropic) |
| `request`       | `dict`    | `None`   | Request payload for API calls                   |
| `response`      | `dict`    | `None`   | Response payload for API calls                  |
| `duration_ms`   | `float`   | `None`   | Duration in milliseconds                        |
| `input_tokens`  | `int`     | `None`   | Input token count                               |
| `output_tokens` | `int`     | `None`   | Output token count                              |
| `total_tokens`  | `int`     | `None`   | Total token count                               |
| `message`       | `str`     | `None`   | General message text                            |
| `data`          | `dict`    | `None`   | Additional structured data                      |
| `error`         | `str`     | `None`   | Error message                                   |

### Attributes

| Attribute    | Type             | Description                                        |
| ------------ | ---------------- | -------------------------------------------------- |
| `id`         | `UUID`           | Unique identifier (inherited from Element, frozen) |
| `created_at` | `datetime`       | UTC timestamp (inherited from Element, frozen)     |
| `metadata`   | `dict[str, Any]` | Arbitrary metadata (inherited from Element)        |
| `log_type`   | `LogType`        | Type of this log entry                             |
| `source`     | `str`            | Source identifier                                  |

### Methods

#### `create()` (classmethod)

Create a new Log from content (Element or dict).

```python
@classmethod
def create(
    cls,
    content: Element | dict[str, Any],
    log_type: LogType = LogType.INFO,
) -> Log
```

**Parameters:**

- `content` (Element or dict): Content to wrap in a log
- `log_type` (LogType, default INFO): Type of log entry

**Returns:**

- `Log`: New Log instance

```python
>>> from lionpride.session.logs import Log, LogType
>>> from lionpride import Message

>>> msg = Message(content={"assistant_message": "Hello!"})
>>> log = Log.create(msg, log_type=LogType.MESSAGE)
>>> log.log_type
<LogType.MESSAGE: 'message'>
```

#### `from_dict()` (classmethod)

Create a Log from a dictionary. The restored log is marked as immutable.

```python
@classmethod
def from_dict(
    cls,
    data: dict[str, Any],
    meta_key: str | None = None,
    **kwargs,
) -> Log
```

**Note:** Logs restored from dict are immutable. Attempting to modify attributes raises
`AttributeError`.

### Immutability

Once created or restored from dict, logs are read-only:

```python
>>> log = Log(log_type=LogType.INFO, message="Test")
>>> log._immutable = True  # Automatically set by from_dict()

>>> log.message = "Modified"  # Raises AttributeError
AttributeError: This Log is immutable.
```

## LogStoreConfig

> Configuration for LogStore persistence

### Class Signature

```python
from lionpride.session.logs import LogStoreConfig

class LogStoreConfig(BaseModel):
    """Configuration for LogStore persistence."""

    persist_dir: str | Path = "./data/logs"
    file_prefix: str | None = None
    capacity: int | None = None
    extension: str = ".json"
    use_timestamp: bool = True
    auto_save_on_exit: bool = True
    clear_after_dump: bool = True
    use_adapter: bool = False
    use_broadcaster: bool = False
```

### Parameters

| Parameter           | Type          | Default         | Description                              |
| ------------------- | ------------- | --------------- | ---------------------------------------- |
| `persist_dir`       | `str \| Path` | `"./data/logs"` | Directory for log files                  |
| `file_prefix`       | `str`         | `None`          | Prefix for log filenames                 |
| `capacity`          | `int`         | `None`          | Max logs before auto-dump (None = 10000) |
| `extension`         | `str`         | `".json"`       | File extension (`.json` or `.jsonl`)     |
| `use_timestamp`     | `bool`        | `True`          | Include timestamp in filename            |
| `auto_save_on_exit` | `bool`        | `True`          | Dump logs on program exit                |
| `clear_after_dump`  | `bool`        | `True`          | Clear logs after dumping to file         |
| `use_adapter`       | `bool`        | `False`         | Using LogAdapter (set automatically)     |
| `use_broadcaster`   | `bool`        | `False`         | Using LogBroadcaster (set automatically) |

### Validation

- `capacity` must be non-negative if provided
- `extension` must be `.json` or `.jsonl` (auto-adds dot if missing)

## LogStore

> Log storage with Pile-based O(1) UUID lookup and async support

### Class Signature

```python
from lionpride.session.logs import LogStore, LogStoreConfig

class LogStore:
    """Log storage with Pile-based O(1) UUID lookup and async support."""

    def __init__(
        self,
        max_logs: int | None = None,
        config: LogStoreConfig | None = None,
        adapter: LogAdapter | None = None,
        broadcaster: LogBroadcaster | None = None,
        **kwargs,
    ) -> None: ...
```

### Parameters

| Parameter     | Type             | Default      | Description                                 |
| ------------- | ---------------- | ------------ | ------------------------------------------- |
| `max_logs`    | `int`            | `None`       | Maximum logs to keep (None = 10000, legacy) |
| `config`      | `LogStoreConfig` | Auto-created | Persistence configuration                   |
| `adapter`     | `LogAdapter`     | `None`       | Adapter for persistent storage              |
| `broadcaster` | `LogBroadcaster` | `None`       | Broadcaster for multi-destination output    |

### Attributes

| Attribute | Type        | Description                           |
| --------- | ----------- | ------------------------------------- |
| `logs`    | `Pile[Log]` | Underlying Pile for direct operations |

### Methods

#### Core Logging Methods

##### `add()`

Add a log entry synchronously. Auto-dumps if capacity reached.

```python
def add(self, log: Log) -> None
```

##### `alog()` (async)

Add a log asynchronously with proper locking.

```python
async def alog(self, log: Log | Any) -> None
```

**Concurrency**: Thread-safe via Pile async context manager

```python
>>> store = LogStore()
>>> log = Log(log_type=LogType.INFO, message="Hello")
>>> await store.alog(log)
```

##### `log_api_call()`

Log an API call with timing and token metrics.

```python
def log_api_call(
    self,
    *,
    source: str = "",
    model: str | None = None,
    provider: str | None = None,
    request: dict[str, Any] | None = None,
    response: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> Log
```

```python
>>> store = LogStore()
>>> log = store.log_api_call(
...     model="gpt-4o-mini",
...     provider="openai",
...     duration_ms=150.5,
...     input_tokens=100,
...     output_tokens=50,
...     total_tokens=150,
... )
>>> log.model
'gpt-4o-mini'
```

##### `log_operation()`

Log an operation event.

```python
def log_operation(
    self,
    *,
    source: str = "",
    message: str = "",
    data: dict[str, Any] | None = None,
) -> Log
```

##### `log_error()`

Log an error.

```python
def log_error(
    self,
    *,
    source: str = "",
    error: str = "",
    data: dict[str, Any] | None = None,
) -> Log
```

##### `log_info()`

Log an info message.

```python
def log_info(
    self,
    *,
    source: str = "",
    message: str = "",
    data: dict[str, Any] | None = None,
) -> Log
```

#### Filtering and Querying

##### `filter()`

Filter logs by multiple criteria.

```python
def filter(
    self,
    *,
    log_type: LogType | None = None,
    source: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    model: str | None = None,
) -> list[Log]
```

**Parameters:**

- `log_type` (LogType): Filter by log type
- `source` (str): Filter by source (substring match)
- `since` (datetime): Filter by created_at >= since
- `until` (datetime): Filter by created_at <= until
- `model` (str): Filter by model (substring match)

**Returns:**

- `list[Log]`: Filtered list of logs

```python
>>> from datetime import datetime, UTC, timedelta

>>> store = LogStore()
>>> store.log_api_call(model="gpt-4o", source="main_branch")
>>> store.log_api_call(model="claude-3", source="experiment")
>>> store.log_error(error="Connection timeout", source="main_branch")

# Filter by type
>>> api_logs = store.filter(log_type=LogType.API_CALL)
>>> len(api_logs)
2

# Filter by source (substring match)
>>> main_logs = store.filter(source="main")
>>> len(main_logs)
2

# Filter by time range
>>> recent = store.filter(since=datetime.now(UTC) - timedelta(hours=1))

# Combine filters
>>> gpt_logs = store.filter(log_type=LogType.API_CALL, model="gpt")
>>> len(gpt_logs)
1
```

##### `get_api_calls()`

Get all API call logs.

```python
def get_api_calls(self) -> list[Log]
```

##### `get_errors()`

Get all error logs.

```python
def get_errors(self) -> list[Log]
```

#### Export and Persistence

##### `to_list()`

Export all logs as list of dicts.

```python
def to_list(self) -> list[dict[str, Any]]
```

##### `dump()`

Dump logs to JSON file synchronously.

```python
def dump(
    self,
    path: str | Path | None = None,
    *,
    clear: bool | None = None,
) -> int
```

**Parameters:**

- `path` (str or Path): File path (uses config if not provided)
- `clear` (bool): Clear logs after dump (uses config default if None)

**Returns:**

- `int`: Number of logs dumped

```python
>>> store = LogStore()
>>> store.log_info(message="Test log")
>>> count = store.dump("./logs/session.json")
>>> count
1
```

##### `adump()` (async)

Asynchronously dump logs to file with proper locking.

```python
async def adump(
    self,
    path: str | Path | None = None,
    *,
    clear: bool | None = None,
) -> int
```

**Concurrency**: Thread-safe via Pile async context manager

```python
>>> count = await store.adump("./logs/session.json", clear=True)
```

##### `aflush()` (async)

Flush logs to adapter and/or broadcaster.

```python
async def aflush(self, *, clear: bool = True) -> dict[str, int]
```

**Returns:**

- `dict`: Counts per destination: `{"adapter": N, "broadcaster": {"sub1": N, ...}}`

**Note:** Logs are only cleared if ALL destinations succeed (prevents data loss on
partial failure).

```python
>>> from lionpride.session.log_adapter import SQLiteWALLogAdapter

>>> store = LogStore()
>>> store.set_adapter(SQLiteWALLogAdapter(db_path="./logs.db"))
>>> store.log_info(message="Important event")
>>> results = await store.aflush()
>>> results
{'adapter': 1}
```

#### Adapter and Broadcaster Integration

##### `set_adapter()`

Set the log adapter for persistent storage.

```python
def set_adapter(self, adapter: LogAdapter) -> None
```

##### `set_broadcaster()`

Set the log broadcaster for multi-destination output.

```python
def set_broadcaster(self, broadcaster: LogBroadcaster) -> None
```

#### Statistics and Utility

##### `summary()`

Get summary statistics of stored logs.

```python
def summary(self) -> dict[str, Any]
```

**Returns:**

- `dict`: Summary with keys: `total_logs`, `api_calls`, `errors`, `total_tokens`,
  `total_duration_ms`, `models_used`

```python
>>> store = LogStore()
>>> store.log_api_call(model="gpt-4o", total_tokens=500, duration_ms=200)
>>> store.log_api_call(model="gpt-4o", total_tokens=300, duration_ms=150)
>>> store.log_error(error="Rate limit hit")

>>> store.summary()
{
    'total_logs': 3,
    'api_calls': 2,
    'errors': 1,
    'total_tokens': 800,
    'total_duration_ms': 350.0,
    'models_used': ['gpt-4o']
}
```

##### `clear()`

Clear all logs.

```python
def clear(self) -> int
```

**Returns:**

- `int`: Count of cleared logs

#### Container Protocol

LogStore supports standard container operations:

```python
>>> len(store)              # Number of logs
>>> for log in store: ...   # Iterate over logs
>>> store[log.id]           # Get log by UUID
>>> store[0]                # Get log by index
```

## Protocol Implementations

**Log** inherits from Element: **Observable** (`id`), **Serializable** (`to_dict()`,
`to_json()`), **Deserializable** (`from_dict()`, `from_json()`), **Hashable**
(`__hash__()` by ID).

## Usage Patterns

### Basic Logging

```python
from lionpride.session.logs import LogStore, LogType

store = LogStore()

# Log API calls
store.log_api_call(
    model="gpt-4o-mini",
    provider="openai",
    duration_ms=150.0,
    input_tokens=100,
    output_tokens=50,
)

# Log operations
store.log_operation(source="chat_branch", message="Branch forked")

# Log errors
store.log_error(source="api_call", error="Rate limit exceeded")

# Get summary
print(store.summary())
```

### Async Usage with Locking

```python
import asyncio
from lionpride.session.logs import LogStore, Log, LogType

async def concurrent_logging():
    store = LogStore()

    async def log_worker(worker_id: int):
        for i in range(10):
            log = Log(
                log_type=LogType.INFO,
                source=f"worker_{worker_id}",
                message=f"Task {i} completed",
            )
            await store.alog(log)

    # Safe concurrent logging
    await asyncio.gather(*[log_worker(i) for i in range(5)])

    # Safe concurrent dump
    await store.adump("./logs/concurrent.json")
```

### Filtering and Analysis

```python
from datetime import datetime, UTC, timedelta

store = LogStore()

# ... add logs ...

# Filter by type
api_logs = store.filter(log_type=LogType.API_CALL)

# Filter by time window
one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
recent_logs = store.filter(since=one_hour_ago)

# Filter by model
gpt4_logs = store.filter(log_type=LogType.API_CALL, model="gpt-4")

# Analyze token usage
total_tokens = sum(log.total_tokens or 0 for log in api_logs)
avg_duration = (
    sum(log.duration_ms or 0 for log in api_logs) / len(api_logs)
    if api_logs else 0
)
```

### SQLite WAL Adapter Integration

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_adapter import SQLiteWALLogAdapter

# Create store with SQLite adapter
adapter = SQLiteWALLogAdapter(db_path="./data/logs.db", wal_mode=True)
store = LogStore()
store.set_adapter(adapter)

# Log as usual
store.log_api_call(model="gpt-4o", duration_ms=200)

# Flush to database
await store.aflush()

# Read back from adapter
logs = await adapter.read(limit=100, log_type="api_call")

# Cleanup
await adapter.close()
```

### PostgreSQL Adapter Integration

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_adapter import PostgresLogAdapter

# Create store with PostgreSQL adapter
adapter = PostgresLogAdapter(
    dsn="postgresql://user:pass@localhost/mydb",
    table="app_logs",
    auto_create=True,
)
store = LogStore()
store.set_adapter(adapter)

# Log and flush
store.log_error(error="Connection failed", source="db_pool")
await store.aflush()

# Query with filters
recent_errors = await adapter.read(
    limit=50,
    log_type="error",
    since="2025-01-01T00:00:00Z",
)

await adapter.close()
```

### Multi-Destination Broadcasting

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    S3LogSubscriber,
    PostgresLogSubscriber,
    WebhookLogSubscriber,
    LogRedactor,
)

# Create broadcaster with redaction (removes API keys, passwords)
broadcaster = LogBroadcaster(redactor=LogRedactor())

# Add S3 subscriber (credentials via environment variables)
broadcaster.add_subscriber(
    S3LogSubscriber(
        bucket="my-logs",
        prefix="app/",
        aws_access_key_id_env="AWS_ACCESS_KEY_ID",
        aws_secret_access_key_env="AWS_SECRET_ACCESS_KEY",
    )
)

# Add PostgreSQL subscriber
broadcaster.add_subscriber(
    PostgresLogSubscriber(dsn="postgresql://user:pass@localhost/logs")
)

# Add webhook subscriber (SSRF-protected)
broadcaster.add_subscriber(
    WebhookLogSubscriber(
        url="https://logs.example.com/ingest",
        headers={"Authorization": "Bearer token"},
    )
)

# Configure store with broadcaster
store = LogStore()
store.set_broadcaster(broadcaster)

# Log and broadcast to all destinations
store.log_api_call(model="gpt-4o", total_tokens=500)
results = await store.aflush()
# results = {"s3:my-logs": 1, "postgres:logs": 1, "webhook:https://...": 1}

await broadcaster.close()
```

### Session Integration

```python
from lionpride import Session, iModel
from lionpride.session.logs import LogStore

# Session with custom log store
log_store = LogStore(max_logs=5000)
session = Session(
    default_generate_model=iModel(provider="openai", model="gpt-4o-mini"),
    default_branch="main",
)

# Session conducts operations (logs automatically generated)
# Access logs via session.logs (if integrated)

# Manual log access
summary = log_store.summary()
errors = log_store.get_errors()
```

## Common Pitfalls

### 1. Forgetting Async Locking for Concurrent Access

LogStore's `add()` is not thread-safe. Use `alog()` for concurrent scenarios.

```python
# WRONG: Race conditions possible
store.add(log)

# CORRECT: Async-safe with locking
await store.alog(log)
```

### 2. Modifying Restored Logs

Logs restored from `from_dict()` are immutable.

```python
# WRONG: Raises AttributeError
log = Log.from_dict(data)
log.message = "New message"

# CORRECT: Create new log
new_log = Log(log_type=log.log_type, message="New message")
```

### 3. Not Handling Partial Flush Failures

When using adapter + broadcaster, `aflush()` only clears logs on full success.

```python
# Results indicate success/failure per destination
results = await store.aflush()
if results.get("adapter", 0) == 0:
    print("Adapter write failed, logs preserved")
```

### 4. Ignoring Capacity Auto-Dump

When capacity is reached, logs auto-dump to file. Configure appropriately:

```python
# Prevent auto-dump (infinite capacity)
store = LogStore(config=LogStoreConfig(capacity=None))

# Or set appropriate capacity
store = LogStore(max_logs=50000)
```

## Design Rationale

**Pile-Backed Storage**: O(1) UUID lookup via underlying Pile, enabling efficient log
retrieval by ID while maintaining insertion order.

**Immutable Logs**: Logs are immutable once created or restored. This ensures log
integrity for audit trails and prevents accidental modification.

**Async Context Manager**: The `alog()` and `adump()` methods use Pile's async context
manager for proper locking, enabling safe concurrent access without manual lock
management.

**Adapter/Broadcaster Pattern**: Separates concerns - LogStore handles in-memory
operations, LogAdapter handles persistence, LogBroadcaster handles distribution. Compose
as needed.

**Auto-Dump on Capacity**: Prevents memory exhaustion in long-running applications by
automatically persisting logs when capacity is reached.

## See Also

- [Session](session.md) - Central orchestrator using LogStore
- [Branch](branch.md) - Branch management with logging
- [LogAdapter](log_adapter.md) - SQLite WAL and PostgreSQL adapters
- [LogBroadcaster](log_broadcaster.md) - Multi-destination broadcasting
- [Element](../core/element.md) - Base class for Log
- [Pile](../core/pile.md) - Underlying collection for LogStore

## Examples

### Complete Logging Pipeline

```python
from lionpride.session.logs import LogStore, LogStoreConfig, LogType
from lionpride.session.log_adapter import SQLiteWALLogAdapter
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    LogBroadcasterConfig,
    S3LogSubscriber,
    LogRedactor,
)

# Configure store with all options
config = LogStoreConfig(
    persist_dir="./data/logs",
    file_prefix="app",
    capacity=10000,
    auto_save_on_exit=True,
)

store = LogStore(config=config)

# Add SQLite adapter for local persistence
adapter = SQLiteWALLogAdapter(db_path="./data/logs.db")
store.set_adapter(adapter)

# Add broadcaster for cloud backup with redaction
broadcaster = LogBroadcaster(
    config=LogBroadcasterConfig(parallel=True, fail_fast=False),
    redactor=LogRedactor(),  # Removes API keys, passwords
)
broadcaster.add_subscriber(
    S3LogSubscriber(
        bucket="app-logs",
        aws_access_key_id_env="AWS_KEY",
        aws_secret_access_key_env="AWS_SECRET",
    )
)
store.set_broadcaster(broadcaster)

# Log various events
store.log_api_call(
    model="gpt-4o",
    provider="openai",
    duration_ms=250.5,
    input_tokens=150,
    output_tokens=80,
    total_tokens=230,
    source="chat_operation",
)

store.log_operation(
    source="session_manager",
    message="Branch forked for A/B testing",
    data={"parent_branch": "main", "child_branch": "experiment_a"},
)

store.log_info(
    source="system",
    message="Cache warmed successfully",
)

# Flush to all destinations
results = await store.aflush()
print(f"Adapter: {results.get('adapter')} logs")
print(f"Broadcaster: {results.get('broadcaster')}")

# Get final summary
print(store.summary())

# Cleanup
await adapter.close()
await broadcaster.close()
```

### Serialization Roundtrip

```python
from lionpride.session.logs import Log, LogType

# Create log
log = Log(
    log_type=LogType.API_CALL,
    model="claude-3-sonnet",
    duration_ms=180.0,
    total_tokens=250,
)

# Serialize to JSON
log_dict = log.to_dict(mode="json")

# Restore (immutable)
restored = Log.from_dict(log_dict)
assert restored.model == log.model
assert restored.total_tokens == log.total_tokens
```
