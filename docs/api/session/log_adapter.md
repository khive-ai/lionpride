# Log Adapters

> Async storage adapters for persistent log storage (SQLite WAL and PostgreSQL)

## Overview

Log adapters provide **async persistence backends** for the lionpride logging system.
They enable durable storage of logs beyond in-memory Pile-backed collections, supporting
both local development (SQLite WAL) and production deployments (PostgreSQL).

**Use Log Adapters for**: Persistent log storage, durable audit trails, cross-session
log analysis, production log infrastructure.

**Key features**:

- Abstract `LogAdapter` protocol for custom implementations
- SQLite WAL mode for concurrent read/write with crash safety
- PostgreSQL with JSONB storage and GIN indexing for flexible queries
- Lazy initialization (connection created on first use)
- SQL injection protection via identifier validation

```text
+-------------------+      +-----------------+      +-------------------+
|     LogStore      | ---> |   LogAdapter    | ---> | Storage Backend   |
| (in-memory Pile)  |      | (abstract)      |      | (SQLite/Postgres) |
+-------------------+      +-----------------+      +-------------------+
         |                        ^
         |                        |
         v                        |
   set_adapter()           +------+------+
                           |             |
                  SQLiteWALLogAdapter  PostgresLogAdapter
```

See [Logs](logs.md) for LogStore usage and [LogBroadcaster](log_broadcaster.md) for
multi-destination output.

## LogAdapterConfig

> Configuration for log adapters

### Class Signature

```python
from lionpride.session.log_adapter import LogAdapterConfig

class LogAdapterConfig(BaseModel):
    """Configuration for log adapters."""

    # SQLite settings
    sqlite_path: str | Path = "./data/logs.db"
    sqlite_wal_mode: bool = True

    # PostgreSQL settings
    postgres_dsn: str | None = None
    postgres_table: str = "logs"

    # Common settings
    batch_size: int = 100  # 1-1000
    auto_create_table: bool = True
```

### Parameters

| Parameter           | Type          | Default            | Description                     |
| ------------------- | ------------- | ------------------ | ------------------------------- |
| `sqlite_path`       | `str \| Path` | `"./data/logs.db"` | Path to SQLite database file    |
| `sqlite_wal_mode`   | `bool`        | `True`             | Enable WAL mode for SQLite      |
| `postgres_dsn`      | `str`         | `None`             | PostgreSQL connection string    |
| `postgres_table`    | `str`         | `"logs"`           | Table name for PostgreSQL logs  |
| `batch_size`        | `int`         | `100`              | Batch size for writes (1-1000)  |
| `auto_create_table` | `bool`        | `True`             | Auto-create table if not exists |

### Validation

- `batch_size` must be between 1 and 1000 (inclusive)

## LogAdapter

> Abstract base class for log storage adapters

### Class Signature

```python
from lionpride.session.log_adapter import LogAdapter

class LogAdapter(ABC):
    """Abstract base class for log storage adapters.

    Adapters provide async read/write interface for log persistence.
    """

    @abstractmethod
    async def write(self, logs: list[Log]) -> int: ...

    @abstractmethod
    async def read(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        log_type: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def close(self) -> None: ...
```

### Methods

#### `write()` (async, abstract)

Write logs to storage.

**Signature:**

```python
async def write(self, logs: list[Log]) -> int
```

**Parameters:**

- `logs` (list[Log]): List of Log objects to persist

**Returns:**

- `int`: Number of logs written

#### `read()` (async, abstract)

Read logs from storage with optional filtering.

**Signature:**

```python
async def read(
    self,
    *,
    limit: int = 100,
    offset: int = 0,
    log_type: str | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]
```

**Parameters:**

- `limit` (int, default 100): Maximum logs to return
- `offset` (int, default 0): Skip this many logs (for pagination)
- `log_type` (str, optional): Filter by log type (e.g., "api_call", "error")
- `since` (str, optional): Filter by created_at >= since (ISO 8601 format)

**Returns:**

- `list[dict]`: List of log dictionaries

#### `close()` (async, abstract)

Close the adapter and release resources.

**Signature:**

```python
async def close(self) -> None
```

## SQLiteWALLogAdapter

> SQLite WAL-mode adapter for local log storage

### Overview

SQLiteWALLogAdapter uses SQLite with **Write-Ahead Logging (WAL)** mode for improved
concurrent access. WAL mode allows readers and writers to operate simultaneously without
blocking, making it ideal for applications with mixed read/write workloads.

**WAL Mode Benefits**:

- Concurrent reads during writes (no reader blocking)
- Better write performance for small transactions
- Crash recovery without database corruption
- Reduced disk I/O via sequential writes

**Dependencies**: Requires `aiosqlite` (optional dependency).

```bash
pip install aiosqlite
```

### Class Signature

```python
from lionpride.session.log_adapter import SQLiteWALLogAdapter

class SQLiteWALLogAdapter(LogAdapter):
    """SQLite WAL-mode adapter for local log storage."""

    def __init__(
        self,
        db_path: str | Path = "./data/logs.db",
        wal_mode: bool = True,
        auto_create: bool = True,
    ) -> None: ...
```

### Parameters

| Parameter     | Type          | Default            | Description                     |
| ------------- | ------------- | ------------------ | ------------------------------- |
| `db_path`     | `str \| Path` | `"./data/logs.db"` | Path to SQLite database file    |
| `wal_mode`    | `bool`        | `True`             | Enable WAL mode                 |
| `auto_create` | `bool`        | `True`             | Auto-create table if not exists |

### Attributes

| Attribute      | Type                   | Description                     |
| -------------- | ---------------------- | ------------------------------- |
| `db_path`      | `Path`                 | Resolved path to database file  |
| `wal_mode`     | `bool`                 | Whether WAL mode is enabled     |
| `auto_create`  | `bool`                 | Whether to auto-create table    |
| `_connection`  | `aiosqlite.Connection` | Database connection (lazy init) |
| `_initialized` | `bool`                 | Whether adapter is initialized  |

### Database Schema

The adapter creates a general-purpose schema with indexed fields for common queries:

```sql
CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    log_type TEXT NOT NULL,
    source TEXT,
    content TEXT NOT NULL,      -- Full log as JSON
    metadata TEXT
);

-- Indexes for efficient filtering
CREATE INDEX idx_logs_created_at ON logs(created_at);
CREATE INDEX idx_logs_type ON logs(log_type);
CREATE INDEX idx_logs_source ON logs(source);
```

### Methods

#### `write()` (async)

Write logs to SQLite database.

**Signature:**

```python
async def write(self, logs: list[Log]) -> int
```

**Parameters:**

- `logs` (list[Log]): List of Log objects to persist

**Returns:**

- `int`: Number of logs written

**Note:** Uses `INSERT OR REPLACE` for upsert behavior (updates existing logs by ID).

```python
>>> from lionpride.session.log_adapter import SQLiteWALLogAdapter
>>> from lionpride.session.logs import Log, LogType

>>> adapter = SQLiteWALLogAdapter(db_path="./logs.db")
>>> log = Log(log_type=LogType.INFO, message="Test message")
>>> count = await adapter.write([log])
>>> count
1
```

#### `read()` (async)

Read logs from SQLite with optional filtering.

**Signature:**

```python
async def read(
    self,
    *,
    limit: int = 100,
    offset: int = 0,
    log_type: str | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]
```

**Parameters:**

- `limit` (int, default 100): Maximum logs to return
- `offset` (int, default 0): Skip this many logs
- `log_type` (str, optional): Filter by log type
- `since` (str, optional): Filter by created_at >= since (ISO 8601)

**Returns:**

- `list[dict]`: List of log dictionaries, ordered by created_at descending

**Note:** Corrupted JSON entries are silently skipped during read.

```python
>>> logs = await adapter.read(limit=50, log_type="api_call")
>>> len(logs)
42

>>> # Paginated read
>>> page1 = await adapter.read(limit=10, offset=0)
>>> page2 = await adapter.read(limit=10, offset=10)
```

#### `close()` (async)

Close SQLite connection and release resources.

**Signature:**

```python
async def close(self) -> None
```

```python
>>> await adapter.close()
>>> adapter._initialized
False
```

### Usage Examples

#### Basic SQLite Logging

```python
from lionpride.session.logs import LogStore, Log, LogType
from lionpride.session.log_adapter import SQLiteWALLogAdapter

# Create adapter with WAL mode
adapter = SQLiteWALLogAdapter(
    db_path="./data/logs.db",
    wal_mode=True,
)

# Create store and attach adapter
store = LogStore()
store.set_adapter(adapter)

# Log events
store.log_api_call(
    model="gpt-4o-mini",
    provider="openai",
    duration_ms=150.0,
    input_tokens=100,
    output_tokens=50,
)

store.log_info(message="Operation completed", source="main")

# Flush to SQLite
results = await store.aflush()
print(f"Wrote {results.get('adapter', 0)} logs to SQLite")

# Read back from adapter
logs = await adapter.read(limit=10, log_type="api_call")
for log in logs:
    print(f"{log['model']}: {log.get('duration_ms')}ms")

# Cleanup
await adapter.close()
```

#### Direct Adapter Usage

```python
from lionpride.session.log_adapter import SQLiteWALLogAdapter
from lionpride.session.logs import Log, LogType

# Create and initialize adapter
adapter = SQLiteWALLogAdapter(db_path="./audit.db")

# Create logs
logs = [
    Log(log_type=LogType.API_CALL, model="claude-3", duration_ms=200.0),
    Log(log_type=LogType.API_CALL, model="gpt-4o", duration_ms=180.0),
    Log(log_type=LogType.ERROR, error="Rate limit hit"),
]

# Write batch
count = await adapter.write(logs)
print(f"Wrote {count} logs")

# Query with filters
api_calls = await adapter.read(log_type="api_call")
errors = await adapter.read(log_type="error")
recent = await adapter.read(since="2025-01-01T00:00:00Z", limit=50)

await adapter.close()
```

## PostgresLogAdapter

> PostgreSQL adapter for production log storage

### Overview

PostgresLogAdapter provides production-grade log storage using PostgreSQL with JSONB
columns and GIN indexing. It uses SQLAlchemy's async engine with asyncpg for high
performance.

**Key Features**:

- JSONB storage for flexible log content
- GIN index for efficient JSON queries
- Async connection pooling via SQLAlchemy
- SQL injection protection via identifier validation
- Automatic DSN conversion (postgresql:// to postgresql+asyncpg://)

**Dependencies**: Requires `sqlalchemy[asyncio]` and `asyncpg`.

```bash
pip install 'sqlalchemy[asyncio]' asyncpg
```

### Class Signature

```python
from lionpride.session.log_adapter import PostgresLogAdapter

class PostgresLogAdapter(LogAdapter):
    """PostgreSQL adapter for production log storage."""

    def __init__(
        self,
        dsn: str,
        table: str = "logs",
        auto_create: bool = True,
    ) -> None: ...
```

### Parameters

| Parameter     | Type   | Default  | Description                                  |
| ------------- | ------ | -------- | -------------------------------------------- |
| `dsn`         | `str`  | Required | PostgreSQL connection string                 |
| `table`       | `str`  | `"logs"` | Table name (validated against SQL injection) |
| `auto_create` | `bool` | `True`   | Auto-create table if not exists              |

### Attributes

| Attribute      | Type          | Description                                     |
| -------------- | ------------- | ----------------------------------------------- |
| `dsn`          | `str`         | Connection string (converted to asyncpg format) |
| `table`        | `str`         | Validated table name                            |
| `auto_create`  | `bool`        | Whether to auto-create table                    |
| `_engine`      | `AsyncEngine` | SQLAlchemy async engine (lazy init)             |
| `_initialized` | `bool`        | Whether adapter is initialized                  |

### Database Schema

The adapter creates a schema optimized for flexible log storage:

```sql
CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    log_type VARCHAR(50) NOT NULL,
    source TEXT,
    content JSONB NOT NULL,     -- Full log as JSONB
    metadata JSONB
);

-- B-tree indexes for common filters
CREATE INDEX idx_logs_created_at ON logs(created_at);
CREATE INDEX idx_logs_type ON logs(log_type);
CREATE INDEX idx_logs_source ON logs(source);

-- GIN index for JSONB content queries
CREATE INDEX idx_logs_content ON logs USING GIN (content);
```

### SQL Injection Protection

Table names are validated against a strict pattern to prevent SQL injection:

- Must start with letter or underscore
- Alphanumeric and underscore only
- Maximum 63 characters (PostgreSQL limit)
- SQL keywords are rejected (SELECT, DROP, etc.)

```python
# Valid table names
adapter = PostgresLogAdapter(dsn=dsn, table="app_logs")      # OK
adapter = PostgresLogAdapter(dsn=dsn, table="logs_2025")     # OK
adapter = PostgresLogAdapter(dsn=dsn, table="_audit_trail")  # OK

# Invalid table names (raises ValueError)
adapter = PostgresLogAdapter(dsn=dsn, table="logs; DROP TABLE users")  # Rejected
adapter = PostgresLogAdapter(dsn=dsn, table="SELECT")                   # Rejected
adapter = PostgresLogAdapter(dsn=dsn, table="")                         # Rejected
```

### Methods

#### `write()` (async)

Write logs to PostgreSQL database.

**Signature:**

```python
async def write(self, logs: list[Log]) -> int
```

**Parameters:**

- `logs` (list[Log]): List of Log objects to persist

**Returns:**

- `int`: Number of logs written

**Raises:**

- `Exception`: On database write failure (logged and re-raised)

**Note:** Uses `ON CONFLICT DO UPDATE` for upsert behavior.

```python
>>> from lionpride.session.log_adapter import PostgresLogAdapter
>>> from lionpride.session.logs import Log, LogType

>>> adapter = PostgresLogAdapter(
...     dsn="postgresql://user:pass@localhost/mydb",
...     table="app_logs",
... )
>>> log = Log(log_type=LogType.API_CALL, model="gpt-4o", duration_ms=180.0)
>>> count = await adapter.write([log])
>>> count
1
```

#### `read()` (async)

Read logs from PostgreSQL with optional filtering.

**Signature:**

```python
async def read(
    self,
    *,
    limit: int = 100,
    offset: int = 0,
    log_type: str | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]
```

**Parameters:**

- `limit` (int, default 100): Maximum logs to return
- `offset` (int, default 0): Skip this many logs
- `log_type` (str, optional): Filter by log type
- `since` (str, optional): Filter by created_at >= since (ISO 8601, converted to
  datetime)

**Returns:**

- `list[dict]`: List of log dictionaries, ordered by created_at descending

**Raises:**

- `Exception`: On database read failure (logged and re-raised)

```python
>>> # Read recent API calls
>>> logs = await adapter.read(log_type="api_call", limit=50)

>>> # Read logs since timestamp
>>> logs = await adapter.read(since="2025-01-01T00:00:00Z")

>>> # Paginated read
>>> page1 = await adapter.read(limit=20, offset=0)
>>> page2 = await adapter.read(limit=20, offset=20)
```

#### `close()` (async)

Close PostgreSQL engine and release connection pool.

**Signature:**

```python
async def close(self) -> None
```

```python
>>> await adapter.close()
>>> adapter._initialized
False
```

### Usage Examples

#### Basic PostgreSQL Logging

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_adapter import PostgresLogAdapter

# Create adapter
adapter = PostgresLogAdapter(
    dsn="postgresql://user:password@localhost:5432/mydb",
    table="app_logs",
    auto_create=True,
)

# Create store and attach adapter
store = LogStore()
store.set_adapter(adapter)

# Log API call
store.log_api_call(
    model="claude-3-sonnet",
    provider="anthropic",
    duration_ms=220.0,
    input_tokens=150,
    output_tokens=80,
    total_tokens=230,
    source="chat_operation",
)

# Log error
store.log_error(
    error="Connection timeout",
    source="db_pool",
    data={"retry_count": 3, "timeout_ms": 5000},
)

# Flush to PostgreSQL
results = await store.aflush()
print(f"Wrote {results.get('adapter', 0)} logs")

# Query logs
recent_errors = await adapter.read(
    log_type="error",
    since="2025-01-01T00:00:00Z",
    limit=100,
)

for log in recent_errors:
    print(f"Error: {log.get('error')} from {log.get('source')}")

# Cleanup
await adapter.close()
```

#### Production Configuration with Environment Variables

```python
import os
from lionpride.session.logs import LogStore
from lionpride.session.log_adapter import PostgresLogAdapter

# Connection from environment
dsn = os.environ.get(
    "DATABASE_URL",
    "postgresql://localhost/logs"
)

adapter = PostgresLogAdapter(
    dsn=dsn,
    table="production_logs",
)

store = LogStore()
store.set_adapter(adapter)

# ... logging operations ...

await adapter.close()
```

## Integration with LogStore

Log adapters integrate with LogStore via `set_adapter()` and `aflush()`.

### Setting an Adapter

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_adapter import SQLiteWALLogAdapter, PostgresLogAdapter

store = LogStore()

# SQLite for local development
store.set_adapter(SQLiteWALLogAdapter(db_path="./logs.db"))

# OR PostgreSQL for production
store.set_adapter(PostgresLogAdapter(
    dsn="postgresql://user:pass@localhost/mydb"
))
```

### Flushing Logs to Adapter

The `aflush()` method writes all in-memory logs to the adapter:

```python
# Log some events
store.log_api_call(model="gpt-4o", duration_ms=150.0)
store.log_info(message="Operation completed")

# Flush to adapter
results = await store.aflush()
# results = {"adapter": 2}

# Logs are cleared from memory after successful flush
len(store)  # 0
```

### Partial Failure Handling

`aflush()` only clears logs on **complete success**. If the adapter write fails, logs
remain in memory:

```python
try:
    results = await store.aflush()
    if results.get("adapter", 0) == 0:
        print("Adapter write failed, logs preserved in memory")
except Exception as e:
    print(f"Flush failed: {e}")
    # Logs still available in store.logs
```

### Combining Adapter and Broadcaster

LogStore supports both adapter (single destination) and broadcaster (multiple
destinations) simultaneously:

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_adapter import PostgresLogAdapter
from lionpride.session.log_broadcaster import LogBroadcaster, S3LogSubscriber

store = LogStore()

# Primary storage (PostgreSQL)
store.set_adapter(PostgresLogAdapter(dsn="postgresql://..."))

# Backup destinations (S3)
broadcaster = LogBroadcaster()
broadcaster.add_subscriber(S3LogSubscriber(bucket="log-backup"))
store.set_broadcaster(broadcaster)

# Flush to both
results = await store.aflush()
# results = {"adapter": 5, "broadcaster": {"s3:log-backup": 5}}
```

## Implementing Custom Adapters

Create custom adapters by subclassing `LogAdapter`:

```python
from lionpride.session.log_adapter import LogAdapter
from lionpride.session.logs import Log
from typing import Any

class RedisLogAdapter(LogAdapter):
    """Redis adapter for real-time log streaming."""

    def __init__(self, redis_url: str, stream_name: str = "logs"):
        self.redis_url = redis_url
        self.stream_name = stream_name
        self._client = None

    async def _ensure_initialized(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url)

    async def write(self, logs: list[Log]) -> int:
        await self._ensure_initialized()
        count = 0
        for log in logs:
            await self._client.xadd(
                self.stream_name,
                log.to_dict(mode="json"),
            )
            count += 1
        return count

    async def read(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        log_type: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        # Implement Redis XREAD/XRANGE
        entries = await self._client.xrange(
            self.stream_name,
            count=limit,
        )
        return [dict(entry[1]) for entry in entries]

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
```

## Common Pitfalls

### 1. Forgetting to Close Adapters

Adapters hold database connections. Always close on shutdown:

```python
# WRONG: Connection leak
adapter = SQLiteWALLogAdapter(db_path="./logs.db")
# ... use adapter ...
# No close() called

# CORRECT: Explicit cleanup
adapter = SQLiteWALLogAdapter(db_path="./logs.db")
try:
    # ... use adapter ...
finally:
    await adapter.close()

# BETTER: Context manager pattern (if implementing custom adapter)
async with MyAdapter() as adapter:
    await adapter.write(logs)
```

### 2. Using Blocking I/O in Async Context

Both adapters are fully async. Do not mix with sync database calls:

```python
# WRONG: Blocking in async
def sync_write():
    adapter.write(logs)  # This is async!

# CORRECT: Use await
async def async_write():
    await adapter.write(logs)
```

### 3. Not Handling Import Errors

Dependencies are optional. Handle import failures gracefully:

```python
try:
    from lionpride.session.log_adapter import SQLiteWALLogAdapter
    adapter = SQLiteWALLogAdapter(db_path="./logs.db")
    await adapter.write(logs)  # May raise ImportError on first use
except ImportError as e:
    print(f"SQLite adapter unavailable: {e}")
    # Fall back to file-based logging
    store.dump("./logs.json")
```

### 4. Invalid Table Names in PostgreSQL

Table names are validated. Use valid SQL identifiers:

```python
# WRONG: Invalid table names
PostgresLogAdapter(dsn=dsn, table="my-logs")    # Hyphen not allowed
PostgresLogAdapter(dsn=dsn, table="2025_logs")  # Cannot start with digit
PostgresLogAdapter(dsn=dsn, table="drop")       # SQL keyword

# CORRECT: Valid identifiers
PostgresLogAdapter(dsn=dsn, table="my_logs")
PostgresLogAdapter(dsn=dsn, table="logs_2025")
PostgresLogAdapter(dsn=dsn, table="app_audit_trail")
```

## Design Rationale

**Lazy Initialization**: Adapters defer connection creation until first use. This
prevents connection errors during import and allows configuration validation before I/O.

**WAL Mode for SQLite**: Write-Ahead Logging provides better concurrent access than the
default rollback journal. Readers never block writers, making it suitable for
applications with mixed read/write patterns.

**JSONB for PostgreSQL**: Storing full logs as JSONB enables flexible queries without
schema migrations. The GIN index supports efficient JSON path queries for log analysis.

**SQL Injection Protection**: Table names are validated against a strict pattern. This
prevents dynamic table name attacks while allowing reasonable naming conventions.

**Upsert Semantics**: Both adapters use upsert (INSERT OR REPLACE / ON CONFLICT DO
UPDATE). This ensures idempotent writes - flushing the same logs twice does not create
duplicates.

## See Also

- [Logs](logs.md) - LogStore and Log entry documentation
- [LogBroadcaster](log_broadcaster.md) - Multi-destination log broadcasting
- [Session](session.md) - Session orchestration with logging
- [Element](../core/element.md) - Base class for Log
