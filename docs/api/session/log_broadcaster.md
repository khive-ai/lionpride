# Log Broadcaster

> Multi-destination log distribution with security controls

## Overview

The LogBroadcaster system provides **fan-out log distribution** to multiple destinations
simultaneously. Built for production environments, it includes comprehensive security
features: SSRF protection for webhooks, credential redaction, and secure secret
handling.

**Use LogBroadcaster for**: Multi-destination logging (S3 + DB + webhook), cloud log
archival, centralized log aggregation, audit trail distribution, secure log
transmission.

**Key features**:

- Fan-out to multiple subscribers (S3, PostgreSQL, webhooks, custom)
- Parallel or sequential broadcast modes
- Built-in SSRF protection for webhook URLs
- Credential redaction via LogRedactor
- Environment variable-based secret management
- Fail-fast or resilient broadcast modes

**Architecture**:

```text
                    +------------------+
                    |  LogBroadcaster  |
                    |  (orchestrator)  |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
+------------------+  +------------------+  +------------------+
| S3LogSubscriber  |  | PostgresLog      |  | WebhookLog       |
|                  |  | Subscriber       |  | Subscriber       |
+------------------+  +------------------+  +------------------+
          |                  |                  |
          v                  v                  v
    +----------+       +-----------+      +------------+
    | S3/MinIO |       | PostgreSQL|      | HTTPS API  |
    +----------+       +-----------+      +------------+
```

See [Logs](logs.md) for LogStore integration and [LogAdapter](log_adapter.md) for
single-destination persistence.

## Security Features

### SSRF Protection

WebhookLogSubscriber validates URLs to prevent Server-Side Request Forgery attacks:

- Blocks private IP ranges (10.x, 172.16-31.x, 192.168.x)
- Blocks link-local addresses (169.254.x.x - AWS metadata endpoint)
- Blocks localhost (127.x.x.x, ::1)
- Requires HTTPS by default
- Optional domain allowlist

### Credential Redaction

LogRedactor removes sensitive information before transmission:

- API keys and tokens
- Bearer authorization headers
- AWS credentials
- Passwords in connection strings
- Generic secrets and tokens

### Secure Secret Handling

S3LogSubscriber uses secure patterns:

- Environment variable references (not hardcoded secrets)
- SecretStr storage (prevents accidental logging)
- Deprecation warnings for direct credential passing

## LogRedactor

> Redact sensitive information from log content

### Class Signature

```python
from lionpride.session.log_broadcaster import LogRedactor, DEFAULT_REDACTION_PATTERNS

class LogRedactor:
    """Redact sensitive information from log content."""

    def __init__(
        self,
        patterns: tuple[re.Pattern[str], ...] | None = None,
        replacement: str = "[REDACTED]",
        include_defaults: bool = True,
    ) -> None: ...
```

### Parameters

| Parameter          | Type                  | Default        | Description                                    |
| ------------------ | --------------------- | -------------- | ---------------------------------------------- |
| `patterns`         | `tuple[Pattern, ...]` | `None`         | Custom regex patterns for redaction            |
| `replacement`      | `str`                 | `"[REDACTED]"` | String to replace sensitive content            |
| `include_defaults` | `bool`                | `True`         | Include DEFAULT_REDACTION_PATTERNS with custom |

### Methods

#### `redact()`

Redact sensitive content from a string.

```python
def redact(self, content: str) -> str
```

**Parameters:**

- `content` (str): String to redact

**Returns:**

- `str`: String with sensitive patterns replaced

```python
>>> from lionpride.session.log_broadcaster import LogRedactor

>>> redactor = LogRedactor()
>>> redactor.redact("api_key=sk-12345abcdef67890")
'api_key=[REDACTED]'

>>> redactor.redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIs...")
'Authorization: [REDACTED]'

>>> redactor.redact("postgresql://user:secret123@localhost/db")
'postgresql://user:[REDACTED]@localhost/db'
```

### Default Redaction Patterns

The `DEFAULT_REDACTION_PATTERNS` constant includes patterns for:

| Pattern Type            | Example Match                    |
| ----------------------- | -------------------------------- |
| API keys                | `api_key=sk-abc123...`           |
| X-API-Key headers       | `x-api-key: abc123...`           |
| Bearer tokens           | `Bearer eyJ...`                  |
| Authorization headers   | `Authorization: Bearer ...`      |
| AWS access key ID       | `aws_access_key_id=AKIA...`      |
| AWS secret access key   | `aws_secret_access_key=wJalr...` |
| Passwords               | `password=secret123`             |
| Generic secrets/tokens  | `secret=abc123...`               |
| Connection string creds | `://user:password@host`          |

### Custom Patterns

```python
import re
from lionpride.session.log_broadcaster import LogRedactor

# Add custom patterns while keeping defaults
custom_patterns = (
    re.compile(r"(?i)(internal_key)[=:\s]+([a-zA-Z0-9_\-]{10,})"),
    re.compile(r"(?i)(session_id)[=:\s]+([a-zA-Z0-9]{32})"),
)

redactor = LogRedactor(
    patterns=custom_patterns,
    replacement="[HIDDEN]",
    include_defaults=True,  # Also apply default patterns
)

# Use only custom patterns (no defaults)
custom_only = LogRedactor(
    patterns=custom_patterns,
    include_defaults=False,
)
```

## LogSubscriber

> Abstract base class for log subscribers

### Class Signature

```python
from lionpride.session.log_broadcaster import LogSubscriber
from abc import ABC, abstractmethod

class LogSubscriber(ABC):
    """Abstract base class for log subscribers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Subscriber name for identification."""
        pass

    @abstractmethod
    async def receive(self, logs: list[Log]) -> int:
        """Receive and process logs."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the subscriber and release resources."""
        pass
```

### Abstract Methods

| Method      | Returns | Description                              |
| ----------- | ------- | ---------------------------------------- |
| `name`      | `str`   | Property: subscriber identification name |
| `receive()` | `int`   | Process logs, return count processed     |
| `close()`   | `None`  | Release resources                        |

### Implementing Custom Subscribers

```python
from lionpride.session.log_broadcaster import LogSubscriber
from lionpride.session.logs import Log

class ElasticsearchLogSubscriber(LogSubscriber):
    """Custom subscriber for Elasticsearch."""

    def __init__(self, hosts: list[str], index: str = "logs"):
        self.hosts = hosts
        self.index = index
        self._client = None

    @property
    def name(self) -> str:
        return f"elasticsearch:{self.index}"

    async def receive(self, logs: list[Log]) -> int:
        if not logs:
            return 0

        # Initialize client lazily
        if self._client is None:
            from elasticsearch import AsyncElasticsearch
            self._client = AsyncElasticsearch(hosts=self.hosts)

        # Bulk index logs
        actions = [
            {"_index": self.index, "_source": log.to_dict(mode="json")}
            for log in logs
        ]
        # ... bulk insert logic ...
        return len(logs)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
```

## S3LogSubscriber

> S3-compatible storage subscriber

### Class Signature

```python
from lionpride.session.log_broadcaster import S3LogSubscriber

class S3LogSubscriber(LogSubscriber):
    """S3-compatible storage subscriber."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "logs/",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,          # Deprecated
        aws_secret_access_key: str | None = None,      # Deprecated
        aws_access_key_id_env: str | None = None,      # Recommended
        aws_secret_access_key_env: str | None = None,  # Recommended
        region_name: str = "us-east-1",
    ) -> None: ...
```

### Parameters

| Parameter                   | Type  | Default       | Description                               |
| --------------------------- | ----- | ------------- | ----------------------------------------- |
| `bucket`                    | `str` | Required      | S3 bucket name                            |
| `prefix`                    | `str` | `"logs/"`     | Key prefix for log files                  |
| `endpoint_url`              | `str` | `None`        | Custom endpoint (MinIO, LocalStack, etc.) |
| `aws_access_key_id`         | `str` | `None`        | AWS access key (deprecated)               |
| `aws_secret_access_key`     | `str` | `None`        | AWS secret key (deprecated)               |
| `aws_access_key_id_env`     | `str` | `None`        | Env var name for access key (recommended) |
| `aws_secret_access_key_env` | `str` | `None`        | Env var name for secret key (recommended) |
| `region_name`               | `str` | `"us-east-1"` | AWS region                                |

### Security: Credential Management

**Recommended**: Use environment variable references:

```python
# Credentials resolved from environment at runtime
subscriber = S3LogSubscriber(
    bucket="my-logs",
    aws_access_key_id_env="AWS_ACCESS_KEY_ID",
    aws_secret_access_key_env="AWS_SECRET_ACCESS_KEY",
)
```

**Deprecated**: Direct credential passing (emits DeprecationWarning):

```python
# Avoid: Credentials may leak in logs/errors
subscriber = S3LogSubscriber(
    bucket="my-logs",
    aws_access_key_id="AKIA...",        # Triggers warning
    aws_secret_access_key="wJalr...",   # Triggers warning
)
```

### File Organization

Logs are written as JSON files with timestamp-based keys:

```text
s3://bucket/prefix/YYYY/MM/DD/HHMMSS.json
```

Example: `s3://my-logs/logs/2025/01/15/143052.json`

### MinIO / S3-Compatible Storage

```python
subscriber = S3LogSubscriber(
    bucket="logs",
    prefix="app/prod/",
    endpoint_url="http://localhost:9000",
    aws_access_key_id_env="MINIO_ACCESS_KEY",
    aws_secret_access_key_env="MINIO_SECRET_KEY",
)
```

### Requirements

Requires `aioboto3`:

```bash
pip install aioboto3
# or
uv add aioboto3
```

## PostgresLogSubscriber

> PostgreSQL database subscriber

### Class Signature

```python
from lionpride.session.log_broadcaster import PostgresLogSubscriber

class PostgresLogSubscriber(LogSubscriber):
    """PostgreSQL database subscriber."""

    def __init__(
        self,
        dsn: str,
        table: str = "logs",
        auto_create: bool = True,
    ) -> None: ...
```

### Parameters

| Parameter     | Type   | Default  | Description                     |
| ------------- | ------ | -------- | ------------------------------- |
| `dsn`         | `str`  | Required | PostgreSQL connection string    |
| `table`       | `str`  | `"logs"` | Table name for logs             |
| `auto_create` | `bool` | `True`   | Auto-create table if not exists |

### DSN Validation

The DSN is validated on construction:

- Must use `postgres://` or `postgresql://` scheme
- Must contain a host component

```python
# Valid DSN formats
PostgresLogSubscriber(dsn="postgresql://user:pass@localhost/mydb")
PostgresLogSubscriber(dsn="postgres://localhost/logs")
PostgresLogSubscriber(dsn="postgresql://user:pass@db.example.com:5432/prod")

# Invalid (raises ValueError)
PostgresLogSubscriber(dsn="mysql://localhost/db")     # Wrong scheme
PostgresLogSubscriber(dsn="postgresql:///localdb")    # Missing host
```

### Usage

```python
from lionpride.session.log_broadcaster import PostgresLogSubscriber

subscriber = PostgresLogSubscriber(
    dsn="postgresql://user:password@localhost:5432/logs",
    table="app_logs",
    auto_create=True,
)

# Name property for identification
print(subscriber.name)  # "postgres:app_logs"

# Used via LogBroadcaster (see below)
```

## WebhookLogSubscriber

> Webhook subscriber with SSRF protection

### Class Signature

```python
from lionpride.session.log_broadcaster import WebhookLogSubscriber

class WebhookLogSubscriber(LogSubscriber):
    """Webhook subscriber for sending logs to external services."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        batch_size: int = 100,
        *,
        require_https: bool = True,
        allowed_domains: frozenset[str] | None = None,
        skip_validation: bool = False,
    ) -> None: ...
```

### Parameters

| Parameter         | Type             | Default  | Description                            |
| ----------------- | ---------------- | -------- | -------------------------------------- |
| `url`             | `str`            | Required | Webhook endpoint URL                   |
| `headers`         | `dict[str, str]` | `None`   | Custom headers (default: Content-Type) |
| `timeout`         | `float`          | `30.0`   | Request timeout in seconds             |
| `batch_size`      | `int`            | `100`    | Maximum logs per request               |
| `require_https`   | `bool`           | `True`   | Require HTTPS URLs                     |
| `allowed_domains` | `frozenset[str]` | `None`   | Restrict to specific domains           |
| `skip_validation` | `bool`           | `False`  | Skip URL validation (testing only)     |

### SSRF Protection

WebhookLogSubscriber validates URLs to prevent Server-Side Request Forgery:

```python
# Blocked: Private IP ranges
WebhookLogSubscriber(url="https://10.0.0.1/logs")       # ValueError
WebhookLogSubscriber(url="https://192.168.1.1/logs")   # ValueError
WebhookLogSubscriber(url="https://172.16.0.1/logs")    # ValueError

# Blocked: AWS metadata endpoint
WebhookLogSubscriber(url="http://169.254.169.254/...")  # ValueError

# Blocked: Localhost
WebhookLogSubscriber(url="https://localhost/logs")      # ValueError
WebhookLogSubscriber(url="https://127.0.0.1/logs")     # ValueError

# Blocked: HTTP without explicit opt-in
WebhookLogSubscriber(url="http://logs.example.com/")   # ValueError

# Allowed: Public HTTPS endpoint
WebhookLogSubscriber(url="https://logs.example.com/ingest")  # OK
```

### Blocked IP Networks

| Network          | Description               |
| ---------------- | ------------------------- |
| `10.0.0.0/8`     | Class A private           |
| `172.16.0.0/12`  | Class B private           |
| `192.168.0.0/16` | Class C private           |
| `169.254.0.0/16` | Link-local (AWS metadata) |
| `127.0.0.0/8`    | Localhost IPv4            |
| `::1/128`        | Localhost IPv6            |
| `fc00::/7`       | IPv6 unique local         |
| `fe80::/10`      | IPv6 link-local           |

### Domain Allowlist

Restrict webhooks to specific domains:

```python
subscriber = WebhookLogSubscriber(
    url="https://logs.mycompany.com/ingest",
    allowed_domains=frozenset({"logs.mycompany.com", "backup.mycompany.com"}),
)

# This would fail validation:
# WebhookLogSubscriber(
#     url="https://attacker.com/steal",
#     allowed_domains=frozenset({"logs.mycompany.com"}),
# )
# ValueError: domain 'attacker.com' not in allowed domains
```

### HTTP (Insecure) Mode

For internal/development use only:

```python
# Development/internal only - NOT recommended for production
subscriber = WebhookLogSubscriber(
    url="http://internal-logs.local/ingest",
    require_https=False,  # Explicit opt-in to insecure
)
```

### Requirements

Requires `httpx`:

```bash
pip install httpx
# or
uv add httpx
```

## LogBroadcasterConfig

> Configuration for LogBroadcaster

### Class Signature

```python
from lionpride.session.log_broadcaster import LogBroadcasterConfig

class LogBroadcasterConfig(BaseModel):
    """Configuration for LogBroadcaster."""

    fail_fast: bool = False
    parallel: bool = True
```

### Parameters

| Parameter   | Type   | Default | Description                      |
| ----------- | ------ | ------- | -------------------------------- |
| `fail_fast` | `bool` | `False` | Stop on first subscriber failure |
| `parallel`  | `bool` | `True`  | Send to subscribers in parallel  |

### Broadcast Modes

**Parallel (default)**: All subscribers receive logs concurrently.

```python
config = LogBroadcasterConfig(parallel=True, fail_fast=False)
# S3, Postgres, Webhook receive logs simultaneously
# If one fails, others continue
```

**Sequential**: Subscribers receive logs one at a time.

```python
config = LogBroadcasterConfig(parallel=False, fail_fast=False)
# S3 first, then Postgres, then Webhook
```

**Fail-fast**: Stop on first failure.

```python
config = LogBroadcasterConfig(fail_fast=True)
# If any subscriber fails, stop broadcasting
# Useful for critical audit logs
```

## LogBroadcaster

> Fan-out log broadcaster to multiple subscribers

### Class Signature

```python
from lionpride.session.log_broadcaster import LogBroadcaster, LogBroadcasterConfig, LogRedactor

class LogBroadcaster:
    """Fan-out log broadcaster to multiple subscribers."""

    def __init__(
        self,
        config: LogBroadcasterConfig | None = None,
        redactor: LogRedactor | None = None,
    ) -> None: ...
```

### Parameters

| Parameter  | Type                   | Default      | Description                    |
| ---------- | ---------------------- | ------------ | ------------------------------ |
| `config`   | `LogBroadcasterConfig` | Auto-created | Broadcast configuration        |
| `redactor` | `LogRedactor`          | `None`       | Redactor for sensitive content |

### Methods

#### `add_subscriber()`

Add a subscriber to the broadcaster.

```python
def add_subscriber(self, subscriber: LogSubscriber) -> None
```

**Note**: Adding a subscriber with the same name replaces the existing one (logs
warning).

```python
>>> broadcaster = LogBroadcaster()
>>> broadcaster.add_subscriber(S3LogSubscriber(bucket="logs"))
>>> broadcaster.add_subscriber(PostgresLogSubscriber(dsn="..."))
>>> broadcaster.list_subscribers()
['s3:logs', 'postgres:logs']
```

#### `remove_subscriber()`

Remove a subscriber by name.

```python
def remove_subscriber(self, name: str) -> bool
```

**Returns:**

- `bool`: True if removed, False if not found

```python
>>> broadcaster.remove_subscriber("s3:logs")
True
>>> broadcaster.remove_subscriber("nonexistent")
False
```

#### `list_subscribers()`

List all subscriber names.

```python
def list_subscribers(self) -> list[str]
```

```python
>>> broadcaster.list_subscribers()
['s3:logs', 'postgres:logs', 'webhook:https://...']
```

#### `broadcast()` (async)

Broadcast logs to all subscribers.

```python
async def broadcast(self, logs: list[Log]) -> dict[str, int]
```

**Parameters:**

- `logs` (list[Log]): Logs to broadcast

**Returns:**

- `dict[str, int]`: Map of subscriber name to count of logs written

```python
>>> results = await broadcaster.broadcast(logs)
>>> results
{'s3:my-logs': 100, 'postgres:logs': 100, 'webhook:https://...': 100}
```

**Behavior**:

- If redactor is configured, logs are redacted before broadcasting
- In parallel mode, all subscribers receive logs concurrently
- Failed subscribers return 0 (unless fail_fast=True)

#### `close()` (async)

Close all subscribers and release resources.

```python
async def close(self) -> None
```

**Important**: Always call `close()` to properly release resources.

```python
>>> await broadcaster.close()
>>> broadcaster.list_subscribers()
[]
```

## Validation Functions

### `_validate_postgres_dsn()`

Validate PostgreSQL DSN format.

```python
from lionpride.session.log_broadcaster import _validate_postgres_dsn

# Valid
_validate_postgres_dsn("postgresql://user:pass@localhost/db")  # OK

# Invalid
_validate_postgres_dsn("mysql://localhost/db")     # ValueError: wrong scheme
_validate_postgres_dsn("postgresql:///db")         # ValueError: missing host
_validate_postgres_dsn("")                          # ValueError: empty string
```

### `_validate_webhook_url()`

Validate webhook URL for SSRF protection.

```python
from lionpride.session.log_broadcaster import _validate_webhook_url

# Valid
_validate_webhook_url("https://logs.example.com/ingest")  # OK

# Invalid
_validate_webhook_url("http://logs.example.com/")         # ValueError: HTTPS required
_validate_webhook_url("https://192.168.1.1/logs")        # ValueError: private IP
_validate_webhook_url("https://localhost/logs")          # ValueError: localhost

# With options
_validate_webhook_url(
    "http://internal/logs",
    require_https=False,  # Allow HTTP
)

_validate_webhook_url(
    "https://logs.myco.com/",
    allowed_domains=frozenset({"logs.myco.com"}),  # Domain allowlist
)
```

## Usage Patterns

### Basic Multi-Destination Logging

```python
from lionpride.session.logs import LogStore
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    S3LogSubscriber,
    PostgresLogSubscriber,
)

# Create broadcaster
broadcaster = LogBroadcaster()

# Add destinations
broadcaster.add_subscriber(
    S3LogSubscriber(
        bucket="app-logs",
        prefix="prod/",
        aws_access_key_id_env="AWS_ACCESS_KEY_ID",
        aws_secret_access_key_env="AWS_SECRET_ACCESS_KEY",
    )
)
broadcaster.add_subscriber(
    PostgresLogSubscriber(dsn="postgresql://user:pass@localhost/logs")
)

# Integrate with LogStore
store = LogStore()
store.set_broadcaster(broadcaster)

# Log events
store.log_api_call(model="gpt-4o", duration_ms=200, total_tokens=500)
store.log_error(error="Rate limit exceeded", source="openai_api")

# Flush to all destinations
results = await store.aflush()
# {'s3:app-logs': 2, 'postgres:logs': 2}

# Cleanup
await broadcaster.close()
```

### Secure Logging with Redaction

```python
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    LogRedactor,
    WebhookLogSubscriber,
)

# Create broadcaster with redaction
broadcaster = LogBroadcaster(redactor=LogRedactor())

# Add webhook (SSRF-protected)
broadcaster.add_subscriber(
    WebhookLogSubscriber(
        url="https://logs.example.com/ingest",
        headers={"Authorization": "Bearer token123"},
    )
)

# Logs containing secrets are automatically redacted
# Before: {"message": "api_key=sk-12345 used"}
# After:  {"message": "api_key=[REDACTED] used"}
```

### Domain-Restricted Webhooks

```python
from lionpride.session.log_broadcaster import WebhookLogSubscriber

# Only allow specific domains
allowed = frozenset({"logs.mycompany.com", "backup.mycompany.com"})

subscriber = WebhookLogSubscriber(
    url="https://logs.mycompany.com/v1/ingest",
    allowed_domains=allowed,
    headers={
        "Authorization": "Bearer secret",
        "X-Source": "lionpride-app",
    },
)
```

### MinIO Local Development

```python
from lionpride.session.log_broadcaster import S3LogSubscriber, LogBroadcaster

# Local MinIO for development
broadcaster = LogBroadcaster()
broadcaster.add_subscriber(
    S3LogSubscriber(
        bucket="dev-logs",
        endpoint_url="http://localhost:9000",
        aws_access_key_id_env="MINIO_ROOT_USER",
        aws_secret_access_key_env="MINIO_ROOT_PASSWORD",
    )
)
```

### Fail-Fast Critical Logging

```python
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    LogBroadcasterConfig,
    PostgresLogSubscriber,
)

# For critical audit logs: fail if any destination fails
broadcaster = LogBroadcaster(
    config=LogBroadcasterConfig(fail_fast=True, parallel=False)
)

broadcaster.add_subscriber(
    PostgresLogSubscriber(dsn="postgresql://audit:pass@primary/audit")
)
broadcaster.add_subscriber(
    PostgresLogSubscriber(dsn="postgresql://audit:pass@replica/audit")
)

# If either database fails, broadcast raises exception
try:
    results = await broadcaster.broadcast(logs)
except Exception as e:
    # Handle critical failure
    print(f"Audit logging failed: {e}")
```

### Custom Subscriber Implementation

```python
from lionpride.session.log_broadcaster import LogSubscriber
from lionpride.session.logs import Log

class ConsoleLogSubscriber(LogSubscriber):
    """Simple console output subscriber for debugging."""

    def __init__(self, prefix: str = "[LOG]"):
        self.prefix = prefix

    @property
    def name(self) -> str:
        return "console"

    async def receive(self, logs: list[Log]) -> int:
        for log in logs:
            print(f"{self.prefix} {log.log_type.value}: {log.message or log.model}")
        return len(logs)

    async def close(self) -> None:
        pass  # No resources to release

# Use in development
broadcaster = LogBroadcaster()
broadcaster.add_subscriber(ConsoleLogSubscriber(prefix="[DEBUG]"))
```

## Common Pitfalls

### 1. Hardcoding Credentials

**Problem**: Credentials in code may leak via logs, errors, or version control.

```python
# WRONG: Credentials may leak
S3LogSubscriber(
    bucket="logs",
    aws_access_key_id="AKIA...",      # In code!
    aws_secret_access_key="wJalr...", # In code!
)

# CORRECT: Use environment variable references
S3LogSubscriber(
    bucket="logs",
    aws_access_key_id_env="AWS_ACCESS_KEY_ID",
    aws_secret_access_key_env="AWS_SECRET_ACCESS_KEY",
)
```

### 2. Forgetting to Close Broadcaster

**Problem**: Resource leaks from unclosed connections.

```python
# WRONG: No cleanup
broadcaster = LogBroadcaster()
broadcaster.add_subscriber(PostgresLogSubscriber(...))
await broadcaster.broadcast(logs)
# Connection left open!

# CORRECT: Always close
broadcaster = LogBroadcaster()
try:
    broadcaster.add_subscriber(PostgresLogSubscriber(...))
    await broadcaster.broadcast(logs)
finally:
    await broadcaster.close()

# OR use async context manager pattern if implemented
```

### 3. Ignoring Broadcast Results

**Problem**: Silent failures when subscribers fail.

```python
# WRONG: Ignoring failures
await broadcaster.broadcast(logs)

# CORRECT: Check results
results = await broadcaster.broadcast(logs)
for name, count in results.items():
    if count == 0:
        logger.warning(f"Subscriber {name} failed to receive logs")
```

### 4. Using HTTP for Production Webhooks

**Problem**: Unencrypted log data in transit.

```python
# WRONG: HTTP in production
WebhookLogSubscriber(
    url="http://logs.example.com/ingest",
    require_https=False,  # Insecure!
)

# CORRECT: Always HTTPS in production
WebhookLogSubscriber(
    url="https://logs.example.com/ingest",
    # require_https=True is default
)
```

### 5. Skipping Validation for "Convenience"

**Problem**: SSRF vulnerabilities.

```python
# WRONG: Never do this in production
WebhookLogSubscriber(
    url=user_provided_url,
    skip_validation=True,  # DANGER!
)

# CORRECT: Let validation protect you
try:
    WebhookLogSubscriber(url=user_provided_url)
except ValueError as e:
    logger.error(f"Invalid webhook URL: {e}")
```

## Design Rationale

**Subscriber Pattern**: Decouples log generation from log storage. Add/remove
destinations without modifying logging code. Each subscriber handles its own connection
lifecycle.

**SSRF Protection**: Webhooks are attack vectors. By default, we block private IPs,
localhost, and require HTTPS. This prevents attackers from using log webhooks to probe
internal networks or steal cloud metadata.

**Credential Redaction**: Logs often accidentally contain secrets (in error messages,
request bodies, etc.). LogRedactor provides defense-in-depth by removing sensitive
patterns before transmission.

**Environment Variable References**: Instead of passing secrets directly, pass the name
of an environment variable. This prevents secrets from appearing in stack traces, logs,
or serialized config.

**SecretStr Storage**: Pydantic's SecretStr prevents accidental logging of credentials.
Even if the subscriber object is logged or repr'd, secrets appear as `'**********'`.

**Fail-Fast Mode**: For critical audit logs, it's better to fail loudly than silently
lose logs. The fail_fast option ensures all destinations receive logs or the broadcast
fails entirely.

## See Also

- [Logs](logs.md) - LogStore with broadcaster integration
- [LogAdapter](log_adapter.md) - Single-destination adapters (SQLite, PostgreSQL)
- [Session](session.md) - Session-level log management
- [Element](../core/element.md) - Base class for Log entries

## Examples

### Complete Production Setup

```python
import os
from lionpride.session.logs import LogStore, LogStoreConfig
from lionpride.session.log_broadcaster import (
    LogBroadcaster,
    LogBroadcasterConfig,
    LogRedactor,
    S3LogSubscriber,
    PostgresLogSubscriber,
    WebhookLogSubscriber,
)

# Environment setup (typically in .env or deployment config)
# AWS_ACCESS_KEY_ID=AKIA...
# AWS_SECRET_ACCESS_KEY=wJalr...
# DATABASE_URL=postgresql://user:pass@db.example.com/logs
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

async def setup_production_logging():
    # Create broadcaster with redaction
    broadcaster = LogBroadcaster(
        config=LogBroadcasterConfig(parallel=True, fail_fast=False),
        redactor=LogRedactor(),
    )

    # S3 for long-term archival
    broadcaster.add_subscriber(
        S3LogSubscriber(
            bucket="prod-logs",
            prefix="app/v1/",
            aws_access_key_id_env="AWS_ACCESS_KEY_ID",
            aws_secret_access_key_env="AWS_SECRET_ACCESS_KEY",
            region_name="us-west-2",
        )
    )

    # PostgreSQL for queryable logs
    broadcaster.add_subscriber(
        PostgresLogSubscriber(
            dsn=os.environ["DATABASE_URL"],
            table="application_logs",
            auto_create=True,
        )
    )

    # Slack webhook for errors only (filtered in webhook handler)
    broadcaster.add_subscriber(
        WebhookLogSubscriber(
            url=os.environ["SLACK_WEBHOOK_URL"],
            headers={"Content-Type": "application/json"},
            timeout=10.0,
            allowed_domains=frozenset({"hooks.slack.com"}),
        )
    )

    # Create store with broadcaster
    store = LogStore(
        config=LogStoreConfig(
            persist_dir="./data/logs",
            capacity=5000,
            auto_save_on_exit=True,
        )
    )
    store.set_broadcaster(broadcaster)

    return store, broadcaster


async def main():
    store, broadcaster = await setup_production_logging()

    try:
        # Application logging
        store.log_api_call(
            model="gpt-4o",
            provider="openai",
            duration_ms=250.0,
            input_tokens=150,
            output_tokens=100,
            total_tokens=250,
            source="chat_endpoint",
        )

        store.log_operation(
            source="session_manager",
            message="New session created",
            data={"session_id": "abc123", "user_tier": "premium"},
        )

        # Flush to all destinations
        results = await store.aflush()
        print(f"Broadcast results: {results}")

    finally:
        await broadcaster.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Testing with Mock Subscriber

```python
from lionpride.session.log_broadcaster import LogSubscriber, LogBroadcaster
from lionpride.session.logs import Log, LogType

class MockLogSubscriber(LogSubscriber):
    """Mock subscriber for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self.received_logs: list[Log] = []
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def receive(self, logs: list[Log]) -> int:
        self.received_logs.extend(logs)
        return len(logs)

    async def close(self) -> None:
        self.closed = True


async def test_broadcaster():
    # Setup
    broadcaster = LogBroadcaster()
    mock1 = MockLogSubscriber("mock1")
    mock2 = MockLogSubscriber("mock2")

    broadcaster.add_subscriber(mock1)
    broadcaster.add_subscriber(mock2)

    # Create test logs
    logs = [
        Log(log_type=LogType.INFO, message="Test 1"),
        Log(log_type=LogType.INFO, message="Test 2"),
    ]

    # Broadcast
    results = await broadcaster.broadcast(logs)

    # Verify
    assert results == {"mock1": 2, "mock2": 2}
    assert len(mock1.received_logs) == 2
    assert len(mock2.received_logs) == 2

    # Cleanup
    await broadcaster.close()
    assert mock1.closed
    assert mock2.closed
```
