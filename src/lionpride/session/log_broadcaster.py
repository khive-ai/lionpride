# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Log broadcasters for multi-destination log distribution.

Provides async broadcasters that fan-out logs to multiple destinations:
- S3 (AWS, MinIO)
- PostgreSQL
- Custom subscribers

Example:
    broadcaster = LogBroadcaster()
    broadcaster.add_subscriber(S3LogSubscriber(bucket="logs"))
    broadcaster.add_subscriber(PostgresLogSubscriber(dsn="..."))
    await broadcaster.broadcast(logs)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import warnings
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr

from lionpride.ln import json_dumps

if TYPE_CHECKING:
    from .log_adapter import PostgresLogAdapter
    from .logs import Log

__all__ = (
    "DEFAULT_REDACTION_PATTERNS",
    "PRIVATE_IP_NETWORKS",
    "LogBroadcaster",
    "LogRedactor",
    "LogSubscriber",
    "PostgresLogSubscriber",
    "S3LogSubscriber",
    "WebhookLogSubscriber",
    "_validate_postgres_dsn",
    "_validate_webhook_url",
)

logger = logging.getLogger(__name__)

# Private/internal IP ranges that should be blocked for SSRF protection
PRIVATE_IP_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),  # Class A private
    ipaddress.ip_network("172.16.0.0/12"),  # Class B private
    ipaddress.ip_network("192.168.0.0/16"),  # Class C private
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local (AWS metadata: 169.254.169.254)
    ipaddress.ip_network("127.0.0.0/8"),  # Localhost
    ipaddress.ip_network("::1/128"),  # IPv6 localhost
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
)

# Default patterns for redacting sensitive information from logs
DEFAULT_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # API keys (generic patterns)
    re.compile(r"(?i)(api[_-]?key|apikey)[=:\s]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?"),
    re.compile(r"(?i)(x-api-key)[=:\s]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?"),
    # Bearer tokens
    re.compile(r"(?i)(bearer)\s+([a-zA-Z0-9_\-\.]+)"),
    re.compile(r"(?i)(authorization)[=:\s]+['\"]?(bearer\s+)?([a-zA-Z0-9_\-\.]+)['\"]?"),
    # AWS credentials
    re.compile(r"(?i)(aws[_-]?access[_-]?key[_-]?id)[=:\s]+['\"]?([A-Z0-9]{16,})['\"]?"),
    re.compile(r"(?i)(aws[_-]?secret[_-]?access[_-]?key)[=:\s]+['\"]?([a-zA-Z0-9/+=]{32,})['\"]?"),
    # Passwords
    re.compile(r"(?i)(password|passwd|pwd)[=:\s]+['\"]?([^\s'\"]{8,})['\"]?"),
    # Generic secrets
    re.compile(r"(?i)(secret|token|credential)[=:\s]+['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"),
    # Connection strings with passwords
    re.compile(r"(://[^:]+:)([^@]+)(@)"),
)


class LogRedactor:
    """Redact sensitive information from log content.

    Applies configurable regex patterns to redact secrets, API keys,
    passwords, and other sensitive data from log messages.

    Args:
        patterns: Custom regex patterns to use for redaction.
            If None, uses DEFAULT_REDACTION_PATTERNS.
        replacement: String to replace redacted content with.
        include_defaults: If True and custom patterns provided,
            also include DEFAULT_REDACTION_PATTERNS.

    Example:
        redactor = LogRedactor()
        safe_message = redactor.redact("api_key=sk-12345abcdef")
        # -> "api_key=[REDACTED]"
    """

    def __init__(
        self,
        patterns: tuple[re.Pattern[str], ...] | None = None,
        replacement: str = "[REDACTED]",
        include_defaults: bool = True,
    ):
        self.replacement = replacement

        if patterns is None:
            self._patterns = DEFAULT_REDACTION_PATTERNS
        elif include_defaults:
            self._patterns = patterns + DEFAULT_REDACTION_PATTERNS
        else:
            self._patterns = patterns

    def redact(self, content: str) -> str:
        """Redact sensitive content from a string.

        Args:
            content: String to redact.

        Returns:
            String with sensitive patterns replaced by replacement text.
        """
        if not content:
            return content

        result = content
        for pattern in self._patterns:
            # Replace matching groups based on pattern structure
            if pattern.groups >= 2:
                # For patterns with capture groups, replace the sensitive part
                result = pattern.sub(self._replacement_func, result)
            else:
                # For simple patterns, replace the entire match
                result = pattern.sub(self.replacement, result)

        return result

    def _replacement_func(self, match: re.Match[str]) -> str:
        """Replace function that preserves non-sensitive parts of the match."""
        groups = match.groups()
        if len(groups) >= 3:
            # Pattern like (prefix)(sensitive)(suffix) - e.g., connection strings
            return f"{groups[0]}{self.replacement}{groups[2]}"
        elif len(groups) >= 2:
            # Pattern like (key)(value) - replace the value
            return f"{groups[0]}={self.replacement}"
        return self.replacement


def _validate_postgres_dsn(dsn: str) -> None:
    """Validate PostgreSQL DSN format.

    Validates that the DSN:
        - Uses postgres:// or postgresql:// scheme
        - Contains a host component

    Args:
        dsn: Database connection string to validate.

    Raises:
        ValueError: If DSN format is invalid.
    """
    if not dsn or not isinstance(dsn, str):
        raise ValueError(f"DSN must be non-empty string, got: {type(dsn).__name__}")

    try:
        parsed = urlparse(dsn)
    except Exception as e:
        raise ValueError(f"Malformed DSN '{dsn}': {e}") from e

    # Check scheme
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError(
            f"DSN must use postgres:// or postgresql:// scheme, got: {parsed.scheme}://"
        )

    # Check host exists
    if not parsed.hostname:
        raise ValueError(f"DSN missing host: {dsn}")


def _validate_webhook_url(
    url: str,
    *,
    require_https: bool = True,
    allowed_domains: frozenset[str] | None = None,
) -> None:
    """Validate webhook URL to prevent SSRF vulnerabilities.

    Security checks:
        - Reject private/internal IP addresses (10.x, 172.16-31.x, 192.168.x, etc.)
        - Reject link-local addresses (169.254.x.x - AWS metadata endpoint)
        - Reject localhost (127.x.x.x)
        - Optionally require HTTPS (default: True)
        - Optionally restrict to allowed domains

    Args:
        url: URL to validate
        require_https: If True, only allow https:// URLs (default: True)
        allowed_domains: If provided, only allow URLs to these domains

    Raises:
        ValueError: If URL fails validation
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"Webhook URL must be non-empty string, got: {type(url).__name__}")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Malformed webhook URL '{url}': {e}") from e

    # Check scheme
    if require_https and parsed.scheme != "https":
        raise ValueError(
            f"Webhook URL must use https:// scheme for security, got: {parsed.scheme}://"
            f"\nSet require_https=False to allow insecure connections (not recommended)"
        )
    elif parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Webhook URL must use http:// or https:// scheme, got: {parsed.scheme}://"
        )

    # Check domain/host exists
    if not parsed.netloc:
        raise ValueError(f"Webhook URL missing domain: {url}")

    # Extract hostname (without port)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Webhook URL missing hostname: {url}")

    # Check allowed domains if specified
    if allowed_domains is not None and hostname not in allowed_domains:
        raise ValueError(
            f"Webhook URL domain '{hostname}' not in allowed domains: {allowed_domains}"
        )

    # Resolve hostname to IP and check for private ranges
    try:
        # Get all IPs for the hostname (handles DNS with multiple IPs)
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                for network in PRIVATE_IP_NETWORKS:
                    if ip in network:
                        raise ValueError(
                            f"Webhook URL resolves to private/internal IP address: {ip_str}"
                            f"\nThis is blocked to prevent SSRF attacks"
                            f"\nBlocked network: {network}"
                        )
            except ipaddress.AddressValueError:
                # Not a valid IP address (shouldn't happen from getaddrinfo)
                continue
    except socket.gaierror as e:
        # DNS resolution failed - could be temporary or invalid hostname
        raise ValueError(f"Cannot resolve webhook hostname '{hostname}': {e}") from e


class LogSubscriber(ABC):
    """Abstract base class for log subscribers.

    Subscribers receive logs from the broadcaster and handle them
    according to their specific destination (S3, DB, webhook, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Subscriber name for identification."""
        pass

    @abstractmethod
    async def receive(self, logs: list[Log]) -> int:
        """Receive and process logs.

        Args:
            logs: List of Log objects to process

        Returns:
            Number of logs successfully processed
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the subscriber and release resources."""
        pass


class S3LogSubscriber(LogSubscriber):
    """S3-compatible storage subscriber.

    Writes logs as JSON files to S3 or S3-compatible storage (MinIO, etc.).

    Security:
        - Use aws_access_key_id_env and aws_secret_access_key_env to resolve
          credentials from environment variables (recommended).
        - Direct credential parameters are deprecated and will emit warnings.
        - Credentials are stored as SecretStr to prevent accidental logging.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix for log files (default: "logs/").
        endpoint_url: Custom S3 endpoint URL (for MinIO, etc.).
        aws_access_key_id: AWS access key ID (deprecated, use env param).
        aws_secret_access_key: AWS secret access key (deprecated, use env param).
        aws_access_key_id_env: Environment variable name for access key ID.
        aws_secret_access_key_env: Environment variable name for secret key.
        region_name: AWS region (default: "us-east-1").
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "logs/",
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_access_key_id_env: str | None = None,
        aws_secret_access_key_env: str | None = None,
        region_name: str = "us-east-1",
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self._client = None

        # Resolve credentials with env vars taking precedence
        resolved_access_key: str | None = None
        resolved_secret_key: str | None = None

        # Check for environment variable params first (preferred)
        if aws_access_key_id_env:
            resolved_access_key = os.environ.get(aws_access_key_id_env)
            if not resolved_access_key:
                logger.warning("AWS access key environment variable not set or empty")

        if aws_secret_access_key_env:
            resolved_secret_key = os.environ.get(aws_secret_access_key_env)
            if not resolved_secret_key:
                logger.warning("AWS secret key environment variable not set or empty")

        # Fall back to direct params with deprecation warning
        if aws_access_key_id and not resolved_access_key:
            warnings.warn(
                "Passing aws_access_key_id directly is deprecated. "
                "Use aws_access_key_id_env to specify an environment variable name instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_access_key = aws_access_key_id

        if aws_secret_access_key and not resolved_secret_key:
            warnings.warn(
                "Passing aws_secret_access_key directly is deprecated. "
                "Use aws_secret_access_key_env to specify an environment variable name instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_secret_key = aws_secret_access_key

        # Store as SecretStr to prevent accidental logging
        self._aws_access_key_id: SecretStr | None = (
            SecretStr(resolved_access_key) if resolved_access_key else None
        )
        self._aws_secret_access_key: SecretStr | None = (
            SecretStr(resolved_secret_key) if resolved_secret_key else None
        )

    @property
    def name(self) -> str:
        return f"s3:{self.bucket}"

    async def _ensure_client(self) -> None:
        """Ensure S3 client is initialized."""
        if self._client is not None:
            return

        try:
            import aioboto3
        except ImportError:
            raise ImportError(
                "aioboto3 is required for S3LogSubscriber. Install with: pip install aioboto3"
            )

        session = aioboto3.Session()
        self._client = session

    async def receive(self, logs: list[Log]) -> int:
        """Write logs to S3 as JSON file."""
        if not logs:
            return 0

        await self._ensure_client()
        assert self._client is not None  # Guaranteed by _ensure_client

        # Generate key with timestamp
        ts = datetime.now(UTC).strftime("%Y/%m/%d/%H%M%S")
        key = f"{self.prefix}{ts}.json"

        # Convert logs to JSON
        log_dicts = [log.to_dict(mode="json") for log in logs]
        content = json_dumps(log_dicts, pretty=True)
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content

        try:
            # Extract secret values for boto3
            access_key = (
                self._aws_access_key_id.get_secret_value() if self._aws_access_key_id else None
            )
            secret_key = (
                self._aws_secret_access_key.get_secret_value()
                if self._aws_secret_access_key
                else None
            )

            async with self._client.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self.region_name,
            ) as client:
                await client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=content_bytes,
                    ContentType="application/json",
                )
            logger.debug(f"Wrote {len(logs)} logs to s3://{self.bucket}/{key}")
            return len(logs)
        except Exception as e:
            logger.error(f"Failed to write logs to S3: {e}")
            return 0

    async def close(self) -> None:
        """S3 client is session-based, no explicit close needed."""
        self._client = None


class PostgresLogSubscriber(LogSubscriber):
    """PostgreSQL database subscriber.

    Uses the PostgresLogAdapter for actual write operations.

    Args:
        dsn: PostgreSQL connection string (postgres:// or postgresql://).
        table: Table name for logs (default: "logs").
        auto_create: Auto-create table if not exists (default: True).

    Raises:
        ValueError: If DSN format is invalid.
    """

    def __init__(
        self,
        dsn: str,
        table: str = "logs",
        auto_create: bool = True,
    ):
        # Validate DSN format
        _validate_postgres_dsn(dsn)

        self.dsn = dsn
        self.table = table
        self.auto_create = auto_create
        self._adapter: PostgresLogAdapter | None = None

    @property
    def name(self) -> str:
        return f"postgres:{self.table}"

    async def _ensure_adapter(self) -> None:
        """Ensure adapter is initialized."""
        if self._adapter is not None:
            return

        from .log_adapter import PostgresLogAdapter

        self._adapter = PostgresLogAdapter(
            dsn=self.dsn,
            table=self.table,
            auto_create=self.auto_create,
        )

    async def receive(self, logs: list[Log]) -> int:
        """Write logs to PostgreSQL."""
        if not logs:
            return 0

        await self._ensure_adapter()
        assert self._adapter is not None  # Guaranteed by _ensure_adapter
        return await self._adapter.write(logs)

    async def close(self) -> None:
        """Close the adapter."""
        if self._adapter:
            await self._adapter.close()
            self._adapter = None


class WebhookLogSubscriber(LogSubscriber):
    """Webhook subscriber for sending logs to external services.

    Sends logs as JSON POST requests to configured endpoint.

    Security:
        - Validates URL to prevent SSRF attacks
        - Blocks private/internal IP addresses by default
        - Requires HTTPS by default (configurable)
        - Optionally restricts to allowed domains

    Args:
        url: Webhook endpoint URL (must be HTTPS by default)
        headers: Custom headers to send with requests
        timeout: Request timeout in seconds
        batch_size: Maximum logs per request
        require_https: If True, only allow HTTPS URLs (default: True)
        allowed_domains: If provided, only allow URLs to these domains
        skip_validation: If True, skip URL validation (DANGER: only for testing)

    Raises:
        ValueError: If URL fails security validation
    """

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
    ):
        # Validate URL for SSRF protection (unless explicitly skipped for testing)
        if not skip_validation:
            _validate_webhook_url(
                url,
                require_https=require_https,
                allowed_domains=allowed_domains,
            )

        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.batch_size = batch_size
        self.require_https = require_https
        self.allowed_domains = allowed_domains
        self._client = None

    @property
    def name(self) -> str:
        return f"webhook:{self.url}"

    async def receive(self, logs: list[Log]) -> int:
        """Send logs to webhook endpoint."""
        if not logs:
            return 0

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for WebhookLogSubscriber. Install with: pip install httpx"
            )

        # Convert logs to JSON
        log_dicts = [log.to_dict(mode="json") for log in logs]

        # Send in batches
        count = 0
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(log_dicts), self.batch_size):
                batch = log_dicts[i : i + self.batch_size]
                try:
                    response = await client.post(
                        self.url,
                        json=batch,
                        headers=self.headers,
                    )
                    if response.is_success:
                        count += len(batch)
                    else:
                        logger.warning(f"Webhook returned {response.status_code}: {response.text}")
                except Exception as e:
                    logger.error(f"Failed to send logs to webhook: {e}")

        logger.debug(f"Sent {count} logs to webhook {self.url}")
        return count

    async def close(self) -> None:
        """No persistent resources to close."""
        pass


class LogBroadcasterConfig(BaseModel):
    """Configuration for LogBroadcaster."""

    fail_fast: bool = Field(
        default=False,
        description="Stop on first subscriber failure",
    )
    parallel: bool = Field(
        default=True,
        description="Send to subscribers in parallel",
    )


class LogBroadcaster:
    """Fan-out log broadcaster to multiple subscribers.

    Distributes logs to all registered subscribers (S3, DB, webhooks, etc.).

    Security:
        - Optional log redaction via LogRedactor to remove sensitive content
          (API keys, credentials, passwords) before sending to subscribers.

    Example:
        broadcaster = LogBroadcaster()
        broadcaster.add_subscriber(S3LogSubscriber(bucket="my-logs"))
        broadcaster.add_subscriber(PostgresLogSubscriber(dsn="..."))

        # Broadcast logs to all subscribers
        results = await broadcaster.broadcast(logs)
        # → {"s3:my-logs": 100, "postgres:logs": 100}

        # With redaction enabled
        broadcaster = LogBroadcaster(redactor=LogRedactor())
        # → Sensitive patterns in log messages are replaced with [REDACTED]

    Args:
        config: Broadcaster configuration (fail_fast, parallel).
        redactor: Optional LogRedactor for sensitive content redaction.
    """

    def __init__(
        self,
        config: LogBroadcasterConfig | None = None,
        redactor: LogRedactor | None = None,
    ):
        self.config = config or LogBroadcasterConfig()
        self.redactor = redactor
        self._subscribers: dict[str, LogSubscriber] = {}

    def add_subscriber(self, subscriber: LogSubscriber) -> None:
        """Add a subscriber."""
        if subscriber.name in self._subscribers:
            logger.warning(f"Replacing existing subscriber: {subscriber.name}")
        self._subscribers[subscriber.name] = subscriber

    def remove_subscriber(self, name: str) -> bool:
        """Remove a subscriber by name. Returns True if removed."""
        if name in self._subscribers:
            del self._subscribers[name]
            return True
        return False

    def list_subscribers(self) -> list[str]:
        """List all subscriber names."""
        return list(self._subscribers.keys())

    def _redact_logs(self, logs: list[Log]) -> list[Log]:
        """Apply redaction to logs if redactor is configured.

        Creates copies of logs with redacted message, error, and data fields.

        Args:
            logs: List of Log objects to redact.

        Returns:
            List of Log objects with sensitive content redacted.
        """
        if not self.redactor:
            return logs

        from .logs import Log as LogClass

        redacted_logs = []
        for log in logs:
            # Convert to dict and redact sensitive fields
            log_dict = log.to_dict(mode="json")

            # Redact message field
            if log_dict.get("message"):
                log_dict["message"] = self.redactor.redact(log_dict["message"])

            # Redact error field
            if log_dict.get("error"):
                log_dict["error"] = self.redactor.redact(log_dict["error"])

            # Redact data field values (recursive)
            if log_dict.get("data"):
                log_dict["data"] = self._redact_dict(log_dict["data"])

            # Create new log from redacted dict
            redacted_logs.append(LogClass.from_dict(log_dict))

        return redacted_logs

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively redact string values in a dictionary."""
        if not self.redactor:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.redactor.redact(value)
            elif isinstance(value, dict):
                result[key] = self._redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self._redact_dict(item)
                        if isinstance(item, dict)
                        else (self.redactor.redact(item) if isinstance(item, str) else item)
                    )
                    for item in value
                ]
            else:
                result[key] = value
        return result

    async def broadcast(self, logs: list[Log]) -> dict[str, int]:
        """Broadcast logs to all subscribers.

        If a redactor is configured, sensitive content in logs is redacted
        before being sent to subscribers.

        Args:
            logs: List of Log objects to broadcast

        Returns:
            Dict mapping subscriber name to count of logs written
        """
        if not logs:
            return {}

        if not self._subscribers:
            logger.warning("No subscribers registered, logs not broadcasted")
            return {}

        # Apply redaction if configured
        broadcast_logs = self._redact_logs(logs)

        results: dict[str, int] = {}

        if self.config.parallel:
            # Parallel broadcast
            from lionpride.libs import concurrency

            async def _send(name: str, sub: LogSubscriber) -> tuple[str, int]:
                try:
                    count = await sub.receive(broadcast_logs)
                    return name, count
                except Exception as e:
                    logger.error(f"Subscriber {name} failed: {e}")
                    if self.config.fail_fast:
                        raise
                    return name, 0

            tasks = [_send(name, sub) for name, sub in self._subscribers.items()]
            outcomes = await concurrency.gather(*tasks, return_exceptions=True)

            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    continue
                name, count = outcome
                results[name] = count
        else:
            # Sequential broadcast
            for name, subscriber in self._subscribers.items():
                try:
                    count = await subscriber.receive(broadcast_logs)
                    results[name] = count
                except Exception as e:
                    logger.error(f"Subscriber {name} failed: {e}")
                    results[name] = 0
                    if self.config.fail_fast:
                        break

        return results

    async def close(self) -> None:
        """Close all subscribers."""
        for subscriber in self._subscribers.values():
            try:
                await subscriber.close()
            except Exception as e:
                logger.error(f"Error closing subscriber {subscriber.name}: {e}")
        self._subscribers.clear()

    def __repr__(self) -> str:
        return f"LogBroadcaster(subscribers={list(self._subscribers.keys())})"
