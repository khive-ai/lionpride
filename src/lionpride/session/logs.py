# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Logging system for conversation and API tracking.

Provides structured logging for:
- API calls (model, request, response, timing, tokens)
- Conversation events (message added, branch created, etc.)
- Errors and warnings

Uses Pile-based storage for O(1) UUID lookup and async context manager
for proper async locking.

Extensible via:
- LogAdapter: Persistent storage backends (SQLite WAL, PostgreSQL)
- LogBroadcaster: Multi-destination fan-out (S3, webhooks)
"""

from __future__ import annotations

import atexit
import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from lionpride.core import Element, Pile

if TYPE_CHECKING:
    from .log_adapter import LogAdapter
    from .log_broadcaster import LogBroadcaster

__all__ = ("Log", "LogStore", "LogStoreConfig", "LogType")

logger = logging.getLogger(__name__)


class LogType(str, Enum):
    """Types of log entries."""

    API_CALL = "api_call"
    MESSAGE = "message"
    OPERATION = "operation"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LogStoreConfig(BaseModel):
    """Configuration for LogStore persistence.

    Supports multiple persistence strategies:
    - File-based: JSON/JSONL files (default)
    - Adapter-based: SQLite WAL, PostgreSQL via LogAdapter
    - Broadcaster-based: Fan-out to S3, webhooks via LogBroadcaster
    """

    persist_dir: str | Path = "./data/logs"
    file_prefix: str | None = None
    capacity: int | None = None
    extension: str = ".json"
    use_timestamp: bool = True
    auto_save_on_exit: bool = True
    clear_after_dump: bool = True

    # Adapter/Broadcaster settings (configured programmatically)
    use_adapter: bool = False
    use_broadcaster: bool = False

    @field_validator("capacity", mode="before")
    @classmethod
    def _validate_non_negative(cls, value):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("Capacity must be non-negative.")
        return value

    @field_validator("extension")
    @classmethod
    def _ensure_dot_extension(cls, value):
        if not value.startswith("."):
            return "." + value
        if value not in {".json", ".jsonl"}:
            raise ValueError("Extension must be '.json' or '.jsonl'.")
        return value


class Log(Element):
    """Immutable log entry that extends Element.

    Inherits UUID identity and created_at from Element.
    Once created or restored from dict, the log is marked as read-only.

    Attributes:
        log_type: Type of log entry (API_CALL, MESSAGE, OPERATION, etc.)
        source: Source of the log (branch ID, operation name, etc.)
        model: Model name for API calls
        provider: Provider name for API calls
        request: Request data for API calls
        response: Response data for API calls
        duration_ms: Duration in milliseconds for API calls
        input_tokens: Input token count
        output_tokens: Output token count
        total_tokens: Total token count
        message: General message text
        data: Additional data dict
        error: Error message
    """

    log_type: LogType
    source: str = Field(default="", description="Source of the log")

    # API call fields
    model: str | None = None
    provider: str | None = None
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # General fields
    message: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None

    _immutable: bool = PrivateAttr(False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutation if log is immutable."""
        if getattr(self, "_immutable", False):
            raise AttributeError("This Log is immutable.")
        super().__setattr__(name, value)

    @classmethod
    def from_dict(cls, data: dict[str, Any], meta_key: str | None = None, **kwargs: Any) -> Log:
        """Create a Log from a dictionary.

        The restored log is marked as immutable.
        """
        # Handle legacy timestamp field
        data = data.copy()
        if "timestamp" in data and "created_at" not in data:
            data["created_at"] = data.pop("timestamp")

        log: Log = super().from_dict(data, meta_key=meta_key, **kwargs)  # type: ignore[assignment]
        log._immutable = True
        return log

    @classmethod
    def create(cls, content: Element | dict[str, Any], log_type: LogType = LogType.INFO) -> Log:
        """Create a new Log from content.

        Args:
            content: Element or dict to wrap in a log
            log_type: Type of log entry (default: INFO)

        Returns:
            New Log instance
        """
        if isinstance(content, Element):
            data = content.to_dict(mode="json")
        elif isinstance(content, dict):
            data = content
        else:
            logger.warning(
                "Invalid content type for log, creating empty log. "
                f"Expected Element or dict, got {type(content)}"
            )
            return cls(log_type=log_type, data={"error": "Invalid content type"})

        return cls(log_type=log_type, data=data)


class LogStore:
    """Log storage with Pile-based O(1) UUID lookup and async support.

    Uses async context manager for proper async locking via `alog()` and `adump()`.

    Extensible via:
    - LogAdapter: For persistent storage (SQLite WAL, PostgreSQL)
    - LogBroadcaster: For multi-destination fan-out (S3, webhooks)

    Example:
        store = LogStore()
        store.log_api_call(model="gpt-4", duration_ms=150.0)

        # Async usage
        await store.alog(some_log)
        await store.adump("/path/to/logs.json")

        # With adapter
        from lionpride.session.log_adapter import SQLiteWALLogAdapter
        store = LogStore()
        store.set_adapter(SQLiteWALLogAdapter())
        await store.aflush()  # Persist to adapter
    """

    def __init__(
        self,
        max_logs: int | None = None,
        config: LogStoreConfig | None = None,
        adapter: LogAdapter | None = None,
        broadcaster: LogBroadcaster | None = None,
        **kwargs,
    ):
        """Initialize log store.

        Args:
            max_logs: Maximum logs to keep (None = 10000, legacy parameter)
            config: LogStoreConfig for persistence settings
            adapter: Optional LogAdapter for persistent storage
            broadcaster: Optional LogBroadcaster for multi-destination output
            **kwargs: Passed to LogStoreConfig if config is None
        """
        if config is None:
            config = LogStoreConfig(capacity=max_logs or 10000, **kwargs)
        elif max_logs is not None:
            # Override config capacity with explicit max_logs
            config = config.model_copy(update={"capacity": max_logs})

        self._config = config
        self._logs: Pile[Log] = Pile(item_type=Log, strict_type=True)
        self._adapter = adapter
        self._broadcaster = broadcaster

        # Auto-dump on exit
        if self._config.auto_save_on_exit:
            atexit.register(self._save_at_exit)

    def set_adapter(self, adapter: LogAdapter) -> None:
        """Set the log adapter for persistent storage."""
        self._adapter = adapter
        self._config = self._config.model_copy(update={"use_adapter": True})

    def set_broadcaster(self, broadcaster: LogBroadcaster) -> None:
        """Set the log broadcaster for multi-destination output."""
        self._broadcaster = broadcaster
        self._config = self._config.model_copy(update={"use_broadcaster": True})

    @property
    def logs(self) -> Pile[Log]:
        """Access underlying Pile for direct operations."""
        return self._logs

    def add(self, log: Log) -> None:
        """Add a log entry.

        If capacity is reached, auto-dumps to file.
        """
        if self._config.capacity and len(self._logs) >= self._config.capacity:
            try:
                self.dump(clear=self._config.clear_after_dump)
            except Exception as e:
                logger.error(f"Failed to auto-dump logs: {e}")

        self._logs.include(log)

    async def alog(self, log: Log | Any) -> None:
        """Add a log asynchronously with proper locking.

        Args:
            log: Log instance or content to create a Log from
        """
        if not isinstance(log, Log):
            log = Log.create(log)

        async with self._logs:
            self.add(log)

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
    ) -> Log:
        """Log an API call."""
        log = Log(
            log_type=LogType.API_CALL,
            source=source,
            model=model,
            provider=provider,
            request=request,
            response=response,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        self.add(log)
        return log

    def log_operation(
        self,
        *,
        source: str = "",
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> Log:
        """Log an operation event."""
        log = Log(
            log_type=LogType.OPERATION,
            source=source,
            message=message,
            data=data,
        )
        self.add(log)
        return log

    def log_error(
        self,
        *,
        source: str = "",
        error: str = "",
        data: dict[str, Any] | None = None,
    ) -> Log:
        """Log an error."""
        log = Log(
            log_type=LogType.ERROR,
            source=source,
            error=error,
            data=data,
        )
        self.add(log)
        return log

    def log_info(
        self,
        *,
        source: str = "",
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> Log:
        """Log an info message."""
        log = Log(
            log_type=LogType.INFO,
            source=source,
            message=message,
            data=data,
        )
        self.add(log)
        return log

    def filter(
        self,
        *,
        log_type: LogType | None = None,
        source: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        model: str | None = None,
    ) -> list[Log]:
        """Filter logs by criteria.

        Args:
            log_type: Filter by log type
            source: Filter by source (substring match)
            since: Filter by created_at >= since
            until: Filter by created_at <= until
            model: Filter by model (substring match)

        Returns:
            Filtered list of logs
        """
        result: list[Log] = list(self._logs)

        if log_type is not None:
            result = [log for log in result if log.log_type == log_type]

        if source is not None:
            result = [log for log in result if source in log.source]

        if since is not None:
            result = [log for log in result if log.created_at >= since]

        if until is not None:
            result = [log for log in result if log.created_at <= until]

        if model is not None:
            result = [log for log in result if log.model and model in log.model]

        return result

    def get_api_calls(self) -> list[Log]:
        """Get all API call logs."""
        return self.filter(log_type=LogType.API_CALL)

    def get_errors(self) -> list[Log]:
        """Get all error logs."""
        return self.filter(log_type=LogType.ERROR)

    def to_list(self) -> list[dict[str, Any]]:
        """Export all logs as list of dicts."""
        return [log.to_dict(mode="json") for log in self._logs]

    def dump(
        self,
        path: str | Path | None = None,
        *,
        clear: bool | None = None,
    ) -> int:
        """Dump logs to JSON file.

        Args:
            path: File path to write (uses config if not provided)
            clear: If True, clear logs after dump

        Returns:
            Number of logs dumped
        """
        import json

        if not self._logs:
            logger.debug("No logs to dump.")
            return 0

        fp = Path(path) if path else self._create_path()

        try:
            # Ensure directory exists
            fp.parent.mkdir(parents=True, exist_ok=True)

            logs = self.to_list()
            with open(fp, "w") as f:
                json.dump(logs, f, indent=2, default=str)

            logger.info(f"Dumped {len(logs)} logs to {fp}")

            do_clear = self._config.clear_after_dump if clear is None else clear
            if do_clear:
                self._logs.clear()

            return len(logs)

        except Exception as e:
            if "JSON serializable" in str(e):
                logger.debug(f"Could not serialize logs to JSON: {e}")
                if clear is not False:
                    self._logs.clear()
                return 0
            logger.error(f"Failed to dump logs: {e}")
            raise

    async def adump(
        self,
        path: str | Path | None = None,
        *,
        clear: bool | None = None,
    ) -> int:
        """Asynchronously dump logs to file with proper locking."""
        async with self._logs:
            return self.dump(path=path, clear=clear)

    async def aflush(self, *, clear: bool = True) -> dict[str, int]:
        """Flush logs to adapter and/or broadcaster.

        Args:
            clear: Clear logs after successful flush

        Returns:
            Dict with counts: {"adapter": N, "broadcaster": {sub: N, ...}}
        """
        results: dict[str, Any] = {}
        logs_list = list(self._logs)

        if not logs_list:
            return results

        async with self._logs:
            # Flush to adapter
            if self._adapter:
                count = await self._adapter.write(logs_list)
                results["adapter"] = count

            # Broadcast to subscribers
            if self._broadcaster:
                broadcast_results = await self._broadcaster.broadcast(logs_list)
                results["broadcaster"] = broadcast_results

            # Clear if requested and ALL destinations succeeded
            # Don't clear on partial success to prevent data loss
            should_clear = clear and results
            if should_clear:
                # Check adapter success (if configured)
                if self._adapter and results.get("adapter", 0) == 0:
                    should_clear = False
                # Check broadcaster success (if configured)
                if self._broadcaster and "broadcaster" in results:
                    broadcast_results = results["broadcaster"]
                    # Any subscriber with 0 count indicates failure
                    if any(count == 0 for count in broadcast_results.values()):
                        should_clear = False

            if should_clear:
                self._logs.clear()

        return results

    async def _acreate_path(self) -> Path:
        """Build a file path asynchronously using ln.acreate_path."""
        from lionpride.ln import acreate_path

        # Build filename parts
        parts = []
        if self._config.file_prefix:
            parts.append(self._config.file_prefix)

        if not parts:
            parts.append("logs")

        filename = "_".join(parts)

        # Use acreate_path for async directory creation
        path = await acreate_path(
            directory=self._config.persist_dir,
            filename=filename,
            extension=self._config.extension.lstrip("."),
            timestamp=self._config.use_timestamp,
            dir_exist_ok=True,
            file_exist_ok=True,
        )

        return Path(path)

    def _create_path(self) -> Path:
        """Build a file path from config settings (sync version)."""
        from datetime import datetime

        path = Path(self._config.persist_dir)

        # Build filename
        parts = []
        if self._config.file_prefix:
            parts.append(self._config.file_prefix)

        if self._config.use_timestamp:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            parts.append(ts)

        if not parts:
            parts.append("logs")

        filename = "_".join(parts) + self._config.extension
        return path / filename

    def _save_at_exit(self) -> None:
        """Dump logs on program exit."""
        if self._logs:
            try:
                self.dump(clear=self._config.clear_after_dump)
            except Exception as e:
                if "JSON serializable" in str(e):
                    logger.debug(f"Could not serialize logs to JSON: {e}")
                else:
                    logger.error(f"Failed to save logs on exit: {e}")

    def clear(self) -> int:
        """Clear all logs. Returns count of cleared logs."""
        count = len(self._logs)
        self._logs.clear()
        return count

    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        api_calls = self.get_api_calls()
        total_tokens = sum(log.total_tokens or 0 for log in api_calls)
        total_duration = sum(log.duration_ms or 0 for log in api_calls)

        return {
            "total_logs": len(self._logs),
            "api_calls": len(api_calls),
            "errors": len(self.get_errors()),
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration,
            "models_used": list({log.model for log in api_calls if log.model}),
        }

    def __len__(self) -> int:
        return len(self._logs)

    def __iter__(self):
        return iter(self._logs)

    def __getitem__(self, key):
        """Get log by UUID or index via Pile."""
        return self._logs[key]

    def __repr__(self) -> str:
        return f"LogStore(logs={len(self._logs)})"
