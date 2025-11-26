# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Logging system for conversation and API tracking.

Provides structured logging for:
- API calls (model, request, response, timing, tokens)
- Conversation events (message added, branch created, etc.)
- Errors and warnings
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

__all__ = ("Log", "LogStore", "LogType")


class LogType(str, Enum):
    """Types of log entries."""

    API_CALL = "api_call"
    MESSAGE = "message"
    OPERATION = "operation"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Log(BaseModel):
    """Single log entry."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    log_type: LogType
    source: str = Field(
        default="", description="Source of the log (branch ID, operation name, etc.)"
    )

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "log_type": self.log_type.value,
            "source": self.source,
        }
        for field in [
            "model",
            "provider",
            "request",
            "response",
            "duration_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "message",
            "data",
            "error",
        ]:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        return result


class LogStore:
    """Thread-safe log storage with filtering and export capabilities."""

    def __init__(self, max_logs: int | None = None):
        """Initialize log store.

        Args:
            max_logs: Maximum logs to keep (None = unlimited, default=10000)
        """
        self._logs: list[Log] = []
        self._max_logs = max_logs or 10000

    def add(self, log: Log) -> None:
        """Add a log entry."""
        self._logs.append(log)
        # Trim if over max
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs :]

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
        """Filter logs by criteria."""
        result = self._logs

        if log_type is not None:
            result = [log for log in result if log.log_type == log_type]

        if source is not None:
            result = [log for log in result if source in log.source]

        if since is not None:
            result = [log for log in result if log.timestamp >= since]

        if until is not None:
            result = [log for log in result if log.timestamp <= until]

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
        return [log.to_dict() for log in self._logs]

    def dump(self, path: str, *, clear: bool = False) -> int:
        """Dump logs to JSON file.

        Args:
            path: File path to write
            clear: If True, clear logs after dump

        Returns:
            Number of logs dumped
        """
        import json

        logs = self.to_list()
        with open(path, "w") as f:
            json.dump(logs, f, indent=2, default=str)

        count = len(logs)
        if clear:
            self.clear()

        return count

    def clear(self) -> int:
        """Clear all logs. Returns count of cleared logs."""
        count = len(self._logs)
        self._logs = []
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

    def __repr__(self) -> str:
        return f"LogStore(logs={len(self._logs)})"
