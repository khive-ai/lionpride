# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Output sinks for streaming content.

Provides abstraction for streaming output destinations,
supporting callbacks, buffers, and file outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import UUID


class OutputSink(ABC):
    """Abstract base for streaming output destinations.

    OutputSinks receive chunks from streaming operations and
    route them to their destinations (callbacks, buffers, files).
    """

    @abstractmethod
    async def write(self, chunk: str, event_id: UUID) -> None:
        """Write a chunk to the output.

        Args:
            chunk: Content chunk to write
            event_id: ID of the event producing this chunk
        """
        ...

    async def close(self) -> None:  # noqa: B027
        """Close the sink (optional cleanup)."""
        pass


class CallbackSink(OutputSink):
    """Output sink that calls an async callback for each chunk.

    Useful for real-time UI updates or custom processing.
    """

    def __init__(self, callback: Callable[[str, UUID], Awaitable[None]]):
        """Initialize with async callback.

        Args:
            callback: Async function called with (chunk, event_id)
        """
        self._callback = callback

    async def write(self, chunk: str, event_id: UUID) -> None:
        """Call the callback with chunk content."""
        await self._callback(chunk, event_id)


class SyncCallbackSink(OutputSink):
    """Output sink that calls a sync callback for each chunk.

    Useful when the callback doesn't need to be async.
    """

    def __init__(self, callback: Callable[[str, UUID], None]):
        """Initialize with sync callback.

        Args:
            callback: Sync function called with (chunk, event_id)
        """
        self._callback = callback

    async def write(self, chunk: str, event_id: UUID) -> None:
        """Call the sync callback with chunk content."""
        self._callback(chunk, event_id)


class BufferSink(OutputSink):
    """Output sink that buffers chunks by event ID.

    Useful for accumulating streamed content for later processing.
    """

    def __init__(self):
        """Initialize empty buffer."""
        self._buffers: dict[UUID, list[str]] = {}

    async def write(self, chunk: str, event_id: UUID) -> None:
        """Buffer chunk content by event ID."""
        if event_id not in self._buffers:
            self._buffers[event_id] = []
        self._buffers[event_id].append(chunk)

    def get_buffer(self, event_id: UUID) -> str:
        """Get accumulated content for an event.

        Args:
            event_id: Event ID to get buffer for

        Returns:
            Accumulated string content
        """
        return "".join(self._buffers.get(event_id, []))

    def get_chunks(self, event_id: UUID) -> list[str]:
        """Get raw chunk list for an event.

        Args:
            event_id: Event ID to get chunks for

        Returns:
            List of chunk strings
        """
        return self._buffers.get(event_id, []).copy()

    def clear(self, event_id: UUID | None = None) -> None:
        """Clear buffer(s).

        Args:
            event_id: Specific event to clear, or None for all
        """
        if event_id is None:
            self._buffers.clear()
        elif event_id in self._buffers:
            del self._buffers[event_id]

    def list_events(self) -> list[UUID]:
        """List all event IDs with buffered content."""
        return list(self._buffers.keys())


class FileSink(OutputSink):
    """Output sink that writes chunks to a file.

    Creates one file per event ID with configurable naming.
    """

    def __init__(
        self,
        output_dir: str | Path,
        filename_pattern: str = "stream_{event_id}.txt",
    ):
        """Initialize file sink.

        Args:
            output_dir: Directory to write files to
            filename_pattern: Pattern for filenames, {event_id} is replaced
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._pattern = filename_pattern
        self._handles: dict[UUID, Any] = {}

    async def write(self, chunk: str, event_id: UUID) -> None:
        """Write chunk to event's file."""
        if event_id not in self._handles:
            filename = self._pattern.format(event_id=str(event_id)[:8])
            filepath = self._output_dir / filename
            self._handles[event_id] = open(filepath, "w")  # noqa: SIM115

        self._handles[event_id].write(chunk)
        self._handles[event_id].flush()

    async def close(self) -> None:
        """Close all file handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class MultiSink(OutputSink):
    """Output sink that fans out to multiple sinks.

    Useful for simultaneously writing to buffer and callback.
    """

    def __init__(self, sinks: list[OutputSink]):
        """Initialize with multiple sinks.

        Args:
            sinks: List of sinks to write to
        """
        self._sinks = sinks

    async def write(self, chunk: str, event_id: UUID) -> None:
        """Write to all sinks."""
        for sink in self._sinks:
            await sink.write(chunk, event_id)

    async def close(self) -> None:
        """Close all sinks."""
        for sink in self._sinks:
            await sink.close()

    def add_sink(self, sink: OutputSink) -> None:
        """Add a sink to the fan-out."""
        self._sinks.append(sink)

    def remove_sink(self, sink: OutputSink) -> bool:
        """Remove a sink from the fan-out."""
        try:
            self._sinks.remove(sink)
            return True
        except ValueError:
            return False


__all__ = (
    "BufferSink",
    "CallbackSink",
    "FileSink",
    "MultiSink",
    "OutputSink",
    "SyncCallbackSink",
)
