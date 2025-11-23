# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .core import Element, Event, Processor

__all__ = (
    "FailingTestEvent",
    "SimpleTestEvent",
    "SlowTestEvent",
    "StreamingTestEvent",
    "TestElement",
    "TestProcessor",
)


class TestElement(Element):
    """Simple Element subclass for testing.

    This class is in _testing.py (not conftest.py) because it needs a proper
    import path for lion_class serialization to work correctly.
    """

    __test__ = False  # Tell pytest not to collect this as a test class

    value: int = 0
    name: str = "test"


class SimpleTestEvent(Event):
    """Simple Event that returns a configurable value."""

    return_value: Any = None
    streaming: bool = False

    async def _invoke(self) -> Any:
        return self.return_value


class FailingTestEvent(Event):
    """Event that raises a configurable exception."""

    error_message: str = "Test error"
    error_type: type[Exception] = ValueError
    streaming: bool = False

    async def _invoke(self) -> Any:
        raise self.error_type(self.error_message)


class SlowTestEvent(Event):
    """Event that takes time to complete."""

    delay: float = 0.1
    return_value: Any = "completed"
    streaming: bool = False

    async def _invoke(self) -> Any:
        import anyio

        await anyio.sleep(self.delay)
        return self.return_value


class StreamingTestEvent(Event):
    """Event that yields values via async generator."""

    stream_count: int = 3
    streaming: bool = True

    async def _invoke(self) -> Any:
        raise NotImplementedError("Use stream() instead")

    async def stream(self):
        import anyio

        from .core.event import EventStatus

        for i in range(self.stream_count):
            await anyio.sleep(0.01)
            yield i
        self.execution.status = EventStatus.COMPLETED
        self.execution.response = f"streamed {self.stream_count} items"


class TestProcessor(Processor):
    """Basic Processor for SimpleTestEvent."""

    event_type = SimpleTestEvent
