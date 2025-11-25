# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from ..libs.concurrency import is_coro_func

logger = logging.getLogger(__name__)

__all__ = ["Broadcaster"]


class Broadcaster:
    """Singleton pub/sub with weakref-based automatic subscriber cleanup."""

    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[
        list[weakref.ref[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]]
    ] = []
    _event_type: ClassVar[type]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def subscribe(cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]) -> None:
        """Add subscriber callback (stored as weakref for automatic cleanup)."""
        # Check if callback already subscribed (compare actual callbacks, not weakrefs)
        for weak_ref in cls._subscribers:
            if weak_ref() is callback:
                return  # Already subscribed

        # Store as weakref for automatic cleanup
        # Use WeakMethod for bound methods, weakref for regular callables
        weak_callback: weakref.ref[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]
        if hasattr(callback, "__self__"):
            weak_callback = weakref.WeakMethod(callback)  # type: ignore[assignment]
        else:
            weak_callback = weakref.ref(callback)
        cls._subscribers.append(weak_callback)

    @classmethod
    def unsubscribe(
        cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
    ) -> None:
        """Remove subscriber callback."""
        # Find and remove weakref that points to this callback
        for weak_ref in list(cls._subscribers):
            if weak_ref() is callback:
                cls._subscribers.remove(weak_ref)
                return

    @classmethod
    def _cleanup_dead_refs(cls) -> list[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]:
        """Remove dead weakrefs and return list of live callbacks."""
        callbacks = []
        alive_refs = []

        for weak_ref in cls._subscribers:
            callback = weak_ref()
            if callback is not None:
                callbacks.append(callback)
                alive_refs.append(weak_ref)

        # In-place update to maintain ClassVar identity
        cls._subscribers[:] = alive_refs

        return callbacks

    @classmethod
    async def broadcast(cls, event: Any) -> None:
        """Broadcast event to all subscribers."""
        if not isinstance(event, cls._event_type):
            raise ValueError(f"Event must be of type {cls._event_type.__name__}")

        callbacks = cls._cleanup_dead_refs()

        # Broadcast to live callbacks
        for callback in callbacks:
            try:
                if is_coro_func(callback):
                    result = callback(event)
                    if result is not None:  # Coroutine functions return awaitable
                        await result
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}", exc_info=True)

    @classmethod
    def get_subscriber_count(cls) -> int:
        """Get live subscriber count (excludes garbage-collected callbacks)."""
        callbacks = cls._cleanup_dead_refs()
        return len(callbacks)
