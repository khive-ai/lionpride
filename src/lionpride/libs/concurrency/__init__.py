# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Concurrency utilities with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # _cancel
    "CancelScope": ("lionpride.libs.concurrency._cancel", "CancelScope"),
    "effective_deadline": ("lionpride.libs.concurrency._cancel", "effective_deadline"),
    "fail_after": ("lionpride.libs.concurrency._cancel", "fail_after"),
    "fail_at": ("lionpride.libs.concurrency._cancel", "fail_at"),
    "move_on_after": ("lionpride.libs.concurrency._cancel", "move_on_after"),
    "move_on_at": ("lionpride.libs.concurrency._cancel", "move_on_at"),
    # _errors
    "get_cancelled_exc_class": (
        "lionpride.libs.concurrency._errors",
        "get_cancelled_exc_class",
    ),
    "is_cancelled": ("lionpride.libs.concurrency._errors", "is_cancelled"),
    "non_cancel_subgroup": (
        "lionpride.libs.concurrency._errors",
        "non_cancel_subgroup",
    ),
    "shield": ("lionpride.libs.concurrency._errors", "shield"),
    # _patterns
    "CompletionStream": ("lionpride.libs.concurrency._patterns", "CompletionStream"),
    "bounded_map": ("lionpride.libs.concurrency._patterns", "bounded_map"),
    "gather": ("lionpride.libs.concurrency._patterns", "gather"),
    "race": ("lionpride.libs.concurrency._patterns", "race"),
    "retry": ("lionpride.libs.concurrency._patterns", "retry"),
    # _primitives
    "CapacityLimiter": ("lionpride.libs.concurrency._primitives", "CapacityLimiter"),
    "Condition": ("lionpride.libs.concurrency._primitives", "Condition"),
    "Event": ("lionpride.libs.concurrency._primitives", "Event"),
    "Lock": ("lionpride.libs.concurrency._primitives", "Lock"),
    "Queue": ("lionpride.libs.concurrency._primitives", "Queue"),
    "Semaphore": ("lionpride.libs.concurrency._primitives", "Semaphore"),
    # _priority_queue
    "PriorityQueue": ("lionpride.libs.concurrency._priority_queue", "PriorityQueue"),
    "QueueEmpty": ("lionpride.libs.concurrency._priority_queue", "QueueEmpty"),
    "QueueFull": ("lionpride.libs.concurrency._priority_queue", "QueueFull"),
    # _resource_tracker
    "LeakInfo": ("lionpride.libs.concurrency._resource_tracker", "LeakInfo"),
    "LeakTracker": ("lionpride.libs.concurrency._resource_tracker", "LeakTracker"),
    "track_resource": (
        "lionpride.libs.concurrency._resource_tracker",
        "track_resource",
    ),
    "untrack_resource": (
        "lionpride.libs.concurrency._resource_tracker",
        "untrack_resource",
    ),
    # _run_async
    "run_async": ("lionpride.libs.concurrency._run_async", "run_async"),
    # _task
    "TaskGroup": ("lionpride.libs.concurrency._task", "TaskGroup"),
    "create_task_group": ("lionpride.libs.concurrency._task", "create_task_group"),
    # _utils
    "current_time": ("lionpride.libs.concurrency._utils", "current_time"),
    "is_coro_func": ("lionpride.libs.concurrency._utils", "is_coro_func"),
    "run_sync": ("lionpride.libs.concurrency._utils", "run_sync"),
    "sleep": ("lionpride.libs.concurrency._utils", "sleep"),
}

_LOADED: dict[str, object] = {}

# Re-export built-in ExceptionGroup
ExceptionGroup = ExceptionGroup


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    if name in _LOADED:
        return _LOADED[name]

    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    # Special case: ConcurrencyEvent is alias for Event
    if name == "ConcurrencyEvent":
        value = __getattr__("Event")
        _LOADED[name] = value
        return value

    raise AttributeError(f"module 'lionpride.libs.concurrency' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from ._cancel import (
        CancelScope,
        effective_deadline,
        fail_after,
        fail_at,
        move_on_after,
        move_on_at,
    )
    from ._errors import (
        get_cancelled_exc_class,
        is_cancelled,
        non_cancel_subgroup,
        shield,
    )
    from ._patterns import CompletionStream, bounded_map, gather, race, retry
    from ._primitives import CapacityLimiter, Condition, Event, Lock, Queue, Semaphore
    from ._priority_queue import PriorityQueue, QueueEmpty, QueueFull
    from ._resource_tracker import (
        LeakInfo,
        LeakTracker,
        track_resource,
        untrack_resource,
    )
    from ._run_async import run_async
    from ._task import TaskGroup, create_task_group
    from ._utils import current_time, is_coro_func, run_sync, sleep

    ConcurrencyEvent = Event

__all__ = (
    "CancelScope",
    "CapacityLimiter",
    "CompletionStream",
    "ConcurrencyEvent",
    "Condition",
    "Event",
    "ExceptionGroup",
    "LeakInfo",
    "LeakTracker",
    "Lock",
    "PriorityQueue",
    "Queue",
    "QueueEmpty",
    "QueueFull",
    "Semaphore",
    "TaskGroup",
    "bounded_map",
    "create_task_group",
    "current_time",
    "effective_deadline",
    "fail_after",
    "fail_at",
    "gather",
    "get_cancelled_exc_class",
    "is_cancelled",
    "is_coro_func",
    "move_on_after",
    "move_on_at",
    "non_cancel_subgroup",
    "race",
    "retry",
    "run_async",
    "run_sync",
    "shield",
    "sleep",
    "track_resource",
    "untrack_resource",
)
