# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core primitives with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # broadcaster
    "Broadcaster": ("lionpride.core.broadcaster", "Broadcaster"),
    # element
    "Element": ("lionpride.core.element", "Element"),
    # event
    "Event": ("lionpride.core.event", "Event"),
    "EventStatus": ("lionpride.core.event", "EventStatus"),
    "Execution": ("lionpride.core.event", "Execution"),
    # eventbus
    "EventBus": ("lionpride.core.eventbus", "EventBus"),
    "Handler": ("lionpride.core.eventbus", "Handler"),
    # flow
    "Flow": ("lionpride.core.flow", "Flow"),
    # graph
    "Edge": ("lionpride.core.graph", "Edge"),
    "EdgeCondition": ("lionpride.core.graph", "EdgeCondition"),
    "Graph": ("lionpride.core.graph", "Graph"),
    # node
    "NODE_REGISTRY": ("lionpride.core.node", "NODE_REGISTRY"),
    "Node": ("lionpride.core.node", "Node"),
    # pile
    "Pile": ("lionpride.core.pile", "Pile"),
    # processor
    "Executor": ("lionpride.core.processor", "Executor"),
    "Processor": ("lionpride.core.processor", "Processor"),
    # progression
    "Progression": ("lionpride.core.progression", "Progression"),
}

_LOADED: dict[str, object] = {}


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

    raise AttributeError(f"module 'lionpride.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from .broadcaster import Broadcaster
    from .element import Element
    from .event import Event, EventStatus, Execution
    from .eventbus import EventBus, Handler
    from .flow import Flow
    from .graph import Edge, EdgeCondition, Graph
    from .node import NODE_REGISTRY, Node
    from .pile import Pile
    from .processor import Executor, Processor
    from .progression import Progression

__all__ = [
    "NODE_REGISTRY",
    "Broadcaster",
    "Edge",
    "EdgeCondition",
    "Element",
    "Event",
    "EventBus",
    "EventStatus",
    "Execution",
    "Executor",
    "Flow",
    "Graph",
    "Handler",
    "Node",
    "Pile",
    "Processor",
    "Progression",
]
