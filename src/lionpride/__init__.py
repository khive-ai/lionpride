# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""lionpride: Production-ready multi-agent workflow orchestration framework.

This module uses lazy loading to minimize import time for CLI usage.
Actual imports happen when attributes are first accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .protocols import implements

__version__ = "1.0.0a11"

# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core primitives
    "Edge": ("lionpride.core", "Edge"),
    "Element": ("lionpride.core", "Element"),
    "Event": ("lionpride.core", "Event"),
    "EventStatus": ("lionpride.core", "EventStatus"),
    "Execution": ("lionpride.core", "Execution"),
    "Flow": ("lionpride.core", "Flow"),
    "Graph": ("lionpride.core", "Graph"),
    "Node": ("lionpride.core", "Node"),
    "Pile": ("lionpride.core", "Pile"),
    "Progression": ("lionpride.core", "Progression"),
    # Session
    "Branch": ("lionpride.session", "Branch"),
    "Message": ("lionpride.session", "Message"),
    "Session": ("lionpride.session", "Session"),
    # Operations
    "Builder": ("lionpride.operations", "Builder"),
    # Services
    "ServiceRegistry": ("lionpride.services", "ServiceRegistry"),
    "Endpoint": ("lionpride.services.types", "Endpoint"),
    "Tool": ("lionpride.services.types", "Tool"),
    "iModel": ("lionpride.services.types", "iModel"),
    # Types
    "ConversionMode": ("lionpride.types", "ConversionMode"),
    "DataClass": ("lionpride.types", "DataClass"),
    "Enum": ("lionpride.types", "Enum"),
    "HashableModel": ("lionpride.types", "HashableModel"),
    "MaybeSentinel": ("lionpride.types", "MaybeSentinel"),
    "MaybeUndefined": ("lionpride.types", "MaybeUndefined"),
    "MaybeUnset": ("lionpride.types", "MaybeUnset"),
    "Meta": ("lionpride.types", "Meta"),
    "ModelConfig": ("lionpride.types", "ModelConfig"),
    "Operable": ("lionpride.types", "Operable"),
    "Params": ("lionpride.types", "Params"),
    "Spec": ("lionpride.types", "Spec"),
    "Undefined": ("lionpride.types", "Undefined"),
    "UndefinedType": ("lionpride.types", "UndefinedType"),
    "Unset": ("lionpride.types", "Unset"),
    "UnsetType": ("lionpride.types", "UnsetType"),
    "is_sentinel": ("lionpride.types", "is_sentinel"),
    "not_sentinel": ("lionpride.types", "not_sentinel"),
}

# Lazy module imports
_LAZY_MODULES: dict[str, str] = {
    "ln": "lionpride.ln",
    "concurrency": "lionpride.libs.concurrency",
    "schema_handlers": "lionpride.libs.schema_handlers",
    "string_handlers": "lionpride.libs.string_handlers",
}

# Cache for loaded attributes
_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    # Check cache first
    if name in _LOADED:
        return _LOADED[name]

    # Check lazy imports
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    # Check lazy modules
    if name in _LAZY_MODULES:
        from importlib import import_module

        module = import_module(_LAZY_MODULES[name])
        _LOADED[name] = module
        return module

    raise AttributeError(f"module 'lionpride' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis (IDE autocomplete, mypy)
if TYPE_CHECKING:
    from . import ln as ln
    from .core import (
        Edge,
        Element,
        Event,
        EventStatus,
        Execution,
        Flow,
        Graph,
        Node,
        Pile,
        Progression,
    )
    from .libs import concurrency, schema_handlers, string_handlers
    from .operations import Builder
    from .services import ServiceRegistry
    from .services.types import Endpoint, Tool, iModel
    from .session import Branch, Message, Session
    from .types import (
        ConversionMode,
        DataClass,
        Enum,
        HashableModel,
        MaybeSentinel,
        MaybeUndefined,
        MaybeUnset,
        Meta,
        ModelConfig,
        Operable,
        Params,
        Spec,
        Undefined,
        UndefinedType,
        Unset,
        UnsetType,
        is_sentinel,
        not_sentinel,
    )

__all__ = (
    "Branch",
    "Builder",
    "ConversionMode",
    "DataClass",
    "Edge",
    "Element",
    "Endpoint",
    "Enum",
    "Event",
    "EventStatus",
    "Execution",
    "Flow",
    "Graph",
    "HashableModel",
    "MaybeSentinel",
    "MaybeUndefined",
    "MaybeUnset",
    "Message",
    "Meta",
    "ModelConfig",
    "Node",
    "Operable",
    "Params",
    "Pile",
    "Progression",
    "ServiceRegistry",
    "Session",
    "Spec",
    "Tool",
    "Undefined",
    "UndefinedType",
    "Unset",
    "UnsetType",
    "__version__",
    "concurrency",
    "iModel",
    "implements",
    "is_sentinel",
    "ln",
    "not_sentinel",
    "schema_handlers",
    "string_handlers",
)
