# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Services types module with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # backend
    "Calling": ("lionpride.services.types.backend", "Calling"),
    "NormalizedResponse": ("lionpride.services.types.backend", "NormalizedResponse"),
    "ServiceBackend": ("lionpride.services.types.backend", "ServiceBackend"),
    "ServiceConfig": ("lionpride.services.types.backend", "ServiceConfig"),
    # endpoint
    "APICalling": ("lionpride.services.types.endpoint", "APICalling"),
    "Endpoint": ("lionpride.services.types.endpoint", "Endpoint"),
    "EndpointConfig": ("lionpride.services.types.endpoint", "EndpointConfig"),
    # hook
    "HookBroadcaster": ("lionpride.services.types.hook", "HookBroadcaster"),
    "HookEvent": ("lionpride.services.types.hook", "HookEvent"),
    "HookPhase": ("lionpride.services.types.hook", "HookPhase"),
    "HookRegistry": ("lionpride.services.types.hook", "HookRegistry"),
    "get_handler": ("lionpride.services.types.hook", "get_handler"),
    "validate_hooks": ("lionpride.services.types.hook", "validate_hooks"),
    "validate_stream_handlers": (
        "lionpride.services.types.hook",
        "validate_stream_handlers",
    ),
    # imodel
    "iModel": ("lionpride.services.types.imodel", "iModel"),
    # registry
    "ServiceRegistry": ("lionpride.services.types.registry", "ServiceRegistry"),
    # tool
    "Tool": ("lionpride.services.types.tool", "Tool"),
    "ToolCalling": ("lionpride.services.types.tool", "ToolCalling"),
    "ToolConfig": ("lionpride.services.types.tool", "ToolConfig"),
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

    raise AttributeError(f"module 'lionpride.services.types' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from .backend import Calling, NormalizedResponse, ServiceBackend, ServiceConfig
    from .endpoint import APICalling, Endpoint, EndpointConfig
    from .hook import (
        HookBroadcaster,
        HookEvent,
        HookPhase,
        HookRegistry,
        get_handler,
        validate_hooks,
        validate_stream_handlers,
    )
    from .imodel import iModel
    from .registry import ServiceRegistry
    from .tool import Tool, ToolCalling, ToolConfig

__all__ = [
    "APICalling",
    "Calling",
    "Endpoint",
    "EndpointConfig",
    "HookBroadcaster",
    "HookEvent",
    "HookPhase",
    "HookRegistry",
    "NormalizedResponse",
    "ServiceBackend",
    "ServiceConfig",
    "ServiceRegistry",
    "Tool",
    "ToolCalling",
    "ToolConfig",
    "get_handler",
    "iModel",
    "validate_hooks",
    "validate_stream_handlers",
]
