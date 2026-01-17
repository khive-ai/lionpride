# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Services module with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Calling": ("lionpride.services.types.backend", "Calling"),
    "ServiceBackend": ("lionpride.services.types.backend", "ServiceBackend"),
    "ServiceRegistry": ("lionpride.services.types.registry", "ServiceRegistry"),
    "Tool": ("lionpride.services.types.tool", "Tool"),
    "iModel": ("lionpride.services.types.imodel", "iModel"),
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

    raise AttributeError(f"module 'lionpride.services' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from .types import Calling, ServiceBackend, ServiceRegistry, Tool
    from .types.imodel import iModel

__all__ = [
    "Calling",
    "ServiceBackend",
    "ServiceRegistry",
    "Tool",
    "iModel",
]
