# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operations module with lazy loading for fast import.

Note: Names that conflict with submodule file names (flow, builder, node, registry)
must be eagerly imported because Python's import machinery finds submodules first,
bypassing __getattr__. See PEP 562.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Eagerly import items that conflict with submodule names
# These MUST be imported at module level to prevent submodule shadowing
from .builder import Builder, OperationGraphBuilder
from .flow import DependencyAwareExecutor, flow, flow_stream
from .node import Operation, OperationType, create_operation
from .registry import OperationRegistry

# Lazy import mapping for non-conflicting names
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # From lionpride.rules (external dependency)
    "ActionRequest": ("lionpride.rules", "ActionRequest"),
    "ActionResponse": ("lionpride.rules", "ActionResponse"),
    "Reason": ("lionpride.rules", "Reason"),
    # operate subpackage (no conflict - it's a directory)
    "ActParams": ("lionpride.operations.operate", "ActParams"),
    "CommunicateParams": ("lionpride.operations.operate", "CommunicateParams"),
    "GenerateParams": ("lionpride.operations.operate", "GenerateParams"),
    "HandleUnmatched": ("lionpride.operations.operate", "HandleUnmatched"),
    "InterpretParams": ("lionpride.operations.operate", "InterpretParams"),
    "OperateParams": ("lionpride.operations.operate", "OperateParams"),
    "ParseParams": ("lionpride.operations.operate", "ParseParams"),
    "ReactParams": ("lionpride.operations.operate", "ReactParams"),
    "ReactResult": ("lionpride.operations.operate", "ReactResult"),
    "ReactStep": ("lionpride.operations.operate", "ReactStep"),
    "ReactStepResponse": ("lionpride.operations.operate", "ReactStepResponse"),
    "communicate": ("lionpride.operations.operate", "communicate"),
    "generate": ("lionpride.operations.operate", "generate"),
    "interpret": ("lionpride.operations.operate", "interpret"),
    "operate": ("lionpride.operations.operate", "operate"),
    "parse": ("lionpride.operations.operate", "parse"),
    "react": ("lionpride.operations.operate", "react"),
    "react_stream": ("lionpride.operations.operate", "react_stream"),
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

    raise AttributeError(f"module 'lionpride.operations' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis (includes all for IDE support)
if TYPE_CHECKING:
    from lionpride.rules import ActionRequest, ActionResponse, Reason

    from .operate import (
        ActParams,
        CommunicateParams,
        GenerateParams,
        HandleUnmatched,
        InterpretParams,
        OperateParams,
        ParseParams,
        ReactParams,
        ReactResult,
        ReactStep,
        ReactStepResponse,
        communicate,
        generate,
        interpret,
        operate,
        parse,
        react,
        react_stream,
    )

__all__ = (
    "ActParams",
    "ActionRequest",
    "ActionResponse",
    "Builder",
    "CommunicateParams",
    "DependencyAwareExecutor",
    "GenerateParams",
    "HandleUnmatched",
    "InterpretParams",
    "OperateParams",
    "Operation",
    "OperationGraphBuilder",
    "OperationRegistry",
    "OperationType",
    "ParseParams",
    "ReactParams",
    "ReactResult",
    "ReactStep",
    "ReactStepResponse",
    "Reason",
    "communicate",
    "create_operation",
    "flow",
    "flow_stream",
    "generate",
    "interpret",
    "operate",
    "parse",
    "react",
    "react_stream",
)
