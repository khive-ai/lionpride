# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lionpride.core import Node
from lionpride.types import HashableModel

__all__ = ("Operation", "OperationContent", "OperationType", "create_operation")

OperationType = Literal[
    "generate",
    "operate",
    "chat",
    "parse",
    "ReAct",
    "select",
    "translate",
    "interpret",
    "act",
]


class OperationContent(HashableModel):
    """Content for Operation nodes."""

    operation: OperationType | str = Field(..., description="Operation type to execute")
    parameters: dict[str, Any] | BaseModel = Field(
        default_factory=dict,
        description="Parameters for the operation",
    )


class Operation(Node):
    """Pure data node for operation graphs."""

    content: OperationContent


def create_operation(
    operation: OperationType | str,
    parameters: dict[str, Any] | BaseModel | None = None,
    **kwargs,
) -> Operation:
    """Create an Operation node."""
    content = OperationContent(
        operation=operation,
        parameters=parameters or {},
    )
    return Operation(content=content, **kwargs)
