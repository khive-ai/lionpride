# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operation - Pure data node for DAG execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from lionpride import HashableModel, Node

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
    """Content for Operation nodes.

    Attributes:
        operation: Operation type (generate, operate, etc.)
        parameters: Operation parameters (dict or Pydantic model)
    """

    operation: OperationType | str = Field(..., description="Operation type to execute")
    parameters: dict[str, Any] | BaseModel = Field(
        default_factory=dict,
        description="Parameters for the operation",
    )


class Operation(Node):
    """Pure data node for operation graphs.

    Node with OperationContent - no execution logic, just data for DAG.

    Attributes:
        content: OperationContent (operation type + parameters)
        metadata: Additional metadata (name, graph_id, etc.)

    Example:
        >>> content = OperationContent(operation="generate", parameters={"instruction": "Hi"})
        >>> op = Operation(content=content)
        >>> op.content.operation  # "generate"
        >>> op.content.parameters  # {"instruction": "Hi"}
    """

    content: OperationContent


def create_operation(
    operation: OperationType | str,
    parameters: dict[str, Any] | BaseModel | None = None,
    **kwargs,
) -> Operation:
    """Create an Operation node.

    Args:
        operation: Operation type
        parameters: Operation parameters
        **kwargs: Additional Operation fields (metadata, etc.)

    Returns:
        Operation instance

    Example:
        >>> op = create_operation("generate", {"instruction": "Hello"})
        >>> op = create_operation("operate", params_model, timeout=30.0)
    """
    content = OperationContent(
        operation=operation,
        parameters=parameters or {},
    )
    return Operation(content=content, **kwargs)
