# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operations - Executable operations for LLM-powered workflows.

Core components:
- ExecutableOperation: Composes Operation (data) + Event (lifecycle)
- OperationDispatcher: Registry mapping operation types to factories
- Operation: Pure data node for operation parameters
- Builder: Fluent API for constructing operation graphs
- flow: Dependency-aware graph execution

Example:
    >>> # Operation graph
    >>> from lionpride.operations import Builder, flow
    >>> builder = Builder()
    >>> builder.add("step1", "custom_op", {"param": "value"})
    >>> builder.add("step2", "custom_op", {"param": "value2"})
    >>> builder.depends_on("step2", "step1")
    >>> graph = builder.build()
    >>> results = await flow(session, branch, graph)
"""

from .builder import Builder, OperationGraphBuilder
from .dispatcher import OperationDispatcher, get_dispatcher, register_operation
from .executable import ExecutableOperation
from .flow import DependencyAwareExecutor, flow, flow_stream
from .models import ActionRequestModel, ActionResponseModel, Reason
from .node import Operation, OperationType, create_operation
from .operate import (
    Operative,
    ReactResult,
    ReactStep,
    create_action_operative,
    create_operative_from_model,
    generate,
    operate,
    react,
)

__all__ = (
    "ActionRequestModel",
    "ActionResponseModel",
    "Builder",
    "DependencyAwareExecutor",
    "ExecutableOperation",
    "Operation",
    "OperationDispatcher",
    "OperationGraphBuilder",
    "OperationType",
    "Operative",
    "ReactResult",
    "ReactStep",
    "Reason",
    "create_action_operative",
    "create_operation",
    "create_operative_from_model",
    "flow",
    "flow_stream",
    "generate",
    "get_dispatcher",
    "operate",
    "react",
    "register_operation",
)
