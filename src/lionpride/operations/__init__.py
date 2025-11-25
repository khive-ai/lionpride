# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

# Core operation primitives
# Builder and flow for operation graphs
from .builder import Builder, OperationGraphBuilder
from .flow import DependencyAwareExecutor, OperationResult, flow, flow_stream

# Models for actions
from .models import ActionRequestModel, ActionResponseModel, Reason

# Operation implementations (operate submodule)
from .operate import (
    Operative,
    ReactResult,
    ReactStep,
    communicate,
    create_action_operative,
    create_operative_from_model,
    generate,
    operate,
    react,
)
from .operation import Operation, OperationType
from .registry import OperationRegistry
from .types import CommunicateParam, GenerateParam, OperateParam, ReactParam

__all__ = (
    # Models
    "ActionRequestModel",
    "ActionResponseModel",
    # Builder/Flow
    "Builder",
    # Params
    "CommunicateParam",
    "DependencyAwareExecutor",
    "GenerateParam",
    "OperateParam",
    # Core
    "Operation",
    "OperationGraphBuilder",
    "OperationRegistry",
    "OperationResult",
    "OperationType",
    # Operate
    "Operative",
    "ReactParam",
    "ReactResult",
    "ReactStep",
    "Reason",
    "communicate",
    "create_action_operative",
    "create_operative_from_model",
    "flow",
    "flow_stream",
    "generate",
    "operate",
    "react",
)
