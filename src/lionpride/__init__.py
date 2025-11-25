# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001
# Import order matters due to circular dependency resolution

# Submodules
from . import ln as ln
from .libs import (
    concurrency as concurrency,
    schema_handlers as schema_handlers,
    string_handlers as string_handlers,
)

# Core primitives (no dependencies)
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

# Protocols (no dependencies)
from .protocols import (
    Adaptable,
    AdapterRegisterable,
    Allowable,
    AsyncAdaptable,
    AsyncAdapterRegisterable,
    Containable,
    Deserializable,
    Hashable,
    Invocable,
    Observable,
    Serializable,
    implements,
)

# Errors (no dependencies)
from .errors import (
    ConfigurationError,
    ConnectionError,
    ExecutionError,
    ExistsError,
    LionprideError,
    NotFoundError,
    QueueFullError,
    TimeoutError,
    ValidationError,
)

# Types (depends on core, protocols)
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

# Services (depends on core, types)
from .services import (
    Calling,
    Endpoint,
    EndpointConfig,
    ServiceBackend,
    ServiceRegistry,
    Tool,
    iModel,
)

# Session (depends on core, types, services)
from .session import (
    ActionRequestContent,
    ActionResponseContent,
    AssistantResponseContent,
    Branch,
    InstructionContent,
    Message,
    MessageContent,
    MessageRole,
    SenderRecipient,
    Session,
    SystemContent,
    prepare_messages_for_chat,
)

# Operations (depends on session, services, types)
from .operations import (
    Builder,
    Operation,
    OperationRegistry,
    OperationType,
    Operative,
    ReactResult,
    ReactStep,
    communicate,
    flow,
    flow_stream,
    generate,
    operate,
    react,
)

__all__ = (
    # Core primitives
    "Edge",
    "Element",
    "Event",
    "EventStatus",
    "Execution",
    "Flow",
    "Graph",
    "Node",
    "Pile",
    "Progression",
    # Protocols
    "Adaptable",
    "AdapterRegisterable",
    "Allowable",
    "AsyncAdaptable",
    "AsyncAdapterRegisterable",
    "Containable",
    "Deserializable",
    "Hashable",
    "Invocable",
    "Observable",
    "Serializable",
    "implements",
    # Errors
    "ConfigurationError",
    "ConnectionError",
    "ExecutionError",
    "ExistsError",
    "LionprideError",
    "NotFoundError",
    "QueueFullError",
    "TimeoutError",
    "ValidationError",
    # Types
    "ConversionMode",
    "DataClass",
    "Enum",
    "HashableModel",
    "MaybeSentinel",
    "MaybeUndefined",
    "MaybeUnset",
    "Meta",
    "ModelConfig",
    "Operable",
    "Params",
    "Spec",
    "Undefined",
    "UndefinedType",
    "Unset",
    "UnsetType",
    "is_sentinel",
    "not_sentinel",
    # Services
    "Calling",
    "Endpoint",
    "EndpointConfig",
    "ServiceBackend",
    "ServiceRegistry",
    "Tool",
    "iModel",
    # Session
    "ActionRequestContent",
    "ActionResponseContent",
    "AssistantResponseContent",
    "Branch",
    "InstructionContent",
    "Message",
    "MessageContent",
    "MessageRole",
    "SenderRecipient",
    "Session",
    "SystemContent",
    "prepare_messages_for_chat",
    # Operations
    "Builder",
    "Operation",
    "OperationRegistry",
    "OperationType",
    "Operative",
    "ReactResult",
    "ReactStep",
    "communicate",
    "flow",
    "flow_stream",
    "generate",
    "operate",
    "react",
    # Submodules
    "concurrency",
    "ln",
    "schema_handlers",
    "string_handlers",
)
