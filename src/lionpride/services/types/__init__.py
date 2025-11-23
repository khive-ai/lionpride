"""Service type definitions and protocols."""

from .backend import (
    Calling,
    NormalizedResponse,
    ServiceBackend,
    ServiceConfig,
)
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
