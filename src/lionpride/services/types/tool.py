# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Self, get_origin, get_type_hints

from pydantic import Field, field_validator, model_validator

from lionpride import concurrency, schema_handlers

from .backend import Calling, NormalizedResponse, ServiceBackend, ServiceConfig

__all__ = ("Tool", "ToolCalling", "ToolConfig")


def _extract_json_schema_from_callable(
    func: Callable[..., Any],
    request_options: type | None = None,
) -> dict[str, Any]:
    """Generate JSON Schema from function signature or Pydantic model.

    Args:
        func: Callable to extract schema from
        request_options: Optional Pydantic model for parameters

    Returns:
        JSON Schema dict with type, properties, required fields
    """
    if request_options is not None:
        # Use Pydantic model schema
        if hasattr(request_options, "model_json_schema"):
            return request_options.model_json_schema()
        raise ValueError(f"request_options must be Pydantic model, got {type(request_options)}")

    # Build schema from function signature
    sig = inspect.signature(func)

    # Use get_type_hints to resolve string annotations
    try:
        type_hints = get_type_hints(func)
    except Exception:
        # Fallback to raw annotations if get_type_hints fails
        type_hints = {}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        # Skip variadic args
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Determine type - prefer type_hints over param.annotation
        if name in type_hints:
            param_type = _python_type_to_json_type(type_hints[name])
        elif param.annotation is not inspect.Parameter.empty:
            param_type = _python_type_to_json_type(param.annotation)
        else:
            param_type = "string"  # Default

        properties[name] = {"type": param_type}

        # Add to required if no default
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_type(annotation: Any) -> str:
    """Convert Python type annotation to JSON Schema type string.

    Args:
        annotation: Python type annotation

    Returns:
        JSON Schema type string (string, number, boolean, array, object, null)
    """
    # Handle None
    if annotation is type(None):
        return "null"

    # Get origin for generic types
    origin = get_origin(annotation)

    # Handle List/list
    if origin in (list, tuple):
        return "array"

    # Handle Dict/dict
    if origin is dict:
        return "object"

    # Handle simple types
    type_map = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    if annotation in type_map:
        return type_map[annotation]

    # Default
    return "string"


class ToolConfig(ServiceConfig):
    """Configuration for Tool backend.

    Extends ServiceConfig with tool-specific defaults.
    """

    provider: str = "tool"


class Tool(ServiceBackend):
    config: ToolConfig
    func_callable: Callable[..., Any] = Field(
        ...,
        frozen=True,
        exclude=True,
    )
    tool_schema: dict[str, Any] | None = None

    @field_validator("func_callable", mode="before")
    def _validate_func_callable(cls, value: Any) -> Callable[..., Any]:  # noqa: N805
        if not callable(value):
            raise ValueError(f"func_callable must be callable, got {type(value)}")
        if not hasattr(value, "__name__"):
            raise ValueError("func_callable must have __name__ attribute")
        return value

    @model_validator(mode="before")
    @classmethod
    def _set_defaults_from_function(cls, data: Any) -> Any:
        """Set config defaults from function if not provided."""
        if isinstance(data, dict) and "func_callable" in data:
            func = data["func_callable"]

            # Ensure func_callable is valid before using it
            # Field validator will run after this and raise if invalid
            if not (callable(func) and hasattr(func, "__name__")):
                return data

            # Initialize config if not present
            if "config" not in data:
                data["config"] = ToolConfig(provider="tool", name=func.__name__)
            elif isinstance(data["config"], dict):
                # Convert dict to ToolConfig, set name from function if not present
                config_dict = data["config"].copy()
                if "name" not in config_dict:
                    config_dict["name"] = func.__name__
                if "provider" not in config_dict:
                    config_dict["provider"] = "tool"
                data["config"] = ToolConfig(**config_dict)
            # If config is already ToolConfig, leave it as is

        return data

    @model_validator(mode="after")
    def _generate_schema(self) -> Self:
        """Generate internal JSON Schema dict with request_options as canonical source.

        Priority order:
        1. request_options (from config) - canonical source
        2. tool_schema - fallback if no request_options
        3. Auto-generate from function signature - if neither provided

        Sets tool_schema to the resolved JSON schema dict.
        """
        # Priority 1: request_options is canonical source
        if self.request_options is not None:
            self.tool_schema = self.request_options.model_json_schema()
            return self

        # Priority 2: tool_schema provided explicitly
        if self.tool_schema is not None:
            if isinstance(self.tool_schema, dict):
                return self
            raise ValueError("tool_schema must be a dict")

        # Priority 3: Auto-generate from function signature
        json_schema = _extract_json_schema_from_callable(self.func_callable, request_options=None)
        self.tool_schema = json_schema
        return self

    @property
    def function_name(self) -> str:
        """Get function name."""
        return self.func_callable.__name__

    @property
    def rendered(self) -> str:
        """Render tool schema as TypeScript for LLM consumption.

        Uses request_options as canonical source if available, otherwise tool_schema.

        Format:
            # Tool description (if present)
            TypeScript parameter definitions

        Returns:
            TypeScript-formatted schema string
        """
        # Priority 1: Use request_options (canonical source)
        if self.request_options is not None:
            schema = self.request_options.model_json_schema()
            desc = schema.get("description", "")
            params_ts = schema_handlers.typescript_schema(schema)
            return f"# {desc}\n{params_ts}" if desc else params_ts

        # Priority 2: Use tool_schema (fallback)
        if self.tool_schema:
            desc = self.tool_schema.get("description", "")
            # Check for properties at top level (auto-generated) or inside parameters (OpenAI format)
            if self.tool_schema.get("properties"):
                params_ts = schema_handlers.typescript_schema(self.tool_schema)
                return f"# {desc}\n{params_ts}" if desc else params_ts
            elif params := self.tool_schema.get("parameters", {}):
                if params.get("properties"):
                    params_ts = schema_handlers.typescript_schema(params)
                    return f"# {desc}\n{params_ts}" if desc else params_ts

            return f"# {desc}" if desc else ""

        return ""

    @property
    def event_type(self) -> type[ToolCalling]:
        """Get Event/Calling type for this backend."""
        return ToolCalling

    @property
    def required_fields(self) -> frozenset[str]:
        """Get required parameter fields.

        Priority order:
        1. request_options (canonical source)
        2. tool_schema (fallback)
        3. Function signature inspection (auto-generated)
        """
        # Priority 1: request_options (canonical source)
        if self.request_options is not None:
            schema = self.request_options.model_json_schema()
            return frozenset(schema.get("required", []))

        # Priority 2: tool_schema (fallback)
        if self.tool_schema and "required" in self.tool_schema:
            return frozenset(self.tool_schema["required"])

        # Priority 3: Inspect function signature
        try:
            sig = inspect.signature(self.func_callable)
            return frozenset(
                {
                    name
                    for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                    and param.kind
                    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                }
            )
        except Exception:
            return frozenset()

    def _to_dict(self, **kwargs) -> dict[str, Any]:
        """Not implemented - Tools contain callables which are not serializable."""
        raise NotImplementedError(
            f"Tool '{self.name}' cannot be serialized - contains callable '{self.function_name}'. "
            "Use Endpoint backends for persistence scenarios."
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs):
        """Not implemented - Tools are created from callables, not dicts."""
        raise NotImplementedError("Tool.from_dict is not supported - create from callable")

    async def call(self, arguments: dict[str, Any]) -> NormalizedResponse:
        """Execute tool callable with sync/async detection.

        Args:
            arguments: Dict of validated parameters to pass to callable

        Returns:
            NormalizedResponse wrapping the tool execution result

        Raises:
            Exception: Any exception raised by the tool callable
        """
        if concurrency.is_coro_func(self.func_callable):
            result = await self.func_callable(**arguments)
        else:
            result = await concurrency.run_sync(lambda: self.func_callable(**arguments))

        return NormalizedResponse(
            status="success",
            data=result,
            raw_response={"result": result, "arguments": arguments},
        )


class ToolCalling(Calling):
    """Tool execution - delegates to Tool.call() which handles sync/async.

    Attributes:
        backend: Tool instance (contains func_callable)
        payload: Dict of validated arguments for the tool (schema-verifiable)
    """

    backend: Tool

    @property
    def call_args(self) -> dict:
        """Get arguments for backend.call(**self.call_args).

        Returns:
            Dict with arguments for Tool.call()
        """
        return {"arguments": self.payload}
