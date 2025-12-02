# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import orjson

from .base import Rule, RuleParams, RuleQualifier
from .models import ActionRequest

__all__ = ("ActionRequestRule",)


def _get_action_request_params() -> RuleParams:
    """Default params for action request rule."""
    return RuleParams(
        apply_types={ActionRequest},
        apply_fields={"action_request", "tool_call", "function_call"},
        default_qualifier=RuleQualifier.FIELD,
        auto_fix=True,
        kw={},
    )


class ActionRequestRule(Rule):
    """Rule for validating tool/function call structures.

    Features:
    - Validates structure has function/name and arguments
    - Validates arguments is a dict
    - Optional allowed functions validation
    - Auto-extraction from various formats (OpenAI, Anthropic, etc.)

    Normalized format:
        {
            "function": "function_name",
            "arguments": {"arg1": "value1", ...}
        }

    Usage:
        rule = ActionRequestRule(allowed_functions={"get_weather", "search"})
        result = await rule.invoke(
            "action_request",
            {"function": "get_weather", "arguments": {"city": "NYC"}},
            dict
        )
    """

    def __init__(
        self,
        allowed_functions: set[str] | None = None,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize action request rule.

        Args:
            allowed_functions: Set of allowed function names (None = any)
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_action_request_params()
        super().__init__(params, **kw)
        self.allowed_functions = allowed_functions

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a valid ActionRequest instance.

        Only ActionRequest instances pass validation. Dict inputs (even in
        canonical format) fail validation to ensure perform_fix() is called,
        which converts them to ActionRequest instances.

        Raises:
            ValueError: If not a valid ActionRequest instance
        """
        # Only accept ActionRequest instances - dicts should go through perform_fix()
        if not isinstance(v, ActionRequest):
            raise ValueError(
                f"Expected ActionRequest instance, got {type(v).__name__}. "
                "Use auto_fix=True to convert dicts to ActionRequest."
            )

        # Check allowed functions
        if self.allowed_functions and v.function not in self.allowed_functions:
            raise ValueError(
                f"Function '{v.function}' not in allowed: {sorted(self.allowed_functions)}"
            )

    async def perform_fix(self, v: Any, t: type) -> ActionRequest:
        """Attempt to extract and normalize action request.

        Handles various formats:
        - OpenAI tool_calls: {"name": str, "arguments": str (JSON)}
        - Anthropic tool_use: {"name": str, "input": dict}
        - Normalized: {"function": str, "arguments": dict}
        - ActionRequest instance (pass-through)

        Returns:
            ActionRequest instance (canonical format)

        Raises:
            ValueError: If extraction fails
        """
        # Already an ActionRequest - return as-is after validation
        if isinstance(v, ActionRequest):
            await self.validate(v, t)
            return v

        # Parse JSON string
        if isinstance(v, str):
            try:
                v = orjson.loads(v)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"Failed to parse action request JSON: {e}") from e

        if not isinstance(v, Mapping):
            raise ValueError(f"Cannot convert {type(v).__name__} to action request")

        # Extract function name (support multiple formats)
        function_name = (
            v.get("function")
            or v.get("name")
            or (v.get("function", {}).get("name") if isinstance(v.get("function"), dict) else None)
        )

        if not function_name:
            raise ValueError("Cannot extract function name from action request")

        # Extract arguments (support multiple formats)
        arguments = (
            v.get("arguments")
            or v.get("input")  # Anthropic format
            or (
                v.get("function", {}).get("arguments")
                if isinstance(v.get("function"), dict)
                else None
            )
            or {}
        )

        # Parse JSON string arguments (OpenAI format)
        if isinstance(arguments, str):
            try:
                arguments = orjson.loads(arguments)
            except orjson.JSONDecodeError:
                # Keep as string if not valid JSON
                arguments = {"raw": arguments}

        if not isinstance(arguments, Mapping):
            arguments = {"value": arguments}

        # Build ActionRequest instance
        result = ActionRequest(
            function=function_name,
            arguments=dict(arguments),
        )

        # Re-validate the fixed value
        await self.validate(result, t)
        return result
