# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Models for rule-based validation.

These models define the canonical formats for structured data
that flows through the validation pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from lionpride.types import HashableModel

__all__ = (
    "ActionRequest",
    "ActionResponse",
)


class ActionRequest(HashableModel):
    """Canonical action/tool call request format.

    This is the normalized format that ActionRequestRule produces.
    Raw LLM outputs in various formats (OpenAI, Anthropic, etc.)
    are converted to this canonical form.

    Usage:
        # Define spec with ActionRequest type
        action_spec = Spec(ActionRequest, name="action_request")

        # Validator auto-assigns ActionRequestRule
        validated = await validator.validate_operable(data, operable)
        # → {"action_request": ActionRequest(function="...", arguments={...})}
    """

    function: str = Field(
        ...,
        description=(
            "Name of function from available tool_schemas. "
            "CRITICAL: Never invent function names. Only use functions "
            "provided in the tool schemas."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments dictionary matching the function's schema. "
            "Keys must match parameter names from tool_schemas."
        ),
    )


class ActionResponse(HashableModel):
    """Action/tool call response after execution.

    Contains the function that was called, arguments passed,
    and the output (result or error).
    """

    function: str = Field(
        default="",
        description="Name of function that was executed",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments that were passed to the function",
    )
    output: Any = Field(
        default=None,
        description="Function output (success) or error message (failure)",
    )
