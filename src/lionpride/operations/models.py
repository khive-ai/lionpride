# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Action models for structured tool calling in operations."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from lionpride.types import HashableModel

__all__ = (
    "ActionRequestModel",
    "ActionResponseModel",
    "Reason",
)


class ActionRequestModel(HashableModel):
    """Action/tool call request from LLM in structured output.

    This represents a function call that the LLM wants to execute.
    It's part of the structured output schema, not a message content type.

    Attributes:
        function: Name of the function to call (must match tool_schemas)
        arguments: Arguments to pass to the function

    Example:
        >>> request = ActionRequestModel(function="multiply", arguments={"a": 5, "b": 3})
    """

    function: str | None = Field(
        None,
        description=(
            "Name of function from available tool_schemas. "
            "CRITICAL: Never invent function names. Only use functions "
            "provided in the tool schemas."
        ),
    )
    arguments: dict[str, Any] | None = Field(
        None,
        description=(
            "Arguments dictionary matching the function's schema. "
            "Keys must match parameter names from tool_schemas."
        ),
    )


class ActionResponseModel(HashableModel):
    """Action/tool call response after execution.

    This represents the result of executing a tool call.
    It's filled in after act() executes the requested tools.

    Attributes:
        function: Name of the function that was executed
        arguments: Arguments that were passed to the function
        output: Result returned by the function (or error if failed)

    Example:
        >>> response = ActionResponseModel(
        ...     function="multiply", arguments={"a": 5, "b": 3}, output=15
        ... )
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


class Reason(HashableModel):
    """Reasoning/explanation field for chain-of-thought.

    Optional field that can be added to structured outputs to capture
    the LLM's reasoning process before taking actions.

    Attributes:
        reasoning: The LLM's explanation of its reasoning
        confidence: Optional confidence score (0.0 to 1.0)

    Example:
        >>> reason = Reason(
        ...     reasoning="I need to multiply 5 by 3 to get the result", confidence=0.95
        ... )
    """

    reasoning: str = Field(
        default="",
        description=(
            "Explain your reasoning step-by-step before taking actions. "
            "Why are you choosing these functions? What is your plan?"
        ),
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score for this reasoning (0.0 to 1.0)",
    )
