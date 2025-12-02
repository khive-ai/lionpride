# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import Field

from lionpride.types import HashableModel

__all__ = (
    "ActionRequest",
    "ActionResponse",
    "Reason",
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


class Reason(HashableModel):
    """Reasoning/explanation for chain-of-thought.

    System spec for structured reasoning output. When 'reason' capability
    is granted to a branch, operations can inject this spec to require
    step-by-step reasoning from the LLM.

    Usage:
        # Grant reason capability to branch
        branch = session.create_branch(capabilities={"reason"})

        # Operations with reason=True inject this spec
        result = await session.conduct(
            "operate", branch=branch, response_model=MyOutput, reason=True
        )
        # → Response includes Reason field with reasoning + confidence
    """

    reasoning: str = Field(
        default="",
        description=(
            "Explain your reasoning step-by-step before providing your answer. "
            "What is your thought process? What factors are you considering?"
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score for this reasoning (0.0 to 1.0)",
    )
