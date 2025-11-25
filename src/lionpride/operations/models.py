# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

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
    """Action/tool call request from LLM structured output."""

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
    """Action/tool call response after execution."""

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
    """Reasoning/explanation field for chain-of-thought."""

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
