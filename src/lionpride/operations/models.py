# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Models generated from Specs via Operable.create_model().

These models match the original hardcoded definitions but are now
dynamically generated from Spec definitions.
"""

from __future__ import annotations

from typing import Any

from lionpride.types import HashableModel, Operable, Spec

__all__ = (
    "ActionRequestModel",
    "ActionResponseModel",
    "Reason",
)

# ---------------------------------------------------------------------------
# ActionRequestModel: function (nullable), arguments (nullable)
# ---------------------------------------------------------------------------
_action_request_specs = (
    Spec(
        str,
        name="function",
        description=(
            "Name of function from available tool_schemas. "
            "CRITICAL: Never invent function names. Only use functions "
            "provided in the tool schemas."
        ),
    ).as_nullable(),
    Spec(
        dict,
        name="arguments",
        description=(
            "Arguments dictionary matching the function's schema. "
            "Keys must match parameter names from tool_schemas."
        ),
    ).as_nullable(),
)

_action_request_operable = Operable(_action_request_specs, name="ActionRequestModel")

ActionRequestModel = _action_request_operable.create_model(
    base_type=HashableModel,
    doc="Action/tool call request from LLM structured output.",
)

# ---------------------------------------------------------------------------
# ActionResponseModel: function (default ""), arguments (default {}), output (nullable)
# ---------------------------------------------------------------------------
_action_response_specs = (
    Spec(
        str,
        name="function",
        default="",
        description="Name of function that was executed",
    ),
    Spec(
        dict,
        name="arguments",
        default_factory=dict,
        description="Arguments that were passed to the function",
    ),
    Spec(
        Any,
        name="output",
        description="Function output (success) or error message (failure)",
    ).as_nullable(),
)

_action_response_operable = Operable(_action_response_specs, name="ActionResponseModel")

ActionResponseModel = _action_response_operable.create_model(
    base_type=HashableModel,
    doc="Action/tool call response after execution.",
)

# ---------------------------------------------------------------------------
# Reason: reasoning (default ""), confidence (nullable with ge/le)
# ---------------------------------------------------------------------------
_reason_specs = (
    Spec(
        str,
        name="reasoning",
        default="",
        description=(
            "Explain your reasoning step-by-step before taking actions. "
            "Why are you choosing these functions? What is your plan?"
        ),
    ),
    Spec(
        float,
        name="confidence",
        ge=0.0,
        le=1.0,
        description="Optional confidence score for this reasoning (0.0 to 1.0)",
    ).as_nullable(),
)

_reason_operable = Operable(_reason_specs, name="Reason")

Reason = _reason_operable.create_model(
    base_type=HashableModel,
    doc="Reasoning/explanation field for chain-of-thought.",
)
