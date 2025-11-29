# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Capabilities system for workflow resource management.

Provides declarative specification of what resources (APIs, tools) each
workflow step can access. Based on capability-based security principles.

DSL Grammar:
    assignment := [branch ":"] [operation "("] inputs "->" outputs [")"] ["|" resources]
    operation := "generate" | "parse" | "communicate" | "operate" | "react"
    resources := resource ("," resource)*
    resource := resource_type ":" name
    resource_type := "api" | "api_gen" | "api_parse" | "api_interpret" | "tool"
    name := identifier | "*"

Examples:
    "context -> plan"
    "orchestrator: operate(context -> plan) | api:gpt4mini"
    "planner: react(a, b -> c) | api_gen:gpt5, api_parse:gpt4, tool:*"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lionpride.session import Branch

__all__ = (
    "AmbiguousResourceError",
    "CapabilityError",
    "FormResources",
    "ParsedAssignment",
    "parse_assignment",
)


class CapabilityError(Exception):
    """Raised when capability validation fails."""

    pass


class AmbiguousResourceError(CapabilityError):
    """Raised when resource resolution is ambiguous."""

    pass


# Valid operation types (must match session.operations registry)
VALID_OPERATIONS = frozenset({"generate", "parse", "communicate", "operate", "react"})

# Valid resource type prefixes
VALID_RESOURCE_TYPES = frozenset({"api", "api_gen", "api_parse", "api_interpret", "tool"})


@dataclass(frozen=True)
class FormResources:
    """Parsed resource declarations from form assignment.

    Immutable specification of what resources a form step can access.

    Attributes:
        api: Default model for all roles (sets gen, parse, interpret if not specified)
        api_gen: Model for generation/completion
        api_parse: Model for structured extraction
        api_interpret: Model for react interpretation
        tools: Set of tool names, or "*" for all branch tools, or None for no tools
    """

    api: str | None = None
    api_gen: str | None = None
    api_parse: str | None = None
    api_interpret: str | None = None
    tools: frozenset[str] | Literal["*"] | None = None

    def get_gen_model(self) -> str | None:
        """Get model for generation (api_gen → api fallback)."""
        return self.api_gen or self.api

    def get_parse_model(self) -> str | None:
        """Get model for parsing (api_parse → api fallback)."""
        return self.api_parse or self.api

    def get_interpret_model(self) -> str | None:
        """Get model for interpretation (api_interpret → api fallback)."""
        return self.api_interpret or self.api

    def resolve_gen_model(self, branch: Branch) -> str:
        """Resolve generation model against branch resources.

        Resolution order:
        1. api_gen if specified
        2. api if specified
        3. Branch's only model (if unambiguous)
        4. Error

        Args:
            branch: Branch to resolve against

        Returns:
            Model name to use

        Raises:
            AmbiguousResourceError: If multiple models and none specified
            CapabilityError: If specified model not in branch resources
        """
        model = self.get_gen_model()
        if model:
            if model not in branch.resources:
                raise CapabilityError(
                    f"Model '{model}' not in branch resources: {branch.resources}"
                )
            return model

        # Count models in branch resources (heuristic: not tools)
        # This requires knowing which are models vs tools
        # For now, assume branch tracks this separately or we check session
        branch_models = _get_branch_models(branch)
        if len(branch_models) == 1:
            return next(iter(branch_models))
        if len(branch_models) == 0:
            raise CapabilityError("No models available in branch resources")
        raise AmbiguousResourceError(
            f"Multiple models available ({branch_models}), must specify api or api_gen"
        )

    def resolve_parse_model(self, branch: Branch) -> str:
        """Resolve parse model against branch resources."""
        model = self.get_parse_model()
        if model:
            if model not in branch.resources:
                raise CapabilityError(
                    f"Model '{model}' not in branch resources: {branch.resources}"
                )
            return model
        return self.resolve_gen_model(branch)  # Fallback to gen model

    def resolve_interpret_model(self, branch: Branch) -> str:
        """Resolve interpret model against branch resources."""
        model = self.get_interpret_model()
        if model:
            if model not in branch.resources:
                raise CapabilityError(
                    f"Model '{model}' not in branch resources: {branch.resources}"
                )
            return model
        return self.resolve_gen_model(branch)  # Fallback to gen model

    def resolve_tools(self, branch: Branch) -> set[str]:
        """Resolve tools against branch resources.

        Args:
            branch: Branch to resolve against

        Returns:
            Set of tool names to use

        Raises:
            CapabilityError: If specified tool not in branch resources
        """
        if self.tools is None:
            return set()
        if self.tools == "*":
            return _get_branch_tools(branch)
        # Validate subset
        tools = set(self.tools)
        branch_tools = _get_branch_tools(branch)
        if not tools <= branch_tools:
            missing = tools - branch_tools
            raise CapabilityError(f"Tools {missing} not in branch resources: {branch_tools}")
        return tools

    def validate_against(self, branch: Branch) -> None:
        """Validate all resources are subset of branch resources.

        Args:
            branch: Branch to validate against

        Raises:
            CapabilityError: If any resource not in branch
        """
        # Validate APIs
        for api in [self.api, self.api_gen, self.api_parse, self.api_interpret]:
            if api and api not in branch.resources:
                raise CapabilityError(f"API '{api}' not in branch resources: {branch.resources}")

        # Validate tools (unless wildcard)
        if self.tools and self.tools != "*":
            branch_resources = branch.resources
            for tool in self.tools:
                if tool not in branch_resources:
                    raise CapabilityError(
                        f"Tool '{tool}' not in branch resources: {branch_resources}"
                    )


def _get_branch_models(branch: Branch) -> set[str]:
    """Get model names from branch resources.

    This is a heuristic - ideally branch would track models vs tools separately.
    For now, we need to check against session's service registry.
    """
    # TODO: Proper separation of models vs tools in branch
    # For now, return all resources (caller may need to filter)
    return set(branch.resources)


def _get_branch_tools(branch: Branch) -> set[str]:
    """Get tool names from branch resources.

    This is a heuristic - ideally branch would track models vs tools separately.
    """
    # TODO: Proper separation of models vs tools in branch
    return set(branch.resources)


@dataclass
class ParsedAssignment:
    """Result of parsing a form assignment DSL string.

    Attributes:
        branch_name: Optional branch prefix (e.g., "orchestrator")
        operation: Operation type (default: "operate")
        input_fields: List of input field names
        output_fields: List of output field names
        resources: Parsed resource declarations
        raw: Original assignment string
    """

    branch_name: str | None
    operation: str
    input_fields: list[str]
    output_fields: list[str]
    resources: FormResources
    raw: str = ""


def parse_assignment(assignment: str) -> ParsedAssignment:
    """Parse assignment DSL into structured result.

    Supports full DSL:
        [branch:] [operation(] inputs -> outputs [)] [| resources]

    Examples:
        "a, b -> c"
        "orchestrator: a -> b"
        "operate(a -> b)"
        "orchestrator: react(a, b -> c) | api:gpt4, tool:search"
        "step: communicate(x -> y) | api_gen:gpt5, api_parse:gpt4mini, tool:*"

    Args:
        assignment: DSL string to parse

    Returns:
        ParsedAssignment with all components

    Raises:
        ValueError: If assignment is invalid
    """
    if not assignment or not assignment.strip():
        raise ValueError("Assignment cannot be empty")

    original = assignment
    assignment = assignment.strip()

    # Split on "|" to separate data flow from resources
    if "|" in assignment:
        flow_part, resources_part = assignment.split("|", 1)
        flow_part = flow_part.strip()
        resources_part = resources_part.strip()
    else:
        flow_part = assignment
        resources_part = ""

    # Parse resources
    resources = _parse_resources(resources_part) if resources_part else FormResources()

    # Parse flow part: [branch:] [operation(] inputs -> outputs [)]
    branch_name = None
    operation = "operate"

    # Check for branch prefix (colon before any parenthesis or arrow)
    arrow_pos = flow_part.find("->")
    if arrow_pos == -1:
        raise ValueError(f"Invalid assignment: '{original}'. Must contain '->'")

    paren_pos = flow_part.find("(")
    colon_pos = flow_part.find(":")

    # Colon is branch prefix only if it's before arrow and before opening paren (if any)
    if colon_pos != -1 and colon_pos < arrow_pos and (paren_pos == -1 or colon_pos < paren_pos):
        branch_name = flow_part[:colon_pos].strip()
        flow_part = flow_part[colon_pos + 1 :].strip()

    # Check for operation prefix: operation(...)
    paren_pos = flow_part.find("(")
    if paren_pos != -1:
        # Has operation prefix
        op_candidate = flow_part[:paren_pos].strip().lower()
        if op_candidate in VALID_OPERATIONS:
            operation = op_candidate
            # Remove operation( and trailing )
            flow_part = flow_part[paren_pos + 1 :].strip()
            if flow_part.endswith(")"):
                flow_part = flow_part[:-1].strip()
        else:
            raise ValueError(
                f"Invalid operation '{op_candidate}'. Valid operations: {sorted(VALID_OPERATIONS)}"
            )

    # Parse inputs -> outputs
    if "->" not in flow_part:
        raise ValueError(f"Invalid assignment: '{original}'. Must contain '->'")

    inputs_str, outputs_str = flow_part.split("->", 1)
    input_fields = [x.strip() for x in inputs_str.split(",") if x.strip()]
    output_fields = [y.strip() for y in outputs_str.split(",") if y.strip()]

    if not output_fields:
        raise ValueError(f"Assignment must have at least one output: '{original}'")

    return ParsedAssignment(
        branch_name=branch_name,
        operation=operation,
        input_fields=input_fields,
        output_fields=output_fields,
        resources=resources,
        raw=original,
    )


def _parse_resources(resources_str: str) -> FormResources:
    """Parse resources string into FormResources.

    Format: "api:name, api_gen:name, tool:name, tool:*"

    Args:
        resources_str: Comma-separated resource declarations

    Returns:
        FormResources instance

    Raises:
        ValueError: If resource format is invalid
    """
    if not resources_str.strip():
        return FormResources()

    api: str | None = None
    api_gen: str | None = None
    api_parse: str | None = None
    api_interpret: str | None = None
    tools: set[str] | Literal["*"] | None = None

    parts = [p.strip() for p in resources_str.split(",") if p.strip()]

    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid resource format '{part}'. Expected 'type:name'")

        res_type, res_name = part.split(":", 1)
        res_type = res_type.strip().lower()
        res_name = res_name.strip()

        if not res_name:
            raise ValueError(f"Resource name cannot be empty: '{part}'")

        if res_type not in VALID_RESOURCE_TYPES:
            raise ValueError(
                f"Invalid resource type '{res_type}'. Valid types: {sorted(VALID_RESOURCE_TYPES)}"
            )

        if res_type == "api":
            api = res_name
        elif res_type == "api_gen":
            api_gen = res_name
        elif res_type == "api_parse":
            api_parse = res_name
        elif res_type == "api_interpret":
            api_interpret = res_name
        elif res_type == "tool":
            if res_name == "*":
                tools = "*"
            else:
                if tools is None:
                    tools = set()
                if tools != "*":
                    tools.add(res_name)

    # Convert tools set to frozenset for immutability
    frozen_tools: frozenset[str] | Literal["*"] | None = None
    if tools == "*":
        frozen_tools = "*"
    elif tools:
        frozen_tools = frozenset(tools)

    return FormResources(
        api=api,
        api_gen=api_gen,
        api_parse=api_parse,
        api_interpret=api_interpret,
        tools=frozen_tools,
    )
