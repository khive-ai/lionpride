# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

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

        # Branch resources contain both models and tools without distinction.
        # When no model is specified, we can only auto-resolve if there's exactly one resource.
        branch_resources = _get_branch_resources(branch)
        if len(branch_resources) == 1:
            return next(iter(branch_resources))
        if len(branch_resources) == 0:
            raise CapabilityError("No resources available in branch")
        raise AmbiguousResourceError(
            f"Multiple resources available ({branch_resources}), must specify api or api_gen"
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

    def resolve_tools(self, branch: Branch) -> list[str] | bool:
        """Resolve tools for ActParams/OperateParams.

        Args:
            branch: Branch to resolve against

        Returns:
            - False: No tools
            - True: All branch tools (tool:*)
            - list[str]: Specific tools

        Raises:
            CapabilityError: If specified tool not in branch resources
        """
        if self.tools is None:
            return False
        if self.tools == "*":
            # tool:* means all tools the branch has access to
            return True
        # Validate subset against branch resources
        tools = set(self.tools)
        branch_resources = _get_branch_resources(branch)
        if not tools <= branch_resources:
            missing = tools - branch_resources
            raise CapabilityError(f"Tools {missing} not in branch resources: {branch_resources}")
        return list(self.tools)

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


def _get_branch_resources(branch: Branch) -> set[str]:
    """Get all resource names from branch.

    Note: Branch.resources contains both models and tools without distinction.
    Callers must disambiguate based on their context (e.g., service registry).
    """
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

    Supports DSL:
        [branch:] inputs -> outputs [| resources]

    Examples:
        "a, b -> c"
        "orchestrator: a -> b"
        "orchestrator: a, b -> c | api:gpt4, tool:search"
        "step: x -> y | api_gen:gpt5, api_parse:gpt4mini, tool:*"

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

    # Parse flow part: [branch:] inputs -> outputs
    branch_name = None

    # Check for branch prefix (colon before arrow)
    arrow_pos = flow_part.find("->")
    if arrow_pos == -1:
        raise ValueError(f"Invalid assignment: '{original}'. Must contain '->'")

    colon_pos = flow_part.find(":")

    # Colon is branch prefix only if it's before arrow
    if colon_pos != -1 and colon_pos < arrow_pos:
        branch_name = flow_part[:colon_pos].strip()
        flow_part = flow_part[colon_pos + 1 :].strip()

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
        operation="operate",  # Always operate - only supported operation
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
            if api is not None:
                raise ValueError(f"Duplicate 'api' declaration: already set to '{api}'")
            api = res_name
        elif res_type == "api_gen":
            if api_gen is not None:
                raise ValueError(f"Duplicate 'api_gen' declaration: already set to '{api_gen}'")
            api_gen = res_name
        elif res_type == "api_parse":
            if api_parse is not None:
                raise ValueError(f"Duplicate 'api_parse' declaration: already set to '{api_parse}'")
            api_parse = res_name
        elif res_type == "api_interpret":
            if api_interpret is not None:
                raise ValueError(
                    f"Duplicate 'api_interpret' declaration: already set to '{api_interpret}'"
                )
            api_interpret = res_name
        elif res_type == "tool":
            if res_name == "*":
                if tools is not None and tools != "*":
                    raise ValueError("Cannot mix 'tool:*' with specific tools")
                tools = "*"
            else:
                if tools == "*":
                    raise ValueError("Cannot mix specific tools with 'tool:*'")
                if tools is None:
                    tools = set()
                if res_name in tools:
                    raise ValueError(f"Duplicate tool declaration: '{res_name}'")
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
