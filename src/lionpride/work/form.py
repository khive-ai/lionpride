# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from lionpride.core import Element

from .capabilities import FormResources, parse_assignment

__all__ = ("Form", "parse_assignment")


class Form(Element):
    """Declarative unit of work with assignment-based field contracts.

    A Form declares:
    - assignment: DSL string with data flow and resources
    - Derived: branch_name, input/output fields, resources

    Forms are data contracts with capability declarations.
    Schema/validation comes from Report class attributes.

    Example:
        form = Form(assignment="orchestrator: context -> plan | api:gpt4, tool:*")
        form.input_fields   # ["context"]
        form.output_fields  # ["plan"]
        form.resources.api  # "gpt4"
        form.resources.tools  # "*"
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # Declaration
    assignment: str = Field(
        ...,
        description="Assignment DSL: '[branch:] inputs -> outputs [| resources]'",
    )

    # Derived fields (computed from assignment)
    branch_name: str | None = Field(default=None)
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    resources: FormResources = Field(default_factory=FormResources)

    # Runtime state
    output: Any = Field(
        default=None,
        description="The structured output after execution",
    )
    filled: bool = Field(
        default=False,
        description="Whether this form has been executed",
    )

    def model_post_init(self, _: Any) -> None:
        """Parse assignment to derive all fields."""
        parsed = parse_assignment(self.assignment)
        self.branch_name = parsed.branch_name
        self.input_fields = parsed.input_fields
        self.output_fields = parsed.output_fields
        self.resources = parsed.resources

    def is_workable(self, available_data: dict[str, Any]) -> bool:
        """Check if all input fields are available.

        Args:
            available_data: Currently available field values

        Returns:
            True if all inputs are available and form not yet filled
        """
        if self.filled:
            return False

        for field in self.input_fields:
            if field not in available_data:
                return False
            # Check for sentinel values
            val = available_data[field]
            if val is None:
                return False

        return True

    def get_inputs(self, available_data: dict[str, Any]) -> dict[str, Any]:
        """Extract input data for this form.

        Args:
            available_data: All available data

        Returns:
            Dict of input field values
        """
        return {f: available_data[f] for f in self.input_fields if f in available_data}

    def fill(self, output: Any) -> None:
        """Mark form as filled with output."""
        self.output = output
        self.filled = True

    def get_output_data(self) -> dict[str, Any]:
        """Extract output field values from the output.

        Returns:
            Dict mapping output field names to values
        """
        if self.output is None:
            return {}

        result = {}
        for field in self.output_fields:
            # Try to get from output model
            if hasattr(self.output, field):
                result[field] = getattr(self.output, field)
            elif hasattr(self.output, "model_dump"):
                data = self.output.model_dump()
                if field in data:
                    result[field] = data[field]

        return result

    def __repr__(self) -> str:
        status = "filled" if self.filled else "pending"
        res_parts = []
        if self.resources.api:
            res_parts.append(f"api:{self.resources.api}")
        if self.resources.tools:
            tools_str = (
                "*" if self.resources.tools == "*" else ",".join(sorted(self.resources.tools))
            )
            res_parts.append(f"tools:{tools_str}")
        res_str = f" [{', '.join(res_parts)}]" if res_parts else ""
        return f"Form('{self.assignment}'{res_str}, {status})"
