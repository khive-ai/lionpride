# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Form - declarative unit of work with assignment-based field contracts.

Assignment DSL: "input1, input2 -> output1, output2"
- Inputs: fields required before work can start
- Outputs: fields produced by the work

Forms derive execution order from field dependencies, not explicit graph edges.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from lionpride.core import Element

__all__ = ("Form", "parse_assignment")


def parse_assignment(assignment: str) -> tuple[str | None, list[str], list[str]]:
    """Parse assignment DSL into (branch, inputs, outputs).

    Supports optional branch prefix:
    - "a, b -> c" -> (None, ["a", "b"], ["c"])
    - "orchestrator: a, b -> c" -> ("orchestrator", ["a", "b"], ["c"])

    Args:
        assignment: DSL string like "branch: a, b -> c, d" or "a, b -> c, d"

    Returns:
        Tuple of (branch_name, input_fields, output_fields)

    Raises:
        ValueError: If assignment is invalid
    """
    if not assignment or "->" not in assignment:
        raise ValueError(f"Invalid assignment: '{assignment}'. Must contain '->'")

    # Check for branch prefix (colon before arrow)
    branch_name = None
    work_assignment = assignment

    arrow_pos = assignment.find("->")
    colon_pos = assignment.find(":")

    if colon_pos != -1 and colon_pos < arrow_pos:
        # Has branch prefix
        branch_name = assignment[:colon_pos].strip()
        work_assignment = assignment[colon_pos + 1 :].strip()

    parts = work_assignment.split("->", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid assignment: '{assignment}'")

    inputs_str, outputs_str = parts
    inputs = [x.strip() for x in inputs_str.split(",") if x.strip()]
    outputs = [y.strip() for y in outputs_str.split(",") if y.strip()]

    if not outputs:
        raise ValueError(f"Assignment must have at least one output: '{assignment}'")

    return branch_name, inputs, outputs


class Form(Element):
    """Declarative unit of work with assignment-based field contracts.

    A Form declares:
    - assignment: "branch: input1, input2 -> output1" (what fields it needs and produces)

    Forms are pure data contracts. Schema/validation comes from Report.form_specs.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # Declaration
    assignment: str = Field(
        ...,
        description="Assignment DSL: 'branch: input1, input2 -> output1, output2'",
    )

    # Derived fields (computed from assignment)
    branch_name: str | None = Field(default=None)
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)

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
        """Parse assignment to derive branch, input/output fields."""
        branch, inputs, outputs = parse_assignment(self.assignment)
        self.branch_name = branch
        self.input_fields = inputs
        self.output_fields = outputs

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
                # Handle nested structure (e.g., {analysis: {analysis: {...}}})
                elif field in data and isinstance(data[field], dict) and field in data[field]:
                    result[field] = data[field][field]

        return result

    def __repr__(self) -> str:
        status = "filled" if self.filled else "pending"
        return f"Form('{self.assignment}', {status})"
