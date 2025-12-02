# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field

from lionpride.core import Element, Pile

from .capabilities import parse_assignment
from .form import Form

__all__ = ("Report",)

# Fields that are part of Report infrastructure, not output schemas
_REPORT_FIELDS = {
    "id",
    "created_at",
    "metadata",
    "assignment",
    "form_assignments",
    "instruction",
    "input_fields",
    "output_fields",
    "forms",
    "completed_forms",
    "available_data",
}


class Report(Element):
    """Workflow orchestrator - schedules forms based on field availability.

    Subclass and define output schemas as class attributes:

        class MyReport(Report):
            analysis: AnalysisModel  # Pydantic model
            score: float             # Primitive type

            assignment = "input -> analysis, score"
            form_assignments = ["input -> analysis", "analysis -> score"]

    The system introspects class annotations to get field -> type mapping.
    """

    # Declaration (can be overridden as class attributes)
    assignment: str = Field(
        default="",
        description="Overall workflow: 'inputs -> final_outputs'",
    )
    form_assignments: list[str] = Field(
        default_factory=list,
        description="List of form assignments: ['a,b->c', 'c->d', ...]",
    )
    instruction: str = Field(
        default="",
        description="Overall workflow goal/instruction for LLM context",
    )

    # Derived
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)

    # Runtime state
    forms: Pile[Form] = Field(
        default_factory=lambda: Pile(item_type=Form),
        description="All forms in the workflow",
    )
    completed_forms: Pile[Form] = Field(
        default_factory=lambda: Pile(item_type=Form),
        description="Completed forms",
    )
    available_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Current state of all field values",
    )

    def model_post_init(self, _: Any) -> None:
        """Parse assignment and create forms."""
        if not self.assignment:
            return

        # Parse overall assignment (ignore branch/operation/resources for report-level)
        parsed = parse_assignment(self.assignment)
        self.input_fields = parsed.input_fields
        self.output_fields = parsed.output_fields

        # Create forms from form_assignments
        for fa in self.form_assignments:
            form = Form(assignment=fa)
            self.forms.include(form)

    def get_field_type(self, field: str) -> type | None:
        """Get the type annotation for an output field.

        Introspects class annotations to find the type.
        Unwraps Optional[X] / X | None to return X.
        Returns None if field is not declared.
        """
        try:
            hints = get_type_hints(self.__class__)
            if field in hints and field not in _REPORT_FIELDS:
                hint = hints[field]
                # Unwrap Optional[X] / Union[X, None] / X | None
                origin = get_origin(hint)
                # Handle both typing.Union and types.UnionType (Python 3.10+ | syntax)
                if origin is Union or isinstance(hint, types.UnionType):
                    args = get_args(hint)
                    # Filter out NoneType
                    non_none = [a for a in args if a is not type(None)]
                    if len(non_none) == 1:
                        return non_none[0]
                return hint
        except Exception:
            pass
        return None

    def get_request_model(self, field: str) -> type[BaseModel] | None:
        """Get request model for an output field.

        Returns the type if it's a BaseModel subclass, None otherwise.
        """
        field_type = self.get_field_type(field)
        if field_type is not None:
            try:
                if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                    return field_type
            except TypeError:
                pass
        return None

    def initialize(self, *, instruction: str | None = None, **inputs: Any) -> None:
        """Provide initial input data and optional workflow instruction.

        Args:
            instruction: Overall workflow goal (passed to LLM for context)
            **inputs: Initial field values

        Raises:
            ValueError: If required input is missing
        """
        if instruction:
            self.instruction = instruction

        for field in self.input_fields:
            if field not in inputs:
                raise ValueError(f"Missing required input: '{field}'")
            self.available_data[field] = inputs[field]

    def next_forms(self) -> list[Form]:
        """Get forms that are ready to execute.

        Returns:
            List of forms with all inputs available
        """
        workable = []
        for form in self.forms:
            if form.is_workable(self.available_data):
                workable.append(form)
        return workable

    def complete_form(self, form: Form) -> None:
        """Mark a form as completed and update available data.

        Args:
            form: The completed form

        Raises:
            ValueError: If form is not filled
        """
        if not form.filled:
            raise ValueError("Form is not filled")

        # Move to completed
        self.completed_forms.include(form)

        # Update available data with form outputs
        output_data = form.get_output_data()
        self.available_data.update(output_data)

        # Also store the raw output under the primary output field
        if form.output_fields and form.output is not None:
            primary_output = form.output_fields[0]
            if primary_output not in self.available_data:
                self.available_data[primary_output] = form.output

    def is_complete(self) -> bool:
        """Check if all output fields are available.

        Returns:
            True if workflow is complete
        """
        return all(field in self.available_data for field in self.output_fields)

    def get_deliverable(self) -> dict[str, Any]:
        """Get final deliverable based on output_fields.

        Returns:
            Dict of output field values
        """
        return {f: self.available_data.get(f) for f in self.output_fields}

    def get_all_data(self) -> dict[str, Any]:
        """Get all accumulated data.

        Returns:
            Complete data dict
        """
        return self.available_data.copy()

    @property
    def progress(self) -> tuple[int, int]:
        """Get progress as (completed, total).

        Returns:
            Tuple of (completed forms, total forms)
        """
        return len(self.completed_forms), len(self.forms)

    def __repr__(self) -> str:
        completed, total = self.progress
        return f"Report('{self.assignment}', {completed}/{total} forms)"
