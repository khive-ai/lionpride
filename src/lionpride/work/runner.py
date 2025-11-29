# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Runner - executes forms using lionpride operations.

Provides:
- flow_report: Run report via compiled graph (parallel-capable)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.operations.operate import (
    GenerateParams,
    OperateParams,
)

from .form import Form
from .report import Report

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("flow_report",)


async def flow_report(
    session: Session,
    branch: Branch | str,
    report: Report,
    *,
    max_concurrent: int | None = None,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute report via compiled graph with DependencyAwareExecutor.

    This compiles the report to an operation graph and executes using
    the standard flow infrastructure with automatic dependency resolution
    and parallel execution.

    Args:
        session: Session for services
        branch: Default branch (forms can override via branch_name in DSL)
        report: Report to execute
        max_concurrent: Max concurrent forms (None = unlimited)
        reason: Include reasoning in outputs
        actions: Enable tool use
        verbose: Print progress

    Returns:
        Final deliverable dict
    """
    from lionpride.operations import Builder
    from lionpride.operations.flow import flow

    # Build operation graph from forms
    builder = Builder()

    # Track which form produces which field
    field_producers: dict[str, str] = {}
    form_by_name: dict[str, Form] = {}

    # First pass: register producers
    for form in report.forms:
        primary_output = form.output_fields[0] if form.output_fields else str(form.id)[:8]
        form_by_name[primary_output] = form
        for output_field in form.output_fields:
            field_producers[output_field] = primary_output

    # Second pass: build operations with dependencies
    for form in report.forms:
        primary_output = form.output_fields[0] if form.output_fields else str(form.id)[:8]

        # Find dependencies from field dataflow
        depends_on = []
        for input_field in form.input_fields:
            if input_field in field_producers:
                producer = field_producers[input_field]
                if producer not in depends_on and producer != primary_output:
                    depends_on.append(producer)

        # Determine branch for this form
        form_branch = form.branch_name if form.branch_name else branch

        # Get request model from report's class annotations
        request_model = report.get_request_model(primary_output)

        # Build instruction from assignment
        instruction = f"Execute: {form.assignment}"

        # Determine capability from request_model name (lowercase convention)
        capability = request_model.__name__.lower() if request_model else primary_output

        params = OperateParams(
            generate=GenerateParams(
                instruction=instruction,
                request_model=request_model,
            ),
            capabilities={capability},  # Explicit capability for security model
            reason=reason,
            actions=actions,
        )

        # Add operation to builder
        builder.add(
            name=primary_output,
            operation="operate",
            parameters=params,
            depends_on=depends_on if depends_on else None,
            branch=form_branch,
            metadata={
                "form_id": str(form.id),
                "assignment": form.assignment,
            },
        )

    # Build graph
    graph = builder.build()

    if verbose:
        print(f"Compiled {len(form_by_name)} forms to graph")
        print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Execute via flow (uses DependencyAwareExecutor + CompletionStream)
    results = await flow(
        session=session,
        branch=branch,
        graph=graph,
        context=report.available_data,
        max_concurrent=max_concurrent,
        verbose=verbose,
    )

    # Update report with results
    for name, result in results.items():
        if name in form_by_name:
            form = form_by_name[name]
            form.fill(output=result)
            report.complete_form(form)

    return report.get_deliverable()
