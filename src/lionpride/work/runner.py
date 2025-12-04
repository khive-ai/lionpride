# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.libs.concurrency import Semaphore, gather
from lionpride.operations.operate import (
    GenerateParams,
    OperateParams,
    operate,
)
from lionpride.types import Operable, Spec

from .form import Form
from .report import Report

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("flow_report",)


async def _execute_form(
    session: Session,
    branch: Branch,
    form: Form,
    report: Report,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
) -> Any:
    """Execute a single form with current context from report.

    Args:
        session: Session for services
        branch: Branch for execution
        form: Form to execute
        report: Report (for context and request_model lookup)
        reason: Include reasoning
        actions: Enable tool use
        verbose: Print progress

    Returns:
        Operation result
    """
    primary_output = form.output_fields[0] if form.output_fields else str(form.id)[:8]

    # Get request model from report's class annotations
    request_model = report.get_request_model(primary_output)

    # Build context from report.available_data (always current)
    # Serialize Pydantic models to dicts for proper YAML rendering
    context = {}
    for field in form.input_fields:
        if field in report.available_data:
            value = report.available_data[field]
            if hasattr(value, "model_dump"):
                context[field] = value.model_dump()
            else:
                context[field] = value

    # Get model from form resources
    imodel = form.resources.get_gen_model()

    # Resolve tools from form resources (False, True, or list[str])
    tools = form.resources.resolve_tools(branch)

    # Build operable for validation
    operable = None
    if request_model:
        spec = Spec(request_model, name=primary_output)
        operable = Operable(specs=(spec,), name=f"{primary_output.title()}Response")

    # Build instruction: use model's docstring if available, else report instruction
    instruction = report.instruction or "Complete the task based on the provided context."
    if request_model and request_model.__doc__:
        # Use model's docstring as the specific instruction for this form
        instruction = request_model.__doc__.strip()

    # Build params - pass request_model for schema rendering
    params = OperateParams(
        generate=GenerateParams(
            instruction=instruction,
            context=context if context else None,
            request_model=request_model,  # Schema rendered into instruction
            imodel=imodel,
        ),
        operable=operable,
        capabilities={primary_output},
        reason=reason,
        actions=actions,
        tools=tools,  # Resolved from form.resources
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Executing form: {primary_output}")
        print(f"{'=' * 60}")
        print(f"INSTRUCTION: {instruction}")
        print(f"CONTEXT KEYS: {list(context.keys()) if context else 'None'}")
        print(f"REQUEST_MODEL: {request_model.__name__ if request_model else 'None'}")

        # Show the actual rendered user message
        msg = params.generate.instruction_message
        rendered = msg.content.render()
        print("\n--- RENDERED USER MESSAGE ---")
        print(rendered)
        print("--- END USER MESSAGE ---\n")
        print(f"{'=' * 60}\n")

    # Execute directly via operate
    result = await operate(session, branch, params)

    if verbose:
        print(f"Completed form: {primary_output}")

    return result


async def flow_report(
    session: Session,
    report: Report,
    *,
    branch: Branch | str | None = None,
    max_concurrent: int | None = None,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute report with explicit context management.

    Uses Report's scheduling (next_forms) to find ready forms,
    executes them with current context, and updates available_data.
    This is simpler than graph-based flow and context is always explicit.

    Args:
        session: Session for services
        report: Report to execute
        branch: Default branch (forms can override via DSL prefix)
        max_concurrent: Max concurrent forms (None = unlimited)
        reason: Include reasoning in outputs
        actions: Enable tool use
        verbose: Print progress

    Returns:
        Final deliverable dict
    """
    # Resolve default branch
    branch = session.default_branch if branch is None else session.get_branch(branch)

    if verbose:
        print(f"Executing report: {report}")
        print(f"Forms: {len(report.forms)}")

    # Execute until complete
    while not report.is_complete():
        ready_forms = report.next_forms()

        if not ready_forms:
            # Check if there are unexecuted forms - indicates a deadlock
            pending = [f for f in report.forms if f not in report.completed_forms]
            if pending:
                raise RuntimeError(
                    f"Deadlock: {len(pending)} forms cannot execute. "
                    f"Missing inputs or circular dependencies."
                )
            break

        if verbose:
            print(f"Ready forms: {[f.output_fields[0] for f in ready_forms]}")

        # Execute ready forms (parallel if multiple)
        if len(ready_forms) == 1:
            # Single form - execute directly
            form = ready_forms[0]
            form_branch = session.get_branch(form.branch_name) if form.branch_name else branch
            result = await _execute_form(
                session, form_branch, form, report, reason=reason, actions=actions, verbose=verbose
            )
            form.fill(output=result)
            report.complete_form(form)
        else:
            # Multiple forms - execute in parallel (respecting max_concurrent)
            async def execute_one(form: Form) -> tuple[Form, Any]:
                form_branch = session.get_branch(form.branch_name) if form.branch_name else branch
                result = await _execute_form(
                    session,
                    form_branch,
                    form,
                    report,
                    reason=reason,
                    actions=actions,
                    verbose=verbose,
                )
                return form, result

            if max_concurrent:
                # Use semaphore for concurrency limit
                sem = Semaphore(max_concurrent)

                async def limited_execute(
                    form: Form, semaphore: Semaphore = sem
                ) -> tuple[Form, Any]:
                    async with semaphore:
                        return await execute_one(form)

                tasks = [limited_execute(f) for f in ready_forms]
            else:
                tasks = [execute_one(f) for f in ready_forms]

            results = await gather(*tasks)

            # Update report with results
            for form, result in results:
                form.fill(output=result)
                report.complete_form(form)

    return report.get_deliverable()
