# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import partial
from typing import TYPE_CHECKING, Any

from lionpride.ln import alcall
from lionpride.operations import ParseParams
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


__all__ = ("flow_report", "stream_flow_report")


async def _execute_form(
    session: Session,
    branch: Branch,
    form: Form,
    report: Report,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
    structure_format: str | None = None,
    custom_parser: Any = None,
    custom_renderer: Any = None,
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
            imodel=form.resources.resolve_gen_model(branch),
            structure_format=structure_format,
            custom_renderer=custom_renderer,
        ),
        parse=ParseParams(
            imodel=form.resources.resolve_parse_model(branch),
            target_keys=list(request_model.model_fields.keys()) if request_model else None,
            structure_format=structure_format,
            custom_parser=custom_parser,
        ),
        operable=operable,
        capabilities={primary_output},
        reason=reason,
        actions=actions,
        tools=form.resources.resolve_tools(branch),  # Resolved from form.resources
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


async def stream_flow_report(
    session: Session,
    report: Report,
    *,
    branch: Branch | str | None = None,
    max_concurrent: int | None = None,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
    structure_format: str | None = None,
    custom_parser: Any = None,
    custom_renderer: Any = None,
    throttle_period: float | None = None,
) -> AsyncGenerator[Form, None]:
    """Execute report forms, yielding each as it completes.

    Useful for streaming progress updates to frontends.
    Report state is mutated in-place; access report directly after completion.

    Yields:
        Form objects as they complete execution.
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

        _execute = partial(
            _execute_form,
            reason=reason,
            actions=actions,
            verbose=verbose,
            structure_format=structure_format,
            custom_parser=custom_parser,
            custom_renderer=custom_renderer,
        )

        if len(ready_forms) == 1:
            # Single form - execute directly
            form = ready_forms[0]
            form_branch = session.get_branch(form.branch_name) if form.branch_name else branch
            result = await _execute(session, form_branch, form, report)
            form.fill(output=result)
            report.complete_form(form)
            yield form

        else:
            # Multiple forms - execute in parallel (respecting max_concurrent)
            async def execute_one(form: Form, _exec=_execute) -> tuple[Form, Any]:
                form_branch = session.get_branch(form.branch_name) if form.branch_name else branch
                result = await _exec(session, form_branch, form, report)
                return form, result

            results = await alcall(
                ready_forms,
                execute_one,
                max_concurrent=max_concurrent,
                throttle_period=throttle_period,
            )

            # Update report with results
            for form, result in results:
                form.fill(output=result)
                report.complete_form(form)
                yield form


async def flow_report(
    session: Session,
    report: Report,
    *,
    branch: Branch | str | None = None,
    max_concurrent: int | None = None,
    reason: bool = False,
    actions: bool = False,
    verbose: bool = False,
    structure_format: str | None = None,
    custom_parser: Any = None,
    custom_renderer: Any = None,
    throttle_period: float | None = None,
) -> dict[str, Any]:
    """Execute all forms in report, return deliverable.

    For streaming/progress updates, use stream_flow_report instead.
    """
    async for _ in stream_flow_report(
        session,
        report,
        branch=branch,
        max_concurrent=max_concurrent,
        reason=reason,
        actions=actions,
        verbose=verbose,
        structure_format=structure_format,
        custom_parser=custom_parser,
        custom_renderer=custom_renderer,
        throttle_period=throttle_period,
    ):
        pass
    return report.get_deliverable()
