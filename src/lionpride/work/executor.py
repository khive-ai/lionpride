# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Report executor with dependency-aware scheduling and context passing.

This executor combines the DependencyAwareExecutor pattern (completion events)
with Report-specific features:
- Permission = inputs available in report.available_data
- Context passing = form outputs → available_data → next form inputs
- Pre-built Operables from Report class annotations
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from lionpride.libs import concurrency
from lionpride.libs.concurrency import CapacityLimiter, CompletionStream
from lionpride.operations import ParseParams
from lionpride.operations.operate import GenerateParams, OperateParams, operate
from lionpride.types import Operable, Spec

from .form import Form

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

    from .report import Report

__all__ = ("FormResult", "ReportExecutor", "execute_report", "stream_report")


@dataclass
class FormResult:
    """Result from a completed form in streaming execution."""

    name: str
    """Primary output field name"""
    result: Any
    """Form output (None if failed)"""
    error: Exception | None = None
    """Exception if form failed"""
    completed: int = 0
    """Number of forms completed so far"""
    total: int = 0
    """Total number of forms"""

    @property
    def success(self) -> bool:
        """Whether the form succeeded."""
        return self.error is None


class ReportExecutor:
    """Executes Report forms with dependency-aware scheduling and context passing.

    This executor:
    - Uses completion events for dependency coordination
    - Checks input availability in report.available_data (not just predecessors)
    - Updates available_data after each form (context passing)
    - Streams form completions via async generator
    """

    def __init__(
        self,
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
    ):
        """Initialize executor."""
        self.session = session
        self.report = report
        self._default_branch = branch
        self.max_concurrent = max_concurrent
        self.reason = reason
        self.actions = actions
        self.verbose = verbose
        self.structure_format = structure_format
        self.custom_parser = custom_parser
        self.custom_renderer = custom_renderer

        # Track completion - keyed by form ID
        self.completion_events: dict[UUID, concurrency.Event] = {}
        self.results: dict[UUID, Any] = {}
        self.errors: dict[UUID, Exception] = {}

        # Concurrency limiter
        self._limiter: CapacityLimiter | None = (
            CapacityLimiter(max_concurrent) if max_concurrent else None
        )

        # Initialize completion events for all forms
        for form in report.forms:
            self.completion_events[form.id] = concurrency.Event()

    def _resolve_branch(self, form: Form) -> Branch:
        """Resolve branch for form execution."""
        if form.branch_name:
            return self.session.get_branch(form.branch_name)

        if self._default_branch is None:
            return self.session.default_branch

        if isinstance(self._default_branch, str):
            return self.session.get_branch(self._default_branch)

        return self._default_branch

    def _check_for_cycles(self) -> None:
        """Raise RuntimeError if form dependencies have cycles."""
        # Map output -> producing form
        producers: dict[str, UUID] = {}
        for form in self.report.forms:
            for out in form.output_fields:
                producers[out] = form.id

        # Build deps: form_id -> set of form_ids it depends on
        deps: dict[UUID, set[UUID]] = {f.id: set() for f in self.report.forms}
        for form in self.report.forms:
            for inp in form.input_fields:
                if inp in producers and producers[inp] != form.id:
                    deps[form.id].add(producers[inp])

        # Kahn's algorithm: if we can't process all nodes, there's a cycle
        in_degree = {fid: len(dep_set) for fid, dep_set in deps.items()}

        queue = [fid for fid, deg in in_degree.items() if deg == 0]
        processed = 0
        while queue:
            node = queue.pop()
            processed += 1
            for fid, dep_set in deps.items():
                if node in dep_set:
                    in_degree[fid] -= 1
                    if in_degree[fid] == 0:
                        queue.append(fid)

        if processed < len(deps):
            raise RuntimeError("Circular dependency detected in form assignments.")

    async def _wait_for_dependencies(self, form: Form) -> None:
        """Wait for all predecessor forms to complete.

        A form's predecessors are forms whose outputs are this form's inputs.
        """
        for other_form in self.report.forms:
            if other_form.id == form.id:
                continue

            # Check if other_form produces any of this form's inputs
            for output in other_form.output_fields:
                if output in form.input_fields:
                    # other_form is a dependency
                    await self.completion_events[other_form.id].wait()
                    break

    async def _execute_form(self, form: Form) -> Form:
        """Execute a single form with dependency coordination and context passing."""
        try:
            # Wait for dependencies (forms that produce our inputs)
            await self._wait_for_dependencies(form)

            # Verify all inputs are available (catches typos, failed upstreams)
            missing = [f for f in form.input_fields if f not in self.report.available_data]
            if missing:
                raise RuntimeError(
                    f"Form '{form.output_fields[0] if form.output_fields else form.id}' "
                    f"missing required inputs: {missing}"
                )

            # Acquire limiter slot when ready to execute
            if self._limiter:
                await self._limiter.acquire()

            try:
                result = await self._invoke_form(form)
                self.results[form.id] = result

                # Context passing: update available_data
                form.fill(output=result)
                self.report.complete_form(form)

            finally:
                if self._limiter:
                    self._limiter.release()

        except Exception as e:
            self.errors[form.id] = e
            if self.verbose:
                print(
                    f"Form {form.output_fields[0] if form.output_fields else form.id} failed: {e}"
                )
            raise

        finally:
            self.completion_events[form.id].set()

        return form

    async def _invoke_form(self, form: Form) -> Any:
        """Invoke form via operate() with context from available_data."""
        primary_output = form.output_fields[0] if form.output_fields else str(form.id)[:8]

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Executing form: {primary_output}")
            print(f"{'=' * 60}")

        # Get request model from report's class annotations
        request_model = self.report.get_request_model(primary_output)

        # Build context from report.available_data
        context = {}
        for field in form.input_fields:
            if field in self.report.available_data:
                value = self.report.available_data[field]
                if hasattr(value, "model_dump"):
                    context[field] = value.model_dump()
                else:
                    context[field] = value

        # Build operable for validation
        operable = None
        if request_model:
            spec = Spec(request_model, name=primary_output)
            operable = Operable(specs=(spec,), name=f"{primary_output.title()}Response")

        # Build instruction
        instruction = self.report.instruction or "Complete the task based on the provided context."
        if request_model and request_model.__doc__:
            instruction = request_model.__doc__.strip()

        # Resolve branch
        branch = self._resolve_branch(form)

        # Build params
        params = OperateParams(
            generate=GenerateParams(
                instruction=instruction,
                context=context if context else None,
                request_model=request_model,
                imodel=form.resources.resolve_gen_model(branch),
                structure_format=self.structure_format,
                custom_renderer=self.custom_renderer,
            ),
            parse=ParseParams(
                imodel=form.resources.resolve_parse_model(branch),
                target_keys=list(request_model.model_fields.keys()) if request_model else None,
                structure_format=self.structure_format,
                custom_parser=self.custom_parser,
            ),
            operable=operable,
            capabilities={primary_output},
            reason=self.reason,
            actions=self.actions,
            tools=form.resources.resolve_tools(branch),
        )

        if self.verbose:
            print(f"INSTRUCTION: {instruction}")
            print(f"CONTEXT KEYS: {list(context.keys()) if context else 'None'}")
            print(f"REQUEST_MODEL: {request_model.__name__ if request_model else 'None'}")

        # Execute
        result = await operate(self.session, branch, params)

        if self.verbose:
            print(f"Completed form: {primary_output}")
            print(f"{'=' * 60}\n")

        return result

    async def execute(self) -> dict[str, Any]:
        """Execute all forms and return the deliverable."""
        async for _ in self.stream_execute():
            pass
        return self.report.get_deliverable()

    async def stream_execute(self) -> AsyncGenerator[FormResult, None]:
        """Execute forms, yielding results as each completes."""
        self._check_for_cycles()

        if self.verbose:
            print(f"Executing report: {self.report}")
            print(f"Forms: {len(self.report.forms)}")

        total = len(self.report.forms)
        forms = list(self.report.forms)

        # Create tasks for all forms (they handle their own dependency waiting)
        tasks = [self._execute_form(form) for form in forms]

        # Stream completions
        completed = 0
        async with CompletionStream(tasks, limit=None) as stream:
            async for idx, _ in stream:
                completed += 1
                form = forms[idx]
                name = form.output_fields[0] if form.output_fields else str(form.id)[:8]

                if form.id in self.errors:
                    yield FormResult(
                        name=name,
                        result=None,
                        error=self.errors[form.id],
                        completed=completed,
                        total=total,
                    )
                else:
                    yield FormResult(
                        name=name,
                        result=self.results.get(form.id),
                        error=None,
                        completed=completed,
                        total=total,
                    )


async def execute_report(
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
) -> dict[str, Any]:
    """Execute all forms in report, return deliverable.

    This is the new executor-based implementation. For backward compatibility,
    use flow_report from runner.py.
    """
    executor = ReportExecutor(
        session=session,
        report=report,
        branch=branch,
        max_concurrent=max_concurrent,
        reason=reason,
        actions=actions,
        verbose=verbose,
        structure_format=structure_format,
        custom_parser=custom_parser,
        custom_renderer=custom_renderer,
    )
    return await executor.execute()


async def stream_report(
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
) -> AsyncGenerator[FormResult, None]:
    """Execute report, yielding FormResult as each form completes.

    This is the new executor-based implementation with richer result metadata.
    For backward compatibility (yielding Form objects), use stream_flow_report.
    """
    executor = ReportExecutor(
        session=session,
        report=report,
        branch=branch,
        max_concurrent=max_concurrent,
        reason=reason,
        actions=actions,
        verbose=verbose,
        structure_format=structure_format,
        custom_parser=custom_parser,
        custom_renderer=custom_renderer,
    )
    async for result in executor.stream_execute():
        yield result
