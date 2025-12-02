# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lionpride.errors import AccessError, ConfigurationError, ValidationError
from lionpride.rules import ActionRequest, ActionResponse
from lionpride.types import Operable, Spec

from .types import GenerateParams, OperateParams, ParseParams, ReactParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = (
    "ReactResult",
    "ReactStep",
    "ReactStepResponse",
    "build_step_operable",
    "react",
    "react_stream",
)

logger = logging.getLogger(__name__)

# React protocol capabilities (required for react to function)
REACT_PROTOCOL_CAPABILITIES = {"reasoning", "action_requests", "is_done"}

REACT_FIRST_STEP_PROMPT = """You can perform multiple reason-action steps.
Set is_done=false to continue. You have {steps_remaining} steps remaining."""

REACT_CONTINUE_PROMPT = """Continue reasoning based on previous results.
You have {steps_remaining} steps remaining. Set is_done=true when complete."""

REACT_FINAL_PROMPT = """This is your last step."""


class ReactStep(BaseModel):
    """Single step in a ReAct loop."""

    step: int = Field(..., description="Step number (1-indexed)")
    reasoning: str | None = Field(default=None, description="LLM reasoning")
    actions_requested: list[ActionRequest] = Field(
        default_factory=list, description="Actions requested"
    )
    actions_executed: list[ActionResponse] = Field(
        default_factory=list, description="Action results"
    )
    intermediate_options: dict[str, Any] | None = Field(
        default=None, description="Intermediate deliverables"
    )
    is_final: bool = Field(default=False, description="Whether this is the final step")


class ReactResult(BaseModel):
    """Result from a ReAct loop."""

    steps: list[ReactStep] = Field(default_factory=list, description="Execution steps")
    total_steps: int = Field(default=0, description="Total steps executed")
    completed: bool = Field(default=False, description="Whether loop completed normally")
    reason_stopped: str = Field(default="", description="Why loop stopped")


class ReactStepResponse(BaseModel):
    """Response model for each ReAct step (LLM output)."""

    reasoning: str | None = Field(default=None, description="Reasoning for this step")
    action_requests: list[ActionRequest] = Field(
        default_factory=list, description="Tool calls to execute"
    )
    is_done: bool = Field(default=False, description="Set true when complete")


def build_intermediate_operable(
    options: list[type[BaseModel]] | type[BaseModel],
    *,
    listable: bool = False,
    nullable: bool = True,
) -> Operable:
    """Build Operable for intermediate response options.

    Each user-provided model becomes a field in the intermediate options
    structure. Uses Spec.from_model() for clean model-to-spec conversion.

    Args:
        options: Model(s) for intermediate deliverables
        listable: Whether options can be lists
        nullable: Whether options default to None

    Returns:
        Operable for the IntermediateOptions nested model
    """
    if not isinstance(options, list):
        options = [options]

    specs = []
    for opt in options:
        spec = Spec.from_model(opt, nullable=nullable, listable=listable, default=None)
        specs.append(spec)

    return Operable(specs=tuple(specs), name="IntermediateOptions")


def build_step_operable(
    intermediate_options: list[type[BaseModel]] | type[BaseModel] | None = None,
    *,
    intermediate_listable: bool = False,
    intermediate_nullable: bool = True,
) -> Operable:
    """Build Operable for ReactStepResponse with optional intermediate options.

    Creates a dynamic Operable that includes:
    - Core step fields (reasoning, action_requests, is_done)
    - Optional intermediate_response_options as nested model

    Args:
        intermediate_options: Model(s) for intermediate deliverables
        intermediate_listable: Whether intermediate options can be lists
        intermediate_nullable: Whether intermediate options default to None

    Returns:
        Operable for the step response model
    """
    specs = [
        Spec(str, name="reasoning", description="Your reasoning").as_optional(),
        Spec(list[ActionRequest], name="action_requests", default_factory=list),
        Spec(bool, name="is_done", default=False, description="Set true when complete"),
    ]

    if intermediate_options:
        intermediate_operable = build_intermediate_operable(
            intermediate_options,
            listable=intermediate_listable,
            nullable=intermediate_nullable,
        )
        IntermediateModel = intermediate_operable.create_model()
        specs.append(Spec(IntermediateModel, name="intermediate_response_options").as_optional())

    return Operable(specs=tuple(specs), name="ReactStepResponse")


async def react_stream(
    session: Session,
    branch: Branch | str,
    params: ReactParams,
) -> AsyncGenerator[ReactStep, None]:
    """ReAct streaming - yields intermediate steps.

    Core async generator that yields ReactStep after each operate() call.
    Use react() wrapper if you just want the final result.

    Args:
        session: Session for services and message storage.
        branch: Branch for conversation history.
        params: ReactParams with flat params (inherits from OperateParams).

    Yields:
        ReactStep for each step in the ReAct loop.

    Raises:
        ValidationError: If params invalid.
        ConfigurationError: If model not configured.
        AccessError: If branch lacks required capabilities.
    """
    from .factory import operate

    # 1. Validate params
    if params._is_sentinel(params.generate):
        raise ValidationError("react requires 'generate' params")

    gen_params = params.generate
    instruction = gen_params.instruction
    if not instruction:
        raise ValidationError("react requires 'instruction' in generate params")

    imodel = gen_params.imodel or session.default_generate_model
    imodel_kwargs = gen_params.imodel_kwargs or {}
    context = gen_params.context

    if params._is_sentinel(imodel) and session.default_generate_model is None:
        raise ConfigurationError(
            "react requires 'imodel' in generate params or session.default_generate_model"
        )

    # Extract model_name for API calls
    model_name = imodel_kwargs.get("model_name") or imodel_kwargs.get("model")
    if not model_name and hasattr(imodel, "name"):
        model_name = imodel.name
    if not model_name:
        raise ConfigurationError(
            "react requires 'model_name' in imodel_kwargs or imodel with .name attribute"
        )

    # 2. Resolve branch
    b_ = session.get_branch(branch)

    # 3. Build step operable and determine required capabilities
    step_operable = build_step_operable(
        intermediate_options=params.intermediate_response_options,
        intermediate_listable=params.intermediate_listable,
        intermediate_nullable=params.intermediate_nullable,
    )

    # Required capabilities = all fields in step operable
    required_capabilities = step_operable.allowed()

    # 4. Validate capabilities against branch (security gate)
    if not required_capabilities.issubset(b_.capabilities):
        missing = required_capabilities - b_.capabilities
        raise AccessError(
            f"Branch '{b_.name}' missing react capabilities: {missing}",
            details={
                "requested": sorted(required_capabilities),
                "available": sorted(b_.capabilities),
            },
        )

    # 5. Prepare step kwargs
    step_kwargs = {k: v for k, v in imodel_kwargs.items() if k not in ("model_name", "model")}
    verbose = params.return_trace
    max_steps = params.max_steps

    if verbose:
        logger.info(f"React starting with capabilities: {required_capabilities}")

    # 6. ReAct loop
    for step_num in range(1, max_steps + 1):
        steps_remaining = max_steps - step_num
        is_last_step = steps_remaining == 0

        if verbose:
            logger.info(f"ReAct Step {step_num}/{max_steps} ({steps_remaining} remaining)")

        step = ReactStep(step=step_num)

        # Build step instruction with remaining steps info
        if step_num == 1:
            step_instruction = f"{instruction}\n\n{REACT_FIRST_STEP_PROMPT.format(steps_remaining=steps_remaining)}"
        elif is_last_step:
            step_instruction = REACT_FINAL_PROMPT
        else:
            step_instruction = REACT_CONTINUE_PROMPT.format(steps_remaining=steps_remaining)

        try:
            # Build OperateParams for this step (flat inheritance)
            operate_params = OperateParams(
                generate=GenerateParams(
                    imodel=imodel,
                    instruction=step_instruction,
                    context=context,
                    imodel_kwargs={"model": model_name, **step_kwargs},
                ),
                parse=ParseParams(),
                operable=step_operable,
                capabilities=required_capabilities,
                auto_fix=params.auto_fix,
                strict_validation=False,
                actions=True,
            )

            # Call operate (raises on error - no dict handling)
            operate_result = await operate(session, b_, operate_params)

            if verbose:
                logger.debug(f"Operate result: {operate_result}")

            # Extract step data
            if hasattr(operate_result, "reasoning"):
                step.reasoning = operate_result.reasoning
                if verbose and step.reasoning:
                    logger.info(f"Reasoning: {step.reasoning[:200]}...")

            if hasattr(operate_result, "action_requests") and operate_result.action_requests:
                step.actions_requested = operate_result.action_requests

            if hasattr(operate_result, "action_responses") and operate_result.action_responses:
                step.actions_executed = operate_result.action_responses
                if verbose:
                    for resp in step.actions_executed:
                        logger.info(f"Tool {resp.function}: {str(resp.output)[:100]}...")

            # Extract intermediate options if present
            if hasattr(operate_result, "intermediate_response_options"):
                iro = operate_result.intermediate_response_options
                if iro is not None:
                    if hasattr(iro, "model_dump"):
                        step.intermediate_options = iro.model_dump(exclude_none=True)
                    elif isinstance(iro, dict):
                        step.intermediate_options = {k: v for k, v in iro.items() if v is not None}
                    if verbose and step.intermediate_options:
                        logger.info(
                            f"Intermediate options: {list(step.intermediate_options.keys())}"
                        )

            # Check if done
            is_done = getattr(operate_result, "is_done", False) or is_last_step
            if is_done:
                step.is_final = True
                yield step
                return

            yield step

        except Exception as e:
            if verbose:
                logger.exception(f"Error at step {step_num}")
            step.reasoning = f"Error: {e}"
            yield step
            return


async def react(
    session: Session,
    branch: Branch | str,
    params: ReactParams,
) -> ReactResult:
    """ReAct loop - collect all steps into ReactResult.

    For final structured output, follow up with communicate().
    """
    result: ReactResult = ReactResult()

    async for step in react_stream(session, branch, params):
        result.steps.append(step)
        if step.is_final:
            result.completed = True
            result.reason_stopped = "Loop completed"
            break

    result.total_steps = len(result.steps)
    if not result.completed and not result.reason_stopped:
        result.reason_stopped = f"Max steps ({params.max_steps}) reached"

    return result
