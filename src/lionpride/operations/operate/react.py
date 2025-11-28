# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""React operation - multi-step reasoning with tool calling.

Stream-first architecture:
    react_stream() - async generator yielding intermediate results
    react() - wrapper collecting all results

Composition:
    react() → react_stream() → operate() → communicate() → generate()
                                        → act() (via actions=True)

Intermediate Response Options:
    Use intermediate_response_options in ReactParams to provide structured
    intermediate deliverables. Each becomes a nullable field in step responses.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field

from lionpride.rules import ActionRequest, ActionResponse
from lionpride.types import Operable, Spec

from .types import (
    CommunicateParams,
    GenerateParams,
    OperateParams,
    ParseParams,
    ReactParams,
)

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

T = TypeVar("T")


# =============================================================================
# Prompt Templates
# =============================================================================

REACT_FIRST_STEP_PROMPT = """You can perform multiple reason-action steps for accuracy.
If you need more steps, set is_done=false. You have {steps_remaining} steps remaining.
Strategize accordingly."""

REACT_CONTINUE_PROMPT = """Continue reasoning based on previous results.
You have {steps_remaining} steps remaining. Set is_done=true when ready to provide final answer."""

REACT_FINAL_PROMPT = """This is your last step. Provide your final answer now."""


# =============================================================================
# Response Models
# =============================================================================


class ReactStep(BaseModel):
    """Single step in a ReAct loop."""

    step: int = Field(..., description="Step number (1-indexed)")
    reasoning: str | None = Field(default=None, description="LLM reasoning for this step")
    actions_requested: list[ActionRequest] = Field(
        default_factory=list, description="Actions requested by LLM"
    )
    actions_executed: list[ActionResponse] = Field(
        default_factory=list, description="Action execution results"
    )
    intermediate_options: dict[str, Any] | None = Field(
        default=None, description="Intermediate deliverables from this step"
    )
    is_final: bool = Field(default=False, description="Whether this is the final step")


class ReactResult(BaseModel, Generic[T]):
    """Result from a ReAct execution."""

    steps: list[ReactStep] = Field(default_factory=list, description="Execution steps")
    final_response: T | None = Field(default=None, description="Final structured response")
    total_steps: int = Field(default=0, description="Total steps executed")
    completed: bool = Field(default=False, description="Whether execution completed normally")
    reason_stopped: str = Field(default="", description="Why execution stopped")


class ReactStepResponse(BaseModel, Generic[T]):
    """Response model for each ReAct step.

    LLM returns this structure with reasoning, actions, and completion status.
    """

    reasoning: str | None = Field(
        default=None, description="Your reasoning about the current state and next action"
    )
    action_requests: list[ActionRequest] | None = Field(
        default=None,
        description="List of tool calls, each with 'function' and 'arguments'. Empty when done.",
    )
    is_done: bool = Field(default=False, description="Set to true when you have the final answer")
    final_answer: T | None = Field(
        default=None, description="Your final answer (only when is_done=true)"
    )


# =============================================================================
# Operable Building
# =============================================================================


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

    Example:
        >>> operable = build_intermediate_operable([ProgressReport, PartialResult])
        >>> Model = operable.create_model()  # IntermediateOptions with nullable fields
    """
    if not isinstance(options, list):
        options = [options]

    specs = []
    for opt in options:
        spec = Spec.from_model(opt, nullable=nullable, listable=listable, default=None)
        specs.append(spec)

    return Operable(specs=tuple(specs), name="IntermediateOptions")


def build_step_operable(
    response_model: type[BaseModel] | None = None,
    intermediate_options: list[type[BaseModel]] | type[BaseModel] | None = None,
    *,
    intermediate_listable: bool = False,
    intermediate_nullable: bool = True,
) -> Operable:
    """Build Operable for ReactStepResponse with optional intermediate options.

    Creates a dynamic Operable that includes:
    - Core step fields (reasoning, action_requests, is_done)
    - Optional final_answer with custom type
    - Optional intermediate_response_options as nested model

    Args:
        response_model: Type for final_answer field
        intermediate_options: Model(s) for intermediate deliverables
        intermediate_listable: Whether intermediate options can be lists
        intermediate_nullable: Whether intermediate options default to None

    Returns:
        Operable for the step response model

    Example:
        >>> operable = build_step_operable(
        ...     response_model=MyAnswer,
        ...     intermediate_options=[ProgressReport],
        ... )
        >>> StepModel = operable.create_model()
    """
    # Core step specs
    specs = [
        Spec(
            str,
            name="reasoning",
            description="Your reasoning about the current state and next action",
        ).as_optional(),
        Spec(
            list[ActionRequest],
            name="action_requests",
            description="List of tool calls. Empty when done.",
        ).as_optional(),
        Spec(
            bool,
            name="is_done",
            default=False,
            description="Set to true when you have the final answer",
        ),
    ]

    # Add final_answer with custom type if provided
    if response_model:
        specs.append(
            Spec.from_model(
                response_model,
                name="final_answer",
                nullable=True,
                default=None,
            )
        )
    else:
        # Generic Any type for final_answer
        specs.append(
            Spec(
                base_type=Any,
                name="final_answer",
                description="Your final answer (only when is_done=true)",
            ).as_optional()
        )

    # Add intermediate_response_options as nested model if provided
    if intermediate_options:
        intermediate_operable = build_intermediate_operable(
            intermediate_options,
            listable=intermediate_listable,
            nullable=intermediate_nullable,
        )
        # Create the nested model type
        IntermediateModel = intermediate_operable.create_model()
        specs.append(
            Spec(
                IntermediateModel,
                name="intermediate_response_options",
                description="Intermediate deliverable outputs. Fill as needed.",
            ).as_optional()
        )

    return Operable(specs=tuple(specs), name="ReactStepResponse")


# =============================================================================
# Stream Implementation
# =============================================================================


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
        params: ReactParams with nested operate params.

    Yields:
        ReactStep for each step in the ReAct loop.
    """
    from .factory import operate

    # Validate nested params structure
    if params._is_sentinel(params.operate):
        raise ValueError("react requires 'operate' params")
    if params._is_sentinel(params.operate.communicate):
        raise ValueError("react requires 'operate.communicate' params")
    if params._is_sentinel(params.operate.communicate.generate):
        raise ValueError("react requires 'operate.communicate.generate' params")

    gen_params = params.operate.communicate.generate

    # Extract base params
    instruction = gen_params.instruction
    imodel = gen_params.imodel or session.default_generate_model
    imodel_kwargs = gen_params.imodel_kwargs or {}
    context = gen_params.context

    if params._is_sentinel(instruction):
        raise ValueError("react requires 'instruction' in operate.communicate.generate")
    if params._is_sentinel(imodel) and session.default_generate_model is None:
        raise ValueError("react requires 'imodel' in operate.communicate.generate")

    # Extract model_name for API calls
    model_name = imodel_kwargs.get("model_name") or imodel_kwargs.get("model")
    if not model_name and hasattr(imodel, "name"):
        model_name = imodel.name
    if not model_name:
        raise ValueError(
            "react requires 'model_name' in imodel_kwargs or imodel with .name attribute"
        )

    # Resolve branch
    b_ = session.get_branch(branch)

    # Prepare kwargs without model_name duplication
    step_kwargs = {k: v for k, v in imodel_kwargs.items() if k not in ("model_name", "model")}

    verbose = params.return_trace
    max_steps = params.max_steps

    # Build step Operable with intermediate options if provided
    step_operable = None
    if params.intermediate_response_options is not None:
        step_operable = build_step_operable(
            response_model=params.response_model,
            intermediate_options=params.intermediate_response_options,
            intermediate_listable=params.intermediate_listable,
            intermediate_nullable=params.intermediate_nullable,
        )
        if verbose:
            logger.info(
                f"Built step Operable with intermediate options: "
                f"{[s.name for s in step_operable.get_specs()]}"
            )

    # ReAct loop
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
            # Build OperateParams for this step
            # Use step_operable if we have intermediate options, else default ReactStepResponse
            operate_params = OperateParams(
                communicate=CommunicateParams(
                    generate=GenerateParams(
                        imodel=imodel,
                        instruction=step_instruction,
                        context=context,
                        # Use operable-built model if available, else static ReactStepResponse
                        request_model=None if step_operable else ReactStepResponse,
                        imodel_kwargs={"model": model_name, **step_kwargs},
                    ),
                    # Pass operable for dynamic schema
                    operable=step_operable,
                    capabilities=step_operable.allowed() if step_operable else None,
                    parse=ParseParams(),
                    strict_validation=False,
                ),
                actions=True,
                reason=True,
            )

            # Call operate
            operate_result = await operate(session, b_, operate_params)

            if verbose:
                logger.debug(f"Operate result: {operate_result}")

            # Handle validation failure
            if isinstance(operate_result, dict) and operate_result.get("validation_failed"):
                step.reasoning = f"Validation failed: {operate_result.get('error')}"
                yield step
                return

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
                    # Convert to dict for storage in ReactStep
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


# =============================================================================
# Wrapper Implementation
# =============================================================================


async def react(
    session: Session,
    branch: Branch | str,
    params: ReactParams,
) -> ReactResult:
    """ReAct operation - multi-step reasoning with tool calling.

    Wrapper around react_stream() that collects all steps and returns ReactResult.

    Composition:
        react() → react_stream() → operate() → communicate() → generate()
                                            → act() (via actions=True)

    Tools must be pre-registered in session.services before calling react().

    Args:
        session: Session for services and message storage.
        branch: Branch for conversation history.
        params: ReactParams with nested operate params.

    Returns:
        ReactResult with steps and final_response.
    """
    result: ReactResult = ReactResult()

    async for step in react_stream(session, branch, params):
        result.steps.append(step)

        if step.is_final:
            result.completed = True
            result.reason_stopped = "Task completed"
            # Extract final_answer from the last operate result
            # (stored in step reasoning or we need to re-extract)
            break

    # Extract final response from last step if completed
    if result.completed and result.steps:
        last_step = result.steps[-1]
        # The final_answer would be in the operate result, but we only stored step data
        # For now, we indicate completion via steps
        result.final_response = last_step.reasoning

    result.total_steps = len(result.steps)

    if not result.completed and not result.reason_stopped:
        result.reason_stopped = f"Max steps ({params.max_steps}) reached"

    return result
