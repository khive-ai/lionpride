# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""React operation - multi-step reasoning with tool calling.

Stream-first architecture:
    react_stream() - async generator yielding intermediate results
    react() - wrapper collecting all results

Composition:
    react() → react_stream() → operate() → communicate() → generate()
                                        → act() (via actions=True)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, Field

from lionpride.rules import ActionRequest, ActionResponse

from .types import (
    CommunicateParams,
    GenerateParams,
    OperateParams,
    ParseParams,
    ReactParams,
)

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("ReactResult", "ReactStep", "ReactStepResponse", "react", "react_stream")

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
            operate_params = OperateParams(
                communicate=CommunicateParams(
                    generate=GenerateParams(
                        imodel=imodel,
                        instruction=step_instruction,
                        context=context,
                        request_model=ReactStepResponse,
                        imodel_kwargs={"model": model_name, **step_kwargs},
                    ),
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
