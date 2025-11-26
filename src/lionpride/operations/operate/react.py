# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""React operation - multi-step reasoning with tool calling.

Composes operate + interpret + analyze in a ReAct loop.

Flow:
    instruction → [operate() → interpret() → analyze()]* → final_response
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lionpride.rules import ActionRequest, ActionResponse

from .types import (
    ReactParams,
)

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("ReactResult", "ReactStep", "react", "react_stream")


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


class ReactResult(BaseModel):
    """Result from a ReAct execution."""

    steps: list[ReactStep] = Field(default_factory=list, description="Execution steps")
    final_response: Any = Field(default=None, description="Final structured response")
    total_steps: int = Field(default=0, description="Total steps executed")
    completed: bool = Field(default=False, description="Whether execution completed normally")
    reason_stopped: str = Field(default="", description="Why execution stopped")


class ReactStepResponse(BaseModel):
    """Response model for each ReAct step."""

    reasoning: str | None = Field(
        default=None, description="Your reasoning about the current state and next action"
    )
    action_requests: list[ActionRequest] | None = Field(
        default=None,
        description=(
            "List of tool calls. Each has 'function' (tool name) and 'arguments' (dict). "
            "Leave empty/null if you have the final answer."
        ),
    )
    is_done: bool = Field(default=False, description="Set to true when you have the final answer")
    final_answer: str | None = Field(
        default=None, description="Your final answer (only when is_done=true)"
    )


def _create_react_response_model(
    response_model: type[BaseModel] | None,
) -> type[BaseModel]:
    """Create dynamic response model for ReAct steps."""
    if response_model is None:
        return ReactStepResponse

    # Create a model with typed final_answer
    class TypedReactStepResponse(BaseModel):
        reasoning: str | None = Field(
            default=None, description="Your reasoning about the current state and next action"
        )
        action_requests: list[ActionRequest] | None = Field(
            default=None,
            description=(
                "List of tool calls. Each has 'function' (tool name) and 'arguments' (dict). "
                "Leave empty/null if you have the final answer."
            ),
        )
        is_done: bool = Field(
            default=False, description="Set to true when you have the final answer"
        )
        final_answer: response_model | None = Field(  # type: ignore[valid-type]
            default=None, description="Your final answer (only when is_done=true)"
        )

    return TypedReactStepResponse


async def react(
    session: Session,
    branch: Branch,
    params: ReactParams | dict,
) -> ReactResult:
    """ReAct operation - multi-step reasoning with tool calling.

    Uses composed params structure:
    - params.operate.communicate.generate for LLM settings
    - params.operate.act for tool settings
    - params.max_steps for loop control

    Args:
        session: Session for services and storage
        branch: Branch for access control
        params: React parameters (ReactParams or dict)

    Returns:
        ReactResult with steps and final response
    """
    from .factory import operate
    from .message_prep import prepare_tool_schemas

    # Convert dict to ReactParams if needed
    if isinstance(params, dict):
        # Handle nested dict conversion for operate
        if "operate" in params and isinstance(params["operate"], dict):
            op = params["operate"]
            # Handle nested communicate in operate
            if "communicate" in op and isinstance(op["communicate"], dict):
                comm = op["communicate"]
                if "generate" in comm and isinstance(comm["generate"], dict):
                    from .types import GenerateParams

                    comm["generate"] = GenerateParams(**comm["generate"])
                from .types import CommunicateParams

                op["communicate"] = CommunicateParams(**comm)
            if "act" in op and isinstance(op["act"], dict):
                from .types import ActParams

                op["act"] = ActParams(**op["act"])
            from .types import OperateParams

            params["operate"] = OperateParams(**op)
        params = ReactParams(**params)

    # Validate params
    if params.operate is None:
        raise ValueError("react requires 'operate' parameter")

    op = params.operate
    if op.communicate is None:
        raise ValueError("react requires 'operate.communicate' parameter")

    comm = op.communicate
    if comm.generate is None:
        raise ValueError("react requires 'operate.communicate.generate' parameter")

    gen = comm.generate
    if gen.instruction is None:
        raise ValueError("react requires instruction")

    if gen.imodel is None:
        raise ValueError("react requires imodel")

    # Get tool config
    from .types import ActParams

    act = op.act or ActParams()
    if not act.tools:
        raise ValueError("react requires tools - set operate.act.tools")

    # Prepare tool schemas
    tool_schemas = act.tool_schemas or prepare_tool_schemas(session, act.tools)
    if not tool_schemas:
        raise ValueError("react requires at least one tool")

    # Create operable for step model
    from lionpride.types import Operable, Spec

    step_operable = Operable(
        specs=(
            Spec(name="reasoning", base_type=str, nullable=True),
            Spec(name="action_requests", base_type=ActionRequest, nullable=True, listable=True),
            Spec(name="is_done", base_type=bool, default=False),
            Spec(name="final_answer", base_type=str, nullable=True),
        ),
        name="ReactStepResponse",
    )
    step_capabilities = {"reasoning", "action_requests", "is_done", "final_answer"}

    # Initialize result
    result = ReactResult()
    max_steps = params.max_steps
    current_instruction = gen.instruction

    # ReAct loop
    for step_num in range(1, max_steps + 1):
        step = ReactStep(step=step_num)

        try:
            # Build operate params for this step
            step_gen = GenerateParams(
                imodel=gen.imodel,
                instruction=current_instruction,
                context={**(gen.context or {}), "tool_schemas": tool_schemas}
                if step_num == 1
                else {"tool_schemas": tool_schemas},
                images=gen.images if step_num == 1 else None,
                image_detail=gen.image_detail if step_num == 1 else None,
                imodel_kwargs=gen.imodel_kwargs,
            )

            step_comm = CommunicateParams(
                generate=step_gen,
                parse=comm.parse,
                operable=step_operable,
                capabilities=step_capabilities,
                max_retries=comm.max_retries,
                strict_validation=comm.strict_validation,
                fuzzy_parse=comm.fuzzy_parse,
            )

            step_op = OperateParams(
                communicate=step_comm,
                act=ActParams(tools=False),  # We execute tools manually
                actions=False,  # Don't auto-execute
                reason=False,  # Already in step model
            )

            # Call operate
            operate_result = await operate(session, branch, step_op)

            # Handle validation failure
            if isinstance(operate_result, dict) and operate_result.get("validation_failed"):
                result.reason_stopped = f"Validation failed: {operate_result.get('error')}"
                result.steps.append(step)
                break

            # Extract step data
            if hasattr(operate_result, "reasoning"):
                step.reasoning = operate_result.reasoning

            # Execute actions if present
            if hasattr(operate_result, "action_requests") and operate_result.action_requests:
                step.actions_requested = operate_result.action_requests

                from ..actions import act as act_func

                action_responses = await act_func(
                    step.actions_requested,
                    session.services,
                    concurrent=act.concurrent,
                    timeout=act.timeout,
                )
                step.actions_executed = action_responses

                # Add action results to conversation
                from lionpride.session.messages import ActionResponseContent, Message

                for action_resp in action_responses:
                    action_content = ActionResponseContent(
                        request_id=action_resp.function,
                        result=action_resp.output,
                    )
                    action_msg = Message(
                        content=action_content,
                        sender="system",
                        recipient=session.id,
                    )
                    session.add_message(action_msg, branches=branch)

            # Check if done
            is_done = getattr(operate_result, "is_done", False)
            if is_done:
                step.is_final = True
                result.final_response = getattr(operate_result, "final_answer", None)
                result.steps.append(step)
                result.completed = True
                result.reason_stopped = "Task completed"
                result.total_steps = step_num
                break

            # Check stop condition
            if params.stop_condition and _check_stop_condition(
                operate_result, params.stop_condition
            ):
                step.is_final = True
                result.steps.append(step)
                result.completed = True
                result.reason_stopped = f"Stop condition met: {params.stop_condition}"
                result.total_steps = step_num
                break

            # Prepare next instruction
            current_instruction = "Continue based on the tool results. What is the final answer?"

        except Exception as e:
            result.reason_stopped = f"Error at step {step_num}: {e}"
            result.steps.append(step)
            break

        result.steps.append(step)

    # If we exhausted max_steps
    if not result.completed:
        result.total_steps = len(result.steps)
        if not result.reason_stopped:
            result.reason_stopped = f"Max steps ({max_steps}) reached"

    return result


def _check_stop_condition(result: Any, condition: str) -> bool:
    """Check if stop condition is met."""
    # Simple attribute check for now
    if hasattr(result, condition):
        return bool(getattr(result, condition))
    return False


async def react_stream(
    session: Session,
    branch: Branch,
    params: ReactParams | dict,
):
    """Streaming ReAct operation - yields ReactStep as each step completes.

    Same parameters as react(), but yields intermediate results.

    Yields:
        ReactStep for each completed step
        ReactResult as final yield when done
    """
    from .factory import operate
    from .message_prep import prepare_tool_schemas

    # Convert dict to ReactParams if needed
    if isinstance(params, dict):
        # Handle nested dict conversion for operate
        if "operate" in params and isinstance(params["operate"], dict):
            op = params["operate"]
            # Handle nested communicate in operate
            if "communicate" in op and isinstance(op["communicate"], dict):
                comm = op["communicate"]
                if "generate" in comm and isinstance(comm["generate"], dict):
                    from .types import GenerateParams

                    comm["generate"] = GenerateParams(**comm["generate"])
                from .types import CommunicateParams

                op["communicate"] = CommunicateParams(**comm)
            if "act" in op and isinstance(op["act"], dict):
                from .types import ActParams

                op["act"] = ActParams(**op["act"])
            from .types import OperateParams

            params["operate"] = OperateParams(**op)
        params = ReactParams(**params)

    # Validate params (same as react)
    if params.operate is None:
        raise ValueError("react_stream requires 'operate' parameter")

    op = params.operate
    if op.communicate is None:
        raise ValueError("react_stream requires 'operate.communicate' parameter")

    comm = op.communicate
    if comm.generate is None:
        raise ValueError("react_stream requires 'operate.communicate.generate' parameter")

    gen = comm.generate
    if gen.instruction is None:
        raise ValueError("react_stream requires instruction")

    if gen.imodel is None:
        raise ValueError("react_stream requires imodel")

    from .types import ActParams

    act = op.act or ActParams()
    if not act.tools:
        raise ValueError("react_stream requires tools")

    tool_schemas = act.tool_schemas or prepare_tool_schemas(session, act.tools)
    if not tool_schemas:
        raise ValueError("react_stream requires at least one tool")

    # Create operable for step model
    from lionpride.types import Operable, Spec

    step_operable = Operable(
        specs=(
            Spec(name="reasoning", base_type=str, nullable=True),
            Spec(name="action_requests", base_type=ActionRequest, nullable=True, listable=True),
            Spec(name="is_done", base_type=bool, default=False),
            Spec(name="final_answer", base_type=str, nullable=True),
        ),
        name="ReactStepResponse",
    )
    step_capabilities = {"reasoning", "action_requests", "is_done", "final_answer"}

    result = ReactResult()
    max_steps = params.max_steps
    current_instruction = gen.instruction

    for step_num in range(1, max_steps + 1):
        step = ReactStep(step=step_num)

        try:
            step_gen = GenerateParams(
                imodel=gen.imodel,
                instruction=current_instruction,
                context={**(gen.context or {}), "tool_schemas": tool_schemas}
                if step_num == 1
                else {"tool_schemas": tool_schemas},
                images=gen.images if step_num == 1 else None,
                image_detail=gen.image_detail if step_num == 1 else None,
                imodel_kwargs=gen.imodel_kwargs,
            )

            step_comm = CommunicateParams(
                generate=step_gen,
                parse=comm.parse,
                operable=step_operable,
                capabilities=step_capabilities,
                max_retries=comm.max_retries,
                strict_validation=comm.strict_validation,
                fuzzy_parse=comm.fuzzy_parse,
            )

            step_op = OperateParams(
                communicate=step_comm,
                act=ActParams(tools=False),
                actions=False,
                reason=False,
            )

            operate_result = await operate(session, branch, step_op)

            if isinstance(operate_result, dict) and operate_result.get("validation_failed"):
                result.reason_stopped = f"Validation failed: {operate_result.get('error')}"
                result.steps.append(step)
                yield step
                break

            if hasattr(operate_result, "reasoning"):
                step.reasoning = operate_result.reasoning

            if hasattr(operate_result, "action_requests") and operate_result.action_requests:
                step.actions_requested = operate_result.action_requests

                from ..actions import act as act_func

                action_responses = await act_func(
                    step.actions_requested,
                    session.services,
                    concurrent=act.concurrent,
                    timeout=act.timeout,
                )
                step.actions_executed = action_responses

                from lionpride.session.messages import ActionResponseContent, Message

                for action_resp in action_responses:
                    action_content = ActionResponseContent(
                        request_id=action_resp.function,
                        result=action_resp.output,
                    )
                    action_msg = Message(
                        content=action_content,
                        sender="system",
                        recipient=session.id,
                    )
                    session.add_message(action_msg, branches=branch)

            is_done = getattr(operate_result, "is_done", False)
            if is_done:
                step.is_final = True
                result.final_response = getattr(operate_result, "final_answer", None)
                result.steps.append(step)
                result.completed = True
                result.reason_stopped = "Task completed"
                result.total_steps = step_num
                yield step
                break

            if params.stop_condition and _check_stop_condition(
                operate_result, params.stop_condition
            ):
                step.is_final = True
                result.steps.append(step)
                result.completed = True
                result.reason_stopped = f"Stop condition met: {params.stop_condition}"
                result.total_steps = step_num
                yield step
                break

            current_instruction = "Continue based on the tool results. What is the final answer?"

        except Exception as e:
            result.reason_stopped = f"Error at step {step_num}: {e}"
            result.steps.append(step)
            yield step
            break

        result.steps.append(step)
        yield step

    if not result.completed:
        result.total_steps = len(result.steps)
        if not result.reason_stopped:
            result.reason_stopped = f"Max steps ({max_steps}) reached"

    yield result
