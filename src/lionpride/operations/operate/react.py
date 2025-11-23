# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..dispatcher import register_operation
from ..models import ActionRequestModel, ActionResponseModel

if TYPE_CHECKING:
    from lionpride.services.types import iModel
    from lionpride.session import Branch, Session

__all__ = ("ReactResult", "ReactStep", "react")


class ReactStep(BaseModel):
    """Single step in a ReAct loop."""

    step: int = Field(..., description="Step number (1-indexed)")
    reasoning: str | None = Field(default=None, description="LLM reasoning for this step")
    actions_requested: list[ActionRequestModel] = Field(
        default_factory=list, description="Actions requested by LLM"
    )
    actions_executed: list[ActionResponseModel] = Field(
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
    action_requests: list[ActionRequestModel] | None = Field(
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
        action_requests: list[ActionRequestModel] | None = Field(
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


@register_operation("react")
async def react(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> ReactResult:
    """ReAct operation - multi-step reasoning with tool calling.

    Args:
        parameters: Must include 'instruction', 'imodel', 'tools', and 'model_name'.
            Optional: response_model, max_steps, use_lndl, verbose.
    """
    from .factory import operate

    # Extract and validate parameters
    instruction = parameters.get("instruction")
    if not instruction:
        raise ValueError("react requires 'instruction' parameter")

    imodel: iModel = parameters.get("imodel")
    if not imodel:
        raise ValueError("react requires 'imodel' parameter")

    tools = parameters.get("tools", [])
    if not tools:
        raise ValueError("react requires 'tools' parameter with at least one tool")

    response_model = parameters.get("response_model")
    context = parameters.get("context", {})
    max_steps = parameters.get("max_steps", 5)
    use_lndl = parameters.get("use_lndl", False)
    lndl_threshold = parameters.get("lndl_threshold", 0.85)
    verbose = parameters.get("verbose", False)

    # Extract model_kwargs - may be nested dict or flat in parameters
    nested_model_kwargs = parameters.get("model_kwargs", {})
    flat_model_kwargs = {
        k: v
        for k, v in parameters.items()
        if k
        not in {
            "instruction",
            "imodel",
            "tools",
            "response_model",
            "context",
            "max_steps",
            "use_lndl",
            "lndl_threshold",
            "verbose",
            "model_kwargs",
            "reason",  # Legacy param
        }
    }
    # Merge: flat params override nested
    model_kwargs = {**nested_model_kwargs, **flat_model_kwargs}

    # Model name is required - check in model_kwargs
    model_name = model_kwargs.pop("model_name", None)
    if not model_name:
        raise ValueError("react requires 'model_name' in model_kwargs")

    # Resolve branch
    if isinstance(branch, str):
        branch = session.conversations.get_progression(branch)

    # Register tools with session
    from lionpride.services.types.imodel import iModel as iModelClass
    from lionpride.services.types.tool import Tool

    tool_names = []
    for tool in tools:
        if isinstance(tool, type) and issubclass(tool, Tool):
            tool_instance = tool()
        elif isinstance(tool, Tool):
            tool_instance = tool
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}")

        tool_names.append(tool_instance.name)
        if not session.services.has(tool_instance.name):
            session.services.register(iModelClass(backend=tool_instance))

    # Build tool descriptions for prompt
    tool_descriptions = []
    for tool in tools:
        tool_instance = tool if isinstance(tool, Tool) else tool()
        schema = tool_instance.tool_schema or {}
        props = schema.get("properties", {})
        params_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        tool_descriptions.append(f"- {tool_instance.name}({params_str})")

    tool_schemas_str = "\n".join(tool_descriptions)

    # Create step response model
    step_model = _create_react_response_model(response_model)

    # Initialize result
    result = ReactResult()

    # Build initial instruction with tool info
    react_instruction = f"""{instruction}

You have access to these tools:
{tool_schemas_str}

Respond with JSON containing:
- "reasoning": explain your thinking
- "action_requests": list of tool calls, each with "function" and "arguments"
- "is_done": true when you have the final answer
- "final_answer": your answer (only when is_done=true)

To call a tool: {{"reasoning": "...", "action_requests": [{{"function": "tool_name", "arguments": {{...}}}}], "is_done": false, "final_answer": null}}
Final answer: {{"reasoning": "...", "action_requests": [], "is_done": true, "final_answer": "..."}}"""

    if context:
        react_instruction += f"\n\nContext:\n{context}"

    # ReAct loop
    current_instruction = react_instruction

    for step_num in range(1, max_steps + 1):
        if verbose:
            print(f"\n--- ReAct Step {step_num}/{max_steps} ---")

        step = ReactStep(step=step_num)

        try:
            # Call operate for this step (no actions=True - we handle actions ourselves)
            operate_result = await operate(
                session,
                branch,
                {
                    "instruction": current_instruction,
                    "imodel": imodel,
                    "response_model": step_model,
                    "use_lndl": use_lndl,
                    "lndl_threshold": lndl_threshold,
                    "model": model_name,
                    **model_kwargs,
                },
            )

            if verbose:
                print(f"Operate result: {operate_result}")

            # Handle validation failure
            if isinstance(operate_result, dict) and operate_result.get("validation_failed"):
                result.reason_stopped = f"Validation failed: {operate_result.get('error')}"
                result.steps.append(step)
                break

            # Extract step data from operate result
            if hasattr(operate_result, "reasoning"):
                step.reasoning = operate_result.reasoning
                if verbose and step.reasoning:
                    print(f"Reasoning: {step.reasoning[:200]}...")

            if hasattr(operate_result, "action_requests") and operate_result.action_requests:
                step.actions_requested = operate_result.action_requests

                # Execute actions manually
                if verbose:
                    print(f"Executing {len(step.actions_requested)} action(s)...")

                from ..actions import act

                action_responses = await act(
                    step.actions_requested,
                    session.services,
                    concurrent=True,
                )
                step.actions_executed = action_responses

                # Add action results to conversation for context
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

                if verbose:
                    for resp in step.actions_executed:
                        print(f"  Tool {resp.function}: {str(resp.output)[:100]}...")

            # Check if done
            is_done = getattr(operate_result, "is_done", False)
            if is_done:
                step.is_final = True
                result.final_response = getattr(operate_result, "final_answer", None)
                result.steps.append(step)
                result.completed = True
                result.reason_stopped = "Task completed"
                result.total_steps = step_num
                if verbose:
                    print(f"Task completed at step {step_num}")
                    print(f"Final answer: {result.final_response}")
                break

            # Prepare next instruction
            current_instruction = "Continue based on the tool results. What is the final answer?"

        except Exception as e:
            if verbose:
                import traceback

                traceback.print_exc()
            result.reason_stopped = f"Error at step {step_num}: {e}"
            result.steps.append(step)
            break

        result.steps.append(step)

    # If we exhausted max_steps without completing
    if not result.completed:
        result.total_steps = len(result.steps)
        if not result.reason_stopped:
            result.reason_stopped = f"Max steps ({max_steps}) reached"

    return result
