# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""React operation - multi-step reasoning with tool calling.

React = Operate in a loop with tool execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lionpride.rules import ActionRequest, ActionResponse

from .types import OperateParams, ReactParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("ReactResult", "ReactStep", "react")


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
    branch: Branch | str,
    params: ReactParams,
) -> ReactResult:
    """ReAct operation - multi-step reasoning with tool calling.

    Args:
        session: Session for services and message storage.
        branch: Branch for conversation history.
        params: ReactParams with nested operate params.

    Returns:
        ReactResult with steps and final_response.
    """
    from .factory import operate
    from .types import CommunicateParams, GenerateParams, ParseParams

    # Validate nested params structure
    if params._is_sentinel(params.operate):
        raise ValueError("react requires 'operate' params")
    if params._is_sentinel(params.operate.communicate):
        raise ValueError("react requires 'operate.communicate' params")
    if params._is_sentinel(params.operate.communicate.generate):
        raise ValueError("react requires 'operate.communicate.generate' params")

    gen_params = params.operate.communicate.generate
    act_params = params.operate.act

    # Extract generation params
    instruction = gen_params.instruction
    imodel = gen_params.imodel or session.default_generate_model
    imodel_kwargs = gen_params.imodel_kwargs or {}
    context = gen_params.context
    response_model = gen_params.request_model

    if params._is_sentinel(instruction):
        raise ValueError("react requires 'instruction' in operate.communicate.generate")
    if params._is_sentinel(imodel) and session.default_generate_model is None:
        raise ValueError("react requires 'imodel' in operate.communicate.generate")

    # Get tools from act params
    tools = act_params.tools if act_params and not params._is_sentinel(act_params) else []
    if not tools:
        raise ValueError("react requires 'tools' in operate.act")

    # Model name is required in imodel_kwargs
    model_name = imodel_kwargs.get("model_name")
    if not model_name:
        raise ValueError("react requires 'model_name' in imodel_kwargs")

    # Resolve branch
    b_ = session.get_branch(branch)

    # Register tools with session
    from lionpride.services.types.imodel import iModel as iModelClass
    from lionpride.services.types.tool import Tool

    # Handle tools - can be list of Tool instances, classes, or string names
    tool_list = tools if isinstance(tools, list) else [tools]
    tool_instances = []
    for tool in tool_list:
        if isinstance(tool, type) and issubclass(tool, Tool):
            tool_instance = tool()
        elif isinstance(tool, Tool):
            tool_instance = tool
        elif isinstance(tool, str):
            # Skip string tool names - they should already be registered
            continue
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}")

        tool_instances.append(tool_instance)
        if not session.services.has(tool_instance.name):
            session.services.register(iModelClass(backend=tool_instance))

    # Build tool descriptions for prompt
    tool_descriptions = []
    for tool_instance in tool_instances:
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

    # Remove model_name from kwargs (it's passed separately)
    step_kwargs = {k: v for k, v in imodel_kwargs.items() if k != "model_name"}

    # ReAct loop
    current_instruction = react_instruction
    verbose = params.return_trace  # Use return_trace for verbose output

    for step_num in range(1, params.max_steps + 1):
        if verbose:
            print(f"\n--- ReAct Step {step_num}/{params.max_steps} ---")

        step = ReactStep(step=step_num)

        try:
            # Build OperateParams for this step using nested structure
            operate_params = OperateParams(
                communicate=CommunicateParams(
                    generate=GenerateParams(
                        imodel=imodel,
                        instruction=current_instruction,
                        request_model=step_model,
                        imodel_kwargs={"model": model_name, **step_kwargs},
                    ),
                    parse=ParseParams(),
                    strict_validation=False,  # Allow validation failures to be handled
                ),
                actions=True,
            )

            # Call operate for this step
            operate_result = await operate(
                session,
                b_,
                operate_params,
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

                from .act import act

                action_responses = await act(
                    step.actions_requested,
                    session.services,
                    concurrent=True,
                )
                step.actions_executed = action_responses

                # Add action results to conversation for context
                from lionpride.session.messages import ActionResponseContent, Message

                for action_resp in action_responses:
                    action_content = ActionResponseContent.create(
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
            result.reason_stopped = f"Max steps ({params.max_steps}) reached"

    return result
