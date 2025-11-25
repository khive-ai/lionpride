# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.types import Operable, Spec

# Operations registered in Session._register_default_operations()
from ..models import ActionRequestModel, ActionResponseModel, Reason
from .tool_executor import execute_tools, has_action_requests


def prepare_tool_schemas(
    session,
    tools: bool | list[str],
) -> list[Any] | None:
    """Prepare tool schemas from session services.

    Args:
        session: Session containing services
        tools: True for all tools, list of names for specific tools, False/None for none

    Returns:
        List of tool schemas or None
    """
    if not tools:
        return None

    if tools is True:
        return session.services.get_tool_schemas()
    elif isinstance(tools, list):
        return session.services.get_tool_schemas(tool_names=tools)
    return None


if TYPE_CHECKING:
    from lionpride.operations.types import OperateParam
    from lionpride.session import Branch, Session


async def operate(
    session: Session,
    branch: Branch | str,
    parameters: OperateParam | dict,
) -> Any:
    """Structured output with optional actions.

    Args:
        parameters: OperateParam or dict with instruction, imodel, response_model/operable, etc.
    """
    # Convert dict to OperateParam for backward compatibility
    if isinstance(parameters, dict):
        from lionpride.operations.types import OperateParam

        parameters = OperateParam(**parameters)

    # 1. Validate required parameters
    if not parameters.instruction:
        raise ValueError("operate requires 'instruction' parameter")
    if not parameters.imodel:
        raise ValueError("operate requires 'imodel' parameter")
    if not parameters.response_model and not parameters.operable:
        raise ValueError("operate requires either 'response_model' or 'operable' parameter")

    # 2. Build Operable from response_model + action/reason specs
    operable, validation_model = _build_operable(
        response_model=parameters.response_model,
        operable=parameters.operable,
        actions=parameters.actions,
        reason=parameters.reason,
    )

    # 3. Prepare tool schemas
    tool_schemas = parameters.tool_schemas or prepare_tool_schemas(session, parameters.tools)

    # 4. Build communicate parameters
    from lionpride.operations.types import CommunicateParam

    # Build context with tool schemas
    context = parameters.context
    if tool_schemas:
        if isinstance(context, dict):
            context = {**context, "tool_schemas": tool_schemas}
        elif context:
            context = {"original": context, "tool_schemas": tool_schemas}
        else:
            context = {"tool_schemas": tool_schemas}

    # 5. Choose mode: LNDL (operable) vs JSON (response_model)
    comm_operable = None
    comm_response_model = None
    comm_return_as = "model"

    if parameters.use_lndl and operable:
        comm_operable = operable
    elif validation_model:
        comm_response_model = validation_model
    else:
        raise ValueError("operate requires response_model or operable")

    # Handle skip_validation
    if parameters.skip_validation:
        comm_return_as = "text"
        comm_operable = None
        comm_response_model = None

    communicate_params = CommunicateParam(
        instruction=parameters.instruction,
        imodel=parameters.imodel,
        context=context,
        images=parameters.images,
        image_detail=parameters.image_detail,
        max_retries=parameters.max_retries,
        return_as=comm_return_as,
        response_model=comm_response_model,
        operable=comm_operable,
        lndl_threshold=parameters.lndl_threshold,
    )

    # 6. Call communicate
    from .communicate import communicate

    result = await communicate(session, branch, communicate_params)

    # Handle validation failure
    if isinstance(result, dict) and result.get("validation_failed"):
        if not parameters.return_message:
            raise ValueError(f"Response validation failed: {result.get('error')}")
        return result, None

    # 7. Execute actions if enabled and present
    if parameters.actions and has_action_requests(result):
        # Resolve branch for tool execution
        if isinstance(branch, str):
            branch = session.conversations.get_progression(branch)

        result, _action_responses = await execute_tools(
            result,
            session,
            branch,
            concurrent=parameters.concurrent_tool_execution,
        )

    # 8. Return result
    if parameters.return_message:
        # Get the last assistant message from branch
        if isinstance(branch, str):
            branch = session.conversations.get_progression(branch)
        branch_msgs = session.messages[branch]
        assistant_msg = branch_msgs[-1] if branch_msgs else None
        return result, assistant_msg

    return result


def _build_operable(
    response_model: type[BaseModel] | None,
    operable: Operable | None,
    actions: bool,
    reason: bool,
) -> tuple[Operable | None, type[BaseModel] | None]:
    """Build Operable from response_model + action/reason specs."""
    # If operable provided, use it directly
    # Handle both Operable (lionpride) and Operative (lionpride wrapper)
    if operable:
        from lionpride.operations.operate.operative import Operative

        if isinstance(operable, Operative):
            # Operative wraps an Operable - return the Operative for validation
            return operable, operable.create_response_model()
        # Raw Operable from lionpride
        return operable, operable.create_model() if operable else None

    # Validate response_model
    if response_model and (
        not isinstance(response_model, type) or not issubclass(response_model, BaseModel)
    ):
        raise ValueError(
            f"response_model must be a Pydantic BaseModel subclass, got {response_model}"
        )

    # If no actions/reason needed, just use response_model directly
    if not actions and not reason:
        return None, response_model

    # Build Operable with additional specs
    specs = []

    # Add base model as a spec
    if response_model:
        spec_name = response_model.__name__.lower()
        specs.append(
            Spec(
                base_type=response_model,
                name=spec_name,
            )
        )

    # Add reason spec
    if reason:
        specs.append(
            Spec(
                base_type=Reason,
                name="reason",
                default=None,
            )
        )

    # Add action specs
    if actions:
        specs.append(
            Spec(
                base_type=list[ActionRequestModel],
                name="action_requests",
                default=None,
            )
        )
        specs.append(
            Spec(
                base_type=list[ActionResponseModel],
                name="action_responses",
                default=None,
            )
        )

    # Create Operable
    name = response_model.__name__ if response_model else "OperateResponse"
    operable = Operable(specs=tuple(specs), name=name)

    # Create validation model (excluding action_responses for request)
    # For JSON mode, we need a model that includes all fields except action_responses
    validation_model = operable.create_model(
        model_name=f"{name}Response",
        exclude={"action_responses"} if actions else None,
    )

    return operable, validation_model
