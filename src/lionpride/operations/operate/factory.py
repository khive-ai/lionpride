# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operate - top-level operation orchestrator.

Composes communicate + tools for structured output with optional actions.

Flow:
    instruction → communicate() → [tool execution] → result
                       │
                       └── parse() → Validator.validate() → typed model

Request/Output Model Separation:
    Request model (to LLM):  (*fields, reason?, action_requests?)
    Output model (returned): (*fields, reason?, action_responses?)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.rules import ActionRequest, ActionResponse, Reason
from lionpride.services.types import iModel

from .communicate import communicate
from .message_prep import prepare_tool_schemas
from .tool_executor import execute_tools, has_action_requests
from .types import CommunicateParams, GenerateParams, OperateParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session
    from lionpride.types import Operable

__all__ = ("OperateParams", "operate")


async def operate(
    session: Session,
    branch: Branch,
    params: OperateParams | dict,
) -> Any:
    """Structured output with optional actions.

    Security:
    - Branch must have access to imodel (branch.resources)
    - Structured output respects capabilities (branch.capabilities or params.communicate.capabilities)

    Request/Output separation:
    - Request model includes action_requests (what LLM proposes)
    - Output model includes action_responses (results of execution)
    - Original operable/capabilities are NOT mutated

    Args:
        session: Session for services and storage
        branch: Branch for access control and message persistence
        params: Operate parameters (OperateParams or dict)

    Returns:
        Validated model instance, or (result, message) if return_message=True

    Raises:
        PermissionError: If branch doesn't have access to imodel
        ValueError: If validation fails with strict_validation=True
    """
    # Convert dict to OperateParams if needed
    if isinstance(params, dict):
        params = OperateParams(**params)

    # Validate composed params
    if params.communicate is None:
        raise ValueError("operate requires 'communicate' parameter")

    comm = params.communicate
    if comm.generate is None:
        raise ValueError("operate requires 'communicate.generate' parameter")

    gen = comm.generate
    if gen.instruction is None:
        raise ValueError("operate requires instruction")

    if gen.imodel is None:
        raise ValueError("operate requires imodel")

    if comm.operable is None:
        raise ValueError("operate requires 'communicate.operable' parameter")

    # 1. Resource access check
    model_name = gen.imodel.name if isinstance(gen.imodel, iModel) else gen.imodel
    if model_name not in branch.resources:
        raise PermissionError(
            f"Branch '{branch.name}' cannot access model '{model_name}'. "
            f"Allowed: {branch.resources or 'none'}"
        )

    # 2. Build REQUEST operable/capabilities (with action_requests for LLM)
    base_capabilities = comm.capabilities if comm.capabilities is not None else branch.capabilities
    request_operable, request_capabilities = _build_request_model(
        operable=comm.operable,
        capabilities=base_capabilities,
        actions=params.actions,
        reason=params.reason,
    )

    # 3. Prepare tool schemas
    act = params.act
    tools = act.tools if act else False
    tool_schemas = (act.tool_schemas if act else None) or prepare_tool_schemas(session, tools)

    # 4. Build context with tool schemas
    context = gen.context
    if tool_schemas:
        existing_context = context or {}
        if isinstance(existing_context, dict):
            context = {**existing_context, "tool_schemas": tool_schemas}
        else:
            context = {"original": existing_context, "tool_schemas": tool_schemas}

    # 5. Build communicate params with request operable
    gen_params = GenerateParams(
        imodel=model_name,
        instruction=gen.instruction,
        context=context,
        images=gen.images,
        image_detail=gen.image_detail,
        imodel_kwargs=gen.imodel_kwargs,
    )

    if params.skip_validation:
        # Skip validation path - return raw text
        comm_params = CommunicateParams(
            generate=gen_params,
            parse=comm.parse,
            max_retries=comm.max_retries,
            return_as="text",
        )
    else:
        # Validation path with request model
        comm_params = CommunicateParams(
            generate=gen_params,
            parse=comm.parse,
            operable=request_operable,
            capabilities=request_capabilities,
            max_retries=comm.max_retries,
            strict_validation=comm.strict_validation,
            fuzzy_parse=comm.fuzzy_parse,
            return_as="model",
        )

    # 6. Call communicate (LLM returns action_requests)
    result = await communicate(session, branch, comm_params)

    # Handle validation failure
    if isinstance(result, dict) and result.get("validation_failed"):
        if not params.return_message:
            raise ValueError(f"Response validation failed: {result.get('error')}")
        return result, None

    # 7. Execute actions if enabled and present
    if params.actions and has_action_requests(result):
        concurrent = act.concurrent if act else True
        # Execute tools and transform action_requests → action_responses
        result, _action_responses = await execute_tools(
            result,
            session,
            branch,
            concurrent=concurrent,
        )
        # Result now has action_responses instead of action_requests

    # 8. Return result
    if params.return_message:
        branch_msgs = session.messages[branch]
        assistant_msg = branch_msgs[-1] if branch_msgs else None
        return result, assistant_msg

    return result


def _build_request_model(
    operable: Operable,
    capabilities: set[str],
    actions: bool,
    reason: bool,
) -> tuple[Operable, set[str]]:
    """Build REQUEST operable/capabilities with injected fields.

    Injects reason and action_requests into operable/capabilities without
    mutating the originals. These are what the LLM will generate.

    Args:
        operable: Original operable (not mutated)
        capabilities: Original capabilities (not mutated)
        actions: Whether to inject action_requests
        reason: Whether to inject reason

    Returns:
        (extended_operable, extended_capabilities) for LLM request
    """
    existing_fields = operable.allowed()

    needs_reason = reason and "reason" not in existing_fields
    needs_actions = actions and "action_requests" not in existing_fields

    # No injection needed
    if not needs_reason and not needs_actions:
        return operable, capabilities

    # Build extended operable
    from lionpride.types import (
        Operable as OperableCls,
        Spec,
    )

    specs = list(operable.get_specs())
    extended_capabilities = set(capabilities)

    if needs_reason:
        specs.append(
            Spec(
                name="reason",
                base_type=Reason,
                nullable=True,
                default=None,
            )
        )
        extended_capabilities.add("reason")

    if needs_actions:
        specs.append(
            Spec(
                name="action_requests",
                base_type=ActionRequest,
                nullable=True,
                listable=True,
                default=None,
            )
        )
        extended_capabilities.add("action_requests")

    # Build model name
    model_name = operable.name or "DynamicModel"
    if needs_actions and needs_reason:
        model_name = f"{model_name}WithReasonAndActions"
    elif needs_actions:
        model_name = f"{model_name}WithActions"
    elif needs_reason:
        model_name = f"{model_name}WithReason"

    extended_operable = OperableCls(
        specs=tuple(specs),
        name=model_name,
        adapter=operable._Operable__adapter_name,
    )

    return extended_operable, extended_capabilities


def _build_output_model(
    operable: Operable,
    capabilities: set[str],
    actions: bool,
    reason: bool,
) -> tuple[Operable, set[str]]:
    """Build OUTPUT operable/capabilities with action_responses.

    Similar to request model but uses action_responses instead of
    action_requests. This is what gets returned after tool execution.

    Args:
        operable: Original operable (not mutated)
        capabilities: Original capabilities (not mutated)
        actions: Whether to inject action_responses
        reason: Whether to inject reason

    Returns:
        (extended_operable, extended_capabilities) for output
    """
    existing_fields = operable.allowed()

    needs_reason = reason and "reason" not in existing_fields
    needs_actions = actions and "action_responses" not in existing_fields

    # No injection needed
    if not needs_reason and not needs_actions:
        return operable, capabilities

    # Build extended operable
    from lionpride.types import (
        Operable as OperableCls,
        Spec,
    )

    specs = list(operable.get_specs())
    extended_capabilities = set(capabilities)

    if needs_reason:
        specs.append(
            Spec(
                name="reason",
                base_type=Reason,
                nullable=True,
                default=None,
            )
        )
        extended_capabilities.add("reason")

    if needs_actions:
        specs.append(
            Spec(
                name="action_responses",
                base_type=ActionResponse,
                nullable=True,
                listable=True,
                default=None,
            )
        )
        extended_capabilities.add("action_responses")

    # Build model name
    model_name = operable.name or "DynamicModel"
    if needs_actions and needs_reason:
        model_name = f"{model_name}OutputWithReasonAndActions"
    elif needs_actions:
        model_name = f"{model_name}OutputWithActions"
    elif needs_reason:
        model_name = f"{model_name}OutputWithReason"

    extended_operable = OperableCls(
        specs=tuple(specs),
        name=model_name,
        adapter=operable._Operable__adapter_name,
    )

    return extended_operable, extended_capabilities
