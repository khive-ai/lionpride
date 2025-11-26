# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operate - top-level operation orchestrator.

Composes communicate + tools for structured output with optional actions.

Flow:
    instruction → communicate() → [tool execution] → result
                       │
                       └── parse() → Validator.validate() → typed model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.rules import ActionRequest, Reason
from lionpride.services.types import iModel
from lionpride.types.base import ModelConfig, Params

from .communicate import CommunicateParams, communicate
from .message_prep import prepare_tool_schemas
from .tool_executor import execute_tools, has_action_requests

if TYPE_CHECKING:
    from lionpride.session import Branch, Session
    from lionpride.types import Operable

__all__ = ("OperateParams", "operate")


@dataclass(init=False, frozen=True, slots=True)
class OperateParams(Params):
    """Parameters for operate - top-level orchestration.

    Operate provides structured output with optional actions/tools.
    Builds on communicate with tool execution layer.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    instruction: str = None
    """User instruction text"""

    imodel: iModel | str = None
    """Model to use for generation"""

    # Structured output - one of these required
    response_model: type[BaseModel] | None = None
    """Pydantic model for validation (simple path)"""

    operable: Operable | None = None
    """Operable for validation (full path with Specs, carries its own adapter)"""

    capabilities: set[str] | None = None
    """Capabilities for validation (defaults to branch.capabilities)"""

    # Context
    context: dict[str, Any] | None = None
    """Additional context for instruction"""

    images: list[str] | None = None
    """Image URLs for multimodal input"""

    image_detail: str | None = None
    """Image detail level"""

    # Actions/tools
    actions: bool = False
    """Enable action_requests in output"""

    reason: bool = False
    """Enable reasoning in output"""

    tools: list[str] | bool = False
    """Tools to include (True=all, list=specific, False=none)"""

    tool_schemas: list[dict] | None = None
    """Pre-computed tool schemas (overrides tools param)"""

    concurrent_tool_execution: bool = True
    """Execute tools concurrently"""

    # Retry/validation
    max_retries: int = 0
    """Retry attempts for validation failures"""

    strict_validation: bool = False
    """Raise on validation failure"""

    fuzzy_parse: bool = True
    """Enable fuzzy JSON parsing"""

    skip_validation: bool = False
    """Skip validation (return raw text)"""

    # Output control
    return_message: bool = False
    """Return (result, message) tuple"""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel"""


async def operate(
    session: Session,
    branch: Branch,
    params: OperateParams,
) -> Any:
    """Structured output with optional actions.

    Security:
    - Branch must have access to imodel (branch.resources)
    - Structured output respects capabilities (branch.capabilities or params.capabilities)

    Args:
        session: Session for services and storage
        branch: Branch for access control and message persistence
        params: Operate parameters

    Returns:
        Validated model instance, or (result, message) if return_message=True

    Raises:
        PermissionError: If branch doesn't have access to imodel
        ValueError: If validation fails with strict_validation=True
    """
    if not params.instruction:
        raise ValueError("operate requires 'instruction' parameter")

    if params.imodel is None:
        raise ValueError("operate requires 'imodel' parameter")

    if params.response_model is None and params.operable is None:
        raise ValueError("operate requires either 'response_model' or 'operable' parameter")

    # 1. Resource access check
    model_name = params.imodel.name if isinstance(params.imodel, iModel) else params.imodel
    if model_name not in branch.resources:
        raise PermissionError(
            f"Branch '{branch.name}' cannot access model '{model_name}'. "
            f"Allowed: {branch.resources or 'none'}"
        )

    # 2. Build operable (validation_model only used for response_model path)
    operable, _validation_model = _build_operable(
        response_model=params.response_model,
        operable=params.operable,
        actions=params.actions,
        reason=params.reason,
    )

    # 3. Prepare tool schemas
    tool_schemas = params.tool_schemas or prepare_tool_schemas(session, params.tools)

    # 4. Build context with tool schemas
    context = params.context
    if tool_schemas:
        existing_context = context or {}
        if isinstance(existing_context, dict):
            context = {**existing_context, "tool_schemas": tool_schemas}
        else:
            context = {"original": existing_context, "tool_schemas": tool_schemas}

    # Determine capabilities
    capabilities = params.capabilities if params.capabilities is not None else branch.capabilities

    # 5. Build communicate params
    if params.skip_validation:
        # Skip validation path - return raw text
        comm_params = CommunicateParams(
            instruction=params.instruction,
            imodel=model_name,
            context=context,
            images=params.images,
            image_detail=params.image_detail,
            max_retries=params.max_retries,
            return_as="text",
            imodel_kwargs=params.imodel_kwargs,
        )
    else:
        # Validation path
        comm_params = CommunicateParams(
            instruction=params.instruction,
            imodel=model_name,
            context=context,
            images=params.images,
            image_detail=params.image_detail,
            operable=operable,
            capabilities=capabilities,
            adapter=params.adapter,
            max_retries=params.max_retries,
            strict_validation=params.strict_validation,
            fuzzy_parse=params.fuzzy_parse,
            return_as="model",
            imodel_kwargs=params.imodel_kwargs,
        )

    # 6. Call communicate
    result = await communicate(session, branch, comm_params)

    # Handle validation failure
    if isinstance(result, dict) and result.get("validation_failed"):
        if not params.return_message:
            raise ValueError(f"Response validation failed: {result.get('error')}")
        return result, None

    # 7. Execute actions if enabled and present
    if params.actions and has_action_requests(result):
        result, _action_responses = await execute_tools(
            result,
            session,
            branch,
            concurrent=params.concurrent_tool_execution,
        )

    # 8. Return result
    if params.return_message:
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
    """Build validation model from response_model or operable.

    Two paths:
    1. operable provided → use operable directly
    2. response_model provided → extend with actions/reason if needed

    No round-trip conversion (Pydantic→Spec→Operable→Pydantic).
    """
    from pydantic import Field, create_model

    # Path 1: Operable provided - use Spec-based validation
    if operable:
        return operable, None

    # Path 2: response_model provided
    if response_model and (
        not isinstance(response_model, type) or not issubclass(response_model, BaseModel)
    ):
        raise ValueError(
            f"response_model must be a Pydantic BaseModel subclass, got {response_model}"
        )

    # No extensions needed - use response_model directly
    if not actions and not reason:
        return None, response_model

    # Need to extend response_model with action/reason fields
    extra_fields: dict[str, Any] = {}
    existing_fields = set(response_model.model_fields.keys()) if response_model else set()

    if reason and "reason" not in existing_fields:
        extra_fields["reason"] = (Reason | None, Field(default=None))

    if actions and "action_requests" not in existing_fields:
        extra_fields["action_requests"] = (
            list[ActionRequest] | None,
            Field(default=None),
        )

    if not extra_fields:
        return None, response_model

    # Create extended model inheriting from response_model
    base = response_model if response_model else BaseModel
    model_name = f"{base.__name__}WithActions" if actions else f"{base.__name__}WithReason"

    extended_model = create_model(
        model_name,
        __base__=base,
        **extra_fields,
    )

    return None, extended_model
