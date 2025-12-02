# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.errors import AccessError, ConfigurationError, ValidationError
from lionpride.rules import ActionRequest, ActionResponse, Reason, Validator
from lionpride.types import Operable, Spec

from .act import execute_tools, has_action_requests
from .communicate import communicate
from .types import CommunicateParams, OperateParams, ParseParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("operate",)


async def operate(
    session: Session,
    branch: Branch | str,
    params: OperateParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
    validator: Validator | None = None,
) -> Any:
    """Structured output with optional actions.

    Security model:
        1. User declares capabilities in params (field names they want)
        2. System adds action_requests/reason if enabled
        3. Total capabilities validated against branch.capabilities
        4. Communicate validates and builds model from operable + capabilities

    Args:
        session: Session for services and message storage.
        branch: Branch for conversation history.
        params: OperateParams with generate/parse params and operate flags.
        poll_timeout: Timeout for model polling.
        poll_interval: Interval for model polling.
        validator: Optional validator instance.

    Returns:
        Validated model instance, or (result, message) tuple if return_message=True.

    Raises:
        ValidationError: If params invalid or capabilities not declared.
        ConfigurationError: If model not configured.
        AccessError: If branch lacks required capabilities.
    """
    # 1. Validate required params
    if params._is_sentinel(params.generate):
        raise ValidationError("operate requires 'generate' params")

    gen_params = params.generate

    if params._is_sentinel(gen_params.imodel) and session.default_generate_model is None:
        raise ConfigurationError(
            "operate requires 'imodel' in generate params or session.default_generate_model"
        )

    # 2. Resolve branch
    b_ = session.get_branch(branch)

    # 3. Handle skip_validation (text path - no structured output)
    if params.skip_validation:
        from .generate import generate

        result = await generate(
            session=session,
            branch=b_,
            params=gen_params.with_updates(return_as="text"),
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )
        if params.return_message:
            branch_msgs = session.messages[b_]
            return result, branch_msgs[-1] if branch_msgs else None
        return result

    # 4. Structured output path - validate capability design
    response_model = gen_params.request_model
    operable = params.operable

    if response_model is None and operable is None:
        raise ValidationError(
            "operate requires 'request_model' in generate or 'operable' in params"
        )

    # 5. Build required capabilities
    # User declares capabilities for their model fields
    if params.capabilities is None:
        raise ValidationError(
            "capabilities must be explicitly declared when using structured output. "
            "This ensures explicit access control over what fields the LLM can produce."
        )

    required_capabilities = set(params.capabilities)

    # System adds action_requests/action_responses/reason if enabled
    if params.actions:
        required_capabilities.add("action_requests")
        required_capabilities.add("action_responses")
    if params.reason:
        required_capabilities.add("reason")

    # 6. Validate capabilities against branch (security gate)
    if not required_capabilities.issubset(b_.capabilities):
        missing = required_capabilities - b_.capabilities
        raise AccessError(
            f"Branch '{b_.name}' missing capabilities: {missing}",
            details={
                "requested": sorted(required_capabilities),
                "available": sorted(b_.capabilities),
            },
        )

    # 7. Build operable (if not provided)
    if operable is None:
        operable = _build_operable(response_model, params.actions, params.reason)

    # 8. Validate capabilities against operable (can only request what exists)
    if not required_capabilities.issubset(operable.allowed()):
        missing = required_capabilities - operable.allowed()
        raise ValidationError(
            f"Requested capabilities not in operable: {missing}",
            details={
                "requested": sorted(required_capabilities),
                "available": sorted(operable.allowed()),
            },
        )

    # 9. Call communicate with validated capabilities
    parse_params = params.parse if not params._is_sentinel(params.parse) else ParseParams()
    communicate_params = CommunicateParams(
        generate=gen_params,
        parse=parse_params,
        operable=operable,
        capabilities=required_capabilities,
        auto_fix=params.auto_fix,
        strict_validation=params.strict_validation,
    )

    result = await communicate(
        session=session,
        branch=b_,
        params=communicate_params,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        validator=validator,
    )

    # 10. Execute actions if enabled and present
    if params.actions and has_action_requests(result):
        result, _ = await execute_tools(
            result,
            session,
            b_,
            max_concurrent=10 if params.tool_concurrent else 1,
        )

    # 11. Return
    if params.return_message:
        branch_msgs = session.messages[b_]
        assistant_msg = branch_msgs[-1] if branch_msgs else None
        return result, assistant_msg

    return result


def _build_operable(
    response_model: type[BaseModel] | None,
    actions: bool,
    reason: bool,
) -> Operable:
    """Build Operable from response_model + action/reason specs.

    Creates an Operable with:
    - response_model as a field (name = lowercase class name)
    - reason spec if enabled
    - action_requests/action_responses specs if enabled

    Args:
        response_model: User's Pydantic model for structured output.
        actions: Whether to include action specs.
        reason: Whether to include reason spec.

    Returns:
        Operable with all required specs.
    """
    specs = []

    # Add response model as a field (lowercase name convention)
    if response_model:
        if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
            raise ValidationError(
                f"response_model must be a Pydantic BaseModel subclass, got {response_model}"
            )
        specs.append(Spec.from_model(response_model))

    # Add reason spec (optional - can be None)
    if reason:
        specs.append(Spec(Reason, name="reason").as_optional())

    # Add action specs (default to empty list)
    if actions:
        specs.append(Spec(list[ActionRequest], name="action_requests", default_factory=list))
        specs.append(Spec(list[ActionResponse], name="action_responses", default_factory=list))

    # Create operable
    name = response_model.__name__ if response_model else "OperateResponse"
    return Operable(specs=tuple(specs), name=name)
