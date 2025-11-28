# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operate operation - structured output with optional actions.

Operate = Communicate + optional Act (tool execution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.rules import ActionRequest, ActionResponse, Reason, Validator
from lionpride.types import Operable, Spec

from .act import execute_tools, has_action_requests
from .communicate import communicate
from .types import OperateParams

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

    Args:
        session: Session for services and message storage.
        branch: Branch for conversation history.
        params: OperateParams with nested communicate params.
        poll_timeout: Timeout for model polling.
        poll_interval: Interval for model polling.
        validator: Optional validator instance (uses default if None).

    Returns:
        Validated model instance, or (result, message) tuple if return_message=True.

    Raises:
        ValueError: If validation fails and strict_validation=True.
    """
    # 1. Validate params - need communicate with generate
    if params._is_sentinel(params.communicate):
        raise ValueError("operate requires 'communicate' params")
    if params._is_sentinel(params.communicate.generate):
        raise ValueError("operate requires 'communicate.generate' params")

    comm_params = params.communicate
    gen_params = comm_params.generate

    # Check instruction and imodel
    if params._is_sentinel(gen_params.instruction):
        raise ValueError("operate requires 'instruction' in communicate.generate")
    if params._is_sentinel(gen_params.imodel) and session.default_generate_model is None:
        raise ValueError(
            "operate requires 'imodel' in communicate.generate or session.default_generate_model"
        )

    # 2. Build Operable from response_model + action/reason specs
    # Get response_model from generate params if present
    response_model = gen_params.request_model
    operable = comm_params.operable

    if response_model is None and operable is None:
        raise ValueError(
            "operate requires 'request_model' in generate or 'operable' in communicate"
        )

    operable, capabilities = _build_operable(
        response_model=response_model,
        operable=operable,
        actions=params.actions,
        reason=params.reason,
    )

    # 3. Update CommunicateParams with built operable/capabilities
    communicate_params = comm_params.with_updates(
        operable=operable,
        capabilities=capabilities,
    )

    # 4. Handle skip_validation (text path)
    if params.skip_validation:
        communicate_params = communicate_params.with_updates(
            operable=None,
            capabilities=None,
        )

    # 5. Resolve branch
    b_ = session.get_branch(branch)

    # 6. Call communicate
    result = await communicate(
        session=session,
        branch=b_,
        params=communicate_params,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        validator=validator,
    )

    # 7. Handle validation failure
    if isinstance(result, dict) and result.get("validation_failed"):
        if not params.return_message:
            raise ValueError(f"Response validation failed: {result.get('error')}")
        return result, None

    # 8. Execute actions if enabled and present
    if params.actions and has_action_requests(result):
        act_params = params.act
        result, _action_responses = await execute_tools(
            result,
            session,
            b_,
            concurrent=act_params.concurrent if act_params else True,
        )

    # 9. Return result
    if params.return_message:
        # Get the last assistant message from branch
        branch_msgs = session.messages[b_]
        assistant_msg = branch_msgs[-1] if branch_msgs else None
        return result, assistant_msg

    return result


def _build_operable(
    response_model: type[BaseModel] | None,
    operable: Operable | None,
    actions: bool,
    reason: bool,
) -> tuple[Operable, set[str]]:
    """Build Operable from response_model + action/reason specs.

    Returns:
        (operable, capabilities) tuple.
    """
    # If operable provided, use it directly
    if operable:
        # Determine capabilities from operable's allowed field names
        capabilities = operable.allowed()
        return operable, capabilities

    # Validate response_model
    if response_model and (
        not isinstance(response_model, type) or not issubclass(response_model, BaseModel)
    ):
        raise ValueError(
            f"response_model must be a Pydantic BaseModel subclass, got {response_model}"
        )

    # Build specs list
    specs = []
    capabilities = set()

    # Add base model as a spec
    if response_model:
        spec_name = response_model.__name__.lower()
        specs.append(
            Spec(
                base_type=response_model,
                name=spec_name,
            )
        )
        capabilities.add(spec_name)

    # Add reason spec
    if reason:
        specs.append(
            Spec(
                base_type=Reason,
                name="reason",
                default=None,
            )
        )
        capabilities.add("reason")

    # Add action specs
    if actions:
        specs.append(
            Spec(
                base_type=list[ActionRequest],
                name="action_requests",
                default=None,
            )
        )
        specs.append(
            Spec(
                base_type=list[ActionResponse],
                name="action_responses",
                default=None,
            )
        )
        capabilities.add("action_requests")
        # Note: action_responses not in capabilities - filled after execution

    # Create Operable
    name = response_model.__name__ if response_model else "OperateResponse"
    operable = Operable(specs=tuple(specs), name=name)

    return operable, capabilities
