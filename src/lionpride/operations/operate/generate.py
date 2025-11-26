# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Generate operation - stateless LLM call.

Lowest-level operation: just calls the model.
No message persistence, no validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lionpride.services.types import iModel
from lionpride.session.messages import AssistantResponseContent, Message, prepare_messages_for_chat

from .types import GenerateParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("GenerateParams", "generate")


async def generate(session: Session, branch: Branch, params: GenerateParams | dict):
    """Stateless LLM call.

    Security:
    - Branch must have access to imodel (branch.resources)

    Args:
        session: Session for services
        branch: Branch for resource access control
        params: Generate parameters (GenerateParams or dict)

    Returns:
        Response in format specified by return_as

    Raises:
        PermissionError: If branch doesn't have access to imodel
        ValueError: If no imodel specified
    """
    # Convert dict to GenerateParams if needed
    if isinstance(params, dict):
        params = GenerateParams(**params)

    b = session.get_branch(branch, session.default_branch)
    if b is None:
        raise ValueError("No branch specified and no default_branch set on session")

    # Get instruction as Message (handles string → Message conversion)
    instruction_msg = params.instruction_message

    # Add instruction to branch if provided
    if instruction_msg is not None:
        session.add_message(instruction_msg, branches=b)

    # Prepare messages for chat API
    msgs = prepare_messages_for_chat(
        session.messages[b], new_instruction=instruction_msg, to_chat=True
    )

    # Resolve imodel
    imodel = params.imodel or session.default_generate_model
    if imodel is None:
        raise ValueError("No imodel specified and no default_generate_model set on session")

    model_name = imodel.name if isinstance(imodel, iModel) else imodel

    # Check branch has access to this model
    if model_name not in b.resources:
        raise PermissionError(
            f"Branch '{b.name}' cannot access model '{model_name}'. "
            f"Allowed: {b.resources or 'none'}"
        )

    # Call model
    calling = await session.request(model_name, messages=msgs, **params.imodel_kwargs)

    # Format result
    match params.return_as:
        case "text":
            return calling.response.data
        case "raw":
            return calling.response.raw_response
        case "message":
            return Message(
                content=AssistantResponseContent(
                    assistant_response=calling.response.data,
                ),
                metadata={
                    "raw_response": calling.response.raw_response,
                    **calling.response.metadata,
                },
            )
        case "calling":
            return calling

    raise ValueError(f"Unsupported return_as: {params.return_as}")
