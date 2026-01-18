# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.session.messages import prepare_messages_for_chat

from .phrases import (
    resolve_branch_exists_in_session,
    resolve_genai_model_exists_in_session,
    resolve_response_is_normalized,
    resource_must_be_accessible_by_branch,
    response_must_be_completed,
)
from .types import GenerateParams, ReturnAs

if TYPE_CHECKING:
    from lionpride.services.types import Calling
    from lionpride.session import Branch, Session

__all__ = ("_handle_return", "generate")


async def generate(
    session: Session,
    branch: Branch | str,
    params: GenerateParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
):
    b_ = resolve_branch_exists_in_session(session, branch)
    imodel_, imodel_kw = resolve_genai_model_exists_in_session(session, params)
    resource_must_be_accessible_by_branch(b_, imodel_.name)

    msgs = session.messages[b_]

    prepared_msgs = prepare_messages_for_chat(
        msgs,
        new_instruction=params.instruction_message,
        to_chat=True,
        structure_format=params.structure_format,
        custom_renderer=params.custom_renderer,
    )
    calling = await session.request(
        imodel_.name,
        messages=prepared_msgs,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
        **imodel_kw,
    )

    return _handle_return(calling, params.return_as)


def _handle_return(calling: Calling, return_as: ReturnAs) -> Any:
    # caller handles status
    if return_as == "calling":
        return calling

    response_must_be_completed(calling)
    response = resolve_response_is_normalized(calling)

    match return_as:
        case "text":
            return response.data
        case "raw":
            return response.raw_response
        case "response":
            return response
        case "message":
            from lionpride.errors import ValidationError
            from lionpride.session.messages import AssistantResponseContent, Message

            metadata_dict: dict[str, Any] = {"raw_response": response.raw_response}
            if response.metadata is not None:
                metadata_dict.update(response.metadata)

            return Message(
                content=AssistantResponseContent.create(
                    assistant_response=response.data,
                ),
                metadata=metadata_dict,
            )
        case _:
            from lionpride.errors import ValidationError

            raise ValidationError(f"Unsupported return_as: {return_as}")
