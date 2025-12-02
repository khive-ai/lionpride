# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.errors import (
    ConfigurationError,
    ExecutionError,
    NotFoundError,
    ValidationError,
)

from .types import GenerateParams, ReturnAs

if TYPE_CHECKING:
    from lionpride.services.types import APICalling
    from lionpride.session import Branch, Session

__all__ = ("_handle_return", "generate")


async def generate(
    session: Session,
    branch: Branch | str,
    params: GenerateParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
):
    # Use 'is None' check - empty Branch evaluates to False due to len() == 0
    if (_b := session.get_branch(branch, None)) is None:
        raise NotFoundError(f"Branch '{branch}' does not exist in session")

    imodel = params.imodel or session.default_generate_model
    imodel = session.services.get(imodel, None)
    imodel_kw = params.imodel_kwargs or {}

    if imodel is None:
        raise ConfigurationError("No valid 'imodel' provided for generate")
    if imodel.name not in _b.resources:
        raise ConfigurationError(
            f"Branch '{_b.name}' has no access to model '{imodel.name}'",
            details={"branch": _b.name, "model": imodel.name, "resources": list(_b.resources)},
        )
    if not isinstance(imodel_kw, dict):
        raise ValidationError("'imodel_kwargs' must be a dict if provided")

    msgs = session.messages[_b]
    from lionpride.session.messages import prepare_messages_for_chat

    prepared_msgs = prepare_messages_for_chat(
        msgs,
        new_instruction=params.instruction_message,
        to_chat=True,
        structure_format=params.structure_format,
        custom_renderer=params.custom_renderer,
    )
    calling = await session.request(
        imodel.name,
        messages=prepared_msgs,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
        **imodel_kw,
    )

    return _handle_return(calling, params.return_as)


def _handle_return(calling: APICalling, return_as: ReturnAs) -> Any:
    # For "calling", always return - caller handles status
    if return_as == "calling":
        return calling

    # For data formats, must succeed
    from lionpride.core.event import EventStatus

    if calling.execution.status != EventStatus.COMPLETED:
        raise ExecutionError(
            "Generation did not complete successfully",
            details=calling.execution.to_dict(),
            retryable=True,  # API failures are often transient
        )

    response = calling.response
    match return_as:
        case "text":
            return response.data
        case "raw":
            return response.raw_response
        case "message":
            from lionpride.session.messages import AssistantResponseContent, Message

            return Message(
                content=AssistantResponseContent.create(
                    assistant_response=response.data,
                ),
                metadata={
                    "raw_response": response.raw_response,
                    **response.metadata,
                },
            )
        case _:
            raise ValidationError(f"Unsupported return_as: {return_as}")
