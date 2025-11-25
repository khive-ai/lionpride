# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lionpride.session.messages import AssistantResponseContent, Message

if TYPE_CHECKING:
    from lionpride.session import Branch, Session


async def generate(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> str | dict | Message | Any:
    """Stateless text generation - does not persist messages.

    Args:
        parameters: Must include 'imodel'. Optional: return_as (text|raw|message|calling).
    """
    imodel_param = parameters.pop("imodel", None)
    if not imodel_param:
        raise ValueError("generate requires 'imodel' parameter")

    return_as: Literal["text", "raw", "message", "calling"] = parameters.pop("return_as", "text")

    # Support both string name and direct iModel object
    imodel = session.services.get(imodel_param) if isinstance(imodel_param, str) else imodel_param

    # Invoke via unified service interface
    calling = await imodel.invoke(**parameters)

    # Check execution status
    if calling.execution.status.value != "completed":
        error = calling.execution.error or f"status: {calling.execution.status}"
        raise RuntimeError(f"Generation failed: {error}")

    response = calling.execution.response

    match return_as:
        case "text":
            return response.data
        case "raw":
            return response.raw_response
        case "message":
            return Message(
                content=AssistantResponseContent(
                    assistant_response=response.data,
                ),
                metadata={
                    "raw_response": response.raw_response,
                    **response.metadata,
                },
            )
        case "calling":
            return calling

    raise ValueError(f"Unsupported return_as: {return_as}")
