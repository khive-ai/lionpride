# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.session.messages import AssistantResponseContent, Message

# Operations registered in Session._register_default_operations()

if TYPE_CHECKING:
    from lionpride.operations.types import GenerateParam
    from lionpride.session import Branch, Session


async def generate(
    session: Session,
    branch: Branch | str,
    parameters: GenerateParam | dict,
) -> str | dict | Message | Any:
    """Stateless text generation - does not persist messages.

    Args:
        parameters: GenerateParam or dict with imodel, messages, return_as
    """
    # Convert dict to GenerateParam for backward compatibility
    if isinstance(parameters, dict):
        from lionpride.operations.types import GenerateParam

        parameters = GenerateParam(**parameters)

    if not parameters.imodel:
        raise ValueError("generate requires 'imodel' parameter")

    # Support both string name and direct iModel object
    imodel = (
        session.services.get(parameters.imodel)
        if isinstance(parameters.imodel, str)
        else parameters.imodel
    )

    # Build invoke kwargs from parameters (exclude our fields)
    invoke_kwargs = {
        k: v
        for k, v in parameters.to_dict().items()
        if k not in ("imodel", "return_as") and v is not None
    }

    # Invoke via unified service interface
    calling = await imodel.invoke(**invoke_kwargs)

    # Check execution status
    if calling.execution.status.value != "completed":
        error = calling.execution.error or f"status: {calling.execution.status}"
        raise RuntimeError(f"Generation failed: {error}")

    response = calling.execution.response

    match parameters.return_as:
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

    raise ValueError(f"Unsupported return_as: {parameters.return_as}")
