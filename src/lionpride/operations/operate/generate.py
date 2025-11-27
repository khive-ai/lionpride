# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Generate - stateless LLM call helper.

Lowest-level operation: just calls the model.
No message persistence, no validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lionpride.session.messages import AssistantResponseContent, Message
from lionpride.types import is_sentinel

from .types import GenerateParams

if TYPE_CHECKING:
    from lionpride.services.types import iModel
    from lionpride.services.types.backend import Calling
    from lionpride.session import Branch, Session

__all__ = ("generate", "generate_operation", "handle_return")

ReturnAs = Literal["text", "raw", "message", "calling"]


def handle_return(calling: Calling, return_as: ReturnAs) -> Any:
    """Handle return based on format.

    Args:
        calling: Completed API calling
        return_as: Output format

    Returns:
        Formatted result based on return_as

    Raises:
        RuntimeError: If calling failed and return_as requires data
        ValueError: If return_as is unsupported
    """
    # For "calling", always return - caller handles status
    if return_as == "calling":
        return calling

    # For data formats, must succeed
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
                content=AssistantResponseContent.create(
                    assistant_response=response.data,
                ),
                metadata={
                    "raw_response": response.raw_response,
                    **response.metadata,
                },
            )
        case _:
            raise ValueError(f"Unsupported return_as: {return_as}")


async def generate(
    imodel: iModel,
    return_as: ReturnAs = "calling",
    **kwargs,
) -> Any:
    """Stateless LLM call helper.

    Args:
        imodel: Model to invoke
        return_as: Output format (text|raw|message|calling)
        **kwargs: Arguments passed to imodel.invoke()

    Returns:
        Result in format specified by return_as
    """
    calling = await imodel.invoke(**kwargs)
    return handle_return(calling, return_as)


async def generate_operation(
    session: Session,
    branch: Branch | str,
    params: GenerateParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
):
    imodel_kw = params.imodel_kwargs or {}
    if not isinstance(imodel_kw, dict):
        raise ValueError(f"imodel_kwargs must be dict, got: {type(imodel_kw).__name__}")
    if "messages" in imodel_kw:
        raise ValueError("generate_operation does not accept 'messages' in imodel_kwargs")

    imodel = params.imodel or session.default_generate_model
    imodel = session.services.get(imodel, None)

    if is_sentinel(imodel, none_as_sentinel=True, empty_as_sentinel=True):
        raise ValueError("generate requires 'imodel' parameter")

    b_ = session.get_branch(branch)

    if imodel.name not in b_.resources:
        raise ValueError(f"Branch '{b_.name}' has no access to model '{imodel.name}'")

    msgs = session.messages[b_]
    from lionpride.session.messages import prepare_messages_for_chat

    prepared_msgs = prepare_messages_for_chat(
        msgs,
        new_instruction=params.instruction_message,
        to_chat=True,
        structure_format=params.structure_format,
    )
    return await generate(
        imodel,
        return_as=params.return_as,
        messages=prepared_msgs,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
        **imodel_kw,
    )
