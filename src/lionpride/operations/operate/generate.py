# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from lionpride.services.types import iModel
from lionpride.session.messages import AssistantResponseContent, Message, prepare_messages_for_chat
from lionpride.types import Params
from lionpride.types.base import ModelConfig

if TYPE_CHECKING:
    from lionpride.session import Branch, Session


@dataclass(init=False, frozen=True, slots=True)
class GenerateParams(Params):
    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    imodel: iModel = None
    instruction: Message = None
    return_as: Literal["text", "raw", "message", "calling"] = "calling"
    imodel_kwargs: dict = field(default_factory=dict)


async def generate(session: Session, branch: Branch, params: GenerateParams):
    b = session.get_branch(branch, session.default_branch)
    if b is None:
        raise ValueError("No branch specified and no default_branch set on session")

    # Add instruction to branch if provided
    if params.instruction is not None:
        session.add_message(params.instruction, branches=b)

    # Prepare messages for chat API
    msgs = prepare_messages_for_chat(
        session.messages[b], new_instruction=params.instruction, to_chat=True
    )

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

    calling = await session.request(model_name, messages=msgs, **params.imodel_kwargs)

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
