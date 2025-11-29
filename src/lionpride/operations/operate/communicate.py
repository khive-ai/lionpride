# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from lionpride.errors import AccessError, ValidationError
from lionpride.rules import Validator
from lionpride.services.types import APICalling
from lionpride.session.messages import AssistantResponseContent, Message

from .generate import generate
from .parse import parse
from .types import CommunicateParams, GenerateParams, ParseParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("communicate",)

logger = logging.getLogger(__name__)


async def communicate(
    session: Session,
    branch: Branch | str,
    params: CommunicateParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
    validator: Validator | None = None,
) -> str | Any:
    b_ = session.get_branch(branch)

    if params._is_sentinel(params.generate):
        raise ValidationError("communicate requires 'generate' params")

    # Determine path based on operable
    has_operable = not params._is_sentinel(params.operable)

    if has_operable:
        return await _communicate_with_operable(
            session=session,
            branch=b_,
            params=params,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            validator=validator,
        )
    else:
        return await _communicate_text(
            session=session,
            branch=b_,
            params=params,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
        )


async def _communicate_text(
    session: Session,
    branch: Branch,
    params: CommunicateParams,
    poll_timeout: float | None,
    poll_interval: float | None,
) -> str:
    """Text path: Generate → persist → return text.

    No operable, no validation - just stateful chat.
    """
    gen_params = params.generate.with_updates(
        copy_containers="deep",
        return_as="calling",
        imodel=params.generate.imodel or session.default_generate_model,
    )

    # 1. Generate
    gen_calling: APICalling = await generate(
        session=session,
        branch=branch,
        params=gen_params,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )

    response_text = gen_calling.response.data

    # 2. Persist messages
    _persist_messages(session, branch, gen_params, gen_calling)

    # 3. Return text
    return response_text


async def _communicate_with_operable(
    session: Session,
    branch: Branch,
    params: CommunicateParams,
    poll_timeout: float | None,
    poll_interval: float | None,
    validator: Validator | None,
) -> Any:
    """IPU path: Generate → Parse → Validate → Persist → Return.

    Requires operable and capabilities. All failures raise errors.
    """
    # 1. Validate capabilities (security gate)
    if params._is_sentinel(params.capabilities):
        raise ValidationError("capabilities must be declared when using structured output")

    capabilities = set(params.capabilities)
    if not capabilities.issubset(branch.capabilities):
        missing = capabilities - branch.capabilities
        raise AccessError(
            f"Branch '{branch.name}' missing capabilities: {missing}",
            details={"requested": list(capabilities), "available": list(branch.capabilities)},
        )

    # 2. Build request_model from operable + capabilities
    request_model = params.operable.create_model(include=capabilities)

    # 3. Generate (with schema)
    gen_params = params.generate.with_updates(
        copy_containers="deep",
        return_as="calling",
        request_model=request_model,
        imodel=params.generate.imodel or session.default_generate_model,
    )
    gen_calling: APICalling = await generate(
        session=session,
        branch=branch,
        params=gen_params,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )

    # 4. Parse (extract structured data) - raises ExecutionError on failure
    parse_params = params.parse if not params._is_sentinel(params.parse) else ParseParams()
    parsed = await parse(
        session=session,
        branch=branch,
        params=parse_params.with_updates(
            copy_containers="deep",
            text=gen_calling.response.data,
            target_keys=list(capabilities),
            imodel=parse_params.imodel or session.default_parse_model,
            structure_format=gen_params.structure_format,  # Must match generate format
        ),
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )

    # 5. Validate (security microkernel) - raises ValidationError on failure
    val_ = validator or Validator()
    validated = await val_.validate(
        parsed,
        params.operable,
        capabilities,
        params.auto_fix,
        params.strict_validation,
    )

    # 6. Persist messages (only on success)
    _persist_messages(session, branch, gen_params, gen_calling)

    return validated


def _persist_messages(
    session: Session,
    branch: Branch,
    gen_params: GenerateParams,
    gen_calling: APICalling,
) -> None:
    """Add instruction and response messages to branch."""
    # Instruction message
    if gen_params.instruction_message is not None:
        session.add_message(
            gen_params.instruction_message.model_copy(
                update={"sender": session.id, "recipient": branch.id}
            ),
            branches=branch,
        )

    # Assistant response message
    session.add_message(
        message=Message(
            content=AssistantResponseContent.create(
                assistant_response=gen_calling.response.data,
            ),
            metadata={
                "raw_response": gen_calling.response.raw_response,
                **gen_calling.response.metadata,
            },
            sender=branch.id,
            recipient=session.id,
        ),
        branches=branch,
    )
