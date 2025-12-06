# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from lionpride.errors import AccessError, ExecutionError, ValidationError
from lionpride.rules import Validator
from lionpride.session.messages import AssistantResponseContent, Message
from lionpride.types import is_sentinel

from .generate import generate
from .parse import parse
from .types import CommunicateParams, GenerateParams, ParseParams

if TYPE_CHECKING:
    from lionpride.services.types import Calling, NormalizedResponse
    from lionpride.session import Branch, Session
    from lionpride.types import Operable

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
    # params.generate is validated to be non-sentinel at this point
    assert params.generate is not None
    gen_params = params.generate.with_updates(
        copy_containers="deep",
        return_as="calling",
        imodel=params.generate.imodel or session.default_generate_model,
    )

    # 1. Generate
    gen_calling: Calling = await generate(
        session=session,
        branch=branch,
        params=gen_params,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )

    # Get response, handling potential UnsetType
    response = gen_calling.response
    if is_sentinel(response):
        raise ExecutionError("Generation completed but no response was returned", retryable=False)
    response = cast("NormalizedResponse", response)
    response_text = response.data

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

    # At this point capabilities is known to be not None/sentinel
    assert params.capabilities is not None
    capabilities = set(params.capabilities)
    if not capabilities.issubset(branch.capabilities):
        missing = capabilities - branch.capabilities
        raise AccessError(
            f"Branch '{branch.name}' missing capabilities: {missing}",
            details={"requested": list(capabilities), "available": list(branch.capabilities)},
        )

    # 2. Build request_model from operable + capabilities
    # params.operable is validated to be non-sentinel in the caller
    assert params.operable is not None
    operable: Operable = params.operable
    request_model = operable.create_model(include=capabilities)

    # 3. Generate (with schema)
    assert params.generate is not None
    gen_params = params.generate.with_updates(
        copy_containers="deep",
        return_as="calling",
        request_model=request_model,
        imodel=params.generate.imodel or session.default_generate_model,
    )
    gen_calling: Calling = await generate(
        session=session,
        branch=branch,
        params=gen_params,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
    )

    # Get response, handling potential UnsetType
    response = gen_calling.response
    if is_sentinel(response):
        raise ExecutionError("Generation completed but no response was returned", retryable=False)
    response = cast("NormalizedResponse", response)

    # 4. Parse (extract structured data) - raises ExecutionError on failure
    parse_params: ParseParams = (
        params.parse
        if params.parse is not None and not params._is_sentinel(params.parse)
        else ParseParams()
    )
    parsed = await parse(
        session=session,
        branch=branch,
        params=parse_params.with_updates(
            copy_containers="deep",
            text=response.data,
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
        operable,
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
    gen_calling: Calling,
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

    # Get response, handling potential UnsetType
    response = gen_calling.response
    if is_sentinel(response):
        # This shouldn't happen if we reach this point, but handle gracefully
        return
    response = cast("NormalizedResponse", response)

    # Build metadata dict
    metadata_dict: dict[str, Any] = {"raw_response": response.raw_response}
    if response.metadata is not None:
        metadata_dict.update(response.metadata)

    # Assistant response message
    session.add_message(
        message=Message(
            content=AssistantResponseContent.create(
                assistant_response=response.data,
            ),
            metadata=metadata_dict,
            sender=branch.id,
            recipient=session.id,
        ),
        branches=branch,
    )
