# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Stateful chat operation with optional structured output and retry.

Main entry point for LLM communication with conversation persistence,
validation (LNDL, JSON), and retry logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
)
from lionpride.session.messages.utils import prepare_messages_for_chat

from ..lndl import prepare_lndl_messages
from ..validation import validate_response
from .generate import generate

if TYPE_CHECKING:
    from lionpride.operations.types import CommunicateParam
    from lionpride.session import Branch, Session

logger = logging.getLogger(__name__)


async def communicate(
    session: Session,
    branch: Branch | str,
    parameters: CommunicateParam | dict,
) -> str | dict | Message | BaseModel:
    """Stateful chat with optional structured output and retry.

    Supports three validation modes:
    - LNDL: Fuzzy-parsed structured output via operable
    - JSON: Schema-validated JSON via response_model
    - Plain: No validation, returns raw text

    Args:
        session: Session with conversation state
        branch: Branch for conversation history
        parameters: CommunicateParam or dict with instruction, imodel, etc.

    Returns:
        Result based on return_as: text, raw, message, or model
    """
    # Convert dict to CommunicateParam for backward compatibility
    if isinstance(parameters, dict):
        from lionpride.operations.types import CommunicateParam

        parameters = CommunicateParam(**parameters)

    # Validate required parameters
    if not parameters.instruction:
        raise ValueError("communicate requires 'instruction' parameter")

    if not parameters.imodel:
        raise ValueError("communicate requires 'imodel' parameter")

    # Support both string name and iModel object
    if isinstance(parameters.imodel, str):
        imodel_name = parameters.imodel
        imodel_direct = None
    else:
        imodel_name = getattr(parameters.imodel, "name", None)
        if not imodel_name:
            raise ValueError("imodel must be a string name or have a 'name' attribute")
        imodel_direct = parameters.imodel

    # Determine validation mode
    use_lndl = parameters.operable is not None
    use_json = parameters.response_model is not None and not use_lndl

    # Validate return_as="model" has a validation target
    if parameters.return_as == "model" and not (parameters.response_model or parameters.operable):
        raise ValueError("return_as='model' requires 'response_model' or 'operable'")

    # Resolve branch
    if isinstance(branch, str):
        branch = session.conversations.get_progression(branch)

    # Get imodel for sender name
    imodel = imodel_direct if imodel_direct else session.services.get(imodel_name)

    # Create initial instruction message
    ins_content = InstructionContent(
        instruction=parameters.instruction,
        context=parameters.context,
        images=parameters.images,
        image_detail=parameters.image_detail,
        response_model=parameters.response_model if use_json else None,
    )
    ins_msg = Message(
        content=ins_content,
        sender=branch.user or session.user or "user",
        recipient=session.id,
    )

    # Retry loop
    last_error: str | None = None
    for attempt in range(parameters.max_retries + 1):
        # Prepare messages based on mode
        if use_lndl:
            messages = prepare_lndl_messages(session, branch, ins_msg, parameters.operable)
        else:
            branch_msgs = session.messages[branch]
            messages = list(
                prepare_messages_for_chat(
                    messages=branch_msgs,
                    progression=branch,
                    new_instruction=ins_msg,
                    to_chat=True,
                )
            )

        # Call generate (stateless)
        from lionpride.operations.types import GenerateParam

        gen_params = GenerateParam(
            imodel=parameters.imodel,
            model=parameters.model,
            messages=messages,
            return_as="message",
        )
        result_msg = await generate(session, branch, gen_params)

        # Extract response data
        response_text = result_msg.content.assistant_response
        raw_response = result_msg.metadata.get("raw_response", {})
        metadata = {k: v for k, v in result_msg.metadata.items() if k != "raw_response"}

        # Create assistant message for persistence
        assistant_msg = Message(
            content=AssistantResponseContent(assistant_response=response_text),
            sender=imodel.name,
            recipient=branch.user or session.user or "user",
            metadata={"raw_response": raw_response, **metadata},
        )

        # Add messages to session/branch (stateful)
        session.add_message(ins_msg, branches=branch)
        session.add_message(assistant_msg, branches=branch)

        # Validate using strategy pattern
        validation_result = validate_response(
            response_text,
            operable=parameters.operable if use_lndl else None,
            response_model=parameters.response_model if use_json else None,
            threshold=parameters.lndl_threshold,
            strict=parameters.strict_validation,
            fuzzy_parse=parameters.fuzzy_parse,
        )

        if validation_result.success:
            return _format_result(
                parameters.return_as,
                validation_result.data,
                response_text,
                raw_response,
                assistant_msg,
            )

        # Validation failed
        last_error = validation_result.error
        logger.warning(f"Validation failed (attempt {attempt + 1}): {last_error}")

        # Retry if we have attempts left
        if attempt < parameters.max_retries:
            ins_content = InstructionContent(
                instruction=(
                    f"Your previous response failed validation:\n{last_error}\n\n"
                    "Please provide a valid response matching the expected format."
                )
            )
            ins_msg = Message(
                content=ins_content,
                sender=branch.user or session.user or "user",
                recipient=session.id,
            )
            logger.info(f"Retrying (attempt {attempt + 2}/{parameters.max_retries + 1})...")

    # All retries exhausted
    if parameters.strict_validation:
        raise ValueError(
            f"Validation failed after {parameters.max_retries + 1} attempts: {last_error}"
        )

    return {"raw": response_text, "validation_failed": True, "error": last_error}


def _format_result(
    return_as: str,
    validated: Any,
    response_text: str,
    raw_response: dict,
    assistant_msg: Message,
) -> Any:
    """Format result based on return_as parameter."""
    match return_as:
        case "text":
            if isinstance(validated, BaseModel):
                return validated.model_dump_json()
            return response_text
        case "raw":
            return raw_response
        case "message":
            return assistant_msg
        case "model":
            return validated

    raise ValueError(f"Unsupported return_as: {return_as}")
