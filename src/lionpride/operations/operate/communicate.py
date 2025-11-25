# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
)
from lionpride.session.messages.utils import prepare_messages_for_chat
from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

from .generate import generate

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

logger = logging.getLogger(__name__)


async def communicate(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> str | dict | Message | BaseModel:
    """Stateful chat with optional structured output and retry.

    Args:
        parameters: Must include 'instruction' and 'imodel'. Optionally:
            response_model for validation, max_retries, return_as.
    """
    # Extract parameters
    instruction = parameters.pop("instruction", None)
    if not instruction:
        raise ValueError("communicate requires 'instruction' parameter")

    imodel_param = parameters.get("imodel")
    if not imodel_param:
        raise ValueError("communicate requires 'imodel' parameter")

    # Support both string name and iModel object
    if isinstance(imodel_param, str):
        imodel_name = imodel_param
        imodel_direct = None
    else:
        imodel_name = getattr(imodel_param, "name", None)
        if not imodel_name:
            raise ValueError("imodel must be a string name or have a 'name' attribute")
        imodel_direct = imodel_param

    # Common params
    context = parameters.pop("context", None)
    images = parameters.pop("images", None)
    image_detail = parameters.pop("image_detail", None)
    return_as: Literal["text", "raw", "message", "model"] = parameters.pop("return_as", "text")

    # JSON mode params
    response_model = parameters.pop("response_model", None)
    strict_validation = parameters.pop("strict_validation", False)
    fuzzy_parse = parameters.pop("fuzzy_parse", True)

    # Retry params
    max_retries = parameters.pop("max_retries", 0)

    # Validate return_as="model" has a validation target
    if return_as == "model" and not response_model:
        raise ValueError("return_as='model' requires 'response_model' parameter")

    # Resolve branch
    if isinstance(branch, str):
        branch = session.conversations.get_progression(branch)

    # Get imodel for sender name
    imodel = imodel_direct if imodel_direct else session.services.get(imodel_name)

    # Create initial instruction message
    ins_content = InstructionContent(
        instruction=instruction,
        context=context,
        images=images,
        image_detail=image_detail,
        response_model=response_model,
    )
    ins_msg = Message(
        content=ins_content,
        sender=branch.user or session.user or "user",
        recipient=session.id,
    )

    # Retry loop
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        # Prepare messages
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
        gen_params = {**parameters, "messages": messages, "return_as": "message"}
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

        # Validate if needed
        if response_model:
            validated, error = _validate_json(
                response_text, response_model, strict_validation, fuzzy_parse
            )
        else:
            # No validation needed
            validated, error = response_text, None

        # Success - return result
        if error is None:
            return _format_result(
                return_as,
                validated,
                response_text,
                raw_response,
                assistant_msg,
                response_model,
            )

        # Validation failed
        last_error = error
        logger.warning(f"Validation failed (attempt {attempt + 1}): {error}")

        # Retry if we have attempts left
        if attempt < max_retries:
            retry_instruction = (
                f"Your previous response failed validation with error:\n{error}\n\n"
                f"Please provide a valid response that matches the expected format."
            )
            ins_content = InstructionContent(instruction=retry_instruction)
            ins_msg = Message(
                content=ins_content,
                sender=branch.user or session.user or "user",
                recipient=session.id,
            )
            logger.info(f"Retrying (attempt {attempt + 2}/{max_retries + 1})...")

    # All retries exhausted
    if strict_validation:
        raise ValueError(
            f"Response validation failed after {max_retries + 1} attempts: {last_error}"
        )

    return {"raw": response_text, "validation_failed": True, "error": last_error}


def _validate_json(
    response_data: str | dict,
    response_model: type[BaseModel],
    strict: bool,
    fuzzy_parse: bool,
) -> tuple[Any, str | None]:
    """Validate JSON response with PydanticSpecAdapter."""
    try:
        if isinstance(response_data, dict):
            validated = response_model.model_validate(response_data)
            return validated, None

        validated = PydanticSpecAdapter.validate_response(
            response_data,
            response_model,
            strict=strict,
            fuzzy_parse=fuzzy_parse,
        )
        if validated is not None:
            return validated, None
        return None, f"Response did not match {response_model.__name__} schema"
    except Exception as e:
        return None, str(e)


def _format_result(
    return_as: str,
    validated: Any,
    response_text: str,
    raw_response: dict,
    assistant_msg: Message,
    response_model: type[BaseModel] | None,
) -> Any:
    """Format the result based on return_as parameter."""
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
