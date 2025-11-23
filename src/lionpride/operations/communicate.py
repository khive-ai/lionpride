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
    SystemContent,
)
from lionpride.session.messages.utils import prepare_messages_for_chat
from lionpride.types import Operable
from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

from .dispatcher import register_operation
from .operate.generate import generate

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

logger = logging.getLogger(__name__)


@register_operation("communicate")
async def communicate(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> str | dict | Message | BaseModel:
    """Stateful chat with optional structured output and retry.

    Args:
        parameters: Must include 'instruction' and 'imodel'. Optionally:
            response_model/operable for validation, max_retries, return_as.
    """
    # Extract parameters
    instruction = parameters.pop("instruction", None)
    if not instruction:
        raise ValueError("communicate requires 'instruction' parameter")

    imodel_param = parameters.get("imodel")
    if not imodel_param:
        raise ValueError("communicate requires 'imodel' parameter")

    # Support both string name and iModel object
    # If string: look up from registry
    # If object: use directly (for testing/direct usage)
    if isinstance(imodel_param, str):
        imodel_name = imodel_param
        imodel_direct = None
    else:
        # Assume it's an iModel object with a name attribute
        imodel_name = getattr(imodel_param, "name", None)
        if not imodel_name:
            raise ValueError("imodel must be a string name or have a 'name' attribute")
        imodel_direct = imodel_param  # Use directly, bypass registry

    # Common params
    context = parameters.pop("context", None)
    images = parameters.pop("images", None)
    image_detail = parameters.pop("image_detail", None)
    return_as: Literal["text", "raw", "message", "model"] = parameters.pop("return_as", "text")

    # JSON mode params
    response_model = parameters.pop("response_model", None)
    strict_validation = parameters.pop("strict_validation", False)
    fuzzy_parse = parameters.pop("fuzzy_parse", True)

    # LNDL mode params
    operable: Operable | None = parameters.pop("operable", None)
    lndl_threshold = parameters.pop("lndl_threshold", 0.85)

    # Retry params
    max_retries = parameters.pop("max_retries", 0)

    # Determine mode
    use_lndl = operable is not None
    use_json_schema = response_model is not None and not use_lndl

    # Validate return_as="model" has a validation target
    if return_as == "model" and not (response_model or operable):
        raise ValueError("return_as='model' requires 'response_model' or 'operable' parameter")

    # Resolve branch
    if isinstance(branch, str):
        branch = session.conversations.get_progression(branch)

    # Get imodel for sender name (use direct reference if provided)
    imodel = imodel_direct if imodel_direct else session.services.get(imodel_name)

    # Create initial instruction message
    ins_content = InstructionContent(
        instruction=instruction,
        context=context,
        images=images,
        image_detail=image_detail,
        response_model=response_model if use_json_schema else None,
    )
    ins_msg = Message(
        content=ins_content,
        sender=branch.user or session.user or "user",
        recipient=session.id,
    )

    # Retry loop
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        # Prepare messages based on mode
        if use_lndl:
            messages = _prepare_lndl_messages(session, branch, ins_msg, operable)
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
        if use_lndl:
            validated, error = _validate_lndl(response_text, operable, lndl_threshold)
        elif response_model:
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
                operable,
            )

        # Validation failed
        last_error = error
        logger.warning(f"Validation failed (attempt {attempt + 1}): {error}")

        # Retry if we have attempts left
        if attempt < max_retries:
            # Create retry instruction (will be used in next iteration)
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

    # Return failure indicator
    return {"raw": response_text, "validation_failed": True, "error": last_error}


# =============================================================================
# LNDL Mode Helpers
# =============================================================================


def _prepare_lndl_messages(
    session: Session,
    branch: Branch,
    ins_msg: Message,
    operable: Operable,
) -> list[dict[str, Any]]:
    """Prepare messages with LNDL system prompt injection."""
    from lionpride.lndl import get_lndl_system_prompt

    # Get base LNDL prompt
    lndl_prompt = get_lndl_system_prompt()

    # Add spec-specific format
    spec_format = _generate_lndl_spec_format(operable)
    if spec_format:
        lndl_prompt = f"{lndl_prompt}\n\n{spec_format}"

    # Create LNDL system message
    lndl_system_msg = _create_lndl_system_message(lndl_prompt, session, branch, ins_msg.sender)

    # Get branch messages and prepare for chat
    branch_msgs = session.messages[branch]
    messages = prepare_messages_for_chat(
        messages=branch_msgs,
        progression=branch,
        new_instruction=ins_msg,
        to_chat=True,
    )

    # Insert LNDL system message at the beginning
    return [lndl_system_msg.chat_msg, *list(messages)]


def _generate_lndl_spec_format(operable: Operable) -> str:
    """Generate LNDL format guidance from Operable specs."""
    # Handle both Operable (lionpride) and Operative (lionpride wrapper)
    from lionpride.operations.operate.operative import Operative

    if isinstance(operable, Operative):
        specs = operable.operable.get_specs()
    else:
        specs = operable.get_specs()
    if not specs:
        return ""

    spec_lines = []
    for spec in specs:
        spec_name = spec.name or "unknown"
        base_type = spec.base_type

        # Check if it's a Pydantic model
        if hasattr(base_type, "model_fields"):
            spec_lines.append(_format_model_spec(spec_name, base_type))
        else:
            spec_lines.append(_format_scalar_spec(spec_name, base_type))

    if not spec_lines:
        return ""

    return (
        "YOUR TASK REQUIRES LNDL FORMAT:\n"
        + "\n".join(spec_lines)
        + "\n\nRemember: You choose the aliases. Fuzzy matching handles typos."
    )


def _format_model_spec(spec_name: str, model_type: Any) -> str:
    """Format LNDL spec for a Pydantic model."""
    model_name = model_type.__name__
    field_info = []

    for field_name, field in model_type.model_fields.items():
        field_type = _get_field_type_str(field.annotation)
        field_info.append(f"{field_name}({field_type})")

    return f"""
Spec: {spec_name}
Model: {model_name}
Fields: {", ".join(field_info)}
Format: <lvar {model_name}.fieldname alias>value</lvar> for each field
OUT: {spec_name}: [your_aliases_in_order]"""


def _format_scalar_spec(spec_name: str, base_type: Any) -> str:
    """Format LNDL spec for a scalar/primitive type."""
    type_name = getattr(base_type, "__name__", str(base_type))
    return f"""
Spec: {spec_name}({type_name})
Format: <lvar alias>value</lvar>
OUT: {spec_name}: [alias] or {spec_name}: literal_value"""


def _get_field_type_str(field_type: Any) -> str:
    """Get readable string representation of field type."""
    if hasattr(field_type, "__origin__"):
        if field_type.__origin__ is list:
            return "array"
        elif field_type.__origin__ is dict:
            return "object"
        elif field_type.__origin__ is tuple:
            return "tuple"

    if hasattr(field_type, "__name__"):
        return field_type.__name__

    type_str = str(field_type)
    if type_str.startswith("typing."):
        type_str = type_str.replace("typing.", "")
    return type_str


def _create_lndl_system_message(
    lndl_prompt: str,
    session: Session,
    branch: Branch,
    recipient: str,
) -> Message:
    """Create LNDL system message, merging with existing if present."""
    system_msg = session.get_branch_system(branch)

    if system_msg:
        existing_message = (
            system_msg.content.system_message
            if hasattr(system_msg.content, "system_message")
            else str(system_msg.content)
        )
        content = SystemContent(system_message=f"{existing_message}\n\n{lndl_prompt}")
    else:
        content = SystemContent(system_message=lndl_prompt)

    return Message(
        content=content,
        sender="system",
        recipient=recipient,
    )


def _validate_lndl(
    response_text: str,
    operable: Operable,
    threshold: float,
) -> tuple[Any, str | None]:
    """Validate LNDL response with fuzzy matching."""
    from lionpride.lndl import parse_lndl_fuzzy
    from lionpride.operations.operate.operative import Operative

    # Extract the actual Operable if wrapped in Operative
    actual_operable = operable.operable if isinstance(operable, Operative) else operable

    try:
        lndl_output = parse_lndl_fuzzy(response_text, actual_operable, threshold=threshold)

        if lndl_output and lndl_output.fields:
            # Get the first field (usually the main result)
            spec_name = next(iter(lndl_output.fields.keys()))
            return lndl_output.fields[spec_name], None

        return None, "LNDL parsing returned no fields"

    except Exception as e:
        # Try fallback: create model and validate as JSON
        try:
            model = actual_operable.create_model()
            validated = PydanticSpecAdapter.validate_response(
                response_text, model, strict=False, fuzzy_parse=True
            )
            if validated is not None:
                return validated, None
        except Exception:
            pass

        return None, f"LNDL parsing failed: {e}"


# =============================================================================
# JSON Mode Helpers
# =============================================================================


def _validate_json(
    response_data: str | dict,
    response_model: type[BaseModel],
    strict: bool,
    fuzzy_parse: bool,
) -> tuple[Any, str | None]:
    """Validate JSON response with PydanticSpecAdapter."""
    try:
        # If response is already a dict, validate directly
        if isinstance(response_data, dict):
            validated = response_model.model_validate(response_data)
            return validated, None

        # Otherwise use PydanticSpecAdapter for string parsing
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


# =============================================================================
# Result Formatting
# =============================================================================


def _format_result(
    return_as: str,
    validated: Any,
    response_text: str,
    raw_response: dict,
    assistant_msg: Message,
    response_model: type[BaseModel] | None,
    operable: Operable | None,
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
