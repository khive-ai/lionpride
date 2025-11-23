from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.session.messages import (
    InstructionContent,
    Message,
    SystemContent,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from lionpride.session import Branch, Session

    from .operative import Operative


# =============================================================================
# LNDL Spec Generation (merged from lndl_spec.py)
# =============================================================================


def generate_lndl_spec_format(operative: Operative) -> str:
    """Generate LNDL format guidance for operative's specs.

    Creates user-friendly LNDL format instructions based on the
    operative's Operable specs.

    Args:
        operative: Operative containing specs

    Returns:
        LNDL format string to append to system prompt
    """
    if not operative or not operative.operable:
        return ""

    specs = operative.operable.get_specs()
    if not specs:
        return ""

    spec_lines = []

    for spec in specs:
        spec_name = spec.name or "unknown"
        base_type = spec.base_type

        # Check if it's a Pydantic model (has model_fields)
        is_pydantic = hasattr(base_type, "model_fields")

        if is_pydantic:
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
        # Handle generic types like list[str]
        if field_type.__origin__ is list:
            return "array"
        elif field_type.__origin__ is dict:
            return "object"
        elif field_type.__origin__ is tuple:
            return "tuple"

    if hasattr(field_type, "__name__"):
        return field_type.__name__

    type_str = str(field_type)
    # Clean up type string for readability
    if type_str.startswith("typing."):
        type_str = type_str.replace("typing.", "")
    return type_str


# =============================================================================
# Message Preparation
# =============================================================================


def create_instruction_message(
    instruction: str | InstructionContent | Message,
    session: Session,
    branch: Branch,
    *,
    sender: str | None = None,
    recipient: str | None = None,
    context: Any = None,
    response_model: type[BaseModel] | None = None,
    tool_schemas: list[Any] | None = None,
    images: list[str] | None = None,
    image_detail: str | None = None,
    use_lndl: bool = False,
) -> Message:
    """Create instruction message from various input types.

    Args:
        instruction: The instruction (str, InstructionContent, or Message)
        session: Current session
        branch: Current branch
        sender: Message sender
        recipient: Message recipient
        context: Additional context
        response_model: Response model for structured output
        tool_schemas: Tool schemas for injection
        images: Image URLs
        image_detail: Image detail level
        use_lndl: Whether to use LNDL format

    Returns:
        Message with appropriate content
    """
    if isinstance(instruction, Message):
        return instruction

    if isinstance(instruction, str):
        content = InstructionContent(
            instruction=instruction,
            context=context,
            response_model=response_model,
            tool_schemas=tool_schemas,
            images=images,
            image_detail=image_detail,
            use_lndl=use_lndl if use_lndl else None,
        )
    elif isinstance(instruction, InstructionContent):
        content = _update_instruction_content(
            instruction,
            response_model=response_model,
            use_lndl=use_lndl,
        )
    else:
        raise TypeError(
            f"instruction must be str, Message, or InstructionContent, got {type(instruction)}"
        )

    return Message(
        content=content,
        sender=sender or branch.user or session.user or "user",
        recipient=recipient or session.id,
    )


def _update_instruction_content(
    content: InstructionContent,
    response_model: type[BaseModel] | None = None,
    use_lndl: bool = False,
) -> InstructionContent:
    """Update InstructionContent fields if needed."""
    needs_update = content._is_sentinel(content.response_model) or (
        use_lndl and content._is_sentinel(content.use_lndl)
    )

    if not needs_update:
        return content

    return InstructionContent(
        instruction=content.instruction if not content._is_sentinel(content.instruction) else None,
        context=content.context if not content._is_sentinel(content.context) else None,
        tool_schemas=content.tool_schemas
        if not content._is_sentinel(content.tool_schemas)
        else None,
        response_model=response_model
        if content._is_sentinel(content.response_model)
        else content.response_model,
        images=content.images if not content._is_sentinel(content.images) else None,
        image_detail=content.image_detail
        if not content._is_sentinel(content.image_detail)
        else None,
        use_lndl=use_lndl if use_lndl else None,
    )


def prepare_lndl_messages(
    session: Session,
    branch: Branch,
    ins_msg: Message,
    operative: Any,
) -> list[dict[str, Any]]:
    """Prepare messages with LNDL system prompt injection.

    Args:
        session: Current session
        branch: Current branch
        ins_msg: Instruction message
        operative: Operative for LNDL spec generation

    Returns:
        List of chat messages with LNDL prompt
    """
    from lionpride.lndl import get_lndl_system_prompt
    from lionpride.session.messages.utils import prepare_messages_for_chat

    # Get base LNDL prompt
    lndl_prompt = get_lndl_system_prompt()

    # Add spec-specific format
    spec_format = generate_lndl_spec_format(operative)
    if spec_format:
        lndl_prompt = f"{lndl_prompt}\n\n{spec_format}"

    # Create LNDL system message
    lndl_system_msg = _create_lndl_system_message(
        lndl_prompt,
        session,
        branch,
        ins_msg.sender,
    )

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


def _create_lndl_system_message(
    lndl_prompt: str,
    session: Session,
    branch: Branch,
    recipient: str,
) -> Message:
    """Create LNDL system message, merging with existing system message if present."""
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


def prepare_tool_schemas(
    session: Session,
    tools: bool | list[str],
) -> list[Any] | None:
    """Prepare tool schemas from session services.

    Args:
        session: Current session
        tools: True for all tools, list for specific tools

    Returns:
        Tool schemas or None
    """
    if not tools:
        return None

    if tools is True:
        return session.services.get_tool_schemas()
    elif isinstance(tools, list):
        return session.services.get_tool_schemas(tool_names=tools)
    return None
