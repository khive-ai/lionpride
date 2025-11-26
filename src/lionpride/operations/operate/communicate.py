# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Communicate operation - stateful chat with optional structured output.

Builds on generate (stateless) to provide:
- Message persistence to branch
- Retry logic for validation failures
- Structured output via parse + Validator

Flow:
    instruction → generate() → response → [parse() → Validator.validate()] → output
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionpride.rules.validator import Validator
from lionpride.services.types import iModel
from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
)
from lionpride.types.base import ModelConfig, Params

from .generate import GenerateParams, generate
from .parse import ParseParams, parse

if TYPE_CHECKING:
    from lionpride.session import Branch, Session
    from lionpride.types import Operable

logger = logging.getLogger(__name__)

__all__ = ("CommunicateParams", "communicate")


@dataclass(init=False, frozen=True, slots=True)
class CommunicateParams(Params):
    """Parameters for communicate operation.

    Communicate is stateful chat: generates response and persists messages
    to branch. Optionally validates structured output.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    instruction: str = None
    """User instruction text"""

    imodel: iModel | str = None
    """Model to use for generation"""

    context: dict[str, Any] | None = None
    """Additional context for instruction"""

    images: list[str] | None = None
    """Image URLs for multimodal input"""

    image_detail: str | None = None
    """Image detail level"""

    # Structured output params
    operable: Operable | None = None
    """Operable for structured output validation (carries its own adapter)"""

    capabilities: set[str] | None = None
    """Capabilities for validation (defaults to branch.capabilities)"""

    # Retry params
    max_retries: int = 0
    """Retry attempts for validation failures"""

    strict_validation: bool = False
    """Raise on validation failure (vs return error dict)"""

    fuzzy_parse: bool = True
    """Enable fuzzy JSON parsing"""

    # Output control
    return_as: Literal["text", "raw", "message", "model"] = "text"
    """Output format"""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel"""


async def communicate(
    session: Session,
    branch: Branch,
    params: CommunicateParams,
) -> str | dict | Message | BaseModel:
    """Stateful chat with optional structured output.

    Security:
    - Branch must have access to imodel (branch.resources)
    - Structured output respects capabilities (branch.capabilities or params.capabilities)

    Args:
        session: Session for services and message storage
        branch: Branch for message persistence and access control
        params: Communicate parameters

    Returns:
        Response in format specified by return_as

    Raises:
        PermissionError: If branch doesn't have access to imodel
        ValueError: If validation fails with strict_validation=True
    """
    if not params.instruction:
        raise ValueError("communicate requires 'instruction' parameter")

    if params.imodel is None:
        raise ValueError("communicate requires 'imodel' parameter")

    # 1. Resource access check
    model_name = params.imodel.name if isinstance(params.imodel, iModel) else params.imodel
    if model_name not in branch.resources:
        raise PermissionError(
            f"Branch '{branch.name}' cannot access model '{model_name}'. "
            f"Allowed: {branch.resources or 'none'}"
        )

    # Resolve imodel for sender name
    imodel = (
        params.imodel if isinstance(params.imodel, iModel) else session.services.get(model_name)
    )

    # Determine capabilities for validation
    capabilities = params.capabilities if params.capabilities is not None else branch.capabilities

    # Create initial instruction message
    ins_content = InstructionContent(
        instruction=params.instruction,
        context=params.context,
        images=params.images,
        image_detail=params.image_detail,
    )
    ins_msg = Message(
        content=ins_content,
        sender=branch.user,
        recipient=session.id,
    )

    # Retry loop
    last_error: str | None = None
    validator = Validator() if params.operable else None

    for attempt in range(params.max_retries + 1):
        # 2. Generate response (stateless)
        gen_params = GenerateParams(
            imodel=model_name,
            instruction=ins_msg,
            return_as="message",
            imodel_kwargs=params.imodel_kwargs,
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
            recipient=branch.user,
            metadata={"raw_response": raw_response, **metadata},
        )

        # 3. Add messages to session/branch (stateful)
        session.add_message(ins_msg, branches=branch)
        session.add_message(assistant_msg, branches=branch)

        # 4. Validate if structured output requested
        if params.operable and validator:
            validated, error = await _validate_structured(
                session=session,
                branch=branch,
                response_text=response_text,
                operable=params.operable,
                capabilities=capabilities,
                validator=validator,
                fuzzy_parse=params.fuzzy_parse,
            )
        else:
            validated, error = response_text, None

        # Success - return result
        if error is None:
            return _format_result(
                return_as=params.return_as,
                validated=validated,
                response_text=response_text,
                raw_response=raw_response,
                assistant_msg=assistant_msg,
            )

        # Validation failed
        last_error = error
        logger.warning(f"Validation failed (attempt {attempt + 1}): {error}")

        # Retry if we have attempts left
        if attempt < params.max_retries:
            retry_instruction = (
                f"Your previous response failed validation with error:\n{error}\n\n"
                f"Please provide a valid response that matches the expected format."
            )
            ins_content = InstructionContent(instruction=retry_instruction)
            ins_msg = Message(
                content=ins_content,
                sender=branch.user,
                recipient=session.id,
            )
            logger.info(f"Retrying (attempt {attempt + 2}/{params.max_retries + 1})...")

    # All retries exhausted
    if params.strict_validation:
        raise ValueError(
            f"Response validation failed after {params.max_retries + 1} attempts: {last_error}"
        )

    return {"raw": response_text, "validation_failed": True, "error": last_error}


async def _validate_structured(
    session: Session,
    branch: Branch,
    response_text: str,
    operable: Operable,
    capabilities: set[str],
    validator: Validator,
    fuzzy_parse: bool,
) -> tuple[Any, str | None]:
    """Validate structured output via parse + Validator.

    Flow: response_text → parse() → dict → Validator.validate() → model
    Operable carries its own adapter for framework-agnostic model creation.
    """
    try:
        # Get expected keys from operable specs
        target_keys = [spec.name for spec in operable.get_specs() if spec.name]

        # 1. Parse: extract JSON dict
        parse_params = ParseParams(
            text=response_text,
            target_keys=target_keys,
            similarity_threshold=0.85 if fuzzy_parse else 1.0,
            handle_unmatched="force" if fuzzy_parse else "raise",
        )
        parsed_dict = await parse(session, branch, parse_params)

        if parsed_dict is None:
            return None, "Failed to extract JSON from response"

        # 2. Validate: dict → model (with capability enforcement)
        # Operable uses its own adapter (op.adapter.create_model, etc.)
        validated = await validator.validate(
            data=parsed_dict,
            operable=operable,
            capabilities=capabilities,
            auto_fix=fuzzy_parse,
            strict=True,
        )
        return validated, None

    except Exception as e:
        return None, str(e)


def _format_result(
    return_as: str,
    validated: Any,
    response_text: str,
    raw_response: dict,
    assistant_msg: Message,
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
