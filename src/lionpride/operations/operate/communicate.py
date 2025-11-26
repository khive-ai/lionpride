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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionpride.rules.validator import Validator
from lionpride.services.types import iModel
from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
)

from .generate import generate
from .parse import parse
from .types import CommunicateParams, GenerateParams, ParseParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session
    from lionpride.types import Operable

logger = logging.getLogger(__name__)

__all__ = ("CommunicateParams", "communicate")


async def communicate(
    session: Session,
    branch: Branch,
    params: CommunicateParams | dict,
) -> str | dict | Message | BaseModel:
    """Stateful chat with optional structured output.

    Security:
    - Branch must have access to imodel (branch.resources)
    - Structured output respects capabilities (branch.capabilities or params.capabilities)

    Args:
        session: Session for services and message storage
        branch: Branch for message persistence and access control
        params: Communicate parameters (CommunicateParams or dict)

    Returns:
        Response in format specified by return_as

    Raises:
        PermissionError: If branch doesn't have access to imodel
        ValueError: If validation fails with strict_validation=True
    """
    # Convert dict to CommunicateParams if needed
    if isinstance(params, dict):
        # Handle top-level context from flow execution (merge into generate.context)
        flow_context = params.pop("context", None)

        # Handle nested dict conversion for generate and parse
        if "generate" in params and isinstance(params["generate"], dict):
            # Merge flow context into generate.context if present
            if flow_context:
                gen_dict = params["generate"]
                existing_ctx = gen_dict.get("context") or {}
                if isinstance(existing_ctx, dict):
                    gen_dict["context"] = {**existing_ctx, **flow_context}
                else:
                    gen_dict["context"] = {"original": existing_ctx, **flow_context}
            params["generate"] = GenerateParams(**params["generate"])
        if "parse" in params and isinstance(params["parse"], dict):
            params["parse"] = ParseParams(**params["parse"])
        params = CommunicateParams(**params)

    # Validate required generate params
    if params.generate is None:
        raise ValueError("communicate requires 'generate' parameter")

    gen = params.generate
    if gen.instruction is None:
        raise ValueError("communicate requires 'generate.instruction' parameter")

    if gen.imodel is None:
        raise ValueError("communicate requires 'generate.imodel' parameter")

    # 1. Resource access check
    model_name = gen.imodel.name if isinstance(gen.imodel, iModel) else gen.imodel
    if model_name not in branch.resources:
        raise PermissionError(
            f"Branch '{branch.name}' cannot access model '{model_name}'. "
            f"Allowed: {branch.resources or 'none'}"
        )

    # Resolve imodel for sender name
    imodel = gen.imodel if isinstance(gen.imodel, iModel) else session.services.get(model_name)

    # Determine capabilities for validation
    capabilities = params.capabilities if params.capabilities is not None else branch.capabilities

    # Get initial instruction message (uses instruction_message property)
    ins_msg = gen.instruction_message
    if ins_msg is None:
        raise ValueError("Failed to create instruction message")

    # Set sender/recipient if not already set
    # Pattern: user sends instruction TO branch (which forwards to model)
    if ins_msg.sender is None:
        ins_msg = Message(
            content=ins_msg.content,
            sender=branch.user or "user",
            recipient=branch.id,
            metadata=ins_msg.metadata,
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
            imodel_kwargs=gen.imodel_kwargs,
        )
        result_msg = await generate(session, branch, gen_params)

        # Extract response data
        response_text = result_msg.content.assistant_response
        raw_response = result_msg.metadata.get("raw_response", {})
        metadata = {k: v for k, v in result_msg.metadata.items() if k != "raw_response"}

        # Create assistant message for persistence
        # Pattern: branch (via model) responds TO user
        assistant_msg = Message(
            content=AssistantResponseContent(assistant_response=response_text),
            sender=branch.id,
            recipient=branch.user or "user",
            metadata={"raw_response": raw_response, "model": imodel.name, **metadata},
        )

        # 3. Add messages to session/branch (stateful)
        # Check if message already added (generate may have added it)
        if ins_msg.id not in session.messages:
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
                parse_params=params.parse,
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
            ins_msg = Message(
                content=InstructionContent(instruction=retry_instruction),
                sender=branch.user or "user",
                recipient=branch.id,
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
    parse_params: ParseParams | None = None,
) -> tuple[Any, str | None]:
    """Validate structured output via parse + Validator.

    Flow: response_text → parse() → dict → Validator.validate() → model
    Operable carries its own adapter for framework-agnostic model creation.
    """
    try:
        # Get expected keys from operable specs
        target_keys = [spec.name for spec in operable.get_specs() if spec.name]

        # 1. Parse: extract JSON dict
        # Use provided parse_params or create default
        if parse_params is not None:
            # Override text with response_text
            p_params = ParseParams(
                text=response_text,
                target_keys=parse_params.target_keys or target_keys,
                imodel=parse_params.imodel,
                similarity_threshold=parse_params.similarity_threshold,
                handle_unmatched=parse_params.handle_unmatched,
                max_retries=parse_params.max_retries,
                imodel_kwargs=parse_params.imodel_kwargs,
            )
        else:
            p_params = ParseParams(
                text=response_text,
                target_keys=target_keys,
                similarity_threshold=0.85 if fuzzy_parse else 1.0,
                handle_unmatched="force" if fuzzy_parse else "raise",
            )

        parsed_dict = await parse(session, branch, p_params)

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
