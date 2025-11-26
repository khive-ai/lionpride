# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Parse operation - fuzzy JSON parsing with LLM fallback.

Tries direct extraction first, falls back to LLM reformatting on failure.
Essential for robust pipelines where LLM output may be malformed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from lionpride.libs.string_handlers import extract_json
from lionpride.ln import fuzzy_validate_mapping

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

# Type alias for handle_unmatched parameter
HandleUnmatched = Literal["ignore", "raise", "remove", "fill", "force"]
HandleValidation = Literal["raise", "return_none", "return_value"]


async def parse(
    session: Session,
    branch: Branch | str,
    parameters: dict[str, Any],
) -> Any:
    """Parse text into structured output with fuzzy matching and LLM fallback.

    Args:
        parameters: Must include:
            - text: str - Text to parse
            - response_model: type[BaseModel] - Target Pydantic model
            Optional:
            - imodel: Model to use for LLM fallback
            - max_retries: int - Retry attempts for LLM fallback (default: 3)
            - similarity_threshold: float - Fuzzy match threshold (default: 0.85)
            - handle_unmatched: str - How to handle unmatched keys (default: "force")
            - handle_validation: str - "raise", "return_none", "return_value" (default: "return_value")

    Returns:
        Validated Pydantic model instance, or fallback based on handle_validation
    """
    # Extract parameters
    text = parameters.get("text")
    if not text:
        raise ValueError("parse requires 'text' parameter")

    response_model = parameters.get("response_model")
    if not response_model:
        raise ValueError("parse requires 'response_model' parameter")

    imodel = parameters.get("imodel")
    max_retries = parameters.get("max_retries", 3)
    similarity_threshold = parameters.get("similarity_threshold", 0.85)
    handle_unmatched = cast(HandleUnmatched, parameters.get("handle_unmatched", "force"))
    handle_validation = cast(HandleValidation, parameters.get("handle_validation", "return_value"))

    # Try direct extraction first (no LLM call needed)
    try:
        result = _try_direct_parse(
            text=text,
            response_model=response_model,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
        )
        if result is not None:
            return result
    except Exception:
        pass  # Fall through to LLM fallback

    # LLM fallback - use communicate to reformat
    if not imodel:
        # No model for fallback
        return _handle_failure(text, handle_validation, "No imodel provided for LLM fallback")

    # Resolve branch
    if isinstance(branch, str):
        branch = session.conversations.get_progression(branch)

    # Try LLM reformatting with retries
    last_error = None
    for _attempt in range(max_retries):
        try:
            result = await _llm_reformat(
                session=session,
                branch=branch,
                text=text,
                response_model=response_model,
                imodel=imodel,
                similarity_threshold=similarity_threshold,
                handle_unmatched=handle_unmatched,
            )
            if result is not None:
                return result
        except Exception as e:
            last_error = e
            continue

    # All retries failed
    return _handle_failure(text, handle_validation, str(last_error))


def _try_direct_parse(
    text: str,
    response_model: type[BaseModel],
    similarity_threshold: float,
    handle_unmatched: HandleUnmatched,
) -> BaseModel | None:
    """Try to parse text directly without LLM."""
    # Extract JSON from text
    extracted = extract_json(text)
    if not extracted:
        return None

    # Handle list results
    if isinstance(extracted, list):
        extracted = extracted[0] if extracted else None
    if not extracted or not isinstance(extracted, dict):
        return None

    # Fuzzy validate keys against model fields
    target_keys = list(response_model.model_fields.keys())
    validated = fuzzy_validate_mapping(
        extracted,
        target_keys,
        similarity_threshold=similarity_threshold,
        handle_unmatched=handle_unmatched,
    )

    # Try to create model instance
    return response_model.model_validate(validated)


async def _llm_reformat(
    session: Session,
    branch: Branch,
    text: str,
    response_model: type[BaseModel],
    imodel: Any,
    similarity_threshold: float,
    handle_unmatched: HandleUnmatched,
) -> BaseModel | None:
    """Use LLM to reformat text into the target structure."""
    from .communicate import communicate

    # Build instruction for reformatting
    schema_str = response_model.model_json_schema()
    instruction = (
        "Reformat the following text into the specified JSON structure. "
        "Extract relevant information and map it to the correct fields. "
        "Return ONLY valid JSON matching the schema, no other text."
    )

    context = {
        "text_to_parse": text,
        "target_schema": schema_str,
    }

    # Use communicate to get reformatted response
    params = {
        "instruction": instruction,
        "context": context,
        "imodel": imodel,
        "response_model": response_model,
        "max_retries": 0,  # We handle retries at the outer level
    }

    result = await communicate(session, branch, params)

    # If we got a model instance, return it
    if isinstance(result, BaseModel):
        return result

    # If we got text, try to parse it
    if isinstance(result, str):
        return _try_direct_parse(
            text=result,
            response_model=response_model,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
        )

    return None


def _handle_failure(
    original_text: str,
    handle_validation: HandleValidation,
    error_msg: str,
) -> Any:
    """Handle parsing failure based on handle_validation setting."""
    if handle_validation == "raise":
        raise ValueError(f"Failed to parse text: {error_msg}")
    elif handle_validation == "return_none":
        return None
    else:  # return_value
        return original_text
