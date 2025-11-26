# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Parse operation - JSON extraction with LLM fallback.

Extracts JSON from raw LLM response text. Falls back to LLM reformatting
if direct extraction fails. Returns dict - validation happens in Validator.

Flow:
    raw_text → parse() → dict → Validator.validate() → typed model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from lionpride.libs.string_handlers import extract_json
from lionpride.ln import fuzzy_validate_mapping
from lionpride.services.types import iModel
from lionpride.types.base import ModelConfig, Params

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("ParseParams", "parse")

# Type alias for handle_unmatched parameter
HandleUnmatched = Literal["ignore", "raise", "remove", "fill", "force"]


@dataclass(init=False, frozen=True, slots=True)
class ParseParams(Params):
    """Parameters for parse operation.

    Parse extracts JSON from raw text. If extraction fails and imodel
    is provided, uses LLM to reformat the text into valid JSON.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    text: str = None
    """Raw text to parse (e.g., LLM response)"""

    target_keys: list[str] = field(default_factory=list)
    """Expected keys for fuzzy matching (optional)"""

    imodel: iModel | str = None
    """Model for LLM reparse fallback (optional)"""

    similarity_threshold: float = 0.85
    """Fuzzy match threshold for key mapping"""

    handle_unmatched: HandleUnmatched = "force"
    """How to handle unmatched keys during fuzzy matching"""

    max_retries: int = 3
    """Retry attempts for LLM reparse"""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel"""


async def parse(
    session: Session,
    branch: Branch,
    params: ParseParams,
) -> dict[str, Any] | None:
    """Parse raw text into JSON dict.

    Tries direct JSON extraction first. Falls back to LLM reformatting
    if direct extraction fails and imodel is provided.

    Security: Branch must have access to imodel if using LLM reparse.

    Args:
        session: Session for service access
        branch: Branch for resource access control
        params: Parse parameters

    Returns:
        Extracted dict, or None if extraction fails

    Raises:
        PermissionError: If branch doesn't have access to imodel
    """
    text = params.text
    if not text:
        return None

    # 1. Try direct JSON extraction
    extracted = _try_direct_extract(
        text=text,
        target_keys=params.target_keys,
        similarity_threshold=params.similarity_threshold,
        handle_unmatched=params.handle_unmatched,
    )
    if extracted is not None:
        return extracted

    # 2. LLM fallback - check resource access first
    if params.imodel is None:
        return None  # No fallback available

    model_name = params.imodel.name if isinstance(params.imodel, iModel) else params.imodel
    if model_name not in branch.resources:
        raise PermissionError(
            f"Branch '{branch.name}' cannot access model '{model_name}' for reparse. "
            f"Allowed: {branch.resources or 'none'}"
        )

    # 3. Try LLM reparse with retries
    for _attempt in range(params.max_retries):
        try:
            result = await _llm_reparse(
                session=session,
                branch=branch,
                text=text,
                target_keys=params.target_keys,
                model_name=model_name,
                similarity_threshold=params.similarity_threshold,
                handle_unmatched=params.handle_unmatched,
                imodel_kwargs=params.imodel_kwargs,
            )
            if result is not None:
                return result
        except Exception:
            continue

    return None


def _try_direct_extract(
    text: str,
    target_keys: list[str],
    similarity_threshold: float,
    handle_unmatched: HandleUnmatched,
) -> dict[str, Any] | None:
    """Try to extract JSON directly from text."""
    extracted = extract_json(text)
    if not extracted:
        return None

    # Handle list results - take first dict
    if isinstance(extracted, list):
        extracted = extracted[0] if extracted else None
    if not extracted or not isinstance(extracted, dict):
        return None

    # Fuzzy validate keys if target_keys provided
    if target_keys:
        extracted = fuzzy_validate_mapping(
            extracted,
            target_keys,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
        )

    return extracted


async def _llm_reparse(
    session: Session,
    branch: Branch,
    text: str,
    target_keys: list[str],
    model_name: str,
    similarity_threshold: float,
    handle_unmatched: HandleUnmatched,
    imodel_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Use LLM to reformat text into valid JSON."""
    from lionpride.session.messages import InstructionContent, Message

    from .generate import GenerateParams, generate

    # Build instruction for reformatting
    instruction_text = (
        "Extract and reformat the following text into valid JSON. "
        "Return ONLY the JSON object, no other text or markdown formatting."
    )
    if target_keys:
        instruction_text += f"\n\nExpected fields: {', '.join(target_keys)}"

    instruction_text += f"\n\nText to parse:\n{text}"

    instruction = Message(
        content=InstructionContent(instruction=instruction_text),
        sender=branch.user,
        recipient=session.id,
    )

    # Generate reformatted response
    gen_params = GenerateParams(
        imodel=model_name,
        instruction=instruction,
        return_as="text",
        imodel_kwargs=imodel_kwargs,
    )

    result = await generate(session, branch, gen_params)

    # Try to extract JSON from LLM response
    if isinstance(result, str):
        return _try_direct_extract(
            text=result,
            target_keys=target_keys,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
        )

    return None
