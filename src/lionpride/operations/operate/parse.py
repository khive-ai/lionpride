# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lionpride.errors import (
    ConfigurationError,
    ExecutionError,
    LionprideError,
    ValidationError,
)
from lionpride.libs.string_handlers import extract_json
from lionpride.ln import fuzzy_validate_mapping
from lionpride.session.messages import InstructionContent, Message

from .types import CustomParser, GenerateParams, HandleUnmatched, ParseParams

if TYPE_CHECKING:
    from lionpride.session import Branch, Session

__all__ = ("parse",)


async def parse(
    session: Session,
    branch: Branch,
    params: ParseParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
):
    if params._is_sentinel(params.text):
        raise ValidationError("No text provided for parsing")

    try:
        return _direct_parse(
            text=params.text,
            target_keys=params.target_keys,
            similarity_threshold=params.similarity_threshold,
            handle_unmatched=params.handle_unmatched,
            structure_format=params.structure_format,
            custom_parser=params.custom_parser,
            **params.match_kwargs,
        )
    except LionprideError as e:
        if e.retryable is False:
            raise
    except Exception:
        pass  # Direct parse failed, will try LLM reparse

    if params.max_retries < 1:
        raise ConfigurationError(
            "Direct parse failed and 'max_retries' is not enabled, no reparse attempted"
        )

    if params.max_retries > 5:
        raise ValidationError("'max_retries' for parse cannot exceed 5 to avoid long delays")

    for _ in range(params.max_retries):
        try:
            return await _llm_reparse(
                session=session,
                branch=branch,
                params=params,
                poll_timeout=poll_timeout,
                poll_interval=poll_interval,
            )
        except LionprideError as e:
            if e.retryable is False:
                raise

    raise ExecutionError(
        "All parse attempts (direct and LLM reparse) failed",
        retryable=True,
    )


def _direct_parse(
    text: str,
    target_keys: list[str],
    similarity_threshold: float,
    handle_unmatched: HandleUnmatched,
    structure_format: str,
    custom_parser: CustomParser | None = None,
    **kwargs,
) -> dict[str, Any]:
    if not target_keys:
        raise ValidationError("No target_keys provided for parse operation")

    # Custom parser path
    if structure_format == "custom":
        if custom_parser is None:
            raise ConfigurationError(
                "structure_format='custom' requires a custom_parser to be provided"
            )
        try:
            return custom_parser(text, target_keys, **kwargs)
        except Exception as e:
            raise ExecutionError(
                "Custom parser failed to extract data from text",
                retryable=True,
                cause=e,
            )

    # JSON parser path
    if structure_format != "json":
        raise ValidationError(f"Unsupported structure_format '{structure_format}' in parse")

    extracted = None
    try:
        extracted = extract_json(text, fuzzy_parse=True, return_one_if_single=False)
    except Exception as e:
        raise ExecutionError(
            "Failed to extract JSON from text during parse",
            retryable=True,
            cause=e,
        )

    if not extracted:
        raise ExecutionError(
            "No JSON object could be extracted from text during parse",
            retryable=True,
        )

    try:
        return fuzzy_validate_mapping(
            extracted[0],
            target_keys,
            similarity_threshold=similarity_threshold,
            handle_unmatched=handle_unmatched,
            **kwargs,
        )
    except Exception as e:
        raise ExecutionError(
            "Failed to validate extracted JSON during parse",
            retryable=True,
            cause=e,
        )


async def _llm_reparse(
    session: Session,
    branch: Branch,
    params: ParseParams,
    poll_timeout: float | None = None,
    poll_interval: float | None = None,
) -> dict[str, Any] | None:
    """Use LLM to reformat text into valid JSON."""
    from .generate import generate

    instruction_text = (
        "Extract and reformat the following text into valid JSON. "
        "Return ONLY the JSON object, no other text or markdown formatting."
        f"\n\nExpected fields: {', '.join(params.target_keys)}"
        f"\n\nText to parse:\n{params.text}"
    )

    instruction = Message(
        content=InstructionContent.create(instruction=instruction_text),
        sender=session.id,
        recipient=branch.id,
    )

    gen_params = GenerateParams(
        instruction=instruction,
        imodel=params.imodel,
        return_as="text",
        imodel_kwargs=params.imodel_kwargs,
    )

    result = await generate(session, branch, gen_params, poll_timeout, poll_interval)
    return _direct_parse(
        result,
        params.target_keys,
        params.similarity_threshold,
        params.handle_unmatched,
        params.structure_format,
        params.custom_parser,
        **params.match_kwargs,
    )
