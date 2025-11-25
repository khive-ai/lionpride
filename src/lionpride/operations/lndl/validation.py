# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL response validation with fuzzy matching.

Functions for validating LLM responses against LNDL specs using
fuzzy parsing with configurable threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lionpride.types import Operable

    from ..operate.operative import Operative


def validate_lndl_response(
    response_text: str,
    operable: Operable | Operative,
    threshold: float = 0.6,
) -> tuple[Any, str | None]:
    """Validate LNDL response with fuzzy matching.

    Parses the response text using LNDL fuzzy parser and validates
    against the operable's specs. Falls back to JSON validation
    if LNDL parsing fails.

    Args:
        response_text: Raw LLM response text
        operable: Operable or Operative with specs
        threshold: Fuzzy matching threshold (0.0-1.0)

    Returns:
        Tuple of (validated_result, error_message).
        If validation succeeds, error_message is None.
        If validation fails, validated_result is None.
    """
    from lionpride.lndl import parse_lndl_fuzzy
    from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

    from ..operate.operative import Operative

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


def extract_lndl_fields(
    response_text: str,
    operable: Operable | Operative,
    threshold: float = 0.6,
) -> dict[str, Any]:
    """Extract all LNDL fields from response.

    Unlike validate_lndl_response which returns the first field,
    this returns all parsed fields as a dictionary.

    Args:
        response_text: Raw LLM response text
        operable: Operable or Operative with specs
        threshold: Fuzzy matching threshold

    Returns:
        Dictionary of spec_name -> validated_value
    """
    from lionpride.lndl import parse_lndl_fuzzy

    from ..operate.operative import Operative

    actual_operable = operable.operable if isinstance(operable, Operative) else operable

    try:
        lndl_output = parse_lndl_fuzzy(response_text, actual_operable, threshold=threshold)
        if lndl_output and lndl_output.fields:
            return lndl_output.fields
    except Exception:
        pass

    return {}


__all__ = (
    "extract_lndl_fields",
    "validate_lndl_response",
)
