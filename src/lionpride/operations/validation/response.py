# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unified response validation entry point.

Provides a single function to validate responses, automatically
selecting the appropriate strategy based on parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .strategies import (
    JSONStrategy,
    LNDLStrategy,
    NoValidationStrategy,
    OperativeStrategy,
    ValidationResult,
    ValidationStrategy,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from lionpride.types import Operable

    from ..operate.operative import Operative


def validate_response(
    response_text: str,
    *,
    operable: Operable | Operative | None = None,
    response_model: type[BaseModel] | None = None,
    threshold: float = 0.6,
    strict: bool = False,
    fuzzy_parse: bool = True,
) -> ValidationResult:
    """Validate response using appropriate strategy.

    Automatically selects validation strategy based on parameters:
    - If operable is Operative: uses OperativeStrategy
    - If operable is Operable: uses LNDLStrategy
    - If response_model: uses JSONStrategy
    - Otherwise: uses NoValidationStrategy (pass-through)

    Args:
        response_text: Raw LLM response to validate
        operable: Operable or Operative for LNDL validation
        response_model: Pydantic model for JSON validation
        threshold: Fuzzy matching threshold for LNDL
        strict: Strict validation mode for JSON
        fuzzy_parse: Enable fuzzy JSON parsing

    Returns:
        ValidationResult with validation outcome
    """
    strategy = get_validation_strategy(
        operable=operable,
        response_model=response_model,
        threshold=threshold,
        strict=strict,
        fuzzy_parse=fuzzy_parse,
    )
    return strategy.validate(response_text)


def get_validation_strategy(
    *,
    operable: Operable | Operative | None = None,
    response_model: type[BaseModel] | None = None,
    threshold: float = 0.6,
    strict: bool = False,
    fuzzy_parse: bool = True,
) -> ValidationStrategy:
    """Get appropriate validation strategy based on parameters.

    Args:
        operable: Operable or Operative for LNDL validation
        response_model: Pydantic model for JSON validation
        threshold: Fuzzy matching threshold for LNDL
        strict: Strict validation mode for JSON
        fuzzy_parse: Enable fuzzy JSON parsing

    Returns:
        Appropriate ValidationStrategy instance
    """
    from ..operate.operative import Operative

    if operable is not None:
        if isinstance(operable, Operative):
            return OperativeStrategy(operable, threshold=threshold)
        else:
            return LNDLStrategy(operable, threshold=threshold)

    if response_model is not None:
        return JSONStrategy(response_model, strict=strict, fuzzy_parse=fuzzy_parse)

    return NoValidationStrategy()


__all__ = (
    "get_validation_strategy",
    "validate_response",
)
