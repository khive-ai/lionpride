# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Validation strategies for different response formats.

Implements the Strategy pattern for response validation, supporting
LNDL, JSON, and Operative validation modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from lionpride.types import Operable

    from ..operate.operative import Operative


@dataclass
class ValidationResult:
    """Result of response validation.

    Attributes:
        success: Whether validation succeeded
        data: Validated data (if success)
        error: Error message (if failure)
        raw_response: Original response text
        metadata: Additional validation metadata
    """

    success: bool
    data: Any = None
    error: str | None = None
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any, raw_response: str = "", **metadata: Any) -> ValidationResult:
        """Create successful validation result."""
        return cls(success=True, data=data, raw_response=raw_response, metadata=metadata)

    @classmethod
    def fail(cls, error: str, raw_response: str = "", **metadata: Any) -> ValidationResult:
        """Create failed validation result."""
        return cls(success=False, error=error, raw_response=raw_response, metadata=metadata)


class ValidationStrategy(ABC):
    """Abstract base for validation strategies."""

    @abstractmethod
    def validate(self, response_text: str, **kwargs: Any) -> ValidationResult:
        """Validate response text.

        Args:
            response_text: Raw LLM response
            **kwargs: Strategy-specific parameters

        Returns:
            ValidationResult with success/failure and data/error
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging/debugging."""
        ...


class LNDLStrategy(ValidationStrategy):
    """Validation strategy for LNDL formatted responses.

    Uses fuzzy parsing to extract structured data from LNDL-formatted
    LLM responses with configurable matching threshold.
    """

    def __init__(self, operable: Operable | Operative, threshold: float = 0.6):
        """Initialize LNDL validation strategy.

        Args:
            operable: Operable or Operative with specs
            threshold: Fuzzy matching threshold (0.0-1.0)
        """
        from ..operate.operative import Operative

        self._operable = operable.operable if isinstance(operable, Operative) else operable
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "lndl"

    def validate(self, response_text: str, **kwargs: Any) -> ValidationResult:
        """Validate LNDL formatted response.

        Args:
            response_text: Raw LLM response with LNDL format
            **kwargs: Unused

        Returns:
            ValidationResult with parsed data or error
        """
        from lionpride.lndl import parse_lndl_fuzzy
        from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

        try:
            lndl_output = parse_lndl_fuzzy(response_text, self._operable, threshold=self._threshold)

            if lndl_output and lndl_output.fields:
                spec_name = next(iter(lndl_output.fields.keys()))
                return ValidationResult.ok(
                    data=lndl_output.fields[spec_name],
                    raw_response=response_text,
                    spec_name=spec_name,
                    all_fields=lndl_output.fields,
                )

            return ValidationResult.fail(
                error="LNDL parsing returned no fields",
                raw_response=response_text,
            )

        except Exception as e:
            # Fallback to JSON validation
            try:
                model = self._operable.create_model()
                validated = PydanticSpecAdapter.validate_response(
                    response_text, model, strict=False, fuzzy_parse=True
                )
                if validated is not None:
                    return ValidationResult.ok(
                        data=validated,
                        raw_response=response_text,
                        fallback="json",
                    )
            except Exception:
                pass

            return ValidationResult.fail(
                error=f"LNDL parsing failed: {e}",
                raw_response=response_text,
            )


class JSONStrategy(ValidationStrategy):
    """Validation strategy for JSON formatted responses.

    Validates JSON responses against a Pydantic model schema with
    optional fuzzy parsing for malformed JSON.
    """

    def __init__(
        self,
        response_model: type[BaseModel],
        strict: bool = False,
        fuzzy_parse: bool = True,
    ):
        """Initialize JSON validation strategy.

        Args:
            response_model: Pydantic model to validate against
            strict: Raise on validation failure vs return error
            fuzzy_parse: Attempt to fix malformed JSON
        """
        self._model = response_model
        self._strict = strict
        self._fuzzy_parse = fuzzy_parse

    @property
    def name(self) -> str:
        return "json"

    def validate(self, response_text: str, **kwargs: Any) -> ValidationResult:
        """Validate JSON formatted response.

        Args:
            response_text: Raw LLM response (JSON or dict)
            **kwargs: Unused

        Returns:
            ValidationResult with validated model or error
        """
        from lionpride.types.spec_adapters.pydantic_field import PydanticSpecAdapter

        try:
            # If response is already a dict, validate directly
            if isinstance(response_text, dict):
                validated = self._model.model_validate(response_text)
                return ValidationResult.ok(
                    data=validated,
                    raw_response=str(response_text),
                )

            # Use PydanticSpecAdapter for string parsing
            validated = PydanticSpecAdapter.validate_response(
                response_text,
                self._model,
                strict=self._strict,
                fuzzy_parse=self._fuzzy_parse,
            )
            if validated is not None:
                return ValidationResult.ok(
                    data=validated,
                    raw_response=response_text,
                )

            return ValidationResult.fail(
                error=f"Response did not match {self._model.__name__} schema",
                raw_response=response_text
                if isinstance(response_text, str)
                else str(response_text),
            )

        except Exception as e:
            return ValidationResult.fail(
                error=str(e),
                raw_response=response_text
                if isinstance(response_text, str)
                else str(response_text),
            )


class OperativeStrategy(ValidationStrategy):
    """Validation strategy for Operative (combined JSON + actions).

    Validates response and extracts action calls if present.
    """

    def __init__(self, operative: Operative, threshold: float = 0.6):
        """Initialize Operative validation strategy.

        Args:
            operative: Operative instance with operable and actions
            threshold: Fuzzy matching threshold
        """
        self._operative = operative
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "operative"

    def validate(self, response_text: str, **kwargs: Any) -> ValidationResult:
        """Validate Operative response with action extraction.

        Args:
            response_text: Raw LLM response
            **kwargs: Unused

        Returns:
            ValidationResult with data and optional action calls
        """
        from lionpride.lndl import has_action_calls, parse_lndl_fuzzy

        try:
            operable = self._operative.operable
            lndl_output = parse_lndl_fuzzy(response_text, operable, threshold=self._threshold)

            if not lndl_output or not lndl_output.fields:
                return ValidationResult.fail(
                    error="Operative parsing returned no fields",
                    raw_response=response_text,
                )

            # Extract action calls if present
            action_calls = []
            if has_action_calls(lndl_output):  # type: ignore[arg-type]
                from lionpride.lndl.types import ActionCall

                for field_data in lndl_output.fields.values():
                    if isinstance(field_data, ActionCall):
                        action_calls.append(field_data)
                    elif isinstance(field_data, dict):
                        for v in field_data.values():
                            if isinstance(v, ActionCall):
                                action_calls.append(v)

            spec_name = next(iter(lndl_output.fields.keys()))
            return ValidationResult.ok(
                data=lndl_output.fields[spec_name],
                raw_response=response_text,
                action_calls=action_calls,
                has_actions=bool(action_calls),
            )

        except Exception as e:
            return ValidationResult.fail(
                error=f"Operative validation failed: {e}",
                raw_response=response_text,
            )


class NoValidationStrategy(ValidationStrategy):
    """Pass-through strategy for unstructured responses."""

    @property
    def name(self) -> str:
        return "none"

    def validate(self, response_text: str, **kwargs: Any) -> ValidationResult:
        """Pass through response without validation."""
        return ValidationResult.ok(
            data=response_text,
            raw_response=response_text,
        )


__all__ = (
    "JSONStrategy",
    "LNDLStrategy",
    "NoValidationStrategy",
    "OperativeStrategy",
    "ValidationResult",
    "ValidationStrategy",
)
