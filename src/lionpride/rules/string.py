# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

from .base import Rule, RuleParams, RuleQualifier

__all__ = ("StringRule",)


def _get_string_params() -> RuleParams:
    """Default params for string rule."""
    return RuleParams(
        apply_types={str},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class StringRule(Rule):
    """Rule for validating and converting string values.

    Features:
    - Type checking (must be str)
    - Length constraints (min_length, max_length)
    - Pattern matching (regex)
    - Auto-conversion from any type to string

    Usage:
        rule = StringRule(min_length=1, max_length=100, pattern=r'^[A-Za-z]+$')
        result = await rule.invoke("name", "Ocean", str)
    """

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize string rule.

        Args:
            min_length: Minimum string length (inclusive)
            max_length: Maximum string length (inclusive)
            pattern: Regex pattern to match. Note: Complex patterns on untrusted
                input could cause ReDoS. Use simple patterns or validate separately.
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_string_params()
        super().__init__(params, **kw)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a string with correct length and pattern.

        Raises:
            ValueError: If not a string or constraints violated
        """
        if not isinstance(v, str):
            raise ValueError(f"Invalid string value: expected str, got {type(v).__name__}")

        # Check minimum length
        if self.min_length is not None and len(v) < self.min_length:
            raise ValueError(
                f"String too short: got {len(v)} characters, minimum {self.min_length}"
            )

        # Check maximum length
        if self.max_length is not None and len(v) > self.max_length:
            raise ValueError(f"String too long: got {len(v)} characters, maximum {self.max_length}")

        # Check pattern matching
        if self.pattern is not None and not re.match(self.pattern, v):
            raise ValueError(f"String does not match required pattern: {self.pattern}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert value to string and re-validate.

        Returns:
            String representation of value (validated)

        Raises:
            ValueError: If conversion or re-validation fails
        """
        try:
            fixed = str(v)
        except Exception as e:
            raise ValueError(f"Failed to convert {v} to string") from e

        # Re-validate the fixed value
        await self.validate(fixed, t)
        return fixed
