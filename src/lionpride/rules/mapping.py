# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

import orjson

from .base import Rule, RuleParams, RuleQualifier

__all__ = ("MappingRule",)


def _get_mapping_params() -> RuleParams:
    """Default params for mapping rule."""
    return RuleParams(
        apply_types={dict},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class MappingRule(Rule):
    """Rule for validating and converting mapping/dict values.

    Features:
    - Type checking (must be dict/Mapping)
    - Required keys validation
    - Optional keys validation
    - Auto-conversion from JSON string
    - Fuzzy key matching (optional, case-insensitive)

    Usage:
        rule = MappingRule(
            required_keys={"name", "value"},
            optional_keys={"description"},
            fuzzy_keys=True
        )
        result = await rule.invoke("config", '{"Name": "test"}', dict)
    """

    def __init__(
        self,
        required_keys: set[str] | None = None,
        optional_keys: set[str] | None = None,
        fuzzy_keys: bool = False,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize mapping rule.

        Args:
            required_keys: Keys that must be present
            optional_keys: Keys that may be present (for validation of known keys)
            fuzzy_keys: Enable case-insensitive key matching
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_mapping_params()
        super().__init__(params, **kw)
        self.required_keys = required_keys or set()
        self.optional_keys = optional_keys or set()
        self.fuzzy_keys = fuzzy_keys

        # Build lowercase key map for fuzzy matching
        if fuzzy_keys:
            all_keys = self.required_keys | self.optional_keys
            self._key_map = {k.lower(): k for k in all_keys}

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a mapping with required keys (exact match).

        For fuzzy_keys mode, validation uses exact key matching so that
        non-canonical keys trigger perform_fix() for normalization.

        Raises:
            ValueError: If not a mapping or missing required keys
        """
        if not isinstance(v, Mapping):
            raise ValueError(
                f"Invalid mapping value: expected dict/Mapping, got {type(v).__name__}"
            )

        # Check required keys (always exact match in validate)
        if self.required_keys:
            missing = self.required_keys - set(v.keys())
            if missing:
                raise ValueError(f"Missing required keys: {sorted(missing)}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert value to mapping and normalize keys.

        Returns:
            Dict with normalized keys (if fuzzy_keys enabled)

        Raises:
            ValueError: If conversion fails
        """
        # Try JSON parsing for strings
        if isinstance(v, str):
            try:
                v = orjson.loads(v)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON string: {e}") from e

        if not isinstance(v, Mapping):
            raise ValueError(f"Cannot convert {type(v).__name__} to mapping")

        # Normalize keys if fuzzy matching enabled
        if self.fuzzy_keys and self._key_map:
            fixed = {}
            for k, val in v.items():
                k_lower = k.lower() if isinstance(k, str) else k
                if k_lower in self._key_map:
                    # Use canonical key name
                    fixed[self._key_map[k_lower]] = val
                else:
                    # Keep original key
                    fixed[k] = val
            v = fixed

        # Convert to dict if not already
        result = dict(v) if not isinstance(v, dict) else v

        # Re-validate the fixed value
        await self.validate(result, t)
        return result
