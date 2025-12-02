# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lionpride.ln._fuzzy_validate import fuzzy_validate_pydantic

from .base import Rule, RuleParams, RuleQualifier

__all__ = ("BaseModelRule",)


def _get_basemodel_params() -> RuleParams:
    """Default params for BaseModel rule."""
    return RuleParams(
        apply_types={BaseModel},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class BaseModelRule(Rule):
    """Rule for validating Pydantic BaseModel subclasses.

    Uses fuzzy_validate_pydantic for flexible parsing:
    - Pass-through if already correct type
    - Dict → model validation
    - String → JSON extraction → model validation
    """

    def __init__(
        self,
        fuzzy_parse: bool = True,
        fuzzy_match: bool = False,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize BaseModel rule.

        Args:
            fuzzy_parse: Enable fuzzy JSON extraction from text
            fuzzy_match: Enable fuzzy key matching for field names
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_basemodel_params()
        super().__init__(params, **kw)
        self.fuzzy_parse = fuzzy_parse
        self.fuzzy_match = fuzzy_match

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate value as a Pydantic model.

        Args:
            v: Value to validate
            t: Expected BaseModel subclass

        Raises:
            ValueError: If value cannot be validated as the model type
        """
        if not isinstance(t, type) or not issubclass(t, BaseModel):
            raise ValueError(f"expected_type must be a BaseModel subclass, got {t}")

        # Already correct type - valid
        if isinstance(v, t):
            return

        # Try dict validation
        if isinstance(v, dict):
            try:
                t.model_validate(v)
                return
            except Exception as e:
                raise ValueError(f"Dict validation failed: {e}") from e

        # Can't validate as-is
        raise ValueError(f"Cannot validate {type(v).__name__} as {t.__name__}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert value to model using fuzzy validation.

        Args:
            v: Value to fix
            t: Expected BaseModel subclass

        Returns:
            Validated model instance

        Raises:
            ValueError: If conversion fails
        """
        return fuzzy_validate_pydantic(
            v,
            model_type=t,
            fuzzy_parse=self.fuzzy_parse,
            fuzzy_match=self.fuzzy_match,
        )
