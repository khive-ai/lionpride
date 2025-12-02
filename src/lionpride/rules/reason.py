# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .base import Rule, RuleParams, RuleQualifier
from .models import Reason

__all__ = ("ReasonRule",)


def _get_reason_params() -> RuleParams:
    """Default params for reason rule."""
    return RuleParams(
        apply_types={Reason},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class ReasonRule(Rule):
    """Rule for validating and normalizing Reason instances.

    Converts various input formats to canonical Reason model:
    - Reason instance → pass through
    - Dict with reasoning/confidence → Reason instance
    - String → Reason(reasoning=string)

    Usage:
        # Via Spec (auto-assigned from type)
        spec = Spec(Reason, name="reason")

        # Direct usage
        rule = ReasonRule()
        result = await rule.invoke("reason", {"reasoning": "..."}, Reason)
    """

    def __init__(self, params: RuleParams | None = None, **kw):
        """Initialize reason rule.

        Args:
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs
        """
        if params is None:
            params = _get_reason_params()
        super().__init__(params, **kw)

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a Reason instance.

        Only accepts Reason instances. Dict/string inputs trigger perform_fix.

        Raises:
            ValueError: If not a Reason instance
        """
        if not isinstance(v, Reason):
            raise ValueError(
                f"Expected Reason instance, got {type(v).__name__}. "
                "Use perform_fix to convert dict/string to Reason."
            )

        # Validate confidence range if provided
        if v.confidence is not None and not (0.0 <= v.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v.confidence}")

    async def perform_fix(self, v: Any, t: type) -> Reason:
        """Convert various formats to Reason instance.

        Handles:
        - Reason instance: pass through
        - Dict: extract reasoning/confidence fields
        - String: use as reasoning text

        Returns:
            Reason instance

        Raises:
            ValueError: If conversion fails
        """
        # Already a Reason instance
        if isinstance(v, Reason):
            return v

        # String → Reason(reasoning=string)
        if isinstance(v, str):
            return Reason(reasoning=v)

        # Dict → extract fields
        if isinstance(v, dict):
            reasoning = v.get("reasoning", "")
            confidence = v.get("confidence")

            # Convert reasoning to string if needed
            if not isinstance(reasoning, str):
                reasoning = str(reasoning) if reasoning else ""

            # Validate confidence if provided
            if confidence is not None:
                try:
                    confidence = float(confidence)
                    if not (0.0 <= confidence <= 1.0):
                        confidence = None  # Invalid, ignore
                except (ValueError, TypeError):
                    confidence = None

            return Reason(reasoning=reasoning, confidence=confidence)

        # Fallback: try to convert to string
        try:
            return Reason(reasoning=str(v))
        except Exception as e:
            raise ValueError(f"Cannot convert {type(v).__name__} to Reason: {e}") from e
