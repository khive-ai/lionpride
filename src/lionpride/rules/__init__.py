# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Rule-based validation system for IPU (Intelligence Processing Unit).

Core of the validation pipeline:
    Spec.base_type → auto Rule assignment → validate spec-by-spec → Operable.create_model()

Features:
- RuleRegistry: Maps types → Rules (auto-assignment from Spec.base_type)
- Built-in rules: StringRule, NumberRule, BooleanRule, MappingRule, etc.
- Validator: Orchestrates rules over Operable specs
- Auto-fix: Convert invalid values to valid ones

Usage:
    from lionpride.rules import Validator
    from lionpride.types import Spec, Operable

    # Define specs
    confidence_spec = Spec(float, name="confidence")
    output_spec = Spec(str, name="output")
    operable = Operable([confidence_spec, output_spec])

    # Validate (auto Rule assignment from base_type)
    validator = Validator()
    validated = await validator.validate_operable(
        data={"confidence": "0.95", "output": 42},  # Raw LLM response
        operable=operable,
        auto_fix=True
    )
    # → {"confidence": 0.95, "output": "42"}

    # Create output model
    OutputModel = operable.create_model()
    result = OutputModel.model_validate(validated)
"""

from .action_request import ActionRequestRule
from .base import Rule, RuleParams, RuleQualifier, ValidationError
from .boolean import BooleanRule
from .choice import ChoiceRule
from .mapping import MappingRule
from .models import ActionRequest, ActionResponse, Reason
from .number import NumberRule
from .reason import ReasonRule
from .registry import RuleRegistry, get_default_registry
from .string import StringRule
from .validator import Validator

__all__ = (
    "ActionRequest",
    "ActionRequestRule",
    "ActionResponse",
    "BooleanRule",
    "ChoiceRule",
    "MappingRule",
    "NumberRule",
    "Reason",
    "ReasonRule",
    "Rule",
    "RuleParams",
    "RuleQualifier",
    "RuleRegistry",
    "StringRule",
    "ValidationError",
    "Validator",
    "get_default_registry",
)
