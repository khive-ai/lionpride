# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unified response validation module.

Provides strategy-based validation for different response formats
(LNDL, JSON, Operative) with a unified entry point.
"""

from .response import (
    get_validation_strategy,
    validate_response,
)
from .strategies import (
    JSONStrategy,
    LNDLStrategy,
    NoValidationStrategy,
    OperativeStrategy,
    ValidationResult,
    ValidationStrategy,
)

__all__ = (
    # Strategies
    "JSONStrategy",
    "LNDLStrategy",
    "NoValidationStrategy",
    "OperativeStrategy",
    "ValidationResult",
    "ValidationStrategy",
    # Entry points
    "get_validation_strategy",
    "validate_response",
)
