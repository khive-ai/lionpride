# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from typing import Any

# Security limit: Maximum string length for number conversion.
# Prevents potential DoS from extremely long numeric strings.
MAX_NUMBER_STRING_LENGTH = 1000


def to_num(
    input_: Any,
    /,
    *,
    upper_bound: int | float | None = None,
    lower_bound: int | float | None = None,
    num_type: type[int] | type[float] = float,
    precision: int | None = None,
) -> int | float:
    """Convert input to numeric type with optional bounds checking.

    Args:
        input_: Value to convert (bool, int, float, Decimal, or str)
        upper_bound: Maximum allowed value (inclusive)
        lower_bound: Minimum allowed value (inclusive)
        num_type: Target type (int or float, default: float)
        precision: Decimal places for float rounding

    Returns:
        Converted numeric value

    Raises:
        ValueError: If string is empty, too long, or out of bounds
        TypeError: If input type is not convertible
    """
    # Validate num_type
    if num_type not in (int, float):
        raise ValueError(f"Invalid number type: {num_type}")

    # Handle boolean (special case - must check before int)
    if isinstance(input_, (bool, int, float, Decimal)):
        value = float(input_)
    # Handle string input
    elif isinstance(input_, str):
        input_ = input_.strip()
        if not input_:
            raise ValueError("Empty string cannot be converted to number")
        # Security: limit string length to prevent DoS
        if len(input_) > MAX_NUMBER_STRING_LENGTH:
            msg = f"String length ({len(input_)}) exceeds maximum ({MAX_NUMBER_STRING_LENGTH})"
            raise ValueError(msg)
        try:
            value = float(input_)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{input_}' to number") from e
    else:
        raise TypeError(f"Cannot convert {type(input_).__name__} to number")

    # Apply bounds checking
    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"Value {value} exceeds upper bound {upper_bound}")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"Value {value} below lower bound {lower_bound}")

    # Apply precision for float
    if precision is not None and num_type is float:
        value = round(value, precision)

    # Convert to target type
    return num_type(value)
