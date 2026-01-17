# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import orjson

# Security limit: Maximum input string size for JSON parsing.
# Large inputs can cause memory exhaustion during parsing and regex operations.
# Default: 10MB - generous for most use cases while preventing DoS.
MAX_JSON_INPUT_SIZE = 10 * 1024 * 1024  # 10 MB


def fuzzy_json(
    str_to_parse: str,
    /,
    *,
    max_size: int = MAX_JSON_INPUT_SIZE,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse JSON string with fuzzy error correction (quotes, spacing, brackets).

    Security Note:
        Input size is limited to prevent memory exhaustion attacks.
        Uses orjson which has built-in recursion depth limits (CVE-2024-27454 fixed).

    Limitation:
        This is designed for "nearly-valid" JSON from LLMs (single quotes, trailing
        commas, unquoted keys). It is NOT a security boundary for adversarial input.

    Args:
        str_to_parse: JSON string to parse
        max_size: Maximum allowed input size in bytes (default: 10MB)

    Returns:
        Either a dict or a list of dicts. Will NOT return primitive types
        (int, float, str, bool, None) or lists of primitives.

    Raises:
        TypeError: If input is not a string or parsed JSON is a primitive type.
        ValueError: If input is empty, exceeds size limit, or parsing fails.
    """
    _check_valid_str(str_to_parse, max_size=max_size)

    # 1. Direct attempt (valid JSON)
    try:
        result = orjson.loads(str_to_parse)
        return _validate_return_type(result)
    except orjson.JSONDecodeError:
        pass

    # 2. Try state-machine cleaning (handles quotes inside strings correctly)
    cleaned = _clean_json_string_safe(str_to_parse)
    try:
        result = orjson.loads(cleaned)
        return _validate_return_type(result)
    except orjson.JSONDecodeError:
        pass

    # 3. Try fixing brackets
    fixed = fix_json_string(cleaned)
    try:
        result = orjson.loads(fixed)
        return _validate_return_type(result)
    except orjson.JSONDecodeError:
        pass

    # If all attempts fail
    raise ValueError("Invalid JSON string")


def _check_valid_str(
    str_to_parse: str,
    /,
    *,
    max_size: int = MAX_JSON_INPUT_SIZE,
) -> None:
    """Validate input string for JSON parsing.

    Args:
        str_to_parse: Input string to validate
        max_size: Maximum allowed size in bytes

    Raises:
        TypeError: If input is not a string
        ValueError: If input is empty or exceeds size limit
    """
    if not isinstance(str_to_parse, str):
        raise TypeError("Input must be a string")
    if not str_to_parse.strip():
        raise ValueError("Input string is empty")
    if len(str_to_parse) > max_size:
        msg = (
            f"Input size ({len(str_to_parse)} bytes) exceeds maximum "
            f"({max_size} bytes). This limit prevents memory exhaustion."
        )
        raise ValueError(msg)


def _validate_return_type(result: Any) -> dict[str, Any] | list[dict[str, Any]]:
    """Validate that parsed JSON matches the declared return type.

    Args:
        result: The result from orjson.loads()

    Returns:
        The validated result (dict or list[dict])

    Raises:
        TypeError: If result is a primitive type or list of non-dict elements
    """
    # Check if it's a dict
    if isinstance(result, dict):
        return result

    # Check if it's a list
    if isinstance(result, list):
        # Ensure all elements are dicts
        if not result:
            # Empty list is valid as list[dict] (vacuous truth)
            return result

        for i, item in enumerate(result):
            if not isinstance(item, dict):
                raise TypeError(
                    f"fuzzy_json returns dict or list[dict], got list with "
                    f"non-dict element at index {i}: {type(item).__name__}"
                )
        return result

    # If we got here, it's a primitive type (int, float, str, bool, None)
    raise TypeError(
        f"fuzzy_json returns dict or list[dict], got primitive type: {type(result).__name__}"
    )


def _clean_json_string_safe(s: str) -> str:
    """Clean JSON string using a state machine to preserve content inside strings.

    This handles the common LLM output issues:
    - Single-quoted strings -> double-quoted (with proper escaping)
    - Trailing commas before ] or }
    - Unquoted keys

    Unlike regex-based approaches, this does NOT corrupt content inside strings
    (e.g., apostrophes like "it's" are preserved).
    """
    result: list[str] = []
    pos = 0
    length = len(s)

    while pos < length:
        char = s[pos]

        # Handle single-quoted strings: convert to double-quoted
        if char == "'":
            result.append('"')  # Start with double quote
            pos += 1
            # Read until closing single quote, escaping inner double quotes
            while pos < length:
                inner_char = s[pos]
                if inner_char == "\\":
                    # Escape sequence
                    if pos + 1 < length:
                        next_char = s[pos + 1]
                        if next_char == "'":
                            # \' in single-quoted string = just an apostrophe
                            result.append("'")
                            pos += 2
                            continue
                        # Other escapes - copy both chars
                        result.append(inner_char)
                        result.append(next_char)
                        pos += 2
                        continue
                    # Trailing backslash
                    result.append(inner_char)
                    pos += 1
                    continue
                if inner_char == "'":
                    # End of single-quoted string
                    result.append('"')
                    pos += 1
                    break
                if inner_char == '"':
                    # Need to escape double quotes inside
                    result.append('\\"')
                    pos += 1
                    continue
                result.append(inner_char)
                pos += 1
            continue

        # Handle double-quoted strings: copy as-is, just track position
        if char == '"':
            result.append(char)
            pos += 1
            while pos < length:
                inner_char = s[pos]
                if inner_char == "\\":
                    # Escape sequence - copy both chars
                    result.append(inner_char)
                    if pos + 1 < length:
                        pos += 1
                        result.append(s[pos])
                    pos += 1
                    continue
                result.append(inner_char)
                if inner_char == '"':
                    pos += 1
                    break
                pos += 1
            continue

        # Handle { and , - check for unquoted keys and trailing commas
        if char in "{,":
            # For comma, first check if it's a trailing comma
            if char == ",":
                lookahead = pos + 1
                while lookahead < length and s[lookahead] in " \t\n\r":
                    lookahead += 1
                if lookahead < length and s[lookahead] in "]}":
                    # Skip the trailing comma
                    pos += 1
                    continue

            # Append the { or , and check for unquoted key
            result.append(char)
            pos += 1

            # Skip whitespace
            while pos < length and s[pos] in " \t\n\r":
                result.append(s[pos])
                pos += 1

            # Check if we have an unquoted key (alphanumeric/underscore followed by :)
            if pos < length and s[pos] not in "\"'{[":
                key_start = pos
                while pos < length and (s[pos].isalnum() or s[pos] == "_"):
                    pos += 1
                if pos < length and key_start < pos:
                    # Skip whitespace before colon
                    key_end = pos
                    while pos < length and s[pos] in " \t\n\r":
                        pos += 1
                    if pos < length and s[pos] == ":":
                        # Found unquoted key - quote it
                        key = s[key_start:key_end]
                        result.append(f'"{key}"')
                        continue
                    else:
                        # Not a key, restore position
                        pos = key_start
            continue

        # Default: copy character as-is
        result.append(char)
        pos += 1

    return "".join(result).strip()


def _clean_json_string(s: str) -> str:
    """Legacy function - use _clean_json_string_safe instead."""
    return _clean_json_string_safe(s)


def fix_json_string(str_to_parse: str, /) -> str:
    """Fix JSON string by balancing unmatched brackets."""
    if not str_to_parse:
        raise ValueError("Input string is empty")

    brackets = {"{": "}", "[": "]"}
    open_brackets = []
    pos = 0
    length = len(str_to_parse)

    while pos < length:
        char = str_to_parse[pos]

        if char == "\\":
            pos += 2  # Skip escaped chars
            continue

        if char == '"':
            pos += 1
            # skip string content
            while pos < length:
                if str_to_parse[pos] == "\\":
                    pos += 2
                    continue
                if str_to_parse[pos] == '"':
                    pos += 1
                    break
                pos += 1
            continue

        if char in brackets:
            open_brackets.append(brackets[char])
        elif char in brackets.values():
            if not open_brackets:
                # Extra closing bracket
                # Better to raise error than guess
                raise ValueError("Extra closing bracket found.")
            if open_brackets[-1] != char:
                # Mismatched bracket
                raise ValueError("Mismatched brackets.")
            open_brackets.pop()

        pos += 1

    # Add missing closing brackets if any
    if open_brackets:
        str_to_parse += "".join(reversed(open_brackets))

    return str_to_parse
