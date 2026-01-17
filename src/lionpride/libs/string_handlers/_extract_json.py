# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

import orjson

from ._fuzzy_json import MAX_JSON_INPUT_SIZE, fuzzy_json

# Precompile the regex for extracting JSON code blocks
_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

# Exceptions that indicate JSON parsing failure (not system errors)
# We intentionally don't catch MemoryError, KeyboardInterrupt, etc.
_JSON_PARSE_ERRORS = (orjson.JSONDecodeError, ValueError, TypeError)


def extract_json(
    input_data: str | list[str],
    /,
    *,
    fuzzy_parse: bool = False,
    return_one_if_single: bool = True,
    max_size: int = MAX_JSON_INPUT_SIZE,
) -> Any | list[Any]:
    """Extract and parse JSON content from a string or markdown code blocks.

    Attempts direct JSON parsing first. If that fails, looks for JSON content
    within markdown code blocks denoted by ```json.

    Security Note:
        Input size is limited to prevent memory exhaustion attacks.

    Args:
        input_data: The input string or list of strings to parse.
        fuzzy_parse: If True, attempts fuzzy JSON parsing on failed attempts.
        return_one_if_single: If True and only one JSON object is found,
            returns a dict instead of a list with one dict.
        max_size: Maximum allowed input size in bytes (default: 10MB)

    Returns:
        Parsed JSON content (dict, list, or empty list if no valid JSON found)

    Raises:
        ValueError: If input exceeds max_size
    """
    # If input_data is a list, join into a single string
    input_str = "\n".join(input_data) if isinstance(input_data, list) else input_data

    # Validate input size to prevent memory exhaustion
    if len(input_str) > max_size:
        msg = (
            f"Input size ({len(input_str)} bytes) exceeds maximum "
            f"({max_size} bytes). This limit prevents memory exhaustion."
        )
        raise ValueError(msg)

    # 1. Try direct parsing
    try:
        parsed = fuzzy_json(input_str) if fuzzy_parse else orjson.loads(input_str)
        return parsed if return_one_if_single else [parsed]
    except _JSON_PARSE_ERRORS:
        pass  # Fall through to markdown extraction

    # 2. Attempt extracting JSON blocks from markdown
    matches = _JSON_BLOCK_PATTERN.findall(input_str)
    if not matches:
        return []

    # If only one match, return single dict; if multiple, return list of dicts
    if return_one_if_single and len(matches) == 1:
        data_str = matches[0]
        try:
            if fuzzy_parse:
                return fuzzy_json(data_str)
            return orjson.loads(data_str)
        except _JSON_PARSE_ERRORS:
            return []

    # Multiple matches
    results: list[Any] = []
    for m in matches:
        try:
            parsed = fuzzy_json(m) if fuzzy_parse else orjson.loads(m)
            # Append valid JSON (dicts, lists, or primitives)
            results.append(parsed)
        except _JSON_PARSE_ERRORS:
            continue  # Skip invalid blocks
    return results
