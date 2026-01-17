# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import orjson
import yaml  # type: ignore[import-untyped]

__all__ = ("minimal_yaml",)

# Maximum recursion depth for pruning to prevent stack overflow
MAX_PRUNE_DEPTH = 100


class MinimalDumper(yaml.SafeDumper):
    """YAML dumper with minimal, readable settings."""

    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        """Disable anchors/aliases (&id001, *id001) for repeated objects."""
        return True


def _represent_str(dumper: yaml.SafeDumper, data: str):
    """Use block scalars for multiline text; plain style otherwise."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


MinimalDumper.add_representer(str, _represent_str)


def _is_empty(x: Any) -> bool:
    """Define 'empty' for pruning. Keeps 0 and False."""
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip() == ""
    if isinstance(x, dict):
        return len(x) == 0
    if isinstance(x, list | tuple | set):
        return len(x) == 0
    return False


def _prune(x: Any, *, _depth: int = 0, _max_depth: int = MAX_PRUNE_DEPTH) -> Any:
    """Recursively remove empty leaves and empty containers.

    Args:
        x: Value to prune
        _depth: Current recursion depth (internal use)
        _max_depth: Maximum recursion depth (default: 100)

    Returns:
        Pruned value with empty containers removed

    Raises:
        RecursionError: If depth exceeds _max_depth
    """
    if _depth > _max_depth:
        msg = f"Pruning depth exceeds maximum ({_max_depth})"
        raise RecursionError(msg)

    if isinstance(x, dict):
        pruned = {
            k: _prune(v, _depth=_depth + 1, _max_depth=_max_depth)
            for k, v in x.items()
            if not _is_empty(v)
        }
        return {k: v for k, v in pruned.items() if not _is_empty(v)}
    if isinstance(x, list):
        pruned_list = [
            _prune(v, _depth=_depth + 1, _max_depth=_max_depth) for v in x if not _is_empty(v)
        ]
        return [v for v in pruned_list if not _is_empty(v)]
    if isinstance(x, tuple):
        pruned_list = [
            _prune(v, _depth=_depth + 1, _max_depth=_max_depth) for v in x if not _is_empty(v)
        ]
        return tuple(v for v in pruned_list if not _is_empty(v))
    if isinstance(x, set):
        pruned_set = {
            _prune(v, _depth=_depth + 1, _max_depth=_max_depth) for v in x if not _is_empty(v)
        }
        return {v for v in pruned_set if not _is_empty(v)}
    return x


def minimal_yaml(
    value: Any,
    *,
    drop_empties: bool = True,
    indent: int = 2,
    line_width: int = 2**31 - 1,
    sort_keys: bool = False,
    unescape_html: bool = False,
) -> str:
    """Convert value to minimal YAML string.

    Args:
        value: Value to convert (dict, list, or JSON string)
        drop_empties: Remove empty/None values recursively (default: True)
        indent: YAML indentation level (default: 2)
        line_width: Maximum line width (default: unlimited)
        sort_keys: Sort dictionary keys alphabetically (default: False)
        unescape_html: Unescape HTML entities in output (default: False)

    Returns:
        YAML-formatted string

    Raises:
        RecursionError: If value nesting exceeds maximum depth (100)

    Security Note:
        This function only DUMPS (outputs) YAML using SafeDumper - it does
        not load/parse YAML, so it is not vulnerable to YAML deserialization
        attacks. However, if unescape_html=True is used and the output is
        later rendered in HTML without proper escaping, XSS vulnerabilities
        may occur. Use caution with unescape_html in web contexts.
    """
    # Auto-parse JSON strings for convenience (fails gracefully on invalid JSON)
    if isinstance(value, str):
        try:
            value = orjson.loads(value)
        except orjson.JSONDecodeError:
            # Not valid JSON - treat as plain string
            pass

    data = _prune(value) if drop_empties else value
    str_ = yaml.dump(
        data,
        Dumper=MinimalDumper,
        default_flow_style=False,
        sort_keys=sort_keys,
        allow_unicode=True,
        indent=indent,
        width=line_width,
    )
    if unescape_html:
        import html

        return html.unescape(str_)
    return str_
