# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import re
from inspect import isclass
from typing import Any, get_args, get_origin

from pydantic import BaseModel

# Pattern to match module-qualified names like __main__.Foo or lionagi.x.y.Bar
_MODULE_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*\.)+([a-zA-Z_][a-zA-Z0-9_]*)")


# Pattern to extract type name from <class 'typename'>
_CLASS_PATTERN = re.compile(r"<class '([^']+)'>")


def _clean_type_repr(t: Any) -> str:
    """Convert a type annotation to a clean Python-like string.

    Handles:
    - <class 'str'> -> "str"
    - <class 'int'> -> "int"
    - Module-qualified names -> just class name
    """
    s = str(t) if not isinstance(t, str) else t

    # Handle <class 'typename'> pattern
    if match := _CLASS_PATTERN.match(s):
        type_name = match.group(1)
        # Strip module prefix if present (e.g., 'builtins.str' -> 'str')
        return type_name.rsplit(".", 1)[-1]

    # Replace module-qualified names with just the class name
    s = _MODULE_PATTERN.sub(r"\2", s)

    return s


def breakdown_pydantic_annotation(
    model: type[BaseModel],
    max_depth: int | None = None,
    clean_types: bool = True,
) -> dict[str, Any]:
    """Break down a Pydantic model's annotations into a nested dict structure.

    Args:
        model: The Pydantic model class to break down.
        max_depth: Maximum recursion depth for nested models.
        clean_types: If True, convert type annotations to clean strings
            without module prefixes (e.g., 'list[CodeModule]' instead of
            'list[__main__.CodeModule]').
        - Lists containing the above for list fields
    """
    result = _breakdown_pydantic_annotation(
        model=model,
        max_depth=max_depth,
        current_depth=0,
    )
    if clean_types:
        return _clean_result(result)
    return result


def _clean_result(result: dict[str, Any]) -> dict[str, Any]:
    """Recursively clean type representations in the result dict."""
    out: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = _clean_result(v)
        elif isinstance(v, list) and v:
            if isinstance(v[0], dict):
                out[k] = [_clean_result(v[0])]
            else:
                out[k] = [_clean_type_repr(v[0])]
        else:
            out[k] = _clean_type_repr(v)
    return out


def _breakdown_pydantic_annotation(
    model: type[BaseModel],
    max_depth: int | None = None,
    current_depth: int = 0,
) -> dict[str, Any]:
    if not is_pydantic_model(model):
        raise TypeError("Input must be a Pydantic model")

    if max_depth is not None and current_depth >= max_depth:
        raise RecursionError("Maximum recursion depth reached")

    out: dict[str, Any] = {}
    for k, v in model.__annotations__.items():
        origin = get_origin(v)
        if is_pydantic_model(v):
            out[k] = _breakdown_pydantic_annotation(v, max_depth, current_depth + 1)
        elif origin is list:
            args = get_args(v)
            if args and is_pydantic_model(args[0]):
                out[k] = [_breakdown_pydantic_annotation(args[0], max_depth, current_depth + 1)]
            else:
                out[k] = [args[0] if args else Any]
        else:
            out[k] = v

    return out


def is_pydantic_model(x: Any) -> bool:
    try:
        return isclass(x) and issubclass(x, BaseModel)
    except TypeError:
        return False
