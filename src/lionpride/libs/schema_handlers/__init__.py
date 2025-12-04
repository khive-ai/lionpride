# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ._breakdown_pydantic_annotation import breakdown_pydantic_annotation, is_pydantic_model
from ._minimal_yaml import minimal_yaml
from ._typescript import typescript_schema

if TYPE_CHECKING:
    from ._schema_to_model import load_pydantic_model_from_schema


__all__ = (
    "breakdown_pydantic_annotation",
    "is_pydantic_model",
    "load_pydantic_model_from_schema",
    "minimal_yaml",
    "typescript_schema",
)


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "load_pydantic_model_from_schema":
        from ._schema_to_model import load_pydantic_model_from_schema

        return load_pydantic_model_from_schema
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
