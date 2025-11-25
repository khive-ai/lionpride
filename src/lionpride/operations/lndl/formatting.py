# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL spec formatting using typescript_schema for consistency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lionpride.libs.schema_handlers import typescript_schema

if TYPE_CHECKING:
    from lionpride.types import Operable

    from ..operate.operative import Operative


def generate_lndl_spec_format(
    operable: Operable | Operative,
    has_tools: bool = False,
) -> str:
    """Generate LNDL output spec using typescript_schema format.

    Uses same schema formatting as InstructionContent for consistency.
    """
    from ..operate.operative import Operative

    if isinstance(operable, Operative):
        specs = operable.operable.get_specs()
    else:
        specs = operable.get_specs()

    if not specs:
        return ""

    parts = []
    for spec in specs:
        spec_name = spec.name or "output"
        base_type = spec.base_type

        if hasattr(base_type, "model_json_schema"):
            # Pydantic model - use typescript_schema
            schema = base_type.model_json_schema()
            ts = typescript_schema(schema)
            # Remove ? markers - LNDL uses field names directly
            ts = ts.replace("?:", ":")
            parts.append(f"{spec_name}({base_type.__name__}):\n{ts}")
        else:
            # Scalar
            type_name = getattr(base_type, "__name__", str(base_type))
            parts.append(f"{spec_name}: {type_name}")

    result = "Output:\n" + "\n".join(parts)
    result += f"\n\nEnd with OUT{{{specs[0].name or 'output'}:[aliases]}}"
    if not has_tools:
        result += "\nNo tools. Don't use <lact>."
    return result


__all__ = ("generate_lndl_spec_format",)
