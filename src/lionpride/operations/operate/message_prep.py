# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lionpride.session import Session


def prepare_tool_schemas(
    session: Session,
    tools: bool | list[str],
) -> list[Any] | None:
    """Prepare tool schemas from session services."""
    if not tools:
        return None

    if tools is True:
        return session.services.get_tool_schemas()
    elif isinstance(tools, list):
        return session.services.get_tool_schemas(tool_names=tools)
    return None
