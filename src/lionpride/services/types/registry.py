# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from uuid import UUID

from lionpride.core import Pile, to_uuid

from .imodel import iModel

__all__ = ("ServiceRegistry",)


class ServiceRegistry:
    def __init__(self):
        """Initialize empty registry with Pile storage and name index."""
        from .imodel import iModel

        self._pile: Pile[iModel] = Pile(item_type=iModel)
        self._name_index: dict[str, UUID] = {}

    def register(self, model: iModel, update: bool = False) -> UUID:
        if model.name in self._name_index:
            if not update:
                raise ValueError(f"Service '{model.name}' already registered")
            # Update: remove old, add new
            old_uid = self._name_index[model.name]
            self._pile.remove(old_uid)

        self._pile.add(model)
        self._name_index[model.name] = model.id

        return model.id

    def unregister(self, name: str) -> iModel:
        """Remove and return service by name."""
        if name not in self._name_index:
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index.pop(name)
        return self._pile.remove(uid)

    def get(self, name: str | UUID | iModel) -> iModel:
        id_ = None
        with contextlib.suppress(ValueError):
            id_ = to_uuid(name)

        if not id_ and name in self._name_index:
            id_ = self._name_index[name]

        with contextlib.suppress(TypeError):
            if id_ is not None:
                return self._pile[id_]
        raise KeyError(f"Service '{name}' not found")

    def __contains__(self, name: str | UUID | iModel) -> bool:
        if name in self._name_index:
            return True
        return name in self._pile

    def list_names(self) -> list[str]:
        """List all registered service names."""
        return list(self._name_index.keys())

    def clear(self) -> None:
        """Remove all registered services."""
        self._pile.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        """Return number of registered services."""
        return len(self._pile)

    def __repr__(self) -> str:
        """String representation."""
        return f"ServiceRegistry(count={len(self)})"

    async def register_mcp_server(
        self,
        server_config: dict,
        tool_names: list[str] | None = None,
        request_options: dict[str, type] | None = None,
        update: bool = False,
    ) -> list[str]:
        from lionpride.services.mcps.loader import load_mcp_tools

        return await load_mcp_tools(
            registry=self,
            server_config=server_config,
            tool_names=tool_names,
            request_options=request_options,
            update=update,
        )

    async def load_mcp_config(
        self,
        config_path: str,
        server_names: list[str] | None = None,
        update: bool = False,
    ) -> dict[str, list[str]]:
        from lionpride.services.mcps.loader import load_mcp_config

        return await load_mcp_config(
            registry=self,
            config_path=config_path,
            server_names=server_names,
            update=update,
        )
