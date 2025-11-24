# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import Field

from lionpride.core import Element, Flow, Pile, Progression, to_uuid
from lionpride.errors import NotFoundError
from lionpride.services import ServiceRegistry

from .messages import Message, SystemContent

if TYPE_CHECKING:
    from lionpride.services.types import Calling

__all__ = ("Branch", "Session")


class Branch(Progression):
    user: str | UUID = Field(frozen=True)
    """The entity giving command to this branch (name or ID) default session_id"""

    system: UUID | None = None
    """System message UUID in corresponding Session conversations.items"""

    capabilities: set[str] = Field(default_factory=set)
    """Structured output allowed capability names during operations"""

    resources: set[str] = Field(default_factory=set)
    """Allowed backend service resources names for this branch"""

    session_id: UUID = Field(frozen=True)
    """Parent Session UUID"""

    def set_system_message(self, message: UUID | Message) -> None:
        old_system = self.system
        msg_id = to_uuid(message)
        self.system = msg_id

        if old_system is not None and len(self) > 0:
            self[0] = msg_id
        else:
            self.insert(0, msg_id)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        bound_str = " bound" if self._session is not None else ""
        return f"Branch(messages={len(self)}, session={self.session_id}{name_str}{bound_str})"


class Session(Element):
    user: str | UUID = "user"
    conversations: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message, progressions=Pile(item_type=Branch))
    )
    services: ServiceRegistry = Field(
        default_factory=ServiceRegistry,
        description="Available services (models, tools)",
    )

    @property
    def messages(self):
        """Read-only view of conversations.items (Pile[Message])."""
        return self.conversations.items

    @property
    def branches(self):
        """Read-only view of conversations.progressions (Pile[Branch])."""
        return self.conversations.progressions

    def create_branch(
        self,
        *,
        name: str | None = None,
        system: Message | UUID | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
        messages: Iterable[UUID | Message] | None = None,
    ):
        if system is not None and system not in self.messages:
            if isinstance(system, UUID):
                raise ValueError(f"System message UUID {system} not found in session messages")
            if not isinstance(system, Message):
                raise ValueError("System message must be a Message instance or UUID")
            self.conversations.add_item(system)

        branch_name = name or f"branch_{len(self.branches)}"
        branch = Branch(
            session_id=self.id,
            user=self.id,
            name=branch_name,
            capabilities=capabilities or set(),
            resources=resources or set(),
            order=messages or [],
        )
        if system is not None:
            branch.set_system_message(system)

        self.conversations.add_progression(branch)
        return branch

    def get_branch(self, branch: UUID | str | Branch) -> Branch:
        if isinstance(branch, Branch) and branch in self.branches:
            return branch
        with contextlib.suppress(KeyError):
            return self.conversations.get_progression(branch)
        raise NotFoundError("Branch not found in session branches")

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        capabilities: set[str] | Literal[True] | None = None,
        resources: set[str] | Literal[True] | None = None,
        system: UUID | Message | Literal[True] | None = None,
    ) -> Branch:
        from_branch = self.get_branch(branch)
        forked = self.create_branch(
            name=name or f"{from_branch.name}_fork",
            messages=from_branch.order,
            capabilities={*from_branch.capabilities}
            if capabilities is True
            else (capabilities or set()),
            resources={*from_branch.resources} if resources is True else (resources or set()),
            system=from_branch.system if system is True else system,
        )
        forked.metadata["forked_from"] = {
            "branch_id": str(branch.id),
            "branch_name": branch.name,
            "created_at": branch.created_at.isoformat(),
            "message_count": len(branch),
        }

        self.conversations.add_progression(forked)
        return forked

    def add_message(
        self,
        message: Message,
        branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
    ):
        if not isinstance(message.content, SystemContent):
            self.conversations.add_item(message, progressions=branches)
            return

        resolved = None
        if branches is not None:
            branches = [branches] if not isinstance(branches, list) else branches
            resolved = [self.get_branch(b) for b in branches]

        self.conversations.add_item(message)
        if resolved is not None:
            for branch in resolved:
                branch.set_system_message(message)

    async def request(
        self,
        service_name: str,
        *,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **kwargs,
    ) -> Calling:
        service = self.services.get(service_name)
        return await service.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **kwargs,
        )
