# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from lionpride.core import Element, Flow, Pile, Progression, to_uuid
from lionpride.errors import NotFoundError
from lionpride.operations import OperationRegistry
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
        return f"Branch(messages={len(self)}, session={self.session_id}{name_str})"


class Session(Element):
    user: str | UUID = "user"
    conversations: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message, progressions=Pile(item_type=Branch))
    )
    services: ServiceRegistry = Field(
        default_factory=ServiceRegistry,
        description="Available services (models, tools)",
    )
    operations: OperationRegistry | None = Field(
        default=None,
        description="Operation registry (OperationRegistry)",
    )

    @model_validator(mode="after")
    def _register_default_operations(self):
        """Auto-register built-in operations on initialization."""
        from lionpride.operations.registry import OperationRegistry

        if self.operations is None:
            self.operations = OperationRegistry()

        # Register default operations
        try:
            from lionpride.operations.operate import communicate, generate, operate, react

            self.operations.register("communicate", communicate, update=True)
            self.operations.register("operate", operate, update=True)
            self.operations.register("generate", generate, update=True)
            self.operations.register("react", react, update=True)
        except ImportError:
            # Operations module may not be fully available during tests
            pass

        return self

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
            order=list(messages) if messages else [],  # type: ignore[arg-type]
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

    def get_branch_system(self, branch: Branch | UUID | str) -> Message | None:
        """Get the system message for a branch.

        Args:
            branch: Branch instance, UUID, or name

        Returns:
            System Message if set, None otherwise
        """
        resolved_branch = self.get_branch(branch)
        if resolved_branch.system is None:
            return None
        return self.messages.get(resolved_branch.system)

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
            "branch_id": str(from_branch.id),
            "branch_name": from_branch.name,
            "created_at": from_branch.created_at.isoformat(),
            "message_count": len(from_branch),
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
        # TODO: Add branch parameter for resource access control
        # Check branch.resources contains service_name before invoking
        service = self.services.get(service_name)
        return await service.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **kwargs,
        )

    async def conduct(
        self,
        branch: Branch | UUID | str,
        operation: str,
        ipu: Any,  # IPU type (avoid circular import)
        params: Any = None,  # Typed Param instance or None
        **kwargs,
    ):
        """Create operation for IPU execution with access control.

        Flow:
        1. Resolve branch
        2. Check operation registered
        3. Validate access control (branch.resources, branch.capabilities)
        4. Convert kwargs → typed Param if needed
        5. Create Operation with Param
        6. Queue to IPU for execution
        7. Return Operation (caller can await op.invoke() or check op.status)

        Args:
            branch: Branch to execute in (UUID, name, or Branch instance)
            operation: Operation type name (must be registered)
            ipu: IPU instance (execution context)
            params: Typed Param instance (GenerateParam, etc.) or None
            **kwargs: Operation parameters (converted to Param if params=None)

        Returns:
            Operation instance (queued for execution)

        Raises:
            NotFoundError: If branch or operation not found
            PermissionError: If branch lacks required resources/capabilities
            ValueError: If neither params nor kwargs provided

        Usage:
            # Flexible kwargs interface
            op = await session.conduct(branch, "generate", ipu, imodel="gpt4", messages=[])

            # Direct Param (for workflows)
            param = GenerateParam(imodel="gpt4", messages=[])
            op = await session.conduct(branch, "generate", ipu, params=param)
        """
        from lionpride.operations import Operation

        # 1. Resolve branch
        resolved_branch = self.get_branch(branch)

        # 2. Check operation registered
        if self.operations is None or operation not in self.operations:
            raise NotFoundError(f"Operation '{operation}' not registered in session")

        # 3. Validate access control (FAIL EARLY)
        metadata = self.operations.get_metadata(operation)

        # Check resources (services)
        missing_resources = metadata["required_resources"] - resolved_branch.resources
        if missing_resources:
            raise PermissionError(
                f"Branch '{resolved_branch.name}' lacks required resources for operation '{operation}': "
                f"{sorted(missing_resources)}. Available: {sorted(resolved_branch.resources)}"
            )

        # Check capabilities (schemas)
        missing_capabilities = metadata["required_capabilities"] - resolved_branch.capabilities
        if missing_capabilities:
            raise PermissionError(
                f"Branch '{resolved_branch.name}' lacks required capabilities for operation '{operation}': "
                f"{sorted(missing_capabilities)}. Available: {sorted(resolved_branch.capabilities)}"
            )

        # 4. Convert kwargs → typed Param if needed
        if params is None:
            if not kwargs:
                raise ValueError(f"Operation '{operation}' requires either params= or **kwargs")
            # Convert kwargs to typed Param based on operation type
            params = self._kwargs_to_param(operation, kwargs)

        # 5. Create operation with Param
        op = Operation(  # type: ignore[call-arg]
            operation_type=operation,
            parameters=params,
            session_id=self.id,
            branch_id=resolved_branch.id,
        )

        # 6. Queue to IPU for execution
        await ipu.queue(op)

        # 7. Return operation (caller can check status or await result)
        return op

    def _kwargs_to_param(self, operation: str, kwargs: dict) -> Any:
        """Convert kwargs dict to typed Param based on operation type.

        Args:
            operation: Operation type name
            kwargs: Parameter dict

        Returns:
            Typed Param instance (GenerateParam, CommunicateParam, etc.)
        """
        from lionpride.operations import (
            CommunicateParam,
            GenerateParam,
            OperateParam,
            ReactParam,
        )

        # Map operation name → Param class
        param_map = {
            "generate": GenerateParam,
            "communicate": CommunicateParam,
            "operate": OperateParam,
            "react": ReactParam,
        }

        param_class = param_map.get(operation)
        if param_class is None:
            # Unknown operation - return raw dict for backward compatibility
            return kwargs

        # Create typed Param from kwargs
        return param_class(**kwargs)
