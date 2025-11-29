# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Session and Branch - Core conversation management.

Session orchestrates messages, branches, services, and operations.
Branch is a named progression of message UUIDs with access control.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr

from lionpride.core import Element, Flow, Graph, Progression
from lionpride.errors import NotFoundError
from lionpride.operations.node import Operation
from lionpride.operations.registry import OperationRegistry
from lionpride.services import ServiceRegistry, iModel
from lionpride.types import Unset, not_sentinel

from .messages import Message, SystemContent

if TYPE_CHECKING:
    from lionpride.services.types import Calling

__all__ = ("Branch", "Session")


# =============================================================================
# Branch
# =============================================================================


class Branch(Progression):
    """Named progression of messages with access control.

    Extends Progression (ordered UUIDs) with session context and permissions.
    Operations are invoked via Session.conduct(), not Branch methods.

    Attributes:
        session_id: Parent session UUID (frozen).
        system_message: System message UUID, always at order[0] if set.
        capabilities: Allowed structured output schema names.
        resources: Allowed backend service names.
    """

    session_id: UUID = Field(..., frozen=True)
    system_message: UUID | None = None
    capabilities: set[str] = Field(default_factory=set)
    resources: set[str] = Field(default_factory=set)

    def set_system_message(self, message_id: UUID | Message) -> None:
        """Set system message at order[0].

        Args:
            message_id: Message UUID or Message instance.
        """
        msg_id = message_id.id if isinstance(message_id, Message) else message_id
        old_system = self.system_message
        self.system_message = msg_id

        if old_system is not None and len(self) > 0:
            self[0] = msg_id
        else:
            self.insert(0, msg_id)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Branch(messages={len(self)}, session={str(self.session_id)[:8]}{name_str})"


# =============================================================================
# Session
# =============================================================================


class Session(Element):
    """Central orchestrator for messages, branches, services, and operations.

    Example:
        session = Session(default_generate_model=my_model)
        branch = session.create_branch(name="main")
        op = await session.conduct("operate", branch, instruction="Analyze this")
        result = op.response
    """

    # -------------------------------------------------------------------------
    # Fields
    # -------------------------------------------------------------------------

    user: str | None = None
    """User identifier for this session."""

    conversations: Flow[Message, Branch] = Field(default_factory=lambda: Flow(item_type=Message))
    """Message storage (items) and branch progressions."""

    services: ServiceRegistry = Field(default_factory=ServiceRegistry)
    """Registered models and tools."""

    operations: OperationRegistry = Field(default_factory=OperationRegistry)
    """Registered operation factories."""

    _default_branch_id: UUID | None = PrivateAttr(None)
    _default_backends: dict[str, str] = PrivateAttr(default_factory=dict)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        user: str | UUID | None = None,
        conversations: Flow[Message, Branch] | None = None,
        services: ServiceRegistry | None = None,
        *,
        default_branch: Branch | UUID | str | None = None,
        default_generate_model: iModel | str | None = None,
        default_parse_model: iModel | str | None = None,
        default_capabilities: set[str] | None = None,
        default_system: Message | None = None,
        **data,
    ):
        """Initialize session with optional defaults.

        Args:
            user: User identifier.
            conversations: Pre-built Flow (rare).
            services: Pre-built ServiceRegistry (rare).
            default_branch: Auto-create or use this branch as default.
            default_generate_model: Default model for generate operations.
            default_parse_model: Default model for parse operations.
            default_capabilities: Capabilities granted to default branch.
            default_system: System message for default branch.
        """
        d_ = {"user": user, "conversations": conversations, "services": services, **data}
        super().__init__(**{k: v for k, v in d_.items() if not_sentinel(v, True, True)})

        # Collect default model names for branch resources
        default_resources: set[str] = set()

        # Register default models
        for model, key in [
            (default_generate_model, "generate"),
            (default_parse_model, "parse"),
        ]:
            if model is not None:
                name = model.name if isinstance(model, iModel) else model
                self._default_backends[key] = name
                default_resources.add(name)
                if isinstance(model, iModel) and not self.services.has(name):
                    self.services.register(model)

        # Set up default branch
        if default_branch is not None:
            if isinstance(default_branch, Branch):
                default_branch.resources.update(default_resources)
                if default_capabilities:
                    default_branch.capabilities.update(default_capabilities)
                self.conversations.add_progression(default_branch)
                self._default_branch_id = default_branch.id
            else:
                branch_obj = self.create_branch(
                    name=str(default_branch),
                    resources=default_resources,
                    capabilities=default_capabilities or set(),
                    system=default_system,
                )
                self._default_branch_id = branch_obj.id

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def messages(self):
        """All messages in session (Pile[Message])."""
        return self.conversations.items

    @property
    def branches(self):
        """All branches in session (Pile[Branch])."""
        return self.conversations.progressions

    @property
    def default_branch(self) -> Branch | None:
        """Default branch for operations, or None."""
        if self._default_branch_id is None:
            return None
        with contextlib.suppress(KeyError, NotFoundError):
            return self.conversations.get_progression(self._default_branch_id)
        return None

    @property
    def default_generate_model(self) -> iModel | None:
        """Default model for generate operations."""
        name = self._default_backends.get("generate")
        return self.services.get(name) if name and self.services.has(name) else None

    @property
    def default_parse_model(self) -> iModel | None:
        """Default model for parse operations."""
        name = self._default_backends.get("parse")
        return self.services.get(name) if name and self.services.has(name) else None

    # -------------------------------------------------------------------------
    # Branch Management
    # -------------------------------------------------------------------------

    def create_branch(
        self,
        *,
        name: str | None = None,
        system: Message | UUID | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
        messages: Iterable[UUID | Message] | None = None,
    ) -> Branch:
        """Create a new branch.

        Args:
            name: Branch name (auto-generated if None).
            system: System message (Message or UUID).
            capabilities: Allowed structured output schemas.
            resources: Allowed backend services.
            messages: Initial message UUIDs to include.

        Returns:
            Created Branch.

        Raises:
            ValueError: If system UUID not found in session.
        """
        if system is not None and system not in self.messages:
            if isinstance(system, UUID):
                raise ValueError(f"System message UUID {system} not found")
            if not isinstance(system, Message):
                raise ValueError("system must be Message or UUID")
            self.conversations.add_item(system)

        branch = Branch(
            session_id=self.id,
            name=name or f"branch_{len(self.branches)}",
            capabilities=capabilities or set(),
            resources=resources or set(),
            order=list(messages) if messages else [],
        )

        if system is not None:
            branch.set_system_message(system)

        self.conversations.add_progression(branch)
        return branch

    def get_branch(self, branch: UUID | str | Branch, default=Unset, /) -> Branch:
        """Get branch by UUID, name, or instance.

        Args:
            branch: Branch identifier.
            default: Return this if not found (else raise).

        Returns:
            Branch instance.

        Raises:
            NotFoundError: If not found and no default.
        """
        if isinstance(branch, Branch) and branch in self.branches:
            return branch
        with contextlib.suppress(KeyError):
            return self.conversations.get_progression(branch)
        if default is not Unset:
            return default
        raise NotFoundError("Branch not found")

    def set_default_branch(self, branch: Branch | UUID | str) -> None:
        """Set the default branch for operations.

        Args:
            branch: Branch to set as default (must exist).

        Raises:
            NotFoundError: If branch not in session.
        """
        resolved = self.get_branch(branch)
        self._default_branch_id = resolved.id

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        capabilities: set[str] | Literal[True] | None = None,
        resources: set[str] | Literal[True] | None = None,
        system: UUID | Message | Literal[True] | None = None,
    ) -> Branch:
        """Fork branch for divergent exploration.

        Args:
            branch: Source branch.
            name: Name for fork (auto-generated if None).
            capabilities: True=copy from source, None=empty, or explicit set.
            resources: True=copy from source, None=empty, or explicit set.
            system: True=copy from source, None=none, or explicit.

        Returns:
            New forked Branch with lineage metadata.
        """
        source = self.get_branch(branch)

        forked = self.create_branch(
            name=name or f"{source.name}_fork",
            messages=source.order,
            capabilities={*source.capabilities}
            if capabilities is True
            else (capabilities or set()),
            resources={*source.resources} if resources is True else (resources or set()),
            system=source.system_message if system is True else system,
        )

        forked.metadata["forked_from"] = {
            "branch_id": str(source.id),
            "branch_name": source.name,
            "created_at": source.created_at.isoformat(),
            "message_count": len(source),
        }
        return forked

    # -------------------------------------------------------------------------
    # Message Management
    # -------------------------------------------------------------------------

    def add_message(
        self,
        message: Message,
        branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
    ) -> None:
        """Add message to session and optionally to branches.

        System messages (SystemContent) are added to items only,
        then set via set_system_message on specified branches.

        Args:
            message: Message to add.
            branches: Target branch(es).
        """
        if isinstance(message.content, SystemContent):
            self.conversations.add_item(message)
            if branches is not None:
                branch_list = [branches] if not isinstance(branches, list) else branches
                for b in branch_list:
                    self.get_branch(b).set_system_message(message)
            return

        self.conversations.add_item(message, progressions=branches)

    def get_branch_system(self, branch: Branch | UUID | str) -> Message | None:
        """Get system message for a branch.

        Args:
            branch: Branch identifier.

        Returns:
            System Message or None.
        """
        resolved = self.get_branch(branch)
        if resolved.system_message is None:
            return None
        return self.messages.get(resolved.system_message)

    def set_branch_system(self, branch: Branch | UUID | str, system: Message | UUID) -> None:
        """Set system message for a branch.

        Args:
            branch: Branch identifier.
            system: Message or UUID to set as system.
        """
        resolved = self.get_branch(branch)
        if isinstance(system, Message):
            if system.id not in self.messages:
                self.conversations.add_item(system)
            resolved.set_system_message(system.id)
        else:
            resolved.set_system_message(system)

    # -------------------------------------------------------------------------
    # Default Model Management
    # -------------------------------------------------------------------------

    def set_default_model(
        self,
        model: iModel | str,
        operation: Literal["generate", "parse"] = "generate",
    ) -> None:
        """Set default model for an operation type.

        Args:
            model: iModel instance or registered name.
            operation: "generate" or "parse".
        """
        name = model.name if isinstance(model, iModel) else model
        self._default_backends[operation] = name
        if isinstance(model, iModel) and not self.services.has(name):
            self.services.register(model)
        if self.default_branch is not None:
            self.default_branch.resources.add(name)

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    async def conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        params: Any | None = None,
    ) -> Operation:
        """Execute an operation on a branch.

        Args:
            operation_type: Operation name (operate, react, communicate, generate).
            branch: Target branch (uses default if None).
            params: Typed operation parameters (GenerateParams, OperateParams, etc.).

        Returns:
            Invoked Operation (access result via op.response).

        Raises:
            RuntimeError: If no branch and no default set.
            KeyError: If operation not registered.
        """
        resolved = self._resolve_branch(branch)
        op = Operation(operation_type=operation_type, parameters=params)
        op.bind(self, resolved)
        await op.invoke()
        return op

    async def flow(
        self,
        graph: Graph,
        branch: Branch | UUID | str | None = None,
        *,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Execute operation graph with dependency scheduling.

        Operations are executed with their given parameters - no context injection.
        For context passing between operations, use flow_report or manage context
        explicitly before adding operations to the graph.

        Args:
            graph: Operation DAG from Builder.
            branch: Default branch for operations (uses default if None).
            max_concurrent: Concurrency limit (None=unlimited).
            stop_on_error: Stop on first error.
            verbose: Print progress.

        Returns:
            Dict mapping operation names to results.
        """
        from lionpride.operations.flow import flow as flow_func

        return await flow_func(
            session=self,
            branch=self._resolve_branch(branch),
            graph=graph,
            max_concurrent=max_concurrent,
            stop_on_error=stop_on_error,
            verbose=verbose,
        )

    async def request(
        self,
        service_name: str,
        *,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **kwargs,
    ) -> Calling:
        """Direct service request (LLM or tool).

        Args:
            service_name: Registered service name.
            poll_timeout: Max wait seconds.
            poll_interval: Poll interval seconds.
            **kwargs: Service-specific args.

        Returns:
            Calling with execution results.
        """
        service = self.services.get(service_name)
        return await service.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **kwargs,
        )

    def register_operation(self, name: str, factory, *, override: bool = False) -> None:
        """Register custom operation factory.

        Args:
            name: Operation name for conduct().
            factory: Async (session, branch, params) -> result.
            override: Allow replacing existing.
        """
        self.operations.register(name, factory, override=override)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _resolve_branch(self, branch: Branch | UUID | str | None) -> Branch:
        """Resolve branch, falling back to default."""
        if branch is not None:
            return self.get_branch(branch)
        if self.default_branch is not None:
            return self.default_branch
        raise RuntimeError("No branch provided and no default branch set")

    def __repr__(self) -> str:
        return (
            f"Session(messages={len(self.messages)}, "
            f"branches={len(self.branches)}, "
            f"services={len(self.services)})"
        )
