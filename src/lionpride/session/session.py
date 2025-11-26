# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr

from lionpride.core import Element, Flow, Graph, Pile, Progression
from lionpride.errors import NotFoundError
from lionpride.operations.registry import OperationRegistry
from lionpride.services import ServiceRegistry
from lionpride.services.types import iModel
from lionpride.types import Unset, not_sentinel

from .messages import Message, SystemContent

if TYPE_CHECKING:
    from lionpride.operations.node import Operation
    from lionpride.services.types import Calling

__all__ = ("Branch", "Session")


class Branch(Progression):
    """Named progression of messages within a session.

    Branch is a Progression (ordered UUIDs) with session context:
    - user: who is commanding this branch
    - system: system message UUID at order[0]
    - capabilities: structured output schemas allowed
    - resources: backend services allowed
    """

    user: str | UUID = Field(frozen=True)
    """The entity giving command to this branch (name or ID), default session_id"""

    system: UUID | None = None
    """System message UUID in corresponding Session conversations.items"""

    capabilities: set[str] = Field(default_factory=set)
    """Structured output allowed capability names during operations"""

    resources: set[str] = Field(default_factory=set)
    """Allowed backend service resource names for this branch"""

    session_id: UUID = Field(frozen=True)
    """Parent Session UUID"""

    def set_system_message(self, message: UUID | Message) -> None:
        """Set system message, ensuring it's at order[0]."""
        old_system = self.system
        msg_id = self._coerce_id(message)
        self.system = msg_id

        if old_system is not None and len(self) > 0:
            self[0] = msg_id
        else:
            self.insert(0, msg_id)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Branch(messages={len(self)}, session={self.session_id}{name_str})"


class Session(Element):
    """Central storage for messages, branches, and services.

    Session composes:
    - conversations: Flow[Message, Branch] for message storage and progressions
    - services: ServiceRegistry for models and tools

    Usage:
        session = Session()
        branch = session.create_branch(name="main")

        # Add messages
        session.add_message(Message(content=...), branches=branch)

        # Service requests
        result = await session.request("openai", model="gpt-4", messages=[...])
    """

    user: str | UUID = "user"
    """Default user identifier"""

    conversations: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message, progressions=Pile(item_type=Branch))
    )
    """Message flow with branch progressions"""
    services: ServiceRegistry = Field(
        default_factory=ServiceRegistry,
        description="Available services (models, tools)",
    )
    operations: OperationRegistry = Field(
        default_factory=OperationRegistry,
        description="Available operations (operate, react, communicate, generate)",
    )
    _default_branch_id: UUID | None = PrivateAttr(None)
    _default_backends: dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        user: str | UUID | None = None,
        conversations: Flow[Message, Branch] | None = None,
        services: ServiceRegistry | None = None,
        default_branch: Branch | UUID | str | None = None,
        default_generate_model: iModel | str | None = None,
        default_parse_model: iModel | str | None = None,
        default_capabilities: set[str] | None = None,
        default_system: Message | None = None,
        /,
        **data,
    ):
        d_ = {
            **data,
            "user": user,
            "conversations": conversations,
            "services": services,
        }

        super().__init__(**{k: v for k, v in d_.items() if not_sentinel(v, True, True)})

        # Collect default model names for branch resources
        default_resources: set[str] = set()

        # Set default backends (store name strings, resolve to iModel on access)
        if default_generate_model is not None:
            name = (
                default_generate_model.name
                if isinstance(default_generate_model, iModel)
                else default_generate_model
            )
            self._default_backends["generate"] = name
            default_resources.add(name)
            # Auto-register if iModel instance provided
            if isinstance(default_generate_model, iModel) and not self.services.has(name):
                self.services.register(default_generate_model)

        if default_parse_model is not None:
            name = (
                default_parse_model.name
                if isinstance(default_parse_model, iModel)
                else default_parse_model
            )
            self._default_backends["parse"] = name
            default_resources.add(name)
            if isinstance(default_parse_model, iModel) and not self.services.has(name):
                self.services.register(default_parse_model)

        # Set default branch (add to session, grant default model resources)
        if default_branch is not None:
            if isinstance(default_branch, Branch):
                # Add branch to session and grant default resources/capabilities
                default_branch.resources.update(default_resources)
                if default_capabilities:
                    default_branch.capabilities.update(default_capabilities)
                self.conversations.add_progression(default_branch)
                self._default_branch_id = default_branch.id
            else:
                # Create new branch with name, grant default resources/capabilities
                branch_obj = self.create_branch(
                    name=str(default_branch),
                    resources=default_resources,
                    capabilities=default_capabilities or set(),
                    system=default_system,
                )
                self._default_branch_id = branch_obj.id

    @property
    def default_branch(self) -> Branch | None:
        """Get default branch, or None if not set."""
        if self._default_branch_id is None:
            return None
        with contextlib.suppress(KeyError):
            return self.conversations.get_progression(self._default_branch_id)
        return None

    @property
    def default_generate_model(self) -> iModel | None:
        """Get default generate model from services."""
        name = self._default_backends.get("generate")
        if name is None:
            return None
        return self.services.get(name) if self.services.has(name) else None

    @property
    def default_parse_model(self) -> iModel | None:
        """Get default parse model from services."""
        name = self._default_backends.get("parse")
        if name is None:
            return None
        return self.services.get(name) if self.services.has(name) else None

    def set_default_branch(self, branch: Branch | UUID | str) -> None:
        """Set the default branch for operations.

        Args:
            branch: Branch instance, UUID, or name (must exist in session)

        Raises:
            NotFoundError: If branch not found in session
        """
        resolved = self.get_branch(branch)  # Raises NotFoundError if not in session
        self._default_branch_id = resolved.id

    def set_default_model(
        self,
        model: iModel | str,
        operation: Literal["generate", "parse"] = "generate",
    ) -> None:
        """Set default model for an operation type.

        Also grants the model as a resource to the default branch if one exists.
        """
        name = model.name if isinstance(model, iModel) else model
        self._default_backends[operation] = name
        if isinstance(model, iModel) and not self.services.has(name):
            self.services.register(model)
        # Grant to default branch if exists
        if self.default_branch is not None:
            self.default_branch.resources.add(name)

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
    ) -> Branch:
        """Create new branch for isolated conversation threads.

        Args:
            name: Optional branch name (auto-generated if None)
            system: Optional system message (Message or UUID)
            capabilities: Structured output schemas allowed
            resources: Backend services allowed
            messages: Initial message UUIDs/Messages to include

        Returns:
            The created Branch
        """
        # Handle system message
        if system is not None and system not in self.messages:
            if isinstance(system, UUID):
                raise ValueError(f"System message UUID {system} not found in session messages")
            if not isinstance(system, Message):
                raise ValueError("System message must be a Message instance or UUID")
            self.conversations.add_item(system)

        branch_name = name or f"branch_{len(self.branches)}"
        branch = Branch(
            session_id=self.id,
            user=self.user,
            name=branch_name,
            capabilities=capabilities or set(),
            resources=resources or set(),
            order=list(messages) if messages else [],  # type: ignore[arg-type]
        )

        if system is not None:
            branch.set_system_message(system)

        self.conversations.add_progression(branch)
        return branch

    def get_branch(self, branch: UUID | str | Branch, default=Unset, /) -> Branch:
        """Get branch by UUID, name, or instance.

        Args:
            branch: Branch UUID, name, or instance

        Returns:
            Branch instance

        Raises:
            NotFoundError: If branch not found
        """
        if isinstance(branch, Branch) and branch in self.branches:
            return branch
        with contextlib.suppress(KeyError):
            return self.conversations.get_progression(branch)
        if default is not Unset:
            return default
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

    def set_branch_system(
        self,
        branch: Branch | UUID | str,
        system: Message | UUID,
    ) -> None:
        """Set or change system message for a branch.

        Args:
            branch: Branch instance, UUID, or name
            system: System message (Message or UUID)
        """
        resolved_branch = self.get_branch(branch)

        if isinstance(system, Message):
            if system.id not in self.messages:
                self.conversations.add_item(system)
            resolved_branch.set_system_message(system.id)
        else:
            resolved_branch.set_system_message(system)

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        capabilities: set[str] | Literal[True] | None = None,
        resources: set[str] | Literal[True] | None = None,
        system: UUID | Message | Literal[True] | None = None,
    ) -> Branch:
        """Fork branch to create divergent conversation path.

        Args:
            branch: Source branch to fork from
            name: Name for forked branch (auto-generated if None)
            capabilities: Capabilities (True = copy from source)
            resources: Resources (True = copy from source)
            system: System message (True = copy from source)

        Returns:
            New forked Branch
        """
        from_branch = self.get_branch(branch)

        forked = self.create_branch(
            name=name or f"{from_branch.name}_fork",
            messages=from_branch.order,
            capabilities=(
                {*from_branch.capabilities} if capabilities is True else (capabilities or set())
            ),
            resources={*from_branch.resources} if resources is True else (resources or set()),
            system=from_branch.system if system is True else system,
        )

        forked.metadata["forked_from"] = {
            "branch_id": str(from_branch.id),
            "branch_name": from_branch.name,
            "created_at": from_branch.created_at.isoformat(),
            "message_count": len(from_branch),
        }

        return forked

    def add_message(
        self,
        message: Message,
        branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
    ) -> None:
        """Add message to session, optionally to specific branches.

        System messages are detected by content type and handled specially
        (added to items only, then set via set_system_message).

        Args:
            message: Message to add
            branches: Branch(es) to add message to (optional)
        """
        # System messages: add to items, then set on branches
        if isinstance(message.content, SystemContent):
            resolved = None
            if branches is not None:
                branches_list = [branches] if not isinstance(branches, list) else branches
                resolved = [self.get_branch(b) for b in branches_list]

            self.conversations.add_item(message)
            if resolved is not None:
                for branch in resolved:
                    branch.set_system_message(message)
            return

        # Regular messages: add to items and progressions
        self.conversations.add_item(message, progressions=branches)

    async def request(
        self,
        service_name: str,
        *,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **kwargs,
    ) -> Calling:
        """Unified service request interface.

        Central entry point for all service interactions (LLM and tools).

        Args:
            service_name: Name of registered service
            poll_timeout: Max seconds to wait for completion
            poll_interval: Seconds between status checks
            **kwargs: Service-specific arguments

        Returns:
            Calling with execution results
        """
        service = self.services.get(service_name)
        return await service.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **kwargs,
        )

    async def conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        **kwargs,
    ) -> Operation:
        """Conduct an operation through the registry.

        Creates an Operation, binds it to session/branch, and invokes it.

        Args:
            operation_type: Operation name (communicate, operate, react, generate)
            branch: Branch for execution (uses default if None)
            **kwargs: Operation parameters

        Returns:
            Operation with execution results (access via operation.response)

        Raises:
            KeyError: If operation not registered
            RuntimeError: If no branch provided and no default set
        """
        from lionpride.operations.node import Operation

        # Resolve branch
        resolved_branch = self._resolve_branch(branch)

        # Create operation
        op = Operation(operation_type=operation_type, parameters=kwargs)
        op.bind(self, resolved_branch)

        # Invoke and return
        await op.invoke()
        return op

    async def flow(
        self,
        graph: Graph,
        branch: Branch | UUID | str | None = None,
        *,
        context: dict | None = None,
        max_concurrent: int | None = None,
        stop_on_error: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Execute operation graph with dependency-aware scheduling.

        Args:
            graph: Operation graph (DAG) to execute
            branch: Branch for execution (uses default if None)
            context: Shared context for all operations
            max_concurrent: Max concurrent operations (None = unlimited)
            stop_on_error: Stop on first error
            verbose: Enable verbose output

        Returns:
            Dictionary mapping operation names to their results
        """
        from lionpride.operations.flow import flow as flow_func

        resolved_branch = self._resolve_branch(branch)
        return await flow_func(
            session=self,
            branch=resolved_branch,
            graph=graph,
            context=context,
            max_concurrent=max_concurrent,
            stop_on_error=stop_on_error,
            verbose=verbose,
        )

    def _resolve_branch(self, branch: Branch | UUID | str | None) -> Branch:
        """Resolve branch parameter, falling back to default."""
        if branch is not None:
            return self.get_branch(branch)
        if self.default_branch is not None:
            return self.default_branch
        raise RuntimeError("No branch provided and no default branch set")

    def register_operation(
        self,
        name: str,
        factory,
        *,
        override: bool = False,
    ) -> None:
        """Register custom operation factory.

        Enables arbitrary workflow patterns including nested DAGs.

        Args:
            name: Operation name for conduct() and Builder
            factory: Async function (session, branch, params) -> result
            override: Allow replacing existing operation

        Example:
            async def my_nested_dag(session, branch, params):
                # Build inner graph
                builder = Builder()
                builder.add("step1", "communicate", {...})
                builder.add("step2", "operate", {...}, depends_on=["step1"])
                graph = builder.build()

                # Execute nested flow
                return await session.flow(graph, branch)

            session.register_operation("nested_dag", my_nested_dag)
            result = await session.conduct("nested_dag", branch, ...)
        """
        self.operations.register(name, factory, override=override)

    def __repr__(self) -> str:
        return (
            f"Session(messages={len(self.messages)}, "
            f"branches={len(self.branches)}, "
            f"services={len(self.services)})"
        )
