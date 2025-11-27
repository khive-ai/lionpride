# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import Field

from lionpride.core import Element, Flow, Progression
from lionpride.operations.registry import OperationRegistry
from lionpride.services import ServiceRegistry

from .messages import Message, SenderRecipient

if TYPE_CHECKING:
    from lionpride.services.types import Calling

__all__ = ("Branch", "Session")


class Branch(Progression):
    """Named progression of messages within a session.

    Branch is a Progression (ordered UUIDs) with session context:
    - user: who is commanding this branch
    - system_message: system message UUID at order[0]
    - capabilities: structured output schemas allowed
    - resources: backend services allowed

    Operations are invoked via Session.conduct(), not Branch methods.
    """

    session_id: UUID = Field(..., frozen=True, description="Parent Session UUID")
    user: str | UUID | None = Field(default=None, description="User identifier")
    system_message: UUID | None = Field(default=None, description="System message UUID")
    capabilities: set[str] = Field(
        default_factory=set, description="Structured output schemas allowed"
    )
    resources: set[str] = Field(default_factory=set, description="Backend service names allowed")

    def set_system_message(self, message_id: UUID) -> None:
        """Set system message, ensuring it's at order[0]."""
        old_system = self.system_message
        self.system_message = message_id

        if old_system is not None and len(self) > 0:
            self[0] = message_id
        else:
            self.insert(0, message_id)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Branch(messages={len(self)}, session={self.session_id}{name_str})"


class Session(Element):
    """Central storage for messages, branches, services, and operations.

    Attributes:
        user: User identifier
        default_imodel: Default iModel for operations (better DX)
        conversations: Flow[Message, Branch] for message storage and branch progressions
        services: ServiceRegistry for models and tools
        operations: OperationRegistry for operation factories
        messages: Read-only view of conversations.items (Pile[Message])
        branches: Read-only view of conversations.progressions (Pile[Branch])

    Example:
        # Better DX with default_imodel
        session = Session(default_imodel=iModel(backend=OpenAI(...)))
        branch = session.create_branch()
        result = await session.conduct("operate", branch, instruction="...", ...)

        # Or explicit imodel per-operation
        result = await session.conduct("operate", branch, imodel=my_model, ...)
    """

    user: str | None = Field(default=None, description="User identifier")
    default_imodel: Any = Field(
        default=None,
        description="Default iModel for operations. Improves DX by eliminating repeated imodel= args.",
    )
    conversations: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message),
        description="Message flow with branches",
    )
    services: ServiceRegistry = Field(
        default_factory=ServiceRegistry,
        description="Available services (models, tools)",
    )
    operations: OperationRegistry = Field(
        default_factory=OperationRegistry,
        description="Operation factories (operate, react, communicate, generate)",
    )
    default_branch_id: UUID | None = Field(
        default=None,
        description="UUID of default branch for operations (auto-set to first created branch)",
    )

    @property
    def messages(self):
        """Read-only view of conversations.items (Pile[Message])."""
        return self.conversations.items

    @property
    def branches(self):
        """Read-only view of conversations.progressions (Pile[Branch])."""
        return self.conversations.progressions

    @property
    def default_branch(self) -> Branch | None:
        """Get the default branch for operations when none specified."""
        if self.default_branch_id is None:
            return None
        try:
            return self.conversations.get_progression(self.default_branch_id)
        except (KeyError, ValueError, Exception):
            # Branch was removed (NotFoundError, KeyError, ValueError)
            # Clear stale reference and return None gracefully
            self.default_branch_id = None
            return None

    def set_default_branch(self, branch: Branch | UUID | str) -> None:
        """Set the default branch for operations.

        Args:
            branch: Branch instance, UUID, or name to set as default

        Raises:
            ValueError: If branch not found in session
        """
        if isinstance(branch, Branch):
            branch_id = branch.id
        elif isinstance(branch, UUID):
            branch_id = branch
        else:
            # Lookup by name
            resolved = self.conversations.get_progression(branch)
            branch_id = resolved.id

        # Verify branch exists
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found in session")

        self.default_branch_id = branch_id

    def create_branch(
        self,
        *,
        name: str | None = None,
        system_message: Message | UUID | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
        set_as_default: bool | None = None,
    ) -> Branch:
        """Create new branch for isolated conversation threads.

        Args:
            name: Optional branch name (auto-generated if None)
            system_message: Optional system message to set
            capabilities: Structured output schemas allowed
            resources: Backend service names allowed
            set_as_default: If True, set as default branch. If None (default),
                           auto-sets as default if this is the first branch.

        Returns:
            The created Branch
        """
        # Auto-set first branch as default if not specified
        is_first_branch = len(self.branches) == 0
        should_set_default = set_as_default if set_as_default is not None else is_first_branch

        branch_name = name or f"branch_{len(self.branches)}"
        branch = Branch(
            session_id=self.id,
            user=self.id,  # Branch user = session id
            name=branch_name,
            capabilities=capabilities or set(),
            resources=resources or set(),
        )

        if system_message is not None:
            if isinstance(system_message, Message):
                if system_message.id not in self.messages:
                    self.conversations.add_item(system_message)
                branch.set_system_message(system_message.id)
            else:
                branch.set_system_message(system_message)

        self.conversations.add_progression(branch)

        # Set as default if appropriate
        if should_set_default:
            self.default_branch_id = branch.id

        return branch

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        sender: SenderRecipient | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
    ) -> Branch:
        """Fork branch to create divergent conversation path.

        Creates a new branch with cloned messages for independent exploration.

        Args:
            branch: Source branch to fork from
            name: Name for forked branch (auto-generated if None)
            sender: Optional sender for cloned messages
            capabilities: Structured output schemas (None = copy from source)
            resources: Backend services (None = copy from source)

        Returns:
            New forked Branch
        """
        if isinstance(branch, (UUID, str)):
            branch = self.conversations.get_progression(branch)

        forked = Branch(
            session_id=self.id,
            user=self.id,  # Branch user = session id
            name=name or f"{branch.name}_fork",
            capabilities=capabilities if capabilities is not None else branch.capabilities.copy(),
            resources=resources if resources is not None else branch.resources.copy(),
        )

        forked_system_id = None
        for msg_id in branch:
            original_msg = self.messages[msg_id]
            cloned_msg = original_msg.clone(sender=sender)
            self.conversations.add_item(cloned_msg)
            forked.append(cloned_msg.id)

            if branch.system_message is not None and msg_id == branch.system_message:
                forked_system_id = cloned_msg.id

        if forked_system_id is not None:
            forked.system_message = forked_system_id

        forked.metadata["forked_from"] = {
            "branch_id": str(branch.id),
            "branch_name": branch.name,
            "created_at": branch.created_at.isoformat(),
            "message_count": len(branch),
        }

        self.conversations.add_progression(forked)
        return forked

    def set_branch_system(
        self,
        branch: Branch | UUID | str,
        system_message: Message | UUID,
    ) -> None:
        """Set or change system message for a branch."""
        if isinstance(branch, (UUID, str)):
            branch = self.conversations.get_progression(branch)

        if isinstance(system_message, Message):
            if system_message.id not in self.messages:
                self.conversations.add_item(system_message)
            msg_id = system_message.id
        else:
            msg_id = system_message

        branch.set_system_message(msg_id)

    def get_branch_system(self, branch: Branch | UUID | str) -> Message | None:
        """Get system message for a branch."""
        if isinstance(branch, Branch):
            branch = self.conversations.get_progression(branch.id)
        else:
            branch = self.conversations.get_progression(branch)

        if branch.system_message is not None:
            return self.messages[branch.system_message]
        return None

    def add_message(
        self,
        message: Message,
        *,
        branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
        is_system: bool = False,
    ) -> None:
        """Add message to session, optionally to specific branches.

        Args:
            message: Message to add
            branches: Branch(es) to add message to (optional)
            is_system: If True, set as system message for all branches
        """
        from .messages import SystemContent

        # Auto-detect system message from content type
        if isinstance(message.content, SystemContent):
            is_system = True

        # Normalize branches to list
        resolved_branches: list[Branch] = []
        if branches is not None:
            if not isinstance(branches, list):
                branches = [branches]
            for b in branches:
                if isinstance(b, (UUID, str)):
                    resolved_branches.append(self.conversations.get_progression(b))
                else:
                    resolved_branches.append(b)

        # Add message to flow
        if resolved_branches:
            if is_system:
                # For system messages, add to items only (not progressions)
                # then use set_system_message which handles ordering
                self.conversations.add_item(message)
                for branch in resolved_branches:
                    branch.set_system_message(message.id)
            else:
                self.conversations.add_item(message, progressions=resolved_branches)
        else:
            self.conversations.add_item(message)

    # =========================================================================
    # Unified Service Interface
    # =========================================================================

    async def request(
        self,
        service_name: str,
        *,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **kwargs,
    ) -> Calling:
        """Unified service request interface.

        Central entry point for all service interactions. Handles both
        LLM endpoints and tool execution through the same interface.

        Args:
            service_name: Name of registered service (LLM or Tool)
            poll_timeout: Max seconds to wait for completion (default: 10s)
            poll_interval: Seconds between status checks (default: 0.1s)
            **kwargs: Service-specific arguments
                - For LLMs: model, messages, temperature, etc.
                - For Tools: function arguments

        Returns:
            Calling with execution results:
                - calling.execution.status (EventStatus)
                - calling.execution.response.data (result data)
                - calling.execution.error (if failed)

        Example:
            LLM request:
                >>> calling = await session.request(
                ...     "openai",
                ...     model="gpt-4.1-mini",
                ...     messages=[{"role": "user", "content": "Hello"}],
                ... )
                >>> result = calling.execution.response.data

            Tool request:
                >>> calling = await session.request("calculator", a=5, b=3)
                >>> result = calling.execution.response.data  # 8
        """
        service = self.services.get(service_name)
        return await service.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **kwargs,
        )

    # =========================================================================
    # Unified Operation Interface
    # =========================================================================

    async def conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        *,
        imodel: Any = None,
        **parameters,
    ) -> Any:
        """Unified operation interface - returns invoked Operation.

        Central entry point for all operations (operate, react, communicate, generate).
        Creates an Operation node, binds it, invokes it, and returns it.

        Args:
            operation_type: Operation type ("operate", "react", "communicate", "generate")
            branch: Target branch (optional, uses default if None)
            imodel: iModel to use (optional, falls back to default_imodel)
            **parameters: Operation-specific parameters

        Returns:
            Operation: Invoked operation with:
                - op.status: EventStatus (COMPLETED, FAILED, etc.)
                - op.response: The operation result
                - op.execution: Full execution details

        Example:
            # With default_imodel set
            session = Session(default_imodel=my_model)
            branch = session.create_branch()

            # Structured output
            op = await session.conduct("operate", branch,
                instruction="Analyze this text",
                response_model=AnalysisResult,
            )
            result = op.response  # The AnalysisResult instance

            # ReAct with tools
            op = await session.conduct("react", branch,
                instruction="Find the answer",
                tools=[SearchTool, CalculatorTool],
            )
            print(op.status)  # EventStatus.COMPLETED

            # Simple chat
            op = await session.conduct("communicate", branch,
                instruction="Hello!",
            )
            print(op.response)  # "Hello! How can I help?"

        Raises:
            KeyError: If operation not registered
            ValueError: If no imodel provided and no default_imodel set
        """
        from lionpride.operations.node import Operation

        # Resolve imodel
        resolved_imodel = imodel or self.default_imodel
        if resolved_imodel is None:
            raise ValueError(
                f"Operation '{operation_type}' requires imodel. Either pass imodel= "
                "or set default_imodel on Session."
            )

        # Resolve branch
        if branch is None:
            # Use default_branch or create one
            if self.default_branch is not None:
                branch = self.default_branch
            else:
                # No default set - create one (which auto-sets as default)
                branch = self.create_branch(name="default")
        elif isinstance(branch, (UUID, str)):
            branch = self.conversations.get_progression(branch)

        # Build parameters with resolved imodel
        params = {"imodel": resolved_imodel, **parameters}

        # Create Operation node
        op = Operation(
            operation_type=operation_type,
            parameters=params,
        )

        # Bind to session/branch and invoke
        op.bind(self, branch)
        await op.invoke()

        return op

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
