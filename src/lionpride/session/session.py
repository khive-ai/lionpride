# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr

from lionpride.core import Element, Flow, Progression
from lionpride.services import ServiceRegistry

from .messages import Message, SenderRecipient

if TYPE_CHECKING:
    from lionpride.services.types import Calling, iModel

__all__ = ("Branch", "Session")


class Branch(Progression):
    """Named progression of messages within a session.

    Attributes:
        session_id: Parent Session UUID
        user: User identifier
        system_message: System message UUID (always at order[0] if set)
        capabilities: Service capability names available to this branch

    Note:
        Branch holds an optional Session reference for convenience methods
        (generate, operate, react). This reference is set automatically when
        creating branches via Session.create_branch() or Session.fork().
    """

    session_id: UUID = Field(..., description="Parent Session UUID")
    user: str | UUID | None = Field(default=None, description="User identifier")
    system_message: UUID | None = Field(default=None, description="System message UUID")
    capabilities: set[str] = Field(default_factory=set, description="Service capabilities")

    # Private session reference for convenience methods
    _session: Any = PrivateAttr(default=None)

    def _bind_session(self, session: Session) -> Branch:
        """Bind session reference for convenience methods.

        Called automatically by Session.create_branch() and Session.fork().
        """
        self._session = session
        return self

    def _require_session(self) -> Session:
        """Get bound session or raise error."""
        if self._session is None:
            raise RuntimeError(
                "Branch not bound to session. Use session.create_branch() "
                "or branch._bind_session(session) first."
            )
        return self._session

    def set_system_message(self, message_id: UUID) -> None:
        """Set system message, ensuring it's at order[0]."""
        old_system = self.system_message
        self.system_message = message_id

        if old_system is not None and len(self) > 0:
            self[0] = message_id
        else:
            self.insert(0, message_id)

    # =========================================================================
    # Convenience methods (require bound session)
    # =========================================================================

    async def generate(
        self,
        *,
        imodel: iModel,
        return_as: str = "text",
        **model_kwargs,
    ) -> Any:
        """Stateless text generation - thin wrapper around operations.generate.

        Args:
            imodel: iModel interface to use
            return_as: "text" | "raw" | "message" (default: "text")
            **model_kwargs: Passed to imodel.invoke() (messages, model, etc.)

        Returns:
            Based on return_as: text string, raw dict, or Message
        """
        from lionpride.operations.operate.generate import generate as generate_op

        if imodel is None:
            raise ValueError("generate requires 'imodel' parameter")

        session = self._require_session()
        return await generate_op(
            session,
            self,
            {"imodel": imodel.name, "return_as": return_as, **model_kwargs},
        )

    async def communicate(
        self,
        instruction: str,
        *,
        imodel: iModel,
        response_model: type[BaseModel] | None = None,
        context: dict[str, Any] | None = None,
        images: list[str] | None = None,
        image_detail: str | None = None,
        return_as: str = "text",
        strict_validation: bool = False,
        fuzzy_parse: bool = True,
        **model_kwargs,
    ) -> Any:
        """Stateful chat - persists messages, optional structured output.

        Args:
            instruction: The instruction/prompt text
            imodel: iModel interface to use
            response_model: Optional Pydantic model for structured output
            context: Optional context dict
            images: Optional image URLs
            image_detail: Optional image detail level
            return_as: "text" | "raw" | "message" | "model" (default: "text")
            strict_validation: If True, raise on validation failure
            fuzzy_parse: Enable fuzzy JSON parsing (default: True)
            **model_kwargs: Additional model parameters

        Returns:
            Based on return_as: text, raw dict, Message, or validated model
        """
        from lionpride.operations.communicate import communicate as communicate_op

        if imodel is None:
            raise ValueError("communicate requires 'imodel' parameter")

        session = self._require_session()
        return await communicate_op(
            session,
            self,
            {
                "instruction": instruction,
                "imodel": imodel.name,
                "response_model": response_model,
                "context": context,
                "images": images,
                "image_detail": image_detail,
                "return_as": return_as,
                "strict_validation": strict_validation,
                "fuzzy_parse": fuzzy_parse,
                **model_kwargs,
            },
        )

    async def operate(
        self,
        instruction: str,
        *,
        imodel: iModel,
        response_model: type[BaseModel] | None = None,
        operative: Any = None,
        context: dict[str, Any] | None = None,
        tools: bool = False,
        actions: bool = False,
        reason: bool = False,
        use_lndl: bool = False,
        lndl_threshold: float = 0.85,
        images: list[str] | None = None,
        image_detail: str | None = None,
        return_message: bool = False,
        **model_kwargs,
    ) -> Any:
        """Operate with structured output (Pydantic model or LNDL).

        Args:
            instruction: The instruction/prompt text
            imodel: iModel interface to use
            response_model: Pydantic BaseModel for structured output
            operative: Optional Operative instance
            context: Optional context dict
            tools: Include tool schemas in prompt
            actions: Enable tool execution
            reason: Include reasoning field in output
            use_lndl: Use LNDL validation (fuzzy fallback)
            lndl_threshold: Confidence threshold for LNDL (default 0.85)
            images: Optional image URLs
            image_detail: Optional image detail level
            return_message: If True, return (result, Message) tuple
            **model_kwargs: Additional model parameters (including model_name)

        Returns:
            Parsed response model, or (result, Message) if return_message=True
        """
        from lionpride.operations.operate.factory import operate as operate_op

        session = self._require_session()
        return await operate_op(
            session,
            self,
            {
                "instruction": instruction,
                "imodel": imodel,
                "response_model": response_model,
                "operative": operative,
                "context": context,
                "tools": tools,
                "actions": actions,
                "reason": reason,
                "use_lndl": use_lndl,
                "lndl_threshold": lndl_threshold,
                "images": images,
                "image_detail": image_detail,
                "return_message": return_message,
                "model_kwargs": model_kwargs,
            },
        )

    async def react(
        self,
        instruction: str,
        *,
        imodel: iModel,
        tools: list[Any],
        response_model: type[BaseModel] | None = None,
        context: dict[str, Any] | None = None,
        max_steps: int = 5,
        reason: bool = True,
        use_lndl: bool = False,
        lndl_threshold: float = 0.85,
        verbose: bool = False,
        **model_kwargs,
    ) -> Any:
        """ReAct: Multi-step reasoning with tool calling.

        Implements the Reason-Action loop:
        1. LLM reasons about task and decides on action
        2. If action requested, execute tool(s)
        3. Feed results back to LLM
        4. Repeat until LLM provides final answer or max_steps reached

        Args:
            instruction: The instruction/prompt text
            imodel: iModel interface to use
            tools: List of tools available for this task
            response_model: Optional Pydantic model for final structured output
            context: Optional context dict
            max_steps: Maximum reasoning steps (default 5)
            reason: Include reasoning in each step (default True)
            use_lndl: Use LNDL validation
            lndl_threshold: Confidence threshold for LNDL
            verbose: Print step-by-step execution
            **model_kwargs: Additional model parameters (including model_name)

        Returns:
            Final response (structured if response_model provided)
        """
        from lionpride.operations.operate.react import react as react_op

        session = self._require_session()
        return await react_op(
            session,
            self,
            {
                "instruction": instruction,
                "imodel": imodel,
                "tools": tools,
                "response_model": response_model,
                "context": context,
                "max_steps": max_steps,
                "reason": reason,
                "use_lndl": use_lndl,
                "lndl_threshold": lndl_threshold,
                "verbose": verbose,
                "model_kwargs": model_kwargs,
            },
        )

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        bound_str = " bound" if self._session is not None else ""
        return f"Branch(messages={len(self)}, session={self.session_id}{name_str}{bound_str})"


class Session(Element):
    """Central storage for messages, branches, and services.

    Attributes:
        user: User identifier
        conversations: Flow[Message, Branch] for message storage and branch progressions
        services: ServiceRegistry for models and tools
        messages: Read-only view of conversations.items (Pile[Message])
        branches: Read-only view of conversations.progressions (Pile[Branch])
    """

    user: str | None = Field(default=None, description="User identifier")
    conversations: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message),
        description="Message flow with branches",
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
        system_message: Message | UUID | None = None,
        capabilities: set[str] | None = None,
    ) -> Branch:
        """Create new branch for isolated conversation threads.

        The branch is automatically bound to this session, enabling
        convenience methods like branch.generate(), branch.operate(), etc.
        """
        branch_name = name or f"branch_{len(self.branches)}"
        branch = Branch(
            session_id=self.id,
            user=self.id,
            name=branch_name,
            capabilities=capabilities or set(),
        )

        # Auto-bind session reference for convenience methods
        branch._bind_session(self)

        if system_message is not None:
            if isinstance(system_message, Message):
                if system_message.id not in self.messages:
                    self.conversations.add_item(system_message)
                branch.set_system_message(system_message.id)
            else:
                branch.set_system_message(system_message)

        self.conversations.add_progression(branch)
        return branch

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        sender: SenderRecipient | None = None,
        capabilities: set[str] | None = None,
    ) -> Branch:
        """Fork branch to create divergent conversation path.

        Creates a new branch with cloned messages for independent exploration.
        The forked branch is automatically bound to this session.
        """
        if isinstance(branch, (UUID, str)):
            branch = self.conversations.get_progression(branch)

        forked = Branch(
            session_id=self.id,
            user=branch.user,
            name=name or f"{branch.name}_fork",
            capabilities=capabilities or branch.capabilities.copy(),
        )

        # Auto-bind session reference for convenience methods
        forked._bind_session(self)

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

    def __repr__(self) -> str:
        return (
            f"Session(messages={len(self.messages)}, "
            f"branches={len(self.branches)}, "
            f"services={len(self.services)})"
        )
