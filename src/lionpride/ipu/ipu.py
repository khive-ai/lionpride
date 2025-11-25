# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""IPU (Intelligence Processing Unit) - Execution boundary with validation.

Resurrected validator pattern from lionagi v0.2.2.
The core insight: "If we cannot validate something, we cannot assume its structure, nor usefulness"

IPU provides:
1. Session registry (multi-tenant context management)
2. Validation layer (rules + Operable specs)
3. Execution context (operations run here, not in Session)
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import Field

from lionpride.core import Element
from lionpride.errors import NotFoundError
from lionpride.ipu.operation_spec import OperationSpec
from lionpride.rules import ValidationError
from lionpride.rules.validator import Validator

if TYPE_CHECKING:
    from lionpride.operations import Operation
    from lionpride.session import Session

__all__ = ("IPU", "get_current_ipu")

# Thread-local IPU context
_current_ipu: contextvars.ContextVar[IPU | None] = contextvars.ContextVar(
    "current_ipu", default=None
)


def get_current_ipu() -> IPU:
    """Get current IPU from context.

    Returns:
        Current IPU instance

    Raises:
        RuntimeError: If no IPU in context
    """
    ipu = _current_ipu.get()
    if ipu is None:
        raise RuntimeError("No IPU in current context")
    return ipu


class IPU(Element):
    """Intelligence Processing Unit - Validated execution context.

    The IPU pattern: validation → structure → usefulness

    Responsibilities:
    1. Manage session registry (multi-tenant)
    2. Pre-validate operation inputs (structural validation)
    3. Execute operations in validated context
    4. Post-validate operation outputs (LNDL OUT{} blocks)
    5. Guarantee all data is valid, structured, usable

    Usage:
        ipu = IPU()
        session = Session()
        ipu.register_session(session)

        # Create operation via session.conduct()
        op = await session.conduct(branch, "communicate", ipu=ipu, instruction="...")

        # IPU executes with validation
        result = await ipu.execute(op)
    """

    session_registry: dict[UUID, Any] = Field(
        default_factory=dict, description="Registered sessions"
    )
    validator: Any = Field(default=None, description="Validator engine")
    operation_specs: dict[str, Any] = Field(
        default_factory=dict, description="Operation specs for validation"
    )

    def __init__(self, validator: Validator | None = None, **kwargs):
        """Initialize IPU.

        Args:
            validator: Custom validator (uses default if None)
            **kwargs: Additional Element kwargs
        """
        super().__init__(**kwargs)
        # Set validator after super().__init__ to avoid Element signature issues
        object.__setattr__(self, "validator", validator or Validator())

    def register_session(self, session: Session) -> None:
        """Register session for operation execution.

        Args:
            session: Session to register

        Raises:
            ValueError: If session already registered
        """
        if session.id in self.session_registry:
            raise ValueError(f"Session {session.id} already registered")
        self.session_registry[session.id] = session

    def unregister_session(self, session_id: UUID) -> Session:
        """Unregister session.

        Args:
            session_id: Session UUID

        Returns:
            Unregistered session

        Raises:
            NotFoundError: If session not found
        """
        if session_id not in self.session_registry:
            raise NotFoundError(f"Session {session_id} not in IPU registry")
        return self.session_registry.pop(session_id)

    def get_session(self, session_id: UUID) -> Session:
        """Get session by ID (O(1) lookup).

        Args:
            session_id: Session UUID

        Returns:
            Session instance

        Raises:
            NotFoundError: If session not found
        """
        if session_id not in self.session_registry:
            raise NotFoundError(f"Session {session_id} not in IPU registry")
        return self.session_registry[session_id]

    def register_operation_spec(self, operation_name: str, spec: OperationSpec) -> None:
        """Register operation spec for validation.

        Args:
            operation_name: Operation name (e.g., "communicate")
            spec: OperationSpec defining input/output Operables
        """
        self.operation_specs[operation_name] = spec

    def get_operation_spec(self, operation_name: str) -> OperationSpec | None:
        """Get operation spec by name.

        Args:
            operation_name: Operation name

        Returns:
            OperationSpec if registered, None otherwise
        """
        return self.operation_specs.get(operation_name)

    async def execute(self, operation: Operation) -> Any:
        """Execute operation with pre/post validation.

        Flow:
        1. Get session and branch from registry
        2. PRE-VALIDATE: Check operation.parameters against input Operable
        3. EXECUTE: Run operation function
        4. POST-VALIDATE: Check output against output Operable
        5. RETURN: Guaranteed valid, structured result

        Args:
            operation: Operation to execute

        Returns:
            Validated operation result

        Raises:
            NotFoundError: If session/branch/operation not found
            ValidationError: If pre/post validation fails
        """
        # Set IPU as current context
        token = _current_ipu.set(self)

        try:
            # 1. Get execution context
            session = self.get_session(operation.session_id)
            branch = session.get_branch(operation.branch_id)

            # Check operation exists
            if session.operations is None or operation.operation_type not in session.operations:
                raise NotFoundError(
                    f"Operation '{operation.operation_type}' not registered in session"
                )

            # 2. PRE-VALIDATE: Check inputs
            operation_spec = self.get_operation_spec(operation.operation_type)
            if operation_spec:
                # Validate field names
                operation_spec.validate_input_names(operation.parameters)

                # Validate types
                try:
                    validated_params = await self.validator.validate(
                        data=operation.parameters,
                        operable=operation_spec.input_operable,
                        auto_fix=True,
                    )
                except ValidationError as e:
                    raise ValidationError(
                        f"Operation '{operation.operation_type}' input validation failed: {e}"
                    ) from e
            else:
                # No spec registered - pass through parameters unchanged
                validated_params = operation.parameters

            # 3. EXECUTE: Run operation function
            if session.operations is None:
                raise NotFoundError("No operations registry in session")
            operation_func = session.operations.get(operation.operation_type)
            result = await operation_func(session, branch, validated_params)

            # 4. POST-VALIDATE: Check outputs (if spec registered)
            if operation_spec:
                # Result should be dict for validation
                result_dict = result if isinstance(result, dict) else {"output": result}

                # Validate field names
                operation_spec.validate_output_names(result_dict)

                # Validate types
                try:
                    validated_output = await self.validator.validate(
                        data=result_dict,
                        operable=operation_spec.output_operable,
                        auto_fix=True,
                    )
                except ValidationError as e:
                    raise ValidationError(
                        f"Operation '{operation.operation_type}' output validation failed: {e}"
                    ) from e

                return validated_output
            else:
                # No spec registered - pass through unchanged
                return result

        finally:
            # Clear IPU context
            _current_ipu.reset(token)

    async def queue(self, operation: Operation) -> None:
        """Queue operation for execution (future: async queue).

        For now, immediately execute. Future: add async queue with worker pool.

        Args:
            operation: Operation to queue
        """
        # TODO: Implement async queue
        # For now, just execute immediately
        await self.execute(operation)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"IPU(sessions={len(self.session_registry)}, "
            f"operation_specs={len(self.operation_specs)}, "
            f"validator={self.validator}, "
            f"id={str(self.id)[:8]})"
        )
