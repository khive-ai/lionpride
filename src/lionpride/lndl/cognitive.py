# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
LNDL v2 Cognitive Runtime - Execution engine for cognitive programming.

This module implements the execution model for LNDL v2, enabling LLMs to:
- Engineer their own context (select/compress/drop messages)
- Invoke capabilities with explicit permissions
- Yield for continuation/approval
- Coordinate with other agents

The key insight: ReAct is an execution model, not a grammar change.
"""

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .ast import ContextBlock, ContextDirective, Program
from .lexer import Lexer
from .parser import Parser


class CognitivePhase(Enum):
    """Current phase of cognitive execution."""

    PARSING = auto()  # Parsing LNDL response
    YIELDING = auto()  # Waiting for action execution
    SENDING = auto()  # Waiting for agent response
    COMPLETE = auto()  # Final output available


@dataclass
class CognitiveState:
    """State of a cognitive execution.

    Tracks:
    - Declared variables and actions
    - Context directives
    - Yield/continuation points
    - Action observations (results)
    """

    # Declared variables (alias -> content)
    lvars: dict[str, str] = field(default_factory=dict)

    # Declared actions (alias -> call string)
    lacts: dict[str, str] = field(default_factory=dict)

    # Action observations (alias -> result)
    observations: dict[str, Any] = field(default_factory=dict)

    # Context directives to apply
    context_directives: list[ContextDirective] = field(default_factory=list)

    # Current phase
    phase: CognitivePhase = CognitivePhase.PARSING


@dataclass
class CognitiveYield:
    """Yield point in cognitive execution.

    When a <yield> is encountered, execution pauses and returns this object.
    The caller executes the referenced action and resumes with the result.
    """

    for_ref: str | None  # Action to execute
    reason: str | None  # Why continuation needed
    keep: str | None  # What to keep from result
    drop_full: bool  # Whether to drop verbose result
    transform: str | None  # Transformation to apply

    # Action details (populated from lact declaration)
    action_call: str | None = None  # Raw call string

    # State at yield point
    state: CognitiveState | None = None


@dataclass
class CognitiveSend:
    """Multi-agent message to send.

    When a <send> is encountered, this describes the message
    to route to another agent.
    """

    to: str  # Target agent
    msg_type: str | None  # Message type for validation
    timeout: str | None  # Timeout duration
    content: list[ContextDirective] | None  # What to send

    # State at send point
    state: CognitiveState | None = None


@dataclass
class CognitiveOutput:
    """Final output from cognitive execution."""

    # Output fields (field_name -> value or list of refs)
    fields: dict[str, list[str] | str | int | float | bool]

    # Complete state
    state: CognitiveState

    # Context engineering applied
    context: ContextBlock | None = None


# Type alias for action executor function
ActionExecutor = Callable[[str, str, dict[str, Any]], Any]


async def execute_cognitive(
    response: str,
    action_executor: ActionExecutor | None = None,
) -> AsyncGenerator[CognitiveYield | CognitiveSend | CognitiveOutput, Any]:
    """Execute LNDL v2 response as a cognitive program.

    This is an async generator that yields at continuation points:
    - CognitiveYield: Action needs execution
    - CognitiveSend: Message needs routing
    - CognitiveOutput: Final output available

    Args:
        response: LNDL v2 response text
        action_executor: Optional function to execute actions.
            Signature: (action_name, call_string, current_observations) -> result

    Yields:
        CognitiveYield for actions needing execution
        CognitiveSend for agent messages needing routing
        CognitiveOutput as final result

    Example:
        async for result in execute_cognitive(lndl_response):
            if isinstance(result, CognitiveYield):
                # Execute the action
                observation = await execute_tool(result.action_call)
                # Send observation back (the generator receives it)
                result = await gen.asend(observation)
            elif isinstance(result, CognitiveOutput):
                print(f"Final output: {result.fields}")
    """
    # Parse the LNDL response
    lexer = Lexer(response)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=response)
    program = parser.parse()

    # Initialize state
    state = CognitiveState()

    # Process context block if present
    if program.context:
        state.context_directives = program.context.directives

    # Collect lvars
    for lvar in program.lvars:
        state.lvars[lvar.alias] = lvar.content

    # Collect lacts
    for lact in program.lacts:
        state.lacts[lact.alias] = lact.call

    # Process yields
    if program.yields:
        for yield_stmt in program.yields:
            # Get action details
            action_call = None
            if yield_stmt.for_ref and yield_stmt.for_ref in state.lacts:
                action_call = state.lacts[yield_stmt.for_ref]

            # Yield control
            state.phase = CognitivePhase.YIELDING
            observation = yield CognitiveYield(
                for_ref=yield_stmt.for_ref,
                reason=yield_stmt.reason,
                keep=yield_stmt.keep,
                drop_full=yield_stmt.drop_full,
                transform=yield_stmt.transform,
                action_call=action_call,
                state=state,
            )

            # Store observation if provided
            if observation is not None and yield_stmt.for_ref:
                state.observations[yield_stmt.for_ref] = observation

    # Process sends
    if program.sends:
        for send_stmt in program.sends:
            state.phase = CognitivePhase.SENDING
            response_from_agent = yield CognitiveSend(
                to=send_stmt.to,
                msg_type=send_stmt.msg_type,
                timeout=send_stmt.timeout,
                content=send_stmt.content,
                state=state,
            )

            # Store agent response if provided
            if response_from_agent is not None:
                state.observations[f"send_{send_stmt.to}"] = response_from_agent

    # Final output
    state.phase = CognitivePhase.COMPLETE
    if program.out_block:
        yield CognitiveOutput(
            fields=program.out_block.fields,
            state=state,
            context=program.context,
        )


def parse_cognitive(response: str) -> tuple[Program, CognitiveState]:
    """Parse LNDL v2 response and extract cognitive state.

    Synchronous parsing without execution - useful for analysis.

    Args:
        response: LNDL v2 response text

    Returns:
        Tuple of (Program AST, CognitiveState)
    """
    lexer = Lexer(response)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=response)
    program = parser.parse()

    state = CognitiveState()

    if program.context:
        state.context_directives = program.context.directives

    for lvar in program.lvars:
        state.lvars[lvar.alias] = lvar.content

    for lact in program.lacts:
        state.lacts[lact.alias] = lact.call

    return program, state


__all__ = (
    "ActionExecutor",
    "CognitiveOutput",
    "CognitivePhase",
    "CognitiveSend",
    "CognitiveState",
    "CognitiveYield",
    "execute_cognitive",
    "parse_cognitive",
)
