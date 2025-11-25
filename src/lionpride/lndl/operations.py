# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
LNDL v2 Cognitive Operations - Real implementations using lionpride primitives.

This module provides:
- compress_messages: Summarize messages via model call using Session/iModel
- transform_observation: Process keep/drop_full/transform on action results
- cognitive_react: Full cognitive ReAct loop using Session, Branch, Message
- CognitivePermission: Rights control for cognitive operations
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lionpride.session.messages import (
    AssistantResponseContent,
    InstructionContent,
    Message,
    SystemContent,
)
from lionpride.session.messages.utils import prepare_messages_for_chat

from .cognitive import CognitiveOutput, CognitiveYield, execute_cognitive

if TYPE_CHECKING:
    from lionpride.core import Pile
    from lionpride.services.types import iModel
    from lionpride.session import Branch, Session


# =============================================================================
# Cognitive Permissions
# =============================================================================


@dataclass
class CognitivePermission:
    """Permissions for cognitive operations.

    Controls what cognitive operations the model is allowed to perform.
    These permissions define the "rights" the model has during execution.
    """

    # Context management permissions
    can_include: bool = True  # Can include specific messages
    can_compress: bool = True  # Can request message compression
    can_drop: bool = True  # Can request message dropping
    can_notice: bool = True  # Can create brief notices
    max_compress_range: int = 100  # Maximum messages in compress range

    # Action permissions
    allowed_actions: set[str] = field(default_factory=set)  # Empty = all allowed

    # Yield/continuation permissions
    can_yield: bool = True  # Can yield for continuation
    auto_approve_yields: bool = True  # Auto-approve yield requests
    max_yields: int = 10  # Maximum yields per session

    # Multi-agent permissions
    can_send: bool = False  # Can send to other agents
    allowed_recipients: set[str] = field(default_factory=set)


# =============================================================================
# Context Management Operations
# =============================================================================


async def compress_messages(
    session: Session,
    branch: Branch,
    start_idx: int = 0,
    end_idx: int | None = None,
    imodel: iModel | str | None = None,
    max_tokens: int = 500,
) -> Message:
    """Compress a range of messages into a summary using lionpride primitives.

    This is the actual implementation of the <compress> directive.
    Uses Session's iModel to generate a summary, returns a Message.

    Args:
        session: Session containing messages and services
        branch: Branch containing the messages to compress
        start_idx: Start index in branch (default 0)
        end_idx: End index in branch (default None = len(branch))
        imodel: iModel name (str) or instance to use for compression
        max_tokens: Maximum tokens in summary

    Returns:
        Message containing the compressed summary as SystemContent
    """
    # Get messages from branch
    branch_messages: Pile[Message] = session.messages[branch]

    # Slice the range
    end = end_idx if end_idx is not None else len(branch_messages)
    messages_to_compress = list(branch_messages)[start_idx:end]

    if not messages_to_compress:
        return Message(
            content=SystemContent(system_message="[No messages to compress]"),
            sender="system",
            recipient=str(session.id),
        )

    # Build compression prompt from messages
    messages_text = "\n".join(
        f"[{msg.role.value}]: {msg.content.rendered[:500]}" for msg in messages_to_compress
    )

    compression_instruction = f"""Summarize the following conversation history into a concise context summary.
Focus on key information, decisions made, and important context.
Keep it brief but preserve essential details.

CONVERSATION:
{messages_text}

SUMMARY (be concise):"""

    # Resolve iModel
    if imodel is None:
        # Try to get first available model from session
        model_names = session.services.list_names()
        if not model_names:
            raise ValueError("No models available in session services for compression")
        resolved_model = session.services.get(model_names[0])
    elif isinstance(imodel, str):
        resolved_model = session.services.get(imodel)
    else:
        resolved_model = imodel

    # Invoke model for compression
    calling = await resolved_model.invoke(
        messages=[{"role": "user", "content": compression_instruction}],
        max_tokens=max_tokens,
    )

    # Check execution status
    if calling.execution.status.value != "completed":
        error = calling.execution.error or f"status: {calling.execution.status}"
        raise RuntimeError(f"Compression failed: {error}")

    response = calling.execution.response
    summary_text = response.data if hasattr(response, "data") else str(response)

    # Create summary message
    return Message(
        content=SystemContent(
            system_message=f"[Compressed context from messages {start_idx}-{end}]\n{summary_text}"
        ),
        sender="system",
        recipient=str(session.id),
        metadata={
            "compression_range": (start_idx, end),
            "original_message_count": len(messages_to_compress),
        },
    )


def transform_observation(
    observation: Any,
    keep: str | None = None,
    drop_full: bool = False,
    transform: str | None = None,
) -> Any:
    """Transform an action observation based on yield parameters.

    This is a pure transformation function - no lionpride dependencies.

    Args:
        observation: Raw action result
        keep: What to keep (e.g., "top_3", "top_5", "summary", "first")
        drop_full: Whether to drop verbose output
        transform: Transformation to apply (e.g., "summarize", "extract_key")

    Returns:
        Transformed observation
    """
    if observation is None:
        return None

    result = observation

    # Handle keep parameter
    if keep:
        if keep.startswith("top_"):
            # Extract top N items if result is a list
            try:
                n = int(keep.split("_")[1])
                if isinstance(result, list):
                    result = result[:n]
                elif isinstance(result, str):
                    # For strings, take first N lines
                    lines = result.split("\n")
                    result = "\n".join(lines[:n])
            except (ValueError, IndexError):
                pass
        elif keep == "summary":
            # For summary, take first portion
            if isinstance(result, str) and len(result) > 500:
                result = result[:500] + "..."
        elif keep == "first" and isinstance(result, list) and result:
            result = result[0]

    # Handle drop_full - return minimal representation
    if drop_full:
        if isinstance(result, str) and len(result) > 200:
            result = result[:200] + "... [truncated]"
        elif isinstance(result, list):
            result = f"[{len(result)} items]"
        elif isinstance(result, dict):
            result = f"{{keys: {list(result.keys())}}}"

    # Handle transform parameter
    if transform and transform == "extract_key" and isinstance(result, dict):
        # Try to extract main value from dict
        for key in ("result", "output", "data", "content", "text"):
            if key in result:
                result = result[key]
                break

    return result


# =============================================================================
# Streaming Support
# =============================================================================


async def stream_cognitive(
    response_stream: AsyncIterator[str],
    on_yield: Any | None = None,
) -> AsyncIterator[str | CognitiveYield | CognitiveOutput]:
    """Stream LNDL v2 response with real-time cognitive processing.

    Processes streaming LLM response and yields cognitive events as they occur.

    Args:
        response_stream: Async iterator of response chunks
        on_yield: Optional callback for yield events

    Yields:
        str chunks for incremental output
        CognitiveYield when yield point detected
        CognitiveOutput when parsing complete
    """
    buffer = ""

    async for chunk in response_stream:
        buffer += chunk
        yield chunk

        # Check for yield markers in buffered content
        if "<yield" in buffer and "/>" in buffer:
            # Parse what we have so far
            try:
                gen = execute_cognitive(buffer)
                async for result in gen:
                    if isinstance(result, CognitiveYield):
                        if on_yield:
                            on_yield(result)
                        yield result
            except Exception:
                # Not enough content yet, continue buffering
                pass

    # Final parse of complete response
    if buffer:
        async for result in execute_cognitive(buffer):
            yield result  # type: ignore[misc]


# =============================================================================
# Cognitive React Loop
# =============================================================================


class CognitiveReactResult(BaseModel):
    """Result from cognitive ReAct execution."""

    phases: list[dict[str, Any]] = Field(default_factory=list)
    final_output: dict[str, Any] | None = None
    observations: dict[str, Any] = Field(default_factory=dict)
    completed: bool = False
    reason_stopped: str = ""


async def cognitive_react(
    session: Session,
    branch: Branch | str,
    instruction: str,
    imodel: iModel | str,
    action_executor: Any | None = None,
    permissions: CognitivePermission | None = None,
    max_iterations: int = 5,
    verbose: bool = False,
) -> CognitiveReactResult:
    """Execute a cognitive ReAct loop using lionpride primitives.

    This is the main entry point for LNDL v2 cognitive programming.
    It handles the full cycle of:
    1. LLM generates LNDL v2 response
    2. Parse and detect yields
    3. Execute actions at yield points
    4. Inject observations as messages
    5. Continue until final output

    Args:
        session: lionpride Session with services and conversations
        branch: Branch name or instance for conversation state
        instruction: User instruction
        imodel: iModel name (str) or instance
        action_executor: Function to execute actions (name, call, observations) -> result
        permissions: Cognitive permissions (default: all allowed)
        max_iterations: Maximum continuation iterations
        verbose: Print debug info

    Returns:
        CognitiveReactResult with execution history and final output
    """
    from lionpride.lndl.prompt import get_lndl_system_prompt

    if permissions is None:
        permissions = CognitivePermission()

    result = CognitiveReactResult()

    # Resolve branch
    if isinstance(branch, str):
        resolved_branch = session.conversations.get_progression(branch)
    else:
        resolved_branch = branch

    # Resolve iModel
    resolved_model = session.services.get(imodel) if isinstance(imodel, str) else imodel

    # Build LNDL v2 system prompt
    lndl_prompt = get_lndl_system_prompt()
    lndl_v2_additions = """

## LNDL v2 Cognitive Extensions

### Context Engineering
<context>
  <include msg="msg_id"/>           - Include specific message
  <compress msgs="0..50" to="ctx"/> - Compress messages to symbol
  <drop msg="verbose_output"/>      - Drop message from context
  <notice msg="tools" brief="..."/> - Brief notice
</context>

### Continuation Control
<yield for="action" reason="why" keep="top_5" drop_full="true"/>

### Multi-Agent (if permitted)
<send to="agent" type="MsgType">
  <include msg="data"/>
</send>

Use these to engineer your cognitive context efficiently."""

    system_message_text = lndl_prompt + lndl_v2_additions

    # Create system message
    system_msg = Message(
        content=SystemContent(system_message=system_message_text),
        sender="system",
        recipient=str(session.id),
    )

    # Create initial instruction message
    user_msg = Message(
        content=InstructionContent(instruction=instruction),
        sender=str(resolved_branch.user),
        recipient=str(session.id),
    )

    # Add to session/branch
    session.add_message(system_msg, branches=resolved_branch)
    session.add_message(user_msg, branches=resolved_branch)

    yield_count = 0

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Cognitive Iteration {iteration + 1}/{max_iterations} ---")

        # Prepare messages for API call using lionpride utilities
        branch_messages: Pile[Message] = session.messages[resolved_branch]
        chat_messages = list(
            prepare_messages_for_chat(
                messages=branch_messages,
                progression=resolved_branch,
                to_chat=True,
            )
        )

        # Call the model via lionpride services
        calling = await resolved_model.invoke(messages=chat_messages)

        if calling.execution.status.value != "completed":
            error = calling.execution.error or f"status: {calling.execution.status}"
            result.reason_stopped = f"Model call failed: {error}"
            return result

        response = calling.execution.response
        response_text = response.data if hasattr(response, "data") else str(response)

        if verbose:
            print(f"Response:\n{response_text[:500]}...")

        # Record phase
        phase: dict[str, Any] = {
            "iteration": iteration + 1,
            "response": response_text,
            "yields": [],
            "observations": {},
        }

        # Add assistant response to conversation
        assistant_msg = Message(
            content=AssistantResponseContent(assistant_response=response_text),
            sender=resolved_model.name,
            recipient=str(resolved_branch.user),
            metadata={"raw_response": getattr(response, "raw_response", {})},
        )
        session.add_message(assistant_msg, branches=resolved_branch)

        # Execute cognitive loop
        gen = execute_cognitive(response_text)

        try:
            async for item in gen:
                if isinstance(item, CognitiveYield):
                    yield_count += 1

                    # Check permissions
                    if not permissions.can_yield:
                        result.reason_stopped = "Yield not permitted"
                        result.phases.append(phase)
                        return result

                    if yield_count > permissions.max_yields:
                        result.reason_stopped = f"Max yields ({permissions.max_yields}) exceeded"
                        result.phases.append(phase)
                        return result

                    if verbose:
                        print(f"Yield: for={item.for_ref}, reason={item.reason}")

                    # Execute the action
                    observation = None
                    if action_executor and item.action_call:
                        try:
                            observation = await action_executor(
                                item.for_ref,
                                item.action_call,
                                result.observations,
                            )
                        except Exception as e:
                            observation = f"Error: {e}"

                    # Transform observation based on yield params
                    if observation:
                        observation = transform_observation(
                            observation,
                            keep=item.keep,
                            drop_full=item.drop_full,
                            transform=item.transform,
                        )

                    if verbose:
                        print(f"Observation: {str(observation)[:200]}...")

                    # Store observation
                    if item.for_ref:
                        result.observations[item.for_ref] = observation
                        phase["yields"].append(item.for_ref)
                        phase["observations"][item.for_ref] = observation

                    # Send observation back to generator
                    item = await gen.asend(observation)

                    # Continue processing if we got another result
                    if isinstance(item, CognitiveOutput):
                        result.final_output = item.fields
                        result.completed = True
                        result.reason_stopped = "Completed"
                        result.phases.append(phase)
                        return result

                elif isinstance(item, CognitiveOutput):
                    result.final_output = item.fields
                    result.completed = True
                    result.reason_stopped = "Completed"
                    result.phases.append(phase)
                    return result

        except StopAsyncIteration:
            pass

        result.phases.append(phase)

        # Add observation feedback as user message for next iteration
        if phase["observations"]:
            obs_text = "\n".join(
                f"Observation for {k}: {v}" for k, v in phase["observations"].items()
            )
            feedback_msg = Message(
                content=InstructionContent(
                    instruction=f"Here are the results:\n{obs_text}\n\nPlease continue and provide your final answer."
                ),
                sender=str(resolved_branch.user),
                recipient=str(session.id),
            )
            session.add_message(feedback_msg, branches=resolved_branch)
        else:
            # No yields, check if we should continue
            break

    result.reason_stopped = f"Max iterations ({max_iterations}) reached"
    return result


__all__ = (
    "CognitivePermission",
    "CognitiveReactResult",
    "cognitive_react",
    "compress_messages",
    "stream_cognitive",
    "transform_observation",
)
