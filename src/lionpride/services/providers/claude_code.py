# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from lionpride.services.third_party.claude_code import (
    ClaudeChunk,
    ClaudeCodeRequest,
    ClaudeSession,
    stream_claude_code_cli,
)
from lionpride.services.types import NormalizedResponse

from ..types.endpoint import Endpoint, EndpointConfig

__all__ = (
    "ClaudeCodeEndpoint",
    "create_claude_code_config",
)


def create_claude_code_config(
    name: str | None = None,
) -> dict:
    """Factory for Claude Code CLI endpoint config.

    Args:
        name: Config name (default: "claude_code_cli")

    Returns:
        Config dict for ClaudeCodeEndpoint
    """
    return {
        "name": name or "claude_code_cli",
        "provider": "claude_code",
        "base_url": "internal",
        "endpoint": "query_cli",
        "api_key": "dummy-key",
        "request_options": ClaudeCodeRequest,
        "timeout": 3600,  # 1 hour max (EndpointConfig limit)
    }


class ClaudeCodeEndpoint(Endpoint):
    """Claude Code CLI endpoint for local AI agent execution.

    Usage:
        endpoint = ClaudeCodeEndpoint()
        response = await endpoint.call({
            "messages": [{"role": "user", "content": "List files"}]
        })
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with Claude Code config."""
        if config is None:
            config = create_claude_code_config()
        elif isinstance(config, EndpointConfig):
            config = config.model_dump()
        if not isinstance(config, dict):
            raise ValueError("Provided config must be a dict or EndpointConfig instance")

        super().__init__(config=config, circuit_breaker=circuit_breaker, **kwargs)

    def create_payload(
        self, request: dict | BaseModel, extra_headers: dict | None = None, **kwargs
    ) -> tuple[dict, dict]:
        """Create payload for Claude Code CLI.

        Args:
            request: Request dict with 'messages' or ClaudeCodeRequest
            extra_headers: Ignored (CLI doesn't use HTTP headers)
            **kwargs: Additional arguments merged into request

        Returns:
            (payload_dict, headers_dict) - headers always empty for CLI
        """
        from lionpride.ln import to_dict

        # Convert request to dict if BaseModel
        request_dict = to_dict(request) if isinstance(request, BaseModel) else request

        # Merge config kwargs, request, and call kwargs (ignore extra_headers for CLI)
        req_dict = {**self.config.kwargs, **request_dict, **kwargs}

        # Extract messages (required)
        messages = req_dict.pop("messages", None)
        if not messages:
            raise ValueError(
                f"'messages' required for Claude Code endpoint. Got keys: {list(req_dict.keys())}"
            )

        # Create ClaudeCodeRequest object
        req_obj = ClaudeCodeRequest(messages=messages, **req_dict)

        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[ClaudeChunk | dict | ClaudeSession]:
        """Stream Claude Code CLI response chunks.

        Yields:
            ClaudeChunk, dict (system messages), or ClaudeSession (final)
        """
        payload, _ = self.create_payload(request, **kwargs)
        request_obj: ClaudeCodeRequest = payload["request"]

        async for chunk in stream_claude_code_cli(request_obj):
            yield chunk

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ) -> dict:
        """Execute Claude Code CLI and return raw result chunk + session data.

        Returns:
            Dict with:
            - raw_result: Raw "result" chunk from Claude Code CLI
            - session: Organized session data from ClaudeSession
        """
        from lionpride.ln import to_dict

        request: ClaudeCodeRequest = payload["request"]
        session = ClaudeSession()
        system: dict | None = None

        # 1. Stream the Claude Code response
        async for chunk in stream_claude_code_cli(request, session, **kwargs):
            if isinstance(chunk, dict):
                if chunk.get("type") == "done":
                    break
                system = chunk

        # 2. Auto-finish if requested and not already finished
        if request.auto_finish and not isinstance(
            session.chunks[-1] if session.chunks else None, ClaudeSession
        ):
            req2 = request.model_copy(deep=True)
            req2.prompt = "Please provide the final result message only"
            req2.max_turns = 1
            req2.continue_conversation = True
            if system:
                req2.resume = system.get("session_id")

            async for chunk in stream_claude_code_cli(req2, session, **kwargs):
                if isinstance(chunk, ClaudeSession):
                    break

        # 3. Combine text results
        texts = []
        for chunk in session.chunks:
            if chunk.text is not None:
                texts.append(chunk.text)

        texts.append(session.result)
        session.result = "\n".join(texts)

        # 4. Populate summary if requested
        if request.cli_include_summary:
            session.populate_summary()

        # 5. Extract raw "result" chunk from session.chunks
        raw_result_chunk = {}
        for chunk in session.chunks:
            if chunk.type == "result":
                raw_result_chunk = chunk.raw
                break

        # 6. Return both raw CLI result and organized session data
        return {
            "raw_result": raw_result_chunk,
            "session": to_dict(session, recursive=True),
        }

    def normalize_response(self, raw_response: dict[str, Any]) -> NormalizedResponse:
        """Normalize Claude Code response to standard format.

        Args:
            raw_response: Dict with:
                - raw_result: Raw "result" chunk from Claude Code CLI
                - session: Organized ClaudeSession data

        Returns:
            NormalizedResponse with:
            - status: "success" or "error"
            - data: Final result text
            - raw_response: Actual raw CLI "result" chunk
            - metadata: Organized session data
        """
        # Extract session data (our organized structure)
        session = raw_response.get("session", {})
        # Extract actual raw CLI result chunk
        raw_cli_result = raw_response.get("raw_result", {})

        # Extract text result from session
        text = session.get("result", "")

        # Extract metadata from organized session
        metadata: dict[str, Any] = {
            "session_id": session.get("session_id"),
            "model": session.get("model"),
            "usage": session.get("usage", {}),
            "total_cost_usd": session.get("total_cost_usd"),
            "num_turns": session.get("num_turns"),
            "duration_ms": session.get("duration_ms"),
            "duration_api_ms": session.get("duration_api_ms"),
            "is_error": session.get("is_error", False),
            "tool_uses": session.get("tool_uses", []),
            "tool_results": session.get("tool_results", []),
            "thinking_log": session.get("thinking_log", []),
            "summary": session.get("summary"),  # Always include (None if not present)
        }

        return NormalizedResponse(
            status="error" if session.get("is_error") else "success",
            data=text,
            raw_response=raw_cli_result,  # Use actual raw CLI result chunk
            metadata=metadata,
        )
