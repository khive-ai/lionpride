# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from lionpride.services.third_party.gemini_models import (
    GeminiChunk,
    GeminiCodeRequest,
    GeminiSession,
    stream_gemini_cli,
)
from lionpride.services.types import NormalizedResponse

from ..types.endpoint import Endpoint, EndpointConfig

__all__ = (
    "GeminiCodeEndpoint",
    "create_gemini_code_config",
)


def create_gemini_code_config(
    name: str | None = None,
    model: str | None = "gemini-2.5-pro",
) -> dict:
    """Factory for Gemini CLI endpoint config.

    Args:
        name: Config name (default: "gemini_code_cli")
        model: Default model to use (gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro)

    Returns:
        Config dict for GeminiCodeEndpoint
    """
    return {
        "name": name or "gemini_code_cli",
        "provider": "gemini_code",
        "base_url": "internal",
        "endpoint": "query_cli",
        "api_key": "dummy-key",  # CLI uses Google OAuth, not API key
        "request_options": GeminiCodeRequest,
        "timeout": 3600,  # 1 hour max
        "kwargs": {"model": model} if model else {},
    }


class GeminiCodeEndpoint(Endpoint):
    """Gemini CLI endpoint for local AI agent execution.

    Usage:
        endpoint = GeminiCodeEndpoint()
        response = await endpoint.call({
            "messages": [{"role": "user", "content": "List files"}]
        })

        # Or with direct prompt:
        response = await endpoint.call({
            "prompt": "Analyze this codebase",
            "yolo": True  # Auto-approve actions
        })
    """

    def __init__(
        self,
        config: dict | EndpointConfig | None = None,
        circuit_breaker: Any | None = None,
        **kwargs,
    ):
        """Initialize with Gemini CLI config."""
        if config is None:
            config = create_gemini_code_config(
                name=kwargs.pop("name", None),
                model=kwargs.pop("model", "gemini-2.5-pro"),
            )
        elif isinstance(config, EndpointConfig):
            config = config.model_dump()
        if not isinstance(config, dict):
            raise ValueError("Provided config must be a dict or EndpointConfig instance")

        super().__init__(config=config, circuit_breaker=circuit_breaker, **kwargs)

    def create_payload(
        self, request: dict | BaseModel, extra_headers: dict | None = None, **kwargs
    ) -> tuple[dict, dict]:
        """Create payload for Gemini CLI.

        Args:
            request: Request dict with 'messages' or 'prompt', or GeminiCodeRequest
            extra_headers: Ignored (CLI doesn't use HTTP headers)
            **kwargs: Additional arguments merged into request

        Returns:
            (payload_dict, headers_dict) - headers always empty for CLI
        """
        from lionpride.ln import to_dict

        # Convert request to dict if BaseModel
        request_dict = to_dict(request) if isinstance(request, BaseModel) else request

        # Merge config kwargs, request, and call kwargs
        req_dict = {**self.config.kwargs, **request_dict, **kwargs}

        # Extract prompt or messages (one is required)
        prompt = req_dict.get("prompt")
        messages = req_dict.get("messages")

        if not prompt and not messages:
            raise ValueError(
                f"'prompt' or 'messages' required for Gemini CLI endpoint. "
                f"Got keys: {list(req_dict.keys())}"
            )

        # Create GeminiCodeRequest object
        req_obj = GeminiCodeRequest(**req_dict)

        return {"request": req_obj}, {}

    async def stream(
        self, request: dict | BaseModel, **kwargs
    ) -> AsyncIterator[GeminiChunk | dict | GeminiSession]:
        """Stream Gemini CLI response chunks.

        Yields:
            GeminiChunk, dict (system messages), or GeminiSession (final)
        """
        payload, _ = self.create_payload(request, **kwargs)
        request_obj: GeminiCodeRequest = payload["request"]

        async for chunk in stream_gemini_cli(request_obj):
            yield chunk

    async def _call(
        self,
        payload: dict,
        headers: dict,
        **kwargs,
    ) -> dict:
        """Execute Gemini CLI and return raw result + session data.

        Returns:
            Dict with:
            - raw_result: Raw result chunk from Gemini CLI
            - session: Organized session data from GeminiSession
        """
        from lionpride.ln import to_dict

        request: GeminiCodeRequest = payload["request"]
        session = GeminiSession()

        # Stream the Gemini response
        async for chunk in stream_gemini_cli(request, session, **kwargs):
            if isinstance(chunk, dict):
                if chunk.get("type") == "done":
                    break
            elif isinstance(chunk, GeminiSession):
                break

        # Use session.result if available (from final "result" event)
        # Only fall back to combining chunk texts if no result was set
        if not session.result:
            texts = []
            for chunk in session.chunks:
                if chunk.text is not None:
                    texts.append(chunk.text)
            session.result = "\n".join(texts)

        # Populate summary if requested
        if request.cli_include_summary:
            session.populate_summary()

        # Extract raw result chunk
        raw_result_chunk = {}
        for chunk in session.chunks:
            if chunk.type in ("result", "response"):
                raw_result_chunk = chunk.raw
                break

        return {
            "raw_result": raw_result_chunk,
            "session": to_dict(session, recursive=True),
        }

    def normalize_response(self, raw_response: dict[str, Any]) -> NormalizedResponse:
        """Normalize Gemini CLI response to standard format.

        Args:
            raw_response: Dict with:
                - raw_result: Raw result chunk from Gemini CLI
                - session: Organized GeminiSession data

        Returns:
            NormalizedResponse with:
            - status: "success" or "error"
            - data: Final result text
            - raw_response: Actual raw CLI result chunk
            - metadata: Organized session data
        """
        session = raw_response.get("session", {})
        raw_cli_result = raw_response.get("raw_result", {})

        text = session.get("result", "")

        metadata: dict[str, Any] = {
            "session_id": session.get("session_id"),
            "model": session.get("model"),
            "usage": session.get("usage", {}),
            "total_cost_usd": session.get("total_cost_usd"),
            "num_turns": session.get("num_turns"),
            "duration_ms": session.get("duration_ms"),
            "is_error": session.get("is_error", False),
            "tool_uses": session.get("tool_uses", []),
            "tool_results": session.get("tool_results", []),
            "summary": session.get("summary"),
        }

        return NormalizedResponse(
            status="error" if session.get("is_error") else "success",
            data=text,
            raw_response=raw_cli_result,
            metadata=metadata,
        )
