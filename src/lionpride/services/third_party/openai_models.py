# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


# ---------- Roles & content parts ----------


class ChatRole(str, Enum):
    system = "system"
    developer = "developer"  # modern system-like role
    user = "user"
    assistant = "assistant"
    tool = "tool"  # for tool results sent back to the model


class TextPart(BaseModel):
    """Text content part for multimodal messages."""

    type: Literal["text"] = "text"
    text: str


class ImageURLObject(BaseModel):
    """Image URL object; 'detail' is optional and model-dependent."""

    url: str
    detail: Literal["auto", "low", "high"] | None = Field(
        default=None,
        description="Optional detail control for vision models (auto/low/high).",
    )


class ImageURLPart(BaseModel):
    """Image content part for multimodal messages."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURLObject


ContentPart = TextPart | ImageURLPart


# ---------- Tool-calling structures ----------


class FunctionDef(BaseModel):
    """JSON Schema function definition for tool-calling."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing function parameters.",
    )


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class FunctionCall(BaseModel):
    """Legacy function_call field on assistant messages."""

    name: str
    arguments: str


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Assistant's tool call (modern)."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolChoiceFunction(BaseModel):
    """Explicit tool selection."""

    type: Literal["function"] = "function"
    function: dict[str, str]  # {"name": "<function_name>"}


ToolChoice = Union[Literal["auto", "none"], ToolChoiceFunction]


# ---------- Response format (structured outputs) ----------


class ResponseFormatText(BaseModel):
    type: Literal["text"] = "text"


class ResponseFormatJSONObject(BaseModel):
    type: Literal["json_object"] = "json_object"


class JSONSchemaFormat(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(alias="schema", description="JSON Schema definition")
    strict: bool | None = Field(
        default=None,
        description="If true, disallow unspecified properties (strict schema).",
    )

    model_config = {"populate_by_name": True}


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchemaFormat


ResponseFormat = Union[
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
]


# ---------- Messages (discriminated by role) ----------


class SystemMessage(BaseModel):
    role: Literal[ChatRole.system] = ChatRole.system
    content: str | list[ContentPart]
    name: str | None = None  # optional per API


class DeveloperMessage(BaseModel):
    role: Literal[ChatRole.developer] = ChatRole.developer
    content: str | list[ContentPart]
    name: str | None = None


class UserMessage(BaseModel):
    role: Literal[ChatRole.user] = ChatRole.user
    content: str | list[ContentPart]
    name: str | None = None


class AssistantMessage(BaseModel):
    role: Literal[ChatRole.assistant] = ChatRole.assistant
    # Either textual content, or only tool_calls (when asking you to call tools)
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None  # modern tool-calling result
    function_call: FunctionCall | None = None  # legacy function-calling result


class ToolMessage(BaseModel):
    role: Literal[ChatRole.tool] = ChatRole.tool
    content: str  # tool output returned to the model
    tool_call_id: str  # must reference the assistant's tool_calls[i].id


ChatMessage = SystemMessage | DeveloperMessage | UserMessage | AssistantMessage | ToolMessage

# ---------- Stream options ----------


class StreamOptions(BaseModel):
    include_usage: bool | None = Field(
        default=None,
        description="If true, a final streamed chunk includes token usage.",
    )


# ---------- Main request model ----------


class OpenAIChatCompletionsRequest(BaseModel):
    """
    Request body for OpenAI Chat Completions.
    Endpoint: POST https://api.openai.com/v1/chat/completions
    """

    # Required
    model: str = Field(..., description="Model name, e.g., 'gpt-4o', 'gpt-4o-mini'.")  # type: ignore
    messages: list[ChatMessage] = Field(
        ...,
        description="Conversation so far, including system/developer context.",
    )

    # Sampling & penalties
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Higher is more random."
    )
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling.")
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Encourages new topics; -2..2.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalizes repetition; -2..2.",
    )

    # Token limits
    max_completion_tokens: int | None = Field(
        default=None,
        description="Preferred cap on generated tokens (newer models).",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Legacy completion cap (still accepted by many models).",
    )

    # Count, stop, logits
    n: int | None = Field(default=None, ge=1, description="# of choices to generate.")
    stop: str | list[str] | None = Field(default=None, description="Stop sequence(s).")
    logit_bias: dict[str, float] | None = Field(
        default=None,
        description="Map of token-id -> bias (-100..100).",
    )
    seed: int | None = Field(
        default=None,
        description="Optional reproducibility seed (model-dependent).",
    )
    logprobs: bool | None = None
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        description="When logprobs is true, how many top tokens to include.",
    )

    # Tool calling (modern)
    tools: list[FunctionTool] | None = None
    tool_choice: ToolChoice | None = Field(
        default=None,
        description="'auto' (default), 'none', or a function selection.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Allow multiple tool calls in a single assistant turn.",
    )

    # Legacy function-calling (still supported)
    functions: list[FunctionDef] | None = None
    function_call: Literal["none", "auto"] | FunctionCall | None = None

    # Structured outputs
    response_format: ResponseFormat | None = None

    # Streaming
    stream: bool | None = None
    stream_options: StreamOptions | None = None

    # Routing / tiering
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = Field(
        default=None,
        description="Processing tier; requires account eligibility.",
    )

    # Misc
    user: str | None = Field(
        default=None,
        description="End-user identifier for abuse monitoring & analytics.",
    )
    store: bool | None = Field(
        default=None,
        description="Whether to store the response server-side (model-dependent).",
    )
    metadata: dict[str, Any] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="For reasoning models: trade-off between speed and accuracy.",
    )
