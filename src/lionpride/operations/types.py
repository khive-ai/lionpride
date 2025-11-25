# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel

from lionpride.types import ModelConfig, Params

if TYPE_CHECKING:
    from lionpride.services.types import iModel
    from lionpride.types import Operable

__all__ = (
    "CommunicateParam",
    "GenerateParam",
    "OperateParam",
    "ReactParam",
)


@dataclass(slots=True, frozen=True, init=False)
class GenerateParam(Params):
    """Parameters for generate operation (stateless text generation).

    Attributes:
        imodel: Model name (str) or iModel instance
        messages: Chat messages for generation
        model: Model name for API (e.g., "gpt-4o", "claude-3-5-sonnet")
        return_as: Output format (text|raw|message|calling)
        **kwargs: Additional iModel.invoke() parameters
    """

    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    imodel: str | iModel | None = None
    messages: list[dict] | None = None
    model: str | None = None  # Model name for API (e.g., "gpt-4o")
    return_as: Literal["text", "raw", "message", "calling"] = "text"


@dataclass(slots=True, frozen=True, init=False)
class CommunicateParam(Params):
    """Parameters for communicate operation (stateful chat).

    Attributes:
        instruction: User instruction/question
        imodel: Model name (str) or iModel instance
        model: Model name for API (e.g., "gpt-4o", "claude-3-5-sonnet")
        context: Additional context (prompt facts)
        images: Image URLs/data for multimodal
        image_detail: Image quality (low|high|auto)
        return_as: Output format (text|raw|message|model)
        response_model: Pydantic model for JSON mode
        operable: Operable for LNDL mode
        strict_validation: Strict JSON schema validation
        fuzzy_parse: Enable fuzzy parsing
        lndl_threshold: LNDL similarity threshold
        max_retries: Retry attempts for validation
    """

    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    instruction: str | None = None
    imodel: str | iModel | None = None
    model: str | None = None  # Model name for API (e.g., "gpt-4o")
    context: Any = None
    images: list | None = None
    image_detail: Literal["low", "high", "auto"] | None = None
    return_as: Literal["text", "raw", "message", "model"] = "text"
    response_model: type[BaseModel] | None = None
    operable: Operable | None = None
    strict_validation: bool = False
    fuzzy_parse: bool = True
    lndl_threshold: float = 0.85
    max_retries: int = 0


@dataclass(slots=True, frozen=True, init=False)
class OperateParam(Params):
    """Parameters for operate operation (structured output with actions).

    Attributes:
        instruction: User instruction
        imodel: Model name (str) or iModel instance
        response_model: Pydantic model for validation
        operable: Operable for validation (alternative to response_model)
        context: Additional context
        images: Image URLs/data for multimodal
        image_detail: Image quality (low|high|auto)
        tool_schemas: Pre-built tool schemas
        tools: Enable tool execution
        actions: Enable action requests
        reason: Enable reasoning field
        use_lndl: Use LNDL mode (vs JSON)
        lndl_threshold: LNDL similarity threshold
        max_retries: Retry attempts for validation
        skip_validation: Skip output validation
        return_message: Return (result, message) tuple
        concurrent_tool_execution: Execute tools concurrently
    """

    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    instruction: str | None = None
    imodel: str | iModel | None = None
    response_model: type[BaseModel] | None = None
    operable: Operable | None = None
    context: Any = None
    images: list | None = None
    image_detail: Literal["low", "high", "auto"] | None = None
    tool_schemas: list[dict] | None = None
    tools: bool = False
    actions: bool = False
    reason: bool = False
    use_lndl: bool = False
    lndl_threshold: float = 0.85
    max_retries: int = 0
    skip_validation: bool = False
    return_message: bool = False
    concurrent_tool_execution: bool = True


@dataclass(slots=True, frozen=True, init=False)
class ReactParam(Params):
    """Parameters for react operation (ReAct reasoning + action loop).

    Attributes:
        instruction: Task instruction
        imodel: Model name (str) or iModel instance
        tools: List of Tool classes/instances
        response_model: Final response schema
        model_name: Model name for invocation
        context: Additional context
        max_steps: Maximum ReAct steps
        use_lndl: Use LNDL mode (vs JSON)
        lndl_threshold: LNDL similarity threshold
        verbose: Verbose output
    """

    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=True)
    instruction: str | None = None
    imodel: str | iModel | None = None
    tools: list | None = None
    response_model: type[BaseModel] | None = None
    model_name: str | None = None
    context: Any = None
    max_steps: int = 5
    use_lndl: bool = False
    lndl_threshold: float = 0.85
    verbose: bool = False
