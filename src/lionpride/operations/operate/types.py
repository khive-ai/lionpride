# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from lionpride.session.messages import InstructionContent, Message
from lionpride.types.base import ModelConfig, Params

if TYPE_CHECKING:
    from lionpride.services.types import iModel
    from lionpride.types import Operable

__all__ = (
    "ActParams",
    "CommunicateParams",
    "CustomParser",
    "CustomRenderer",
    "GenerateParams",
    "HandleUnmatched",
    "OperateParams",
    "ParseParams",
    "ReturnAs",
)

# Type aliases
HandleUnmatched = Literal["ignore", "raise", "remove", "fill", "force"]
ReturnAs = Literal["text", "raw", "message", "calling"]


# =============================================================================
# Custom Parser/Renderer Protocols
# =============================================================================


@runtime_checkable
class CustomParser(Protocol):
    """Protocol for custom output parsers (e.g., LNDL).

    Implementations extract structured data from LLM text responses.

    Args:
        text: Raw LLM response text
        target_keys: Expected field names for fuzzy matching
        **kwargs: Additional parser-specific options

    Returns:
        Dict mapping field names to extracted values
    """

    def __call__(self, text: str, target_keys: list[str], **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class CustomRenderer(Protocol):
    """Protocol for custom instruction renderers.

    Implementations format request_model schema for custom output formats.

    Args:
        model: Pydantic model class defining expected response schema
        **kwargs: Additional renderer-specific options

    Returns:
        Formatted instruction string for the custom output format
    """

    def __call__(self, model: type[BaseModel], **kwargs: Any) -> str: ...


# =============================================================================
# Standalone Params (no inheritance)
# =============================================================================


@dataclass(frozen=True, slots=True)
class GenerateParams(Params):
    """Parameters for generate operation (stateless LLM call).

    Generate is the lowest-level helper - just calls the model.
    No message persistence, no validation.

    Required:
        instruction: The instruction text or Message (must be provided).
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    # Required field (no default)
    instruction: str | Message
    """Instruction text or Message (required)."""

    imodel: iModel | str | None = None
    """Model to use for generation."""

    context: dict[str, Any] | None = None
    """Additional context for instruction."""

    images: list[str] | None = None
    """Image URLs for multimodal input."""

    image_detail: Literal["low", "high", "auto"] | None = None
    """Image detail level."""

    tool_schemas: list[str] | None = None
    """Tool schemas for function calling (pass-through to instruction)."""

    request_model: type[BaseModel] | None = None
    """Pydantic model for structured output schema."""

    structure_format: Literal["json", "custom"] = "json"
    """Format for structured output rendering ('json' or 'custom')."""

    custom_renderer: CustomRenderer | None = None
    """Custom renderer for structure_format='custom'. Formats request_model schema."""

    return_as: ReturnAs = "calling"
    """Output format."""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel."""

    @property
    def instruction_message(self) -> Message:
        """Get instruction as Message."""
        if isinstance(self.instruction, Message):
            return self.instruction

        # Create Message from string instruction
        # Note: context here is dict but InstructionContent.create expects list
        # We wrap the context dict in a list if present
        context_list = [self.context] if self.context is not None else None
        content = InstructionContent.create(
            instruction=self.instruction,
            context=context_list,
            images=self.images,
            image_detail=self.image_detail,
            tool_schemas=self.tool_schemas,
            request_model=self.request_model,
        )
        return Message(content=content)


@dataclass(frozen=True, slots=True)
class ParseParams(Params):
    """Parameters for parse operation (JSON extraction).

    Parse extracts JSON from raw text. Falls back to LLM if needed.
    Returns dict - validation happens in Validator.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    text: str | None = None
    """Raw text to parse."""

    target_keys: list[str] = field(default_factory=list)
    """Expected keys for fuzzy matching."""

    imodel: iModel | str | None = None
    """Model for LLM reparse fallback."""

    similarity_threshold: float = 0.85
    """Fuzzy match threshold."""

    handle_unmatched: HandleUnmatched = "force"
    """How to handle unmatched keys."""

    max_retries: int = 3
    """Retry attempts for LLM reparse. should be less than 5 to avoid long delays."""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel."""

    structure_format: Literal["json", "custom"] = "json"
    """Format for parsing output ('json' or 'custom')."""

    custom_parser: CustomParser | None = None
    """Custom parser for structure_format='custom'. Extracts dict from text."""

    match_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for direct matching."""


@dataclass(frozen=True, slots=True)
class ActParams(Params):
    """Parameters for act operation (tool execution)."""

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    tools: list[str] | bool = False
    """Tools to include (True=all, list=specific, False=none)."""

    tool_schemas: list[dict] | None = None
    """Pre-computed tool schemas."""

    concurrent: bool = True
    """Execute tools concurrently."""

    timeout: float | None = None
    """Timeout for tool execution."""


# =============================================================================
# Inherited Params (flat hierarchy)
# =============================================================================


@dataclass(frozen=True, slots=True)
class CommunicateParams(Params):
    """Parameters for communicate operation (stateful chat).

    Communicate = Generate + Parse + Validate.

    Access: communicate.generate.instruction (1 level)

    Validation:
        When using structured output (operable or generate.request_model),
        capabilities MUST be explicitly declared.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    # Composed params (1 level deep)
    generate: GenerateParams | None = None
    """Generate parameters."""

    parse: ParseParams | None = None
    """Parse parameters."""

    # Validation (IPU pipeline)
    operable: Operable | None = None
    """Operable for structured output (defines schema + validation rules)."""

    capabilities: set[str] | None = None
    """Capabilities - allowed fields (REQUIRED when using structured output)."""

    auto_fix: bool = True
    """Auto-fix validation issues when possible."""

    strict_validation: bool = True
    """Raise on validation failure."""

    def __post_init__(self) -> None:
        """Validate that capabilities are declared when using structured output."""
        has_structured_output = self.operable is not None or (
            not self._is_sentinel(self.generate)
            and self.generate is not None
            and self.generate.request_model is not None
        )
        if has_structured_output and self.capabilities is None:
            raise ValueError(
                "capabilities must be explicitly declared when using structured output "
                "(operable or request_model). This ensures explicit access control."
            )


@dataclass(frozen=True)
class OperateParams(CommunicateParams):
    """Parameters for operate operation (structured output + actions).

    Inherits from CommunicateParams - access generate/parse directly:
        operate.generate.instruction  (1 level, not operate.communicate.generate)

    Operate = Communicate + Act.
    """

    # Act params (flattened from ActParams for convenience)
    tools: list[str] | bool = False
    """Tools to include (True=all, list=specific, False=none)."""

    tool_schemas: list[dict] | None = None
    """Pre-computed tool schemas."""

    tool_concurrent: bool = True
    """Execute tools concurrently."""

    tool_timeout: float | None = None
    """Timeout for tool execution."""

    # Operate-specific
    actions: bool = False
    """Enable action_requests in output."""

    reason: bool = False
    """Enable reasoning in output."""

    skip_validation: bool = False
    """Skip validation (return raw text)."""

    return_message: bool = False
    """Return (result, message) tuple."""

    @property
    def act_params(self) -> ActParams:
        """Get ActParams for tool execution."""
        return ActParams(
            tools=self.tools,
            tool_schemas=self.tool_schemas,
            concurrent=self.tool_concurrent,
            timeout=self.tool_timeout,
        )

    @property
    def communicate(self) -> CommunicateParams:
        """Get CommunicateParams for communicate operation."""
        return CommunicateParams(
            generate=self.generate,
            parse=self.parse,
            operable=self.operable,
            capabilities=self.capabilities,
            auto_fix=self.auto_fix,
            strict_validation=self.strict_validation,
        )
