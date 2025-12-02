# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
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
    "InterpretParams",
    "OperateParams",
    "ParseParams",
    "ReactParams",
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

    imodel: iModel | str = None
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
        content = InstructionContent.create(
            instruction=self.instruction,
            context=self.context,
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

    text: str = None
    """Raw text to parse."""

    target_keys: list[str] = field(default_factory=list)
    """Expected keys for fuzzy matching."""

    imodel: iModel | str = None
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


@dataclass(frozen=True, slots=True)
class InterpretParams(Params):
    """Parameters for interpret operation (refine user instructions).

    Rewrites raw user input into clearer, more structured prompts.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    text: str = None
    """Raw user instruction to refine."""

    imodel: iModel | str = None
    """Model to use for interpretation."""

    domain: str = "general"
    """Domain hint for interpretation."""

    style: str = "concise"
    """Desired style of output."""

    sample_writing: str | None = None
    """Example of desired output style."""

    temperature: float = 0.1
    """Temperature for generation."""


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
    generate: GenerateParams = None
    """Generate parameters."""

    parse: ParseParams = None
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


@dataclass(frozen=True)
class ReactParams(OperateParams):
    """Parameters for react operation (multi-step reasoning loop).

    Inherits from OperateParams - access everything directly:
        react.generate.instruction  (1 level)
        react.tools                 (direct)
        react.max_steps             (direct)

    React is a pure loop: reasoning + actions + optional intermediate outputs.
    """

    max_steps: int = 10
    """Maximum react steps."""

    return_trace: bool = False
    """Return full execution trace."""

    # Intermediate response options
    intermediate_response_options: list[type[BaseModel]] | type[BaseModel] | None = None
    """Models for intermediate deliverables (e.g., ProgressReport, PartialResult).

    Each model becomes a nullable field in step responses. The model can
    populate these during multi-step reasoning to provide structured
    intermediate outputs.
    """

    intermediate_listable: bool = False
    """Whether intermediate options can be lists (e.g., list[CodeBlock])."""

    intermediate_nullable: bool = True
    """Whether intermediate options default to None (usually True)."""

    @property
    def operate(self) -> OperateParams:
        """Get OperateParams for operate operation."""
        return OperateParams(
            generate=self.generate,
            parse=self.parse,
            operable=self.operable,
            capabilities=self.capabilities,
            auto_fix=self.auto_fix,
            strict_validation=self.strict_validation,
            tools=self.tools,
            tool_schemas=self.tool_schemas,
            tool_concurrent=self.tool_concurrent,
            tool_timeout=self.tool_timeout,
            actions=self.actions,
            reason=self.reason,
            skip_validation=self.skip_validation,
            return_message=self.return_message,
        )
