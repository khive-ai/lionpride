# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operation parameter types - composable params for all operations.

Hierarchy (composition):
    GenerateParams ─────────────────┐
    ParseParams ────────────────────┼─► CommunicateParams
                                    │
    CommunicateParams ──────────────┤
    ActParams ──────────────────────┼─► OperateParams
                                    │
    OperateParams ──────────────────┴─► ReactParams
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from lionpride.session.messages import InstructionContent, Message
from lionpride.types.base import ModelConfig, Params

if TYPE_CHECKING:
    from lionpride.services.types import iModel
    from lionpride.types import Operable

__all__ = (
    "ActParams",
    "CommunicateParams",
    "GenerateParams",
    "HandleUnmatched",
    "InterpretParams",
    "OperateParams",
    "ParseParams",
    "ReactParams",
)

# Type aliases
HandleUnmatched = Literal["ignore", "raise", "remove", "fill", "force"]


# =============================================================================
# Base Operation Params
# =============================================================================


@dataclass(frozen=True, slots=True)
class GenerateParams(Params):
    """Parameters for generate operation (stateless LLM call).

    Generate is the lowest-level helper - just calls the model.
    No message persistence, no validation.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    imodel: iModel | str = None
    """Model to use for generation."""

    instruction: str | Message = None
    """Instruction text or Message."""

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

    structure_format: Literal["json", "lndl"] = "json"
    """Format for structured output rendering."""

    return_as: Literal["text", "raw", "message", "calling"] = "calling"
    """Output format."""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel."""

    @property
    def instruction_message(self) -> Message | None:
        """Get instruction as Message."""
        if self._is_sentinel(self.instruction):
            return None

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
    """Retry attempts for LLM reparse."""

    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs for imodel."""


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
# Composed Operation Params
# =============================================================================


@dataclass(frozen=True, slots=True)
class CommunicateParams(Params):
    """Parameters for communicate operation (stateful chat).

    Communicate = Generate + message persistence + Validate (via IPU).

    Validation flow (when operable set):
    1. operable.create_model(include=capabilities) → schema for LLM prompt
    2. LLM returns response
    3. adapter.parse_json() → dict
    4. validator.validate(dict, operable, capabilities) → typed model
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    # Composed params
    generate: GenerateParams = None
    """Generate parameters."""

    parse: ParseParams = None
    """Parse parameters."""

    # Validation (IPU pipeline)
    operable: Operable | None = None
    """Operable for structured output (defines schema + validation rules)."""

    capabilities: set[str] | None = None
    """Capabilities - allowed fields (defaults to branch.capabilities)."""

    auto_fix: bool = True
    """Auto-fix validation issues when possible."""

    strict_validation: bool = True
    """Raise on validation failure."""


@dataclass(frozen=True, slots=True)
class OperateParams(Params):
    """Parameters for operate operation (structured output + actions).

    Operate = Communicate + Act.
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    # Composed params
    communicate: CommunicateParams = None
    """Communicate parameters."""

    act: ActParams = None
    """Act parameters (for tool execution)."""

    # Operate-specific
    actions: bool = False
    """Enable action_requests in output."""

    reason: bool = False
    """Enable reasoning in output."""

    skip_validation: bool = False
    """Skip validation (return raw text)."""

    return_message: bool = False
    """Return (result, message) tuple."""


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


@dataclass(frozen=True, slots=True)
class ReactParams(Params):
    """Parameters for react operation (multi-step reasoning loop).

    React is a pure loop: reasoning + actions + optional intermediate outputs.
    For final structured output, follow up with communicate().

    Step response fields:
        - reasoning: str | None
        - action_requests: list[ActionRequest] | None
        - is_done: bool
        - intermediate_response_options: nested model | None (if configured)

    Example:
        class ProgressReport(BaseModel):
            progress_pct: int
            current_status: str

        params = ReactParams(
            intermediate_response_options=[ProgressReport],
        )
    """

    _config = ModelConfig(none_as_sentinel=True, empty_as_sentinel=True)

    operate: OperateParams = None
    """Operate parameters."""

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
