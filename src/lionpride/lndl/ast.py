# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


# Base Nodes
class ASTNode:
    """Base AST node for all LNDL constructs."""

    __slots__ = ()  # Empty slots for proper inheritance


# Expressions (evaluate to values)
class Expr(ASTNode):
    """Base expression node."""

    __slots__ = ()


@dataclass(slots=True)
class Literal(Expr):
    """Literal scalar value.

    Examples:
        - "AI safety"
        - 42
        - 0.85
        - true
    """

    value: str | int | float | bool


@dataclass(slots=True)
class Identifier(Expr):
    """Variable reference.

    Examples:
        - [title]
        - [summary]
    """

    name: str


# Statements (declarations, no return value)
class Stmt(ASTNode):
    """Base statement node."""

    __slots__ = ()


@dataclass(slots=True)
class Lvar(Stmt):
    """Namespaced variable declaration - maps to Pydantic model field.

    Syntax:
        <lvar Model.field alias>content</lvar>
        <lvar Model.field>content</lvar>  # Uses field as alias

    Examples:
        <lvar Report.title t>AI Safety Analysis</lvar>
        → Lvar(model="Report", field="title", alias="t", content="AI Safety Analysis")

        <lvar Report.score>0.95</lvar>
        → Lvar(model="Report", field="score", alias="score", content="0.95")
    """

    model: str  # Model name (e.g., "Report")
    field: str  # Field name (e.g., "title", "score")
    alias: str  # Local variable name (e.g., "t", defaults to field)
    content: str  # Raw string value


@dataclass(slots=True)
class RLvar(Stmt):
    """Raw variable declaration - simple string capture without model mapping.

    Syntax:
        <lvar alias>content</lvar>

    Examples:
        <lvar reasoning>The analysis shows...</lvar>
        → RLvar(alias="reasoning", content="The analysis shows...")

        <lvar score>0.95</lvar>
        → RLvar(alias="score", content="0.95")

    Usage:
        - Use for intermediate LLM outputs not mapped to Pydantic models
        - Can only resolve to scalar OUT{} fields (str, int, float, bool)
        - Cannot be used in BaseModel OUT{} fields (no type validation)
    """

    alias: str  # Local variable name
    content: str  # Raw string value


@dataclass(slots=True)
class Lact(Stmt):
    """Action declaration.

    Syntax:
        - Namespaced: <lact Model.field alias>func(...)</lact>
        - Direct: <lact alias>func(...)</lact>

    Examples:
        <lact Report.summary s>generate_summary(prompt="...")</lact>
        → Lact(model="Report", field="summary", alias="s", call="generate_summary(...)")

        <lact search>search(query="AI")</lact>
        → Lact(model=None, field=None, alias="search", call="search(...)")
    """

    model: str | None  # Model name or None for direct actions
    field: str | None  # Field name or None for direct actions
    alias: str  # Local reference name
    call: str  # Raw function call string


# =================================================================== #
# v2: Context Management Nodes                                         #
# =================================================================== #


@dataclass(slots=True)
class IncludeDirective(Stmt):
    """Include full message in context.

    Syntax: <include msg="msg_id"/>

    Example:
        <include msg="inst_001"/>
        → IncludeDirective(msg_ref="inst_001")
    """

    msg_ref: str  # Message ID to include


@dataclass(slots=True)
class CompressDirective(Stmt):
    """Compress messages to symbolic representation.

    Syntax: <compress msgs="range" to="alias"/>

    Example:
        <compress msgs="0..50" to="summary"/>
        → CompressDirective(msg_refs="0..50", alias="summary")
    """

    msg_refs: str  # Range like "0..50" or list
    alias: str  # Store compressed result as


@dataclass(slots=True)
class DropDirective(Stmt):
    """Explicitly drop message from context.

    Syntax: <drop msg="msg_id"/>

    Example:
        <drop msg="verbose_output"/>
        → DropDirective(msg_ref="verbose_output")
    """

    msg_ref: str  # Message ID to drop


@dataclass(slots=True)
class NoticeDirective(Stmt):
    """Include brief notice instead of full content.

    Syntax: <notice msg="msg_id" brief="description"/>

    Example:
        <notice msg="tools" brief="search, calculate, retrieve"/>
        → NoticeDirective(msg_ref="tools", brief="search, calculate, retrieve")
    """

    msg_ref: str  # Message ID
    brief: str  # Short description


# Type alias for all context directives
ContextDirective = IncludeDirective | CompressDirective | DropDirective | NoticeDirective


@dataclass(slots=True)
class ContextBlock(Stmt):
    """Context engineering block.

    Syntax:
        <context>
          <include msg="..."/>
          <compress msgs="..." to="..."/>
        </context>

    Example:
        <context>
          <include msg="inst_001"/>
          <compress msgs="0..50" to="summary"/>
        </context>
        → ContextBlock(directives=[IncludeDirective(...), CompressDirective(...)])
    """

    directives: list[ContextDirective]


# =================================================================== #
# v2: Continuation Control Nodes                                       #
# =================================================================== #


@dataclass(slots=True)
class YieldStmt(Stmt):
    """Yield control for approval/continuation.

    Syntax: <yield for="ref" reason="..." keep="..." drop_full="true|false" transform="..."/>

    Example:
        <yield for="search" reason="need results" keep="top_5" drop_full="true"/>
        → YieldStmt(for_ref="search", reason="need results", keep="top_5", drop_full=True)
    """

    for_ref: str | None = None  # What we're waiting for
    reason: str | None = None  # Why continuation needed
    drop_full: bool = False  # Drop verbose result
    keep: str | None = None  # What to keep (e.g., "top_5", "summary")
    transform: str | None = None  # Transformation to apply


# =================================================================== #
# v2: Multi-Agent Communication Nodes                                  #
# =================================================================== #


@dataclass(slots=True)
class SendStmt(Stmt):
    """Send message to another agent.

    Syntax:
        <send to="agent" type="MsgType" timeout="30s">
          <include msg="..."/>
        </send>

    Example:
        <send to="critic" type="ReviewRequest" timeout="30s">
          <include msg="analysis"/>
        </send>
        → SendStmt(to="critic", msg_type="ReviewRequest", timeout="30s", content=[...])
    """

    to: str  # Target agent
    msg_type: str | None = None  # Message type for validation
    timeout: str | None = None  # Timeout duration (e.g., "30s")
    content: list[ContextDirective] | None = None  # What to send


# =================================================================== #
# v2: Reference Expression                                             #
# =================================================================== #


@dataclass(slots=True)
class Ref(Expr):
    """Reference to prior result: {{alias.path}}

    Syntax: {{alias}} or {{alias.path.to.field}}

    Examples:
        {{search}}
        → Ref(alias="search", path=None)

        {{search.result.items}}
        → Ref(alias="search", path=["result", "items"])
    """

    alias: str  # Variable/action name
    path: list[str] | None = None  # Optional path for nested access


@dataclass(slots=True)
class OutBlock(Stmt):
    """Output specification block.

    Syntax: OUT{field: value, field2: [ref1, ref2]}

    Values can be:
        - Literal: 0.85, "text", true
        - Single reference: [alias]
        - Multiple references: [alias1, alias2]

    Example:
        OUT{title: [t], summary: [s], confidence: 0.85}
        → OutBlock(fields={"title": ["t"], "summary": ["s"], "confidence": 0.85})
    """

    fields: dict[str, list[str] | str | int | float | bool]


@dataclass(slots=True)
class Program:
    """Root AST node containing all declarations.

    A complete LNDL program consists of:
        - Variable declarations (lvars + rlvars)
        - Action declarations (lacts)
        - Output specification (out_block)

    v2 additions:
        - Context block (context engineering)
        - Yield statements (continuation control)
        - Send statements (multi-agent communication)

    Example (v1):
        <lvar Report.title t>Title</lvar>
        <lvar reasoning>Analysis text</lvar>
        <lact Report.summary s>summarize()</lact>
        OUT{title: [t], summary: [s], reasoning: [reasoning]}

        → Program(
            lvars=[Lvar(...), RLvar(...)],
            lacts=[Lact(...)],
            out_block=OutBlock(...)
        )

    Example (v2):
        <context>
          <include msg="user_question"/>
          <compress msgs="0..50" to="ctx"/>
        </context>
        <lvar reasoning r>...</lvar>
        <lact search s>search(query="AI")</lact>
        <yield for="s" reason="need results"/>
        OUT{answer: [r]}
    """

    lvars: list[Lvar | RLvar]  # Both namespaced and raw lvars
    lacts: list[Lact]
    out_block: OutBlock | None
    # v2 additions
    context: ContextBlock | None = None  # Context engineering
    yields: list[YieldStmt] | None = None  # Continuation control
    sends: list[SendStmt] | None = None  # Multi-agent communication


__all__ = (
    "ASTNode",
    # v2: Context Management
    "CompressDirective",
    "ContextBlock",
    "ContextDirective",
    "DropDirective",
    "Expr",
    "Identifier",
    "IncludeDirective",
    "Lact",
    "Literal",
    "Lvar",
    "NoticeDirective",
    "OutBlock",
    "Program",
    "RLvar",
    "Ref",
    # v2: Multi-Agent
    "SendStmt",
    "Stmt",
    # v2: Continuation
    "YieldStmt",
)
