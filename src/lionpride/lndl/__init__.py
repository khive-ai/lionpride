# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from .ast import (
    # v2 nodes - Context Management
    CompressDirective,
    ContextBlock,
    ContextDirective,
    DropDirective,
    # v1 nodes
    Identifier,
    IncludeDirective,
    Lact,
    Literal,
    Lvar,
    NoticeDirective,
    OutBlock,
    Program,
    # v2 nodes - References
    Ref,
    RLvar,
    # v2 nodes - Multi-Agent
    SendStmt,
    # v2 nodes - Continuation
    YieldStmt,
)
from .cognitive import (
    CognitiveOutput,
    CognitiveSend,
    CognitiveState,
    CognitiveYield,
    execute_cognitive,
    parse_cognitive,
)
from .errors import (
    AmbiguousMatchError,
    InvalidConstructorError,
    LNDLError,
    MissingFieldError,
    MissingLvarError,
    MissingOutBlockError,
    TypeMismatchError,
)
from .fuzzy import parse_lndl_fuzzy
from .lexer import Lexer, Token, TokenType
from .operations import (
    CognitivePermission,
    CognitiveReactResult,
    cognitive_react,
    compress_messages,
    stream_cognitive,
    transform_observation,
)
from .parser import ParseError, Parser
from .prompt import LNDL_SYSTEM_PROMPT, get_lndl_system_prompt
from .resolver import parse_lndl, resolve_references_prefixed
from .types import (
    ActionCall,
    LactMetadata,
    LNDLOutput,
    LvarMetadata,
    ParsedConstructor,
    RLvarMetadata,
    Scalar,
    ensure_no_action_calls,
    has_action_calls,
    revalidate_with_action_results,
)

__all__ = (
    "LNDL_SYSTEM_PROMPT",
    "ActionCall",
    "AmbiguousMatchError",
    # v2: Cognitive Runtime
    "CognitiveOutput",
    "CognitivePermission",
    "CognitiveReactResult",
    "CognitiveSend",
    "CognitiveState",
    "CognitiveYield",
    # v2: Context Management
    "CompressDirective",
    "ContextBlock",
    "ContextDirective",
    "DropDirective",
    "Identifier",
    "IncludeDirective",
    "InvalidConstructorError",
    "LNDLError",
    "LNDLOutput",
    "Lact",
    "LactMetadata",
    "Lexer",
    "Literal",
    "Lvar",
    "LvarMetadata",
    "MissingFieldError",
    "MissingLvarError",
    "MissingOutBlockError",
    "NoticeDirective",
    "OutBlock",
    "ParseError",
    "ParsedConstructor",
    "Parser",
    "Program",
    "RLvar",
    "RLvarMetadata",
    # v2: References
    "Ref",
    "Scalar",
    # v2: Multi-Agent
    "SendStmt",
    "Token",
    "TokenType",
    "TypeMismatchError",
    # v2: Continuation
    "YieldStmt",
    # v2: Cognitive Operations
    "cognitive_react",
    "compress_messages",
    "ensure_no_action_calls",
    "execute_cognitive",
    "get_lndl_system_prompt",
    "has_action_calls",
    "parse_cognitive",
    "parse_lndl",
    "parse_lndl_fuzzy",
    "resolve_references_prefixed",
    "revalidate_with_action_results",
    "stream_cognitive",
    "transform_observation",
)
