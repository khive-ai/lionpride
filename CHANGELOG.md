# Changelog

All notable changes to lionpride will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0a1] - 2025-11-28

Initial alpha release of lionpride - foundational primitives for production AI agents.

### Core Primitives

- **Element**: Base identity with UUID, timestamps, metadata
- **Node**: Polymorphic content container with adapter support
- **Pile[T]**: Type-safe O(1) collections with UUID lookup
- **Progression**: Ordered UUID sequences for workflow state
- **Flow[E, P]**: Composition pattern (items + progressions)
- **Graph**: Directed graphs with conditional edges and pathfinding
- **Event**: Async lifecycle tracking with timeout support

### Session System

- **Session**: Central orchestrator for messages, branches, services
- **Branch**: Named progression with capability/resource access control
- **Message**: Universal container with auto-derived roles
- **MessageContent**: Discriminated union (System, Instruction, Assistant, Action)

### Services

- **iModel**: Unified LLM interface (OpenAI, Anthropic, Gemini)
- **Tool**: Callable wrapper for LLM tool use
- **ServiceRegistry**: O(1) name-indexed service management
- **MCP Integration**: Model Context Protocol support

### Operations

- **generate**: Low-level LLM calls
- **parse**: Structured data extraction
- **communicate**: Generate + parse composition
- **operate**: Full structured output with validation
- **react**: Multi-turn ReAct pattern
- **flow**: Graph-based parallel execution

### Validation System

- **Rule/Validator**: Type-based validation with auto-fix
- **RuleRegistry**: Type to Rule auto-assignment
- **Built-in Rules**: String, Number, Boolean, Mapping, Choice, Reason, BaseModel

### Type System

- **Spec**: Field specifications with validators
- **Operable**: Spec collections generating Pydantic models

### Work System

- **Form**: Declarative unit of work with assignment DSL
- **Report**: Workflow orchestrator with schema introspection
- **flow_report**: Graph-compiled parallel execution

### Utilities

- **ln module**: alcall, bcall, fuzzy_match, json_dumps, to_dict, to_list, hash_dict
- **LNDL**: Lion Natural Description Language parser
- **Concurrency**: TaskGroup, CancelScope, async patterns
- **Schema handlers**: TypeScript notation, function call parser

### Documentation

- Comprehensive CLAUDE.md and AGENTS.md guides
- Interactive Jupyter notebooks
- 99%+ test coverage

[Unreleased]: https://github.com/khive-ai/lionpride/compare/v1.0.0a1...HEAD
[1.0.0a1]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a1
