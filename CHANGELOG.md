# Changelog

All notable changes to lionpride will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0a10] - 2025-12-31

### Added

- Comprehensive API documentation for session module (#148)
  - `mail.md` - Mail, Exchange, OUTBOX multi-agent communication
  - `logs.md` - Log, LogStore, LogType, LogStoreConfig
  - `log_adapter.md` - LogAdapter, SQLiteWALLogAdapter, PostgresLogAdapter
  - `log_broadcaster.md` - LogBroadcaster, subscribers, LogRedactor
- Comprehensive API documentation for services module (#148)
  - `backend.md` - ServiceBackend, ServiceConfig, NormalizedResponse, Calling
  - `endpoint.md` - Endpoint, EndpointConfig, APICalling
  - `utilities.md` - RateLimitConfig, TokenBucket, CircuitBreaker, RetryConfig
  - `mcps.md` - MCPConnectionPool, MCPSecurityConfig, create_mcp_pool
  - `hook.md` - HookRegistry, HookEvent, HookPhase
- Export `Mail`, `Exchange`, `OUTBOX` from `lionpride.session` (#148)

### Fixed

- Documentation linting issues across existing docs (#148)

## [1.0.0a8] - 2025-12-12

### Added

- `ReportExecutor` retry/resume support for fault-tolerant workflows (#134)
- `deterministic_sets` parameter to `Element.to_json` for reproducible serialization (#131)
- `rapidfuzz` optional dependency for O([n/64]*m) string similarity algorithms (#137)
- `RAPIDFUZZ_AVAILABLE` flag for runtime detection of rapidfuzz availability (#137)
- `allow_inf` parameter to `to_num` for explicit infinity handling (#137)

### Security

- **NaN/Infinity bypass fix**: `to_num` now rejects NaN (always) and infinity (by default) to prevent bounds check bypass (#137)
- **Fuzzy JSON state machine**: Replaced regex-based parsing with state machine to prevent inside-string content corruption (#137)
- **Narrowed exception handling**: `extract_json` now catches only JSON-specific errors, not `MemoryError`/`KeyboardInterrupt` (#137)
- **Input size limits**: Added `MAX_STRING_LENGTH`, `MAX_JSON_INPUT_SIZE`, `MAX_NUMBER_STRING_LENGTH` constants (#137)
- **Schema handler hardening**: Added recursion depth limits, size caps, and cycle detection (#136)
- **Concurrency safety**: Fixed race conditions and improved error propagation (#135)

### Fixed

- Form error propagation from `ReportExecutor.execute()` (#133)
- Exception handling in `CompletionStream` and `is_sentinel` (#132)
- Cache files now properly ignored via `.gitignore`

### Changed

- String similarity functions use rapidfuzz when available, pure Python with size limits as fallback (#137)
- `fuzzy_json` preserves apostrophes and quotes inside string values correctly (#137)

## [1.0.0a7] - 2025-12-06

### Security

- Module allowlist for `load_type_from_string()` - blocks arbitrary code loading (#129)
- Session-scoped `MCPConnectionPoolInstance` with `create_mcp_pool()` factory (#91)

### Performance

- Session lazy initialization for `conversations`, `services`, `operations` (#112)
- `alcall` complexity reduced from 33 to 10; extract `LazyInit` utility (#115, #116)

### Changed

- Replace `print()` with `logging` module (#99)

### Deprecated

- `MCPConnectionPool.configure_security()` - use `create_mcp_pool()` instead

### Internal

- Test coverage for `factory.py` and `act.py` edge cases (#85)

## [1.0.0a6] - 2025-12-05

### Fixed

- Resolve 16 discovered issues across core, security, and observability (#60)
- Remove obsolete `_testing.py` module from package (#84)

### Documentation

- Remove AlcallParams/BcallParams references from API docs (#86)

### Internal

- Consolidate MockNormalizedResponse fixture to conftest.py (#87)
- Consolidate test suite structure (#84)

## [1.0.0a5] - 2025-12-05

### Added

- `ReportExecutor`: Event-driven executor using completion events (#43)
- `FormResult`: Rich result dataclass with progress tracking (#43)
- `execute_report`/`stream_report`: New executor-based API (#43)
- Cycle detection for form dependencies (#43)
- Input availability validation before form execution (#43)

### Changed

- Replace polling-based runner with event-driven executor (#43)
- Simplify `runner.py` to alias executor (`flow_report = execute_report`) (#43)
- Simplify work DSL: remove `operation` field, consolidate tool resolution (#41)
- Add streaming API to work module (#42)

### Removed

- `function_call_parser` module (obsolete) (#40)
- Unused `api_interpret`, `get_*` wrapper methods from FormResources (#41)

## [1.0.0a4] - 2025-12-02

### Added

- `CustomParser` protocol for extensible output parsing (#39)
- `CustomRenderer` protocol for extensible instruction rendering (#39)
- `custom_parser` param to `ParseParams` for external parser injection (#39)
- `custom_renderer` param to `GenerateParams` for external renderer injection (#39)

### Changed

- Replace `structure_format="lndl"` with `structure_format="custom"` (#39)

### Removed

- LNDL tutorial docs (lndl_architecture.md, structured_llm_outputs.md) (#39)
- Broken CLAUDE.md and AGENTS.md links from README (#39)

## [1.0.0a3] - 2025-12-02

### Added

- `Operable.from_model()` to disassemble Pydantic models into Spec fields (#32)
- `unescape_html` param to `minimal_yaml` with improved type hints (#34)
- `return_one` and `extract_all` params to `fuzzy_validate_pydantic` (#37)

### Changed

- Move spec extraction logic to `SpecAdapter` for cleaner separation (#38)
- Use internal `ln` and concurrency modules instead of naked json/asyncio (#31)
- Use `textwrap.indent` for cleaner message line indentation (#35)
- Pass provider and endpoint kwargs to OAIChatEndpoint fallback (#36)

### Documentation

- Comprehensive API reference for session, services, operations, rules, work (#33)
- Cleaned README.md by removing redundant example sections
- Removed outdated docstrings from workflow modules

### Removed

- Repo-level CLAUDE.md (consolidated into project docs) (#38)

## [1.0.0a2] - 2025-11-29

### Added

- **Declarative multi-agent workflows**: Pydantic model docstrings as agent instructions, schema as output contract
- **Explicit capability-based security**: Branch capabilities gate structured output fields
- **code_review_report.py example**: Working multi-agent code review workflow

### Changed

- **flow_report**: Simplified to use Report's `next_forms()` scheduling instead of graph compilation
- **Operations refactor**: Cleaner parameter hierarchy (GenerateParams → CommunicateParams → OperateParams → ReactParams)

### Fixed

- Include `action_responses` capability when `actions=True` (tool outputs now properly attached)
- Fixed 5 API errors in README.md, CLAUDE.md, AGENTS.md documentation
- Fixed broken notebook link in processor/executor docs

### Removed

- Pruned 44 redundant notebooks (51 → 7 essential primitives)
- Removed buggy `form_report_demo.py` example

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
- **Custom Parser Interface**: Extensible parser protocol for structured output formats
- **Concurrency**: TaskGroup, CancelScope, async patterns
- **Schema handlers**: TypeScript notation, function call parser

### Documentation

- Comprehensive CLAUDE.md and AGENTS.md guides
- Interactive Jupyter notebooks
- 99%+ test coverage

[Unreleased]: https://github.com/khive-ai/lionpride/compare/v1.0.0a8...HEAD
[1.0.0a8]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a8
[1.0.0a7]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a7
[1.0.0a6]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a6
[1.0.0a5]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a5
[1.0.0a4]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a4
[1.0.0a3]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a3
[1.0.0a2]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a2
[1.0.0a1]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0a1
