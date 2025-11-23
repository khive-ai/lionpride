# Changelog

All notable changes to lionpride will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-alpha] - 2025-11-23

### Added

- Initial alpha release
- Core primitives: Element, Node, Pile, Progression, Flow, Graph, Event
- Protocol system: Observable, Serializable, Adaptable
- Type system: Spec, Operable, Meta, Sentinel types
- Concurrency utilities: TaskGroup, CancelScope, patterns
- LN utilities: async_call, fuzzy_match, json_dump, to_dict, to_list
- LNDL parser for structured LLM outputs
- Schema handlers: TypeScript notation, function call parser
- String handlers: JSON extraction, fuzzy matching, similarity
- Documentation and notebooks

### Changed

- Migrated from lionherd-core with full package rename to lionpride
- Import paths updated: `lionpride.base` → `lionpride.core`

[Unreleased]: https://github.com/khive-ai/lionpride/compare/v1.0.0-alpha...HEAD
[1.0.0-alpha]: https://github.com/khive-ai/lionpride/releases/tag/v1.0.0-alpha
