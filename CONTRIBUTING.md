# Contributing to lionpride

## Code Standards

### File Headers

Every `.py` file MUST start with this exact header:

```python
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
```

No module-level docstrings in `src/`. The header is sufficient.

### Docstring Standards

**Philosophy**: Docstrings should be succinct, informationally dense, and API-focused. Design rationale and implementation details belong in test files, not source code.

#### Classes

```python
class Element(BaseModel):
    """Base identity for all lionpride entities.

    Attrs:
        id: Immutable UUID4 identifier.
        created_at: UTC timestamp, frozen at creation.
        metadata: Mutable key-value store.
    """
```

- One-line summary (what it is)
- `Attrs:` section listing public attributes with brief descriptions
- No implementation details, no "Design Philosophy" sections

#### Methods/Functions

```python
def to_dict(self, mode: str = "python") -> dict[str, Any]:
    """Serialize to dictionary.

    Args:
        mode: "python" (native types) or "json" (JSON-safe).

    Returns:
        Dict with lion_class for polymorphic deserialization.
    """
```

- One-line summary (what it does)
- `Args:` only for non-obvious parameters
- `Returns:` only if non-obvious
- `Raises:` only for expected exceptions users should handle
- No examples in src (put them in tests or docs)

#### What NOT to Include in src/ Docstrings

- Design philosophy or rationale
- Mathematical properties or invariants
- Implementation strategy explanations
- "Why" explanations (save for tests)
- Verbose examples (use tests)
- ASCII diagrams
- Historical context

#### Where Design Rationale Goes

Test files document the "why":

```python
# tests/core/test_element.py

"""Element test suite.

Design Rationale:
    Element enforces immutable identity (UUID + timestamp) for entity lifecycle
    tracking. The lion_class field enables polymorphic deserialization.

Invariants Verified:
    - Identity: e.id in UUID, e.created_at in DateTime(UTC)
    - Immutability: id and created_at are frozen post-creation
    - Roundtrip: from_dict(to_dict(e)) == e (by ID equality)
"""
```

### Test File Structure

Test files CAN have module-level docstrings explaining:

- What the test suite covers
- Design rationale moved from src
- Mathematical invariants being verified
- Edge cases and why they matter

### Import Order

1. Standard library
2. Third-party packages
3. Local imports (relative preferred within package)

Sorted alphabetically within each group. Use `isort` defaults.

## Development Workflow

### Running Tests

```bash
uv run pytest tests/ -v
```

### Coverage

```bash
uv run pytest tests/ --cov=lionpride --cov-report=term-missing
```

**Note**: Coverage must be run against the full test suite. Running coverage on individual test files (e.g., `pytest tests/ln/test_alcall.py --cov=...`) will show incorrect results due to module import ordering. Always use `pytest tests/` for accurate coverage reports.

### Linting

```bash
uv run ruff check . --fix
uv run ruff format .
```

### Pre-commit

Pre-commit hooks run automatically. If they modify files, re-stage and commit.

## Pull Request Guidelines

1. One logical change per PR
2. All tests must pass
3. Follow docstring standards above
4. Copyright headers on all new files
