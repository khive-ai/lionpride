# Validator Enhancements from lionagi v0.2.2

## Overview

The `Validator` class has been enhanced with features from lionagi v0.2.2 to provide better validation control, error tracking, and rule precedence management. These enhancements make the validator more suitable for complex validation scenarios in the IPU (Intelligence Processing Unit).

## Key Enhancements

### 1. Validation Logging (`validation_log`)

**Purpose**: Track all validation attempts and errors for auditing and debugging.

**Implementation**:

- Added `validation_log: list[dict[str, Any]] = []` attribute to `__init__`
- Each log entry contains: `field`, `value`, `error`, `timestamp` (ISO format)

**Example**:

```python
validator = Validator()
validator.log_validation_error("age", -5, "Must be positive")

# Access the log
print(validator.validation_log)
# Output: [{'field': 'age', 'value': -5, 'error': 'Must be positive', 'timestamp': '2025-11-24T16:46:24.271807'}]
```

### 2. Rule Order (`rule_order`)

**Purpose**: Control the order in which rules are applied (deterministic, not random dict iteration).

**Implementation**:

- Added `rule_order: list[str]` parameter to `__init__`
- Rules are applied in the order specified, enabling predictable validation behavior
- If not specified, uses the order of keys in the `rules` dict

**Example**:

```python
# Custom rule order: try number validation before string
validator = Validator(rule_order=["number", "string", "boolean"])

# Rules will be checked in this specific order
result = await validator.validate_field("value", "42", str)
```

### 3. Strict Mode (`strict` parameter)

**Purpose**: Control behavior when no rule applies to a field.

**Implementation**:

- Added `strict: bool = True` parameter to `validate_field()` and `validate()`
- **strict=True**: Raises `ValidationError` if no rule applies (default, defensive)
- **strict=False**: Returns the value as-is if no rule applies (permissive)

**Example**:

```python
validator = Validator()

# Strict mode (raises)
try:
    await validator.validate_field("unknown_type", [], list, strict=True)
except ValidationError as e:
    print("No rule applied:", e)

# Permissive mode (returns value as-is)
result = await validator.validate_field("unknown_type", [1, 2, 3], list, strict=False)
assert result == [1, 2, 3]
```

### 4. Validation Error Logging Method

**Purpose**: Manually log validation errors with timestamps.

**Implementation**:

- Added `log_validation_error(field: str, value: Any, error: str)` method
- Automatically called when strict mode raises ValidationError
- Creates timestamped log entries for error tracking

**Example**:

```python
validator = Validator()
validator.log_validation_error("name", "", "Value cannot be empty")
```

### 5. Validation Summary (`get_validation_summary()`)

**Purpose**: Get an overview of validation history.

**Implementation**:

- Returns dict with:
  - `total_errors`: Number of logged errors
  - `fields_with_errors`: Sorted list of fields that had errors
  - `error_entries`: Full list of all error log entries

**Example**:

```python
validator = Validator()
validator.log_validation_error("field1", "value1", "error1")
validator.log_validation_error("field2", "value2", "error2")

summary = validator.get_validation_summary()
# {
#     'total_errors': 2,
#     'fields_with_errors': ['field1', 'field2'],
#     'error_entries': [...]
# }
```

## Design Decisions

### 1. Deterministic Rule Order

**Why**: The original implementation relied on dict key iteration order, which while deterministic in Python 3.7+ is still implicit. Making it explicit via `rule_order` parameter ensures:

- Predictable validation behavior
- Clear precedence in rule selection
- Easier debugging and testing

### 2. Dual Mode Validation (strict vs. permissive)

**Why**: Different use cases require different behaviors:

- **Strict mode**: Ensures all fields have applicable rules (fail-fast, good for structured APIs)
- **Permissive mode**: Allows passthrough of unmapped fields (good for flexible schemas)

### 3. Lightweight Logging

**Why**:

- Simple list of dicts for flexibility (can be easily serialized, analyzed, or persisted)
- Timestamps in ISO format for standards compliance
- No complex logging infrastructure (respects CLAUDE.md philosophy: keep it simple)

## Migration Guide

### From Original Validator

**Before**:

```python
validator = Validator()
result = await validator.validate_field("name", "Ocean", str)
```

**After** (with enhancements):

```python
# Define rule order explicitly
validator = Validator(rule_order=["string", "number", "boolean"])

# Validate with strict mode control
result = await validator.validate_field("name", "Ocean", str, strict=True)

# Access validation history
if validator.validation_log:
    summary = validator.get_validation_summary()
    print(f"Validation errors: {summary['total_errors']}")
```

## API Reference

### Constructor

```python
Validator(
    rules: dict[str, Rule] | None = None,
    rule_order: list[str] | None = None
)
```

### Methods

#### `log_validation_error(field: str, value: Any, error: str) -> None`

Log a validation error with timestamp.

#### `get_validation_summary() -> dict[str, Any]`

Get summary of validation log with error count and affected fields.

#### `async validate_field(..., strict: bool = True) -> Any`

Validate single field with strict mode control.

#### `async validate(..., strict: bool = True) -> dict[str, Any]`

Validate data dict with strict mode control propagated to all fields.

## Testing

All enhancements have been tested with:

- validation_log initialization and logging
- rule_order parameter and precedence
- strict mode with both raise and passthrough
- validation summary generation
- integration with existing rule system

Tests demonstrate:

1. Errors are logged with timestamps
2. Rules are applied in specified order
3. strict=True raises ValidationError for unmapped types
4. strict=False returns values as-is
5. Summary accurately reports error counts and affected fields

## Backward Compatibility

**Breaking changes**: None

- New parameters have defaults matching original behavior
- Existing code using `Validator()` continues to work
- New features are opt-in via parameters

## Files Modified

1. **src/lionpride/rules/validator.py** - Core enhancements
2. **src/lionpride/rules/**init**.py** - Export Validator
3. **src/lionpride/rules/base.py** - Added @dataclass to RuleParams
4. **src/lionpride/operations/operation.py** - Fixed imports, added create_operation
5. **src/lionpride/operations/builder.py** - Fixed imports
6. **src/lionpride/operations/flow.py** - Fixed imports
7. **src/lionpride/**init**.py** - Commented broken Builder import (temporary)

## References

- **lionagi v0.2.2**: `/Users/lion/projects/references/lionagi-0-2-2/lionagi_v02/core/validator/validator.py`
- Pattern: Validation → Structure → Usefulness (IPU pipeline)
- Design Philosophy: Wall Street rigor + Cornell physics precision
