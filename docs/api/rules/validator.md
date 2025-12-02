# Validator

> Orchestrates rules over Operable specs for field-by-field validation with auto-correction

## Overview

`Validator` is the **orchestration engine** for lionpride's rule-based validation system. It bridges the gap between raw data (e.g., LLM responses) and validated, type-safe output by applying appropriate validation rules to each field defined in an `Operable` specification.

**Key Capabilities:**

- **Automatic Rule Assignment**: Resolves rules based on Spec metadata, field name, or base type
- **Capability-Based Access Control**: Validates only fields within allowed capabilities set
- **Auto-Correction**: Optionally fixes invalid values using rule-specific correction logic
- **Error Tracking**: Maintains validation log with timestamps for debugging and auditing
- **Async Support**: All validation methods are async for compatibility with async validators

**Use for**: LLM output parsing, capability-filtered validation, type coercion pipelines.
**Skip for**: Simple Pydantic validation (use `model_validate`) or single-value checks (use Rule directly).

See [Rule](rule.md) for rule implementations and [Operable](../types/operable.md) for specifications.

## Class Signature

```python
from lionpride.rules import Validator

class Validator:
    """Validates data spec-by-spec using auto-assigned Rules from Spec.base_type."""

    # Constructor signature
    def __init__(
        self,
        registry: RuleRegistry | None = None,
    ) -> None: ...
```

## Parameters

### Constructor Parameters

**registry** : RuleRegistry, optional

Registry for type-to-rule and name-to-rule mappings. If `None`, uses the default registry with standard rules (StringRule, NumberRule, BooleanRule, MappingRule, etc.).

- Default: `None` (uses `get_default_registry()`)
- Usage: Custom registries for domain-specific validation rules

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `registry` | `RuleRegistry` | Rule registry for type/name-to-rule lookup |
| `validation_log` | `list[dict[str, Any]]` | Log of validation errors with timestamps |

## Methods

### Core Validation

#### `validate_operable()`

Validate data against an Operable specification, validating all fields.

**Signature:**

```python
async def validate_operable(
    self,
    data: dict[str, Any],
    operable: Operable,
    auto_fix: bool = True,
    strict: bool = True,
) -> dict[str, Any]: ...
```

**Parameters:**

- `data` (dict[str, Any]): Raw data dictionary to validate (e.g., from LLM response)
- `operable` (Operable): Specification defining expected field structure
- `auto_fix` (bool, default True): Enable auto-correction for invalid values
- `strict` (bool, default True): Raise ValidationError if no rule found for a field

**Returns:**

- dict[str, Any]: Dictionary of validated (and possibly corrected) field values

**Raises:**

- ValidationError: If validation fails and cannot be auto-fixed

**Examples:**

```python
>>> from lionpride.rules import Validator
>>> from lionpride.types import Spec, Operable

>>> operable = Operable([
...     Spec(float, name="confidence"),
...     Spec(str, name="output")
... ])

>>> validator = Validator()
>>> validated = await validator.validate_operable(
...     data={"confidence": "0.95", "output": 42},
...     operable=operable,
...     auto_fix=True
... )
>>> validated
{'confidence': 0.95, 'output': '42'}
```

**Notes:**

This is a convenience wrapper around `_validate_data` that validates all fields defined in the operable. For capability-filtered validation, use `validate()` instead.

#### `validate()`

Validate data with capability filtering and return a model instance.

**Signature:**

```python
async def validate(
    self,
    data: dict[str, Any],
    operable: Operable,
    capabilities: set[str],
    auto_fix: bool = True,
    strict: bool = True,
) -> Any: ...
```

**Parameters:**

- `data` (dict[str, Any]): Raw data dictionary to validate
- `operable` (Operable): Specification defining expected field structure
- `capabilities` (set[str]): Set of field names allowed to be validated (access control)
- `auto_fix` (bool, default True): Enable auto-correction for invalid values
- `strict` (bool, default True): Raise ValidationError if no rule found

**Returns:**

- Any: Validated model instance (type depends on Operable's adapter)

**Raises:**

- ValidationError: If validation fails

**Examples:**

```python
>>> from lionpride.rules import Validator
>>> from lionpride.types import Spec, Operable

>>> operable = Operable([
...     Spec(str, name="public_field"),
...     Spec(str, name="private_field"),
...     Spec(int, name="admin_field"),
... ])

>>> validator = Validator()

# User with limited capabilities
>>> user_capabilities = {"public_field"}
>>> result = await validator.validate(
...     data={"public_field": "value", "private_field": "secret", "admin_field": 42},
...     operable=operable,
...     capabilities=user_capabilities
... )
# Only public_field is validated and included in result
```

**Notes:**

This is the **security microkernel** - all capability-based access control flows through these few lines. The flow is:

1. Validate data field-by-field with rules (respects capabilities)
2. Create model from operable with allowed capabilities
3. Validate dict into model instance

Empty capabilities set means nothing is validated.

#### `validate_spec()`

Validate a single value against a Spec.

**Signature:**

```python
async def validate_spec(
    self,
    spec: Spec,
    value: Any,
    auto_fix: bool = True,
    strict: bool = True,
) -> Any: ...
```

**Parameters:**

- `spec` (Spec): Specification defining the field
- `value` (Any): Value to validate
- `auto_fix` (bool, default True): Enable auto-correction
- `strict` (bool, default True): Raise if no rule applies

**Returns:**

- Any: Validated (and possibly corrected) value

**Raises:**

- ValidationError: If validation fails

**Examples:**

```python
>>> from lionpride.rules import Validator
>>> from lionpride.types import Spec

>>> validator = Validator()
>>> spec = Spec(float, name="score")

# String to float conversion with auto_fix
>>> result = await validator.validate_spec(spec, "0.85", auto_fix=True)
>>> result
0.85

# Nullable spec with None value
>>> nullable_spec = Spec(str, name="optional", nullable=True)
>>> result = await validator.validate_spec(nullable_spec, None)
>>> result is None
True
```

**Notes:**

Handles:

- **nullable**: Returns None if value is None and spec is nullable
- **default**: Uses sync/async default factory if value is None
- **listable**: Validates each item in list against base_type
- **validator**: Applies Spec's custom validators after rule validation

### Rule Lookup

#### `get_rule_for_spec()`

Get the appropriate Rule for a Spec based on lookup priority.

**Signature:**

```python
def get_rule_for_spec(self, spec: Spec) -> Rule | None: ...
```

**Parameters:**

- `spec` (Spec): Specification to find rule for

**Returns:**

- Rule | None: Rule instance if found, None otherwise

**Examples:**

```python
>>> from lionpride.rules import Validator, NumberRule
>>> from lionpride.types import Spec

>>> validator = Validator()

# Priority 1: Spec metadata override
>>> custom_rule = NumberRule(lower_bound=0, upper_bound=1)
>>> spec_with_override = Spec(float, name="score", rule=custom_rule)
>>> rule = validator.get_rule_for_spec(spec_with_override)
>>> rule is custom_rule
True

# Priority 2/3: Registry lookup by name then type
>>> spec_standard = Spec(float, name="confidence")
>>> rule = validator.get_rule_for_spec(spec_standard)
>>> type(rule).__name__
'NumberRule'
```

**Rule Lookup Priority:**

1. **Spec metadata "rule" override**: Explicit Rule instance in spec's metadata
2. **Field name match in registry**: Registry lookup by `spec.name`
3. **Base type match in registry**: Registry lookup by `spec.base_type`

### Error Tracking

#### `log_validation_error()`

Log a validation error with timestamp.

**Signature:**

```python
def log_validation_error(
    self,
    field: str,
    value: Any,
    error: str,
) -> None: ...
```

**Parameters:**

- `field` (str): Field name that failed validation
- `value` (Any): Value that failed validation
- `error` (str): Error message

**Examples:**

```python
>>> validator = Validator()
>>> validator.log_validation_error("age", "invalid", "Expected int, got str")
>>> validator.validation_log
[{'field': 'age', 'value': 'invalid', 'error': 'Expected int, got str', 'timestamp': '...'}]
```

#### `get_validation_summary()`

Get summary of validation errors.

**Signature:**

```python
def get_validation_summary(self) -> dict[str, Any]: ...
```

**Returns:**

- dict[str, Any]: Dictionary with:
  - `total_errors` (int): Total count of logged errors
  - `fields_with_errors` (list[str]): Sorted list of field names with errors
  - `error_entries` (list[dict]): Full error log entries

**Examples:**

```python
>>> validator = Validator()
>>> validator.log_validation_error("age", "abc", "Invalid integer")
>>> validator.log_validation_error("age", "xyz", "Invalid integer")
>>> validator.log_validation_error("score", "bad", "Invalid float")

>>> summary = validator.get_validation_summary()
>>> summary["total_errors"]
3
>>> summary["fields_with_errors"]
['age', 'score']
```

## Usage Patterns

### Basic Validation with Auto-Fix

```python
from lionpride.rules import Validator
from lionpride.types import Spec, Operable

operable = Operable([
    Spec(str, name="task"),
    Spec(float, name="confidence"),
])

validator = Validator()
validated = await validator.validate_operable(
    data={"task": "analyze", "confidence": "0.92"},  # String to float
    operable=operable,
    auto_fix=True
)  # {"task": "analyze", "confidence": 0.92}
```

### Capability-Based Access Control

```python
# Role-based field filtering
public_caps = {"username", "email"}
admin_caps = {"username", "email", "password_hash", "is_admin"}

public_result = await validator.validate(data, operable, capabilities=public_caps)
admin_result = await validator.validate(data, operable, capabilities=admin_caps)
```

### Error Handling

```python
try:
    await validator.validate_operable(data, operable, auto_fix=False, strict=True)
except ValidationError:
    summary = validator.get_validation_summary()
    for entry in summary['error_entries']:
        print(f"{entry['field']}: {entry['error']}")
```

## Common Pitfalls

- **Missing await**: All validation methods are async. `validator.validate_operable(...)` returns a coroutine.

- **Empty capabilities**: `validate(..., capabilities=set())` validates nothing. Use `validate_operable()` for all fields.

- **Strict mode without rules**: `strict=True` raises error if no rule exists for a type. Register custom rule or use `strict=False`.

## Design Rationale

1. **Spec-by-Spec validation**: Field-level granularity enables different rules per field, precise error localization, and partial success on multi-field data.

2. **Rule lookup priority (metadata > name > type)**: Specs can override defaults, named fields get domain-specific rules, types provide sensible fallback.

3. **Capability-based access control**: Security microkernel pattern - `validate()` enforces field-level permissions at the validation layer.

## See Also

- [Rule](rule.md): Base validation rule class
- [Operable](../types/operable.md): Specification container
- [Spec](../types/spec.md): Field specification

## Examples

### LLM Response Validation

```python
from lionpride.rules import Validator
from lionpride.types import Spec, Operable

operable = Operable([
    Spec(str, name="task"),
    Spec(float, name="confidence"),
])

validator = Validator()
llm_response = {"task": "Analyze", "confidence": "0.92"}  # String

validated = await validator.validate_operable(
    data=llm_response, operable=operable, auto_fix=True
)
# {"task": "Analyze", "confidence": 0.92}  # Float
```

### Custom Registry with Domain Rules

```python
from lionpride.rules import Validator, RuleRegistry, NumberRule

registry = RuleRegistry()
registry.register("confidence", NumberRule(ge=0.0, le=1.0, auto_fix=True))

validator = Validator(registry=registry)
result = await validator.validate_operable(
    data={"confidence": 1.5},  # Out of range
    operable=operable,
    auto_fix=True
)  # {"confidence": 1.0}  # Clamped
```
