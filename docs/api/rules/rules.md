# Rules

> Validation and auto-correction system for the Intelligence Processing Unit (IPU).

## Overview

The `lionpride.rules` module provides a rule-based validation system with automatic type coercion and error correction. It serves as the validation layer of the IPU pipeline, enabling structured output validation from LLM responses.

**Key capabilities:**

- **Rule-based validation**: Define validation constraints (length, range, pattern) declaratively
- **Auto-correction**: Automatically fix invalid values when possible (string conversion, type coercion)
- **Type-driven assignment**: Rules auto-assigned based on `Spec.base_type` via `RuleRegistry`
- **Qualifier system**: Control when rules apply (by field name, type annotation, or custom condition)

**Architecture:**

```text
Spec.base_type --> RuleRegistry --> Rule.invoke() --> validated value
                        |                  |
                   auto-assign      validate + fix
```

## Module Exports

```python
from lionpride.rules import (
    # Base classes
    Rule,
    RuleParams,
    RuleQualifier,
    ValidationError,

    # Built-in rules
    StringRule,
    NumberRule,
    BooleanRule,
    ChoiceRule,
    MappingRule,
    BaseModelRule,

    # Registry and validation
    RuleRegistry,
    get_default_registry,
    Validator,
)
```

---

## Rule Base Class

The `Rule` class defines the validation interface with three core methods.

### Class Signature

```python
class Rule:
    """Base validation rule with auto-correction support."""

    def __init__(
        self,
        params: RuleParams,
        **kw: Any,
    ) -> None: ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | `RuleParams` | Immutable configuration defining rule behavior |
| `**kw` | `Any` | Additional validation kwargs (merged with `params.kw`) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `RuleParams` | Rule configuration |
| `apply_types` | `set[type]` | Types this rule applies to (e.g., `{str, int}`) |
| `apply_fields` | `set[str]` | Field names this rule applies to |
| `default_qualifier` | `RuleQualifier` | Preferred qualifier type |
| `auto_fix` | `bool` | Whether auto-fixing is enabled |
| `validation_kwargs` | `dict` | Additional validation parameters |

### Methods

#### `apply()`

Check if rule applies to a field using qualifier precedence.

**Signature:**

```python
async def apply(
    self,
    k: str,
    v: Any,
    t: type | None = None,
    qualifier: str | RuleQualifier | None = None,
    **kw: Any,
) -> bool: ...
```

**Parameters:**

- `k` (str): Field name
- `v` (Any): Field value
- `t` (type, optional): Field type (defaults to `type(v)`)
- `qualifier` (str or RuleQualifier, optional): Override default qualifier order
- `**kw` (Any): Additional kwargs for condition checking

**Returns:**

- `bool`: True if rule applies (any qualifier matches)

**Qualifier precedence:** `FIELD > ANNOTATION > CONDITION`

#### `validate()`

Abstract method - validate value (implement in subclass).

**Signature:**

```python
async def validate(self, v: Any, t: type, **kw: Any) -> None: ...
```

**Parameters:**

- `v` (Any): Value to validate
- `t` (type): Expected type
- `**kw` (Any): Additional validation parameters

**Raises:**

- `Exception`: If validation fails

#### `perform_fix()`

Attempt to fix invalid value (optional, override in subclass).

**Signature:**

```python
async def perform_fix(self, v: Any, t: type) -> Any: ...
```

**Parameters:**

- `v` (Any): Value to fix
- `t` (type): Expected type

**Returns:**

- `Any`: Fixed value

**Raises:**

- `NotImplementedError`: If `auto_fix=True` but `perform_fix()` not implemented

#### `invoke()`

Execute validation with optional auto-fixing.

**Signature:**

```python
async def invoke(
    self,
    k: str,
    v: Any,
    t: type | None = None,
    *,
    auto_fix: bool | None = None,
) -> Any: ...
```

**Parameters:**

- `k` (str): Field name (for error messages)
- `v` (Any): Value to validate
- `t` (type, optional): Field type
- `auto_fix` (bool, optional): Override `self.auto_fix` for this invocation

**Returns:**

- `Any`: Validated (and possibly fixed) value

**Raises:**

- `ValidationError`: If validation fails and auto_fix disabled

---

## RuleParams

Immutable configuration for rules.

### Class Signature

```python
@dataclass(slots=True, frozen=True)
class RuleParams(Params):
    apply_types: set[type] = field(default_factory=set)
    apply_fields: set[str] = field(default_factory=set)
    default_qualifier: RuleQualifier = RuleQualifier.FIELD
    auto_fix: bool = False
    kw: dict = field(default_factory=dict)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `apply_types` | `set[type]` | Types this rule applies to (e.g., `{str, int}`) |
| `apply_fields` | `set[str]` | Field names this rule applies to |
| `default_qualifier` | `RuleQualifier` | Preferred qualifier type |
| `auto_fix` | `bool` | Enable automatic fixing on validation failure |
| `kw` | `dict` | Additional validation parameters |

**Constraint:** Must set exactly one of `apply_types` or `apply_fields` (unless using `CONDITION` qualifier).

---

## RuleQualifier

Determines when a rule applies.

```python
class RuleQualifier(IntEnum):
    FIELD = auto()       # Match by field name
    ANNOTATION = auto()  # Match by type annotation
    CONDITION = auto()   # Match by custom condition
```

**Default precedence:** `FIELD > ANNOTATION > CONDITION`

---

## Built-in Rules

### StringRule

Validates and converts string values with length and pattern constraints.

**Signature:**

```python
class StringRule(Rule):
    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        params: RuleParams | None = None,
        **kw: Any,
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_length` | `int`, optional | Minimum string length (inclusive) |
| `max_length` | `int`, optional | Maximum string length (inclusive) |
| `pattern` | `str`, optional | Regex pattern to match |
| `params` | `RuleParams`, optional | Custom params (uses default if None) |

**Auto-fix behavior:** Converts any type to string via `str()`, then re-validates.

**Default configuration:**

- `apply_types`: `{str}`
- `default_qualifier`: `ANNOTATION`
- `auto_fix`: `True`

**Examples:**

```python
from lionpride.rules import StringRule

# Type coercion with auto-fix
rule = StringRule()
result = await rule.invoke("name", 42, str, auto_fix=True)  # "42"

# Length + pattern constraints
rule = StringRule(min_length=1, max_length=100, pattern=r'^[A-Za-z]+$')
await rule.invoke("code", "ABC123", str)  # Raises ValidationError
```

---

### NumberRule

Validates and converts numeric values with range constraints.

**Signature:**

```python
class NumberRule(Rule):
    def __init__(
        self,
        ge: int | float | None = None,
        gt: int | float | None = None,
        le: int | float | None = None,
        lt: int | float | None = None,
        params: RuleParams | None = None,
        **kw: Any,
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ge` | `int` or `float`, optional | Greater than or equal to (>=) |
| `gt` | `int` or `float`, optional | Greater than (>) |
| `le` | `int` or `float`, optional | Less than or equal to (<=) |
| `lt` | `int` or `float`, optional | Less than (<) |
| `params` | `RuleParams`, optional | Custom params (uses default if None) |

**Auto-fix behavior:** Converts strings to `int()` or `float()` based on target type.

**Default configuration:**

- `apply_types`: `{int, float}`
- `default_qualifier`: `ANNOTATION`
- `auto_fix`: `True`

**Examples:**

```python
from lionpride.rules import NumberRule

# Range validation with string-to-float coercion
rule = NumberRule(ge=0.0, le=1.0)
result = await rule.invoke("confidence", "0.85", float, auto_fix=True)  # 0.85

# Positive integer constraint
rule = NumberRule(gt=0)
await rule.invoke("count", -5, int)  # Raises ValidationError
```

---

### BooleanRule

Validates and converts boolean values with flexible string parsing.

**Signature:**

```python
class BooleanRule(Rule):
    def __init__(
        self,
        params: RuleParams | None = None,
        **kw: Any,
    ) -> None: ...
```

**Auto-fix behavior:**

- Strings: `"true"`, `"yes"`, `"1"`, `"on"` --> `True` (case-insensitive)
- Strings: `"false"`, `"no"`, `"0"`, `"off"` --> `False` (case-insensitive)
- Numbers: `0` --> `False`, non-zero --> `True`

**Default configuration:**

- `apply_types`: `{bool}`
- `default_qualifier`: `ANNOTATION`
- `auto_fix`: `True`

**Examples:**

```python
from lionpride.rules import BooleanRule

rule = BooleanRule()
await rule.invoke("active", "yes", bool, auto_fix=True)   # True
await rule.invoke("enabled", "FALSE", bool, auto_fix=True) # False
await rule.invoke("flag", 1, bool, auto_fix=True)          # True
```

---

### ChoiceRule

Validates values against an allowed set with optional case-insensitive matching.

**Signature:**

```python
class ChoiceRule(Rule):
    def __init__(
        self,
        choices: set[Any] | list[Any],
        case_sensitive: bool = True,
        apply_fields: set[str] | None = None,
        apply_types: set[type] | None = None,
        params: RuleParams | None = None,
        **kw: Any,
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `choices` | `set` or `list` | Allowed values |
| `case_sensitive` | `bool`, default `True` | Whether string matching is case-sensitive |
| `apply_fields` | `set[str]`, optional | Field names to apply to |
| `apply_types` | `set[type]`, optional | Types to apply to |

**Auto-fix behavior:** For `case_sensitive=False`, normalizes to canonical case.

**Examples:**

```python
from lionpride.rules import ChoiceRule

# Case-insensitive normalization
rule = ChoiceRule(choices=["low", "medium", "high"], case_sensitive=False)
await rule.invoke("priority", "HIGH", str, auto_fix=True)  # "high"

# Exact match (case-sensitive)
rule = ChoiceRule(choices={"pending", "active"})
await rule.invoke("status", "PENDING", str)  # Raises ValidationError
```

---

### MappingRule

Validates dict/mapping values with key constraints and JSON parsing.

**Signature:**

```python
class MappingRule(Rule):
    def __init__(
        self,
        required_keys: set[str] | None = None,
        optional_keys: set[str] | None = None,
        fuzzy_keys: bool = False,
        params: RuleParams | None = None,
        **kw: Any,
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `required_keys` | `set[str]`, optional | Keys that must be present |
| `optional_keys` | `set[str]`, optional | Keys that may be present |
| `fuzzy_keys` | `bool`, default `False` | Enable case-insensitive key matching |

**Auto-fix behavior:**

- JSON string --> parsed dict
- Fuzzy key normalization (if enabled)

**Default configuration:**

- `apply_types`: `{dict}`
- `default_qualifier`: `ANNOTATION`
- `auto_fix`: `True`

**Examples:**

```python
from lionpride.rules import MappingRule

# JSON string parsing + required keys
rule = MappingRule(required_keys={"name", "value"})
await rule.invoke("data", '{"name": "test", "value": 42}', dict, auto_fix=True)

# Fuzzy key matching (case-insensitive)
rule = MappingRule(required_keys={"name"}, fuzzy_keys=True)
await rule.invoke("config", {"NAME": "test"}, dict, auto_fix=True)  # {"name": "test"}
```

---

### BaseModelRule

Validates Pydantic BaseModel subclasses with fuzzy parsing support.

**Signature:**

```python
class BaseModelRule(Rule):
    def __init__(
        self,
        fuzzy_parse: bool = True,
        fuzzy_match: bool = False,
        params: RuleParams | None = None,
        **kw: Any,
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `fuzzy_parse` | `bool`, default `True` | Enable fuzzy JSON extraction from text |
| `fuzzy_match` | `bool`, default `False` | Enable fuzzy key matching for field names |

**Auto-fix behavior:**

- Dict --> model validation
- String --> JSON extraction --> model validation
- Uses `fuzzy_validate_pydantic()` for flexible parsing

**Default configuration:**

- `apply_types`: `{BaseModel}`
- `default_qualifier`: `ANNOTATION`
- `auto_fix`: `True`

**Examples:**

```python
from pydantic import BaseModel
from lionpride.rules import BaseModelRule

class Person(BaseModel):
    name: str
    age: int

rule = BaseModelRule()

# Dict or JSON string -> model (fuzzy extraction from surrounding text)
await rule.invoke("person", {"name": "Ocean", "age": 30}, Person, auto_fix=True)
await rule.invoke("person", 'Data: {"name": "Ocean", "age": 30}', Person, auto_fix=True)
```

---

## RuleRegistry

Maps types to Rule instances for automatic rule assignment.

### Class Signature

```python
class RuleRegistry:
    def __init__(self) -> None: ...
```

### Methods

#### `register()`

Register a Rule for a type or field name.

**Signature:**

```python
def register(
    self,
    key: type | str,
    rule: Rule,
    *,
    replace: bool = False,
) -> None: ...
```

**Parameters:**

- `key` (type or str): Type or field name to register
- `rule` (Rule): Rule instance to use
- `replace` (bool): Allow replacing existing registration

#### `get_rule()`

Get Rule for a type or field name.

**Signature:**

```python
def get_rule(
    self,
    base_type: type | None = None,
    field_name: str | None = None,
) -> Rule | None: ...
```

**Priority:**

1. Exact field name match
2. Exact type match
3. Inheritance-based type match (subclasses inherit parent rules)

#### `has_rule()`

Check if a rule is registered.

```python
def has_rule(self, key: type | str) -> bool: ...
```

#### `list_types()` / `list_names()`

List registered types or field names.

```python
def list_types(self) -> list[type]: ...
def list_names(self) -> list[str]: ...
```

### Default Registry

`get_default_registry()` returns a pre-configured registry with standard mappings:

| Type | Rule |
|------|------|
| `str` | `StringRule()` |
| `int` | `NumberRule()` |
| `float` | `NumberRule()` |
| `bool` | `BooleanRule()` |
| `dict` | `MappingRule()` |
| `BaseModel` | `BaseModelRule()` |

**Examples:**

```python
from lionpride.rules import RuleRegistry, NumberRule, get_default_registry

registry = RuleRegistry()
registry.register(int, NumberRule(ge=0))                      # By type
registry.register("confidence", NumberRule(ge=0.0, le=1.0))   # By field name (higher priority)

rule = registry.get_rule(field_name="confidence")  # Field name > type > inheritance
default = get_default_registry()                   # Pre-configured with standard rules
```

---

## Validator

Orchestrates rule-based validation over Operable specs.

### Class Signature

```python
class Validator:
    def __init__(
        self,
        registry: RuleRegistry | None = None,
    ) -> None: ...
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `registry` | `RuleRegistry` | Rule registry for type-to-rule lookup |
| `validation_log` | `list[dict]` | Log of validation errors |

### Methods

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

**Handles:**

- `nullable`: Returns None if value is None and spec is nullable
- `default`: Uses sync/async default factory if value is None
- `listable`: Validates each item in list against base_type
- `validator`: Applies Spec's custom validators after rule validation

#### `validate_operable()`

Validate data spec-by-spec against an Operable.

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

- `data` (dict): Raw data dict (e.g., from LLM response)
- `operable` (Operable): Operable defining expected structure
- `auto_fix` (bool): Enable auto-correction for each field
- `strict` (bool): Raise if validation fails

**Returns:**

- `dict`: Validated field values

#### `validate()`

Validate data and return a model instance (security microkernel).

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

**Flow:**

1. Validate data field-by-field with rules (respects capabilities)
2. Create model from operable with allowed capabilities
3. Validate dict into model instance

#### `get_validation_summary()`

Get summary of validation errors.

```python
def get_validation_summary(self) -> dict[str, Any]: ...
```

**Returns:** Dict with `total_errors`, `fields_with_errors`, and `error_entries`.

### Examples

```python
from lionpride.rules import Validator
from lionpride.types import Spec, Operable

# Define specs
operable = Operable([
    Spec(float, name="confidence"),
    Spec(str, name="output"),
])

# Validate raw LLM response
validator = Validator()
validated = await validator.validate_operable(
    data={"confidence": "0.95", "output": 42},  # Raw types
    operable=operable,
    auto_fix=True
)
# validated: {"confidence": 0.95, "output": "42"}

# With capability-based access control
result = await validator.validate(
    data={"confidence": "0.95", "output": "result"},
    operable=operable,
    capabilities={"confidence", "output"},
)
# result: Pydantic model instance
```

---

## Usage Patterns

### Custom Rule

```python
from lionpride.rules import Rule, RuleParams, RuleQualifier

class EmailRule(Rule):
    def __init__(self):
        super().__init__(RuleParams(apply_types={str}, auto_fix=False))

    async def validate(self, v, t, **kw):
        if not isinstance(v, str) or "@" not in v:
            raise ValueError(f"Invalid email: {v}")

await EmailRule().invoke("email", "user@example.com", str)
```

### LLM Response Validation

```python
from lionpride.rules import Validator
from lionpride.types import Spec, Operable

operable = Operable([
    Spec(str, name="reasoning"),
    Spec(float, name="confidence"),
])

validator = Validator()
result = await validator.validate_operable(
    data={"reasoning": "...", "confidence": "0.92"},  # String to float
    operable=operable,
    auto_fix=True,
)  # {"reasoning": "...", "confidence": 0.92}
```

---

## Common Pitfalls

- **Missing await**: All rule methods are async. Always use `await rule.invoke(...)`.

- **Auto-fix default**: Rules default to `auto_fix=True`. Set `auto_fix=False` for strict validation without coercion.

---

## Design Rationale

1. **Separation of concerns**: Rules define validation; Registry maps types to rules; Validator orchestrates.

2. **Qualifier precedence (FIELD > ANNOTATION > CONDITION)**: Field-specific rules override type-based rules.

3. **Auto-fix by default**: LLM outputs are imprecise. Auto-correction provides resilience without sacrificing type safety.

---

## See Also

- [`Spec`](../types/spec.md): Field specification with validation
- [`Operable`](../types/operable.md): Collection of specs that generates Pydantic models
- [`Validator`](../rules/validator.md): Full validator documentation
- [User Guide: Validation](../../user_guide/validation.md): Conceptual overview
