# Rules

> Type-based validation with auto-correction for LLM outputs

## Overview

The `rules` module provides a validation pipeline for structured LLM outputs. It automatically assigns validation rules based on field types and supports auto-correction of invalid values.

**Validation Pipeline:**

```text
Spec.base_type -> auto Rule assignment -> validate spec-by-spec -> Operable.create_model()
```

**Key Capabilities:**

- **RuleRegistry**: Maps Python types to Rule classes (auto-assignment from Spec.base_type)
- **Built-in Rules**: StringRule, NumberRule, BooleanRule, MappingRule, ChoiceRule, etc.
- **Validator**: Orchestrates rules over Operable specs with parallel execution
- **Auto-Fix**: Convert invalid LLM outputs to valid values (e.g., `"0.95"` -> `0.95`)

The rules module integrates with the type system (`Spec`, `Operable`) to provide end-to-end validation from schema definition to validated Pydantic model instances.

## Classes

### Core

| Class | Description |
|-------|-------------|
| [Validator](validator.md) | Orchestrates validation over Operable specs |
| [RuleRegistry](registry.md) | Maps types to Rule classes |
| `get_default_registry` | Get the default rule registry |

### Rule Base

| Class | Description |
|-------|-------------|
| [Rule](rules.md) | Base class for all validation rules |
| `RuleParams` | Parameters for rule configuration |
| `RuleQualifier` | Rule matching qualifiers |
| `ValidationError` | Validation failure exception |

### Built-in Rules

| Class | Type | Description |
|-------|------|-------------|
| `StringRule` | `str` | Strip, case conversion, length limits, patterns |
| `NumberRule` | `int`, `float` | Bounds checking, precision, rounding |
| `BooleanRule` | `bool` | Truthy/falsy conversion |
| `ChoiceRule` | Enum/Literal | Discrete value validation |
| `MappingRule` | `dict` | Key/value validation |
| `BaseModelRule` | `BaseModel` | Pydantic model validation |
| `ActionRequestRule` | `ActionRequest` | Tool call validation |
| `ReasonRule` | `Reason` | Reasoning chain validation |

### Output Models

| Class | Description |
|-------|-------------|
| `ActionRequest` | Tool/function call request model |
| `ActionResponse` | Tool execution response model |
| `Reason` | Reasoning chain with confidence |

## Quick Start

```python
from lionpride.rules import Validator, NumberRule, StringRule, RuleRegistry
from lionpride.types import Spec, Operable

# Define specs with base types
confidence_spec = Spec(float, name="confidence")
output_spec = Spec(str, name="output")
operable = Operable([confidence_spec, output_spec])

# Validate raw LLM response (auto Rule assignment from base_type)
validator = Validator()
validated = await validator.validate_operable(
    data={"confidence": "0.95", "output": 42},  # Raw LLM response
    operable=operable,
    auto_fix=True
)
# -> {"confidence": 0.95, "output": "42"}

# Create Pydantic model from operable
OutputModel = operable.create_model()
result = OutputModel.model_validate(validated)

# Direct rule usage for custom validation
rule = NumberRule(ge=0, le=1)
result = await rule.invoke("confidence", "0.95", float, auto_fix=True)
# -> 0.95 (string converted to float)

rule = StringRule(min_length=1, max_length=100)
result = await rule.invoke("name", 42, str, auto_fix=True)
# -> "42" (int converted to string)
```

## Rule Configuration

### StringRule Options

```python
StringRule(
    min_length=1,         # Minimum string length (inclusive)
    max_length=100,       # Maximum string length (inclusive)
    pattern=r"^\w+$",     # Regex pattern to match
)
```

**Auto-fix behavior:** Converts any type to string via `str()`, then re-validates against constraints.

### NumberRule Options

```python
NumberRule(
    ge=0,                 # Greater than or equal to (>=)
    gt=None,              # Greater than (>)
    le=100,               # Less than or equal to (<=)
    lt=None,              # Less than (<)
)
```

**Auto-fix behavior:** Converts strings to `int()` or `float()` based on target type, then validates against bounds.

### ChoiceRule Options

```python
ChoiceRule(
    choices=["low", "medium", "high"],  # Valid options
    case_sensitive=False,               # Case-insensitive matching
)
```

**Auto-fix behavior:** For `case_sensitive=False`, normalizes to canonical case from the choices set.

## See Also

- [Types](../types/index.md) - Spec and Operable definitions
- [Operations](../operations/index.md) - Operations use Validator for output validation
- [Session](../session/index.md) - Session conducts operations with validation
- [User Guide: Validation](../../user_guide/validation.md) - Tutorial and patterns
