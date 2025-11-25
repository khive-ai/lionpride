# operate() and Operative

> Structured outputs with optional actions and two-tier validation

## Overview

The `operate()` function provides structured LLM outputs with optional action requests and reasoning. It combines stateful chat (via `communicate()`) with automatic tool execution and validation strategies. The `Operative` class implements a two-tier validation strategy (strict then fuzzy) for robust LLM output handling.

**Key Capabilities:**

- **Structured Outputs**: Validate LLM responses against Pydantic models or Operables
- **Action Support**: Enable tool calling with automatic execution
- **Reasoning**: Optional reasoning field for chain-of-thought outputs
- **Two-Tier Validation**: Strict validation with fuzzy fallback
- **Flexible Modes**: JSON (strict) or LNDL (fuzzy) validation

## operate() Function

### Signature

```python
async def operate(
    session: Session,
    branch: Branch | str,
    parameters: OperateParam | dict,
) -> Any
```

### Parameters

- `session` (Session): Session with conversation state and services
- `branch` (Branch | str): Branch for message context
- `parameters` (OperateParam | dict): Operation parameters

### OperateParam Fields

```python
@dataclass(slots=True, frozen=True, init=False)
class OperateParam(Params):
    instruction: str = None              # User instruction (required)
    imodel: str | iModel = None          # Model name or instance (required)
    response_model: type[BaseModel] = None  # Pydantic model for validation
    operable: Operable = None            # Operable for LNDL validation
    context: Any = None                  # Additional context
    images: list = None                  # Image URLs/data for multimodal
    image_detail: Literal["low", "high", "auto"] = None
    tool_schemas: list[dict] = None      # Pre-built tool schemas
    tools: bool = False                  # Enable tool execution
    actions: bool = False                # Enable action requests
    reason: bool = False                 # Enable reasoning field
    use_lndl: bool = False              # Use LNDL mode (vs JSON)
    lndl_threshold: float = 0.85        # LNDL similarity threshold
    max_retries: int = 0                # Retry attempts for validation
    skip_validation: bool = False        # Skip output validation
    return_message: bool = False         # Return (result, message) tuple
    concurrent_tool_execution: bool = True  # Execute tools concurrently
```

### Returns

- **By default**: Validated model instance matching `response_model` or `operable`
- **If `return_message=True`**: Tuple of `(result, assistant_message)`
- **If `skip_validation=True`**: Raw text response
- **If validation fails**: Raises `ValueError` or returns dict with error

### Raises

- `ValueError`: Missing required parameters or validation failure
- `RuntimeError`: Generation failure or tool execution error

## Validation Modes

### JSON Mode (response_model)

Uses Pydantic strict validation. Best for simple, well-defined structures.

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    age: int
    email: str

result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Extract user: John Doe, 30, john@example.com",
        imodel="gpt-4o",
        response_model=UserProfile,
    )
)
# result = UserProfile(name="John Doe", age=30, email="john@example.com")
```

### LNDL Mode (operable)

Uses LNDL fuzzy parsing. Best for complex nested structures or tolerating LLM variations.

```python
from lionpride.operations import create_operative_from_model

operative = create_operative_from_model(UserProfile)
result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Extract user: John Doe, 30, john@example.com",
        imodel="gpt-4o",
        operable=operative.operable,
        use_lndl=True,
    )
)
```

## Action Support

Enable action requests for tool calling:

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
    query: str
    results: list[str]

result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Search for Python typing best practices",
        imodel="gpt-4o",
        response_model=SearchResult,
        actions=True,
        tools=True,  # Enable all registered tools
    )
)

# If LLM requests actions:
# 1. action_requests field contains ActionRequestModel instances
# 2. Tools are automatically executed
# 3. action_responses field contains ActionResponseModel instances
# 4. Result includes both base model data and action results
```

### Action Models

```python
class ActionRequestModel(BaseModel):
    function: str  # Tool name
    arguments: dict[str, Any]  # Tool arguments

class ActionResponseModel(BaseModel):
    function: str  # Tool name
    output: Any  # Tool result
    error: str | None  # Error message if failed
```

## Reasoning Support

Enable reasoning field for chain-of-thought:

```python
class Analysis(BaseModel):
    conclusion: str
    confidence: float

result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Analyze sentiment: This movie was amazing!",
        imodel="gpt-4o",
        response_model=Analysis,
        reason=True,
    )
)

# Result includes:
# - analysis (Analysis model)
# - reason (Reason model with reasoning text)
```

## Operative Class

The `Operative` class wraps an `Operable` with validation logic, implementing a two-tier strategy.

### Signature

```python
class Operative:
    def __init__(
        self,
        operable: Operable,
        *,
        name: str | None = None,
        adapter: str = "pydantic",
        strict: bool = False,
        auto_retry_parse: bool = True,
        max_retries: int = 3,
        request_exclude: set[str] | None = None,
    ): ...
```

### Parameters

- `operable` (Operable): Operable defining the validation schema
- `name` (str, optional): Operative name (defaults to operable.name)
- `adapter` (str): Adapter type (default: "pydantic")
- `strict` (bool): Enable strict validation only (default: False)
- `auto_retry_parse` (bool): Enable fuzzy fallback (default: True)
- `max_retries` (int): Max validation retry attempts (default: 3)
- `request_exclude` (set[str], optional): Fields to exclude from request model

### Key Methods

#### `create_request_model()`

Materialize request model (excludes runtime fields like `action_responses`).

```python
request_model = operative.create_request_model()
```

**Returns**: `type[BaseModel]` - Pydantic model for LLM request

#### `create_response_model()`

Materialize response model (includes all fields).

```python
response_model = operative.create_response_model()
```

**Returns**: `type[BaseModel]` - Pydantic model for validation

#### `validate_response()`

Validate LLM response with two-tier strategy.

```python
def validate_response(
    self,
    text: str,
    strict: bool | None = None,
) -> Any: ...
```

**Parameters**:

- `text` (str): LLM response text
- `strict` (bool, optional): Override strict mode

**Returns**: Validated model instance or None (if validation fails in non-strict mode)

**Raises**: Exception if validation fails in strict mode

### Two-Tier Validation Strategy

1. **Tier 1 - Strict Validation**:
   - Attempts exact schema matching
   - Fast and precise
   - Raises exception on failure if `strict=True`

2. **Tier 2 - Fuzzy Fallback**:
   - Activates if strict fails and `auto_retry_parse=True`
   - Uses fuzzy parsing to handle LLM variations
   - Returns None if both tiers fail (non-strict mode)

**Example**:

```python
operative = Operative(operable, strict=False, auto_retry_parse=True)

# LLM returns slightly malformed JSON
text = '{"name": "John", age: 30}'  # Missing quotes around age

# Tier 1: Strict validation fails (invalid JSON)
# Tier 2: Fuzzy fallback succeeds (corrects formatting)
result = operative.validate_response(text)
# result = validated model instance
```

## Factory Functions

### create_operative_from_model()

Create Operative from a single Pydantic model.

```python
def create_operative_from_model(
    response_model: type[BaseModel],
    *,
    name: str | None = None,
    strict: bool = False,
    auto_retry_parse: bool = True,
) -> Operative: ...
```

**Example**:

```python
from pydantic import BaseModel
from lionpride.operations import create_operative_from_model

class UserInfo(BaseModel):
    name: str
    age: int

operative = create_operative_from_model(UserInfo)
# operative.operable has single spec for UserInfo
```

### create_action_operative()

Create Operative with action and reasoning support.

```python
def create_action_operative(
    base_model: type[BaseModel] | None = None,
    *,
    reason: bool = False,
    actions: bool = True,
    name: str | None = None,
    strict: bool = False,
    auto_retry_parse: bool = True,
) -> Operative: ...
```

**Parameters**:

- `base_model` (type[BaseModel], optional): Base response model
- `reason` (bool): Include reasoning field (default: False)
- `actions` (bool): Include action request/response fields (default: True)
- `name` (str, optional): Operative name
- `strict` (bool): Strict validation (default: False)
- `auto_retry_parse` (bool): Fuzzy fallback (default: True)

**Example**:

```python
from lionpride.operations import create_action_operative

class SearchQuery(BaseModel):
    query: str
    filters: list[str]

operative = create_action_operative(
    base_model=SearchQuery,
    reason=True,
    actions=True,
)

# Operative includes:
# - searchquery field (SearchQuery model)
# - reason field (Reason model)
# - action_requests field (list[ActionRequestModel])
# - action_responses field (list[ActionResponseModel])
```

## Usage Patterns

### Basic Structured Output

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    category: str

result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Extract product: iPhone 15 Pro, $999, Electronics",
        imodel="gpt-4o",
        response_model=Product,
    )
)
# result = Product(name="iPhone 15 Pro", price=999.0, category="Electronics")
```

### With Reasoning

```python
class Sentiment(BaseModel):
    label: str  # positive, negative, neutral
    confidence: float

result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Analyze: This product exceeded my expectations!",
        imodel="gpt-4o",
        response_model=Sentiment,
        reason=True,
    )
)
# result.sentiment = Sentiment(label="positive", confidence=0.95)
# result.reason = Reason(reasoning="The phrase 'exceeded expectations' indicates...")
```

### With Actions

```python
class WeatherQuery(BaseModel):
    location: str
    date: str

result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Get weather for Tokyo tomorrow",
        imodel="gpt-4o",
        response_model=WeatherQuery,
        actions=True,
        tools=True,
    )
)

# If LLM calls weather tool:
# result.weatherquery = WeatherQuery(location="Tokyo", date="2025-11-25")
# result.action_requests = [ActionRequestModel(function="get_weather", ...)]
# result.action_responses = [ActionResponseModel(function="get_weather", output=...)]
```

### Custom Operative

```python
from lionpride.types import Operable, Spec
from lionpride.operations import Operative

# Define custom operable
specs = [
    Spec(base_type=UserInfo, name="user"),
    Spec(base_type=str, name="summary", default=None),
]
operable = Operable(specs=tuple(specs), name="UserSummary")

# Create operative
operative = Operative(
    operable=operable,
    strict=False,
    auto_retry_parse=True,
)

# Use in operate()
result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Extract user and summarize: John, 30, engineer",
        imodel="gpt-4o",
        operable=operative.operable,
        use_lndl=True,
    )
)
```

### Return Message

```python
result, message = await operate(
    session,
    "main",
    OperateParam(
        instruction="Extract data",
        imodel="gpt-4o",
        response_model=DataModel,
        return_message=True,
    )
)

# result = validated model instance
# message = assistant Message with full metadata
print(message.metadata["raw_response"])  # Full API response
```

## Common Pitfalls

- **Both validation modes**: Providing both `response_model` and `operable`
  - **Solution**: Use one validation mode - JSON OR LNDL, not both

- **Actions without tools**: Setting `actions=True` but no tools registered
  - **Solution**: Set `tools=True` or register tools in session.services

- **Missing instruction**: Not providing required `instruction` parameter
  - **Solution**: Always provide `instruction` (what you want the LLM to do)

- **Validation confusion**: Using `use_lndl=True` with `response_model` only
  - **Solution**: LNDL mode requires `operable` parameter

- **Strict mode failures**: Setting `strict=True` on Operative causes exceptions
  - **Solution**: Use `strict=False` (default) for fuzzy fallback

## Design Rationale

### Why operate() vs communicate()?

`operate()` extends `communicate()` with structured output guarantees and action support. It's designed for programmatic LLM interaction (APIs, data pipelines) where validation is critical.

### Why Two-Tier Validation?

LLMs produce probabilistic outputs. Strict validation catches perfect responses quickly (performance), while fuzzy fallback handles edge cases (robustness). This maximizes both speed and reliability.

### Why Separate Request/Response Models?

Request models exclude runtime fields (like `action_responses`) that the LLM shouldn't generate. Response models include all fields for validation. This prevents LLM hallucination of action results.

## See Also

- **Related Functions**:
  - [`communicate`](communicate.md): Underlying stateful chat operation
  - [`react`](react.md): Multi-step reasoning with operate() internally
  - [`generate`](generate.md): Stateless generation without validation

- **Related Types**:
  - [Operable](../types/operable.md): Schema definition for LNDL validation
  - [Spec](../types/spec.md): Field specification for Operables

- **User Guide**:
  - Validation strategies: JSON vs LNDL (documentation pending)
  - Tool calling patterns (documentation pending)

## Examples

### Complete Example: Product Extraction with Actions

```python
from pydantic import BaseModel, Field
from lionpride.operations import operate, OperateParam

class Product(BaseModel):
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Price in USD")
    availability: str = Field(..., description="In stock or out of stock")

# With action support for inventory lookup
result = await operate(
    session,
    "main",
    OperateParam(
        instruction="Extract product and check availability: iPhone 15 Pro",
        imodel="gpt-4o",
        response_model=Product,
        actions=True,
        tools=True,
        reason=True,
        max_retries=2,
    )
)

# Result structure:
# result.product = Product(name="iPhone 15 Pro", price=999.0, availability="In stock")
# result.reason = Reason(reasoning="Checked inventory database...")
# result.action_requests = [ActionRequestModel(function="check_inventory", ...)]
# result.action_responses = [ActionResponseModel(function="check_inventory", output={"in_stock": True})]
```
