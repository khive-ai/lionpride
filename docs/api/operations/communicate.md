# communicate()

> Stateful chat with optional structured output and retry

## Overview

The `communicate()` function is the primary interface for stateful LLM conversations in lionpride. It manages conversation history, validates outputs, and handles retry logic for failed validations. Unlike `generate()`, messages are automatically persisted to the session, maintaining conversation context across calls.

**Key Capabilities:**

- **Conversation State**: Automatic message persistence to session branches
- **Structured Outputs**: Optional JSON or LNDL validation
- **Retry Logic**: Automatic retry on validation failure
- **Multimodal Support**: Images with configurable detail levels
- **Flexible Returns**: Text, raw, message, or validated model

## Signature

```python
async def communicate(
    session: Session,
    branch: Branch | str,
    parameters: CommunicateParam | dict,
) -> str | dict | Message | BaseModel
```

## Parameters

- `session` (Session): Session with conversation state and services
- `branch` (Branch | str): Branch name or instance for conversation history
- `parameters` (CommunicateParam | dict): Communication parameters

### CommunicateParam Fields

```python
@dataclass(slots=True, frozen=True, init=False)
class CommunicateParam(Params):
    instruction: str = None              # User instruction (required)
    imodel: str | iModel = None          # Model name or instance (required)
    model: str = None                    # API model name (e.g., "gpt-4o")
    context: Any = None                  # Additional context
    images: list = None                  # Image URLs/data for multimodal
    image_detail: Literal["low", "high", "auto"] = None
    return_as: Literal["text", "raw", "message", "model"] = "text"
    response_model: type[BaseModel] = None  # Pydantic model for JSON validation
    operable: Operable = None            # Operable for LNDL validation
    strict_validation: bool = False      # Raise on validation failure
    fuzzy_parse: bool = True             # Enable fuzzy parsing
    lndl_threshold: float = 0.85         # LNDL similarity threshold
    max_retries: int = 0                 # Retry attempts for validation
```

## Returns

Return type depends on `return_as` parameter:

- **"text"** (default): Plain text response or JSON string (if validated model)
- **"raw"**: Raw API response dictionary
- **"message"**: `Message` object with full metadata
- **"model"**: Validated model instance (requires `response_model` or `operable`)

## Raises

- `ValueError`: Missing required parameters or validation failure (if `strict_validation=True`)

## Validation Modes

### Plain Text (No Validation)

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="What is the capital of France?",
        imodel="gpt-4o",
    )
)
# result = "The capital of France is Paris."
```

### JSON Mode (response_model)

Validates LLM output against Pydantic model:

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    country: str
    population: int

result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract: Paris is in France with 2.2M people",
        imodel="gpt-4o",
        response_model=City,
        return_as="model",
    )
)
# result = City(name="Paris", country="France", population=2200000)
```

### LNDL Mode (operable)

Uses fuzzy parsing for more forgiving validation:

```python
from lionpride.operations import create_operative_from_model

operative = create_operative_from_model(City)

result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract: Paris is in France with 2.2M people",
        imodel="gpt-4o",
        operable=operative.operable,
        return_as="model",
    )
)
# More tolerant of LLM formatting variations
```

## Retry Logic

When validation fails, `communicate()` automatically retries:

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract user data",
        imodel="gpt-4o",
        response_model=UserInfo,
        max_retries=2,
        strict_validation=True,
    )
)

# If validation fails:
# 1. Adds error message to conversation
# 2. Requests LLM to correct the response
# 3. Retries up to max_retries times
# 4. Raises ValueError if all retries exhausted (strict mode)
```

### Non-Strict Mode

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract data",
        imodel="gpt-4o",
        response_model=DataModel,
        max_retries=2,
        strict_validation=False,  # Don't raise on failure
    )
)

# If validation fails after retries:
# result = {
#     "raw": "original response text",
#     "validation_failed": True,
#     "error": "validation error message"
# }
```

## Conversation Flow

### Message Persistence

Every `communicate()` call adds two messages to the session:

1. **User Message**: Contains instruction and context
2. **Assistant Message**: Contains LLM response

```python
# Before communicate
print(len(session.messages[branch]))  # 0

await communicate(session, "main", CommunicateParam(...))

# After communicate
print(len(session.messages[branch]))  # 2 (user + assistant)
```

### Context Accumulation

Each call builds on previous conversation:

```python
# First message
await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="My name is Alice",
        imodel="gpt-4o",
    )
)

# Second message uses history
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="What's my name?",
        imodel="gpt-4o",
    )
)
# result = "Your name is Alice."
```

## Multimodal Support

### Images

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Describe this image",
        imodel="gpt-4o",
        images=["https://example.com/image.jpg"],
        image_detail="high",
    )
)
```

### Image Detail Levels

- **"low"**: Faster, less detailed analysis
- **"high"**: Slower, more detailed analysis
- **"auto"**: Model decides based on image complexity

## Usage Patterns

### Simple Chat

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Tell me a joke",
        imodel="gpt-4o",
    )
)
print(result)  # "Why did the chicken cross the road?..."
```

### Structured Extraction

```python
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(..., description="Article title")
    author: str = Field(..., description="Author name")
    published: str = Field(..., description="Publication date")

result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="""
        Extract article info:
        "Understanding Python Typing" by John Doe, published 2025-11-20
        """,
        imodel="gpt-4o",
        response_model=Article,
        return_as="model",
    )
)
# result = Article(title="Understanding Python Typing", author="John Doe", ...)
```

### With Context

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Summarize the key points",
        imodel="gpt-4o",
        context={
            "document": "Long document text...",
            "max_length": 100,
        },
    )
)
```

### Getting Full Message

```python
message = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Analyze sentiment",
        imodel="gpt-4o",
        return_as="message",
    )
)

# Access metadata
print(message.metadata["raw_response"])  # Full API response
print(message.sender)  # Model name
print(message.created_at)  # Timestamp
```

### Raw API Response

```python
raw = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Generate text",
        imodel="gpt-4o",
        return_as="raw",
    )
)

# Access API-specific fields
print(raw["usage"])  # Token usage
print(raw["model"])  # Model used
print(raw["choices"])  # Response choices
```

## Advanced Patterns

### Multi-Branch Conversations

```python
# Main conversation
await communicate(session, "main", CommunicateParam(instruction="Hello", imodel="gpt-4o"))

# Alternative branch
await communicate(session, "alternative", CommunicateParam(instruction="Hi", imodel="gpt-4o"))

# Each branch maintains separate history
print(len(session.messages["main"]))  # 2
print(len(session.messages["alternative"]))  # 2
```

### Custom Validation Threshold

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract data",
        imodel="gpt-4o",
        operable=operative.operable,
        lndl_threshold=0.9,  # More strict fuzzy matching
        return_as="model",
    )
)
```

### Fuzzy Parsing Control

```python
result = await communicate(
    session,
    "main",
    CommunicateParam(
        instruction="Extract JSON",
        imodel="gpt-4o",
        response_model=DataModel,
        fuzzy_parse=False,  # Disable fuzzy JSON parsing
        return_as="model",
    )
)
```

## Common Pitfalls

- **Missing imodel**: Not providing `imodel` parameter
  - **Solution**: Always specify model name or instance: `imodel="gpt-4o"`

- **Validation without model**: Using `return_as="model"` without `response_model` or `operable`
  - **Solution**: Provide either `response_model` or `operable` for validation

- **Both validation modes**: Providing both `response_model` and `operable`
  - **Solution**: Use one validation mode - JSON OR LNDL, not both

- **Branch confusion**: Different branch names create separate conversations
  - **Solution**: Use consistent branch names for continuous conversation

- **Image format**: Invalid image URLs or data
  - **Solution**: Use valid URLs or base64-encoded image data

## Design Rationale

### Why Automatic Persistence?

Conversation state is fundamental to chat applications. Automatic persistence eliminates boilerplate and ensures consistency. For stateless generation, use `generate()` instead.

### Why Validation Modes?

Different use cases need different validation strategies:

- **JSON mode**: Fast, strict, good for simple structures
- **LNDL mode**: Flexible, fuzzy, good for complex or variable outputs
- **No validation**: Fast, flexible, good for free-form chat

### Why Retry Logic?

LLMs don't always produce valid outputs on first attempt. Automatic retry with error feedback significantly improves success rates without manual handling.

## See Also

- **Related Functions**:
  - [`generate`](generate.md): Stateless generation (no conversation state)
  - [`operate`](operate.md): Structured outputs with actions
  - [`react`](react.md): Multi-step reasoning (uses communicate internally)

- **Related Types**:
  - [Message](../session/message.md): Message container with content types
  - [CommunicateParam](../types/params.md): Parameter specification

- **User Guide**:
  - Building chatbots (documentation pending)
  - Validation strategies (documentation pending)

## Examples

### Complete Example: Multi-Turn Conversation with Validation

```python
from pydantic import BaseModel, Field

# Define response structure
class Task(BaseModel):
    description: str = Field(..., description="Task description")
    priority: str = Field(..., description="high, medium, or low")
    deadline: str | None = Field(None, description="Optional deadline")

# Turn 1: Create task
task = await communicate(
    session,
    "tasks",
    CommunicateParam(
        instruction="Create a task: Review PR by Friday",
        imodel="gpt-4o",
        response_model=Task,
        return_as="model",
    )
)
print(task)
# Task(description="Review PR", priority="high", deadline="Friday")

# Turn 2: Modify task (uses conversation history)
updated = await communicate(
    session,
    "tasks",
    CommunicateParam(
        instruction="Change priority to medium",
        imodel="gpt-4o",
        response_model=Task,
        return_as="model",
    )
)
print(updated)
# Task(description="Review PR", priority="medium", deadline="Friday")

# Turn 3: Query task (history-aware)
result = await communicate(
    session,
    "tasks",
    CommunicateParam(
        instruction="What's the deadline?",
        imodel="gpt-4o",
    )
)
print(result)
# "The deadline is Friday."
```
