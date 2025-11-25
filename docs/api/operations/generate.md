# generate()

> Stateless text generation without conversation persistence

## Overview

The `generate()` function provides stateless LLM text generation. Unlike `communicate()`, it does not persist messages to the session, making it ideal for one-off generations, templates, and scenarios where conversation history is not needed.

**Key Capabilities:**

- **Stateless Execution**: No message persistence to session
- **Direct Model Access**: Minimal abstraction over iModel.invoke()
- **Flexible Returns**: Text, raw, message, or calling object
- **Custom Messages**: Full control over message history

## Signature

```python
async def generate(
    session: Session,
    branch: Branch | str,
    parameters: GenerateParam | dict,
) -> str | dict | Message | Any
```

## Parameters

- `session` (Session): Session for service access (not modified)
- `branch` (Branch | str): Branch reference (not used for message persistence)
- `parameters` (GenerateParam | dict): Generation parameters

### GenerateParam Fields

```python
@dataclass(slots=True, frozen=True, init=False)
class GenerateParam(Params):
    imodel: str | iModel = None          # Model name or instance (required)
    messages: list[dict] = None          # Chat messages (required)
    model: str = None                    # API model name (e.g., "gpt-4o")
    return_as: Literal["text", "raw", "message", "calling"] = "text"
    # **kwargs: Additional iModel.invoke() parameters
```

## Returns

Return type depends on `return_as` parameter:

- **"text"** (default): Plain text response
- **"raw"**: Raw API response dictionary
- **"message"**: `Message` object (not persisted to session)
- **"calling"**: Full `Calling` object with execution metadata

## Raises

- `ValueError`: Missing required parameters
- `RuntimeError`: Generation failure

## Basic Usage

### Simple Generation

```python
result = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
    )
)
print(result)  # "2 + 2 equals 4."
```

### With System Message

```python
result = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "Explain calculus simply."},
        ],
    )
)
```

### Multi-Turn Context

```python
result = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What's my name?"},
        ],
    )
)
print(result)  # "Your name is Alice."
```

## Return Modes

### Text (Default)

```python
text = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        return_as="text",
    )
)
# text = "Hello! How can I help you today?"
```

### Raw API Response

```python
raw = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        return_as="raw",
    )
)

# Access API-specific fields
print(raw["usage"])  # Token usage
print(raw["model"])  # Model used
print(raw["id"])  # Request ID
```

### Message Object

```python
message = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        return_as="message",
    )
)

# Access message fields
print(message.content.assistant_response)  # Response text
print(message.metadata["raw_response"])  # Full API response
print(message.created_at)  # Timestamp
```

### Calling Object

```python
calling = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        return_as="calling",
    )
)

# Access execution metadata
print(calling.execution.status)  # ExecutionStatus.COMPLETED
print(calling.execution.response.data)  # Response text
print(calling.execution.duration)  # Execution time
```

## Advanced Parameters

### Custom Model Name

Override the default model for the iModel:

```python
result = await generate(
    session,
    "main",
    GenerateParam(
        imodel="openai",  # Service name
        model="gpt-4o-mini",  # Specific model
        messages=[{"role": "user", "content": "Hello"}],
    )
)
```

### Additional iModel Parameters

Pass extra parameters to `iModel.invoke()`:

```python
from lionpride.operations.types import GenerateParam

result = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )
)
```

## Usage Patterns

### Template Generation

```python
async def fill_template(template: str, **kwargs) -> str:
    """Fill template using LLM."""
    prompt = template.format(**kwargs)
    return await generate(
        session,
        "main",
        GenerateParam(
            imodel="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
    )

result = await fill_template(
    "Summarize this article: {article}",
    article="Long article text..."
)
```

### Batch Processing

```python
async def batch_generate(prompts: list[str]) -> list[str]:
    """Generate responses for multiple prompts."""
    from lionpride.ln import alcall

    async def gen_one(prompt: str) -> str:
        return await generate(
            session,
            "main",
            GenerateParam(
                imodel="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
        )

    return await alcall(prompts, gen_one)

results = await batch_generate([
    "What is Python?",
    "What is TypeScript?",
    "What is Rust?",
])
```

### Custom Message Builder

```python
def build_messages(system: str, *turns: tuple[str, str]) -> list[dict]:
    """Build message list from system prompt and turns."""
    messages = [{"role": "system", "content": system}]
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages

result = await generate(
    session,
    "main",
    GenerateParam(
        imodel="gpt-4o",
        messages=build_messages(
            "You are a helpful assistant.",
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well!"),
        ),
    )
)
```

### Error Handling

```python
try:
    result = await generate(
        session,
        "main",
        GenerateParam(
            imodel="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )
except RuntimeError as e:
    print(f"Generation failed: {e}")
    # Handle error (retry, fallback, etc.)
```

## Common Pitfalls

- **Missing messages**: Not providing `messages` parameter
  - **Solution**: Always provide at least one message: `messages=[{"role": "user", "content": "..."}]`

- **Missing imodel**: Not providing `imodel` parameter
  - **Solution**: Specify model name or instance: `imodel="gpt-4o"`

- **State confusion**: Expecting conversation state
  - **Solution**: Use `communicate()` for stateful conversations

- **Invalid message format**: Incorrect message role or structure
  - **Solution**: Use standard chat format: `{"role": "user"|"assistant"|"system", "content": "..."}`

- **Direct iModel access**: Using session.services.get() instead of generate()
  - **Solution**: Use `generate()` for consistent error handling and return formatting

## Design Rationale

### Why Stateless?

Stateless generation is useful for:

- **Templates**: One-off text generation without conversation
- **Batch Processing**: Parallel generation without state conflicts
- **Testing**: Isolated generation for unit tests
- **Custom Workflows**: Full control over message history

For conversational applications, use `communicate()` instead.

### Why Message List Parameter?

Direct message control enables:

- **Custom Histories**: Build arbitrary conversation contexts
- **Message Filtering**: Select specific messages from session
- **Cross-Session Context**: Combine messages from multiple sessions
- **Testing**: Precise control for reproducible tests

### Why Multiple Return Types?

Different use cases need different information:

- **Text**: Simple applications, templates
- **Raw**: API analysis, token counting, debugging
- **Message**: Metadata inspection, logging
- **Calling**: Execution monitoring, performance analysis

## See Also

- **Related Functions**:
  - [`communicate`](communicate.md): Stateful chat with conversation persistence
  - [`operate`](operate.md): Structured outputs with validation
  - [`react`](react.md): Multi-step reasoning

- **Related Types**:
  - [GenerateParam](../types/params.md): Parameter specification
  - [Message](../session/message.md): Message container
  - [iModel](../services/imodel.md): Model service interface

- **User Guide**:
  - Template generation patterns (documentation pending)
  - Batch processing with alcall (documentation pending)

## Examples

### Complete Example: Template System

```python
from dataclasses import dataclass

@dataclass
class EmailTemplate:
    subject: str
    greeting: str
    body: str
    closing: str

async def generate_email(
    template: EmailTemplate,
    recipient: str,
    **context,
) -> str:
    """Generate personalized email from template."""
    prompt = f"""
    Generate an email with:
    Subject: {template.subject}
    Recipient: {recipient}
    Greeting: {template.greeting}
    Body: {template.body}
    Closing: {template.closing}

    Context: {context}
    """

    result = await generate(
        session,
        "main",
        GenerateParam(
            imodel="gpt-4o",
            messages=[
                {"role": "system", "content": "You write professional emails."},
                {"role": "user", "content": prompt},
            ],
        )
    )
    return result

# Usage
template = EmailTemplate(
    subject="Project Update",
    greeting="Dear {recipient},",
    body="Update about {project}",
    closing="Best regards,",
)

email = await generate_email(
    template,
    recipient="Alice",
    project="lionpride documentation",
)
print(email)
```

### Complete Example: Batch Analysis

```python
async def analyze_batch(texts: list[str], model: str = "gpt-4o") -> list[dict]:
    """Analyze sentiment for multiple texts in parallel."""
    from lionpride.ln import alcall

    async def analyze_one(text: str) -> dict:
        raw = await generate(
            session,
            "main",
            GenerateParam(
                imodel=model,
                messages=[
                    {"role": "system", "content": "Analyze sentiment as positive/negative/neutral."},
                    {"role": "user", "content": text},
                ],
                return_as="raw",
            )
        )
        return {
            "text": text,
            "sentiment": raw["choices"][0]["message"]["content"],
            "tokens": raw["usage"]["total_tokens"],
        }

    return await alcall(texts, analyze_one, max_concurrent=5)

# Analyze 100 reviews
reviews = ["Great product!", "Terrible experience.", ...]
results = await analyze_batch(reviews)

for r in results:
    print(f"{r['sentiment']}: {r['text'][:50]} ({r['tokens']} tokens)")
```
