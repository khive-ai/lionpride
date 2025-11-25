# MessageContent Types

> Discriminated union of typed message content variants with auto-derived roles

## Overview

`MessageContent` is the base class for a discriminated union of message content types, each representing a different kind of message in conversational workflows. Each content type has specific fields, custom rendering logic, and an associated `MessageRole` that automatically determines the message's role in chat APIs.

**Content Types:**

- [`SystemContent`](#systemcontent): System instructions with optional timestamps
- [`InstructionContent`](#instructioncontent): User instructions with structured outputs, tools, images
- [`AssistantResponseContent`](#assistantresponsecontent): Assistant text responses
- [`ActionRequestContent`](#actionrequestcontent): Function/tool call requests
- [`ActionResponseContent`](#actionresponsecontent): Function/tool call results

**Key Capabilities:**

- **Type Safety**: Pydantic validates fields per content type
- **Auto-Role Assignment**: Each type has a `ClassVar[MessageRole]` for automatic role derivation
- **Custom Rendering**: Each type implements content-specific rendering logic
- **Immutability**: Dataclass with `slots=True` prevents accidental mutations
- **Functional Updates**: `with_updates()` creates new instances with modifications

**When to Use Each Type:**

| Type | Use Case |
|------|----------|
| `SystemContent` | Model behavior configuration, environment context |
| `InstructionContent` | User queries, commands, instructions with structured outputs |
| `AssistantResponseContent` | Model-generated text responses |
| `ActionRequestContent` | Model requesting tool/function execution |
| `ActionResponseContent` | Tool/function execution results |

---

## MessageContent (Base Class)

Base class for message content variants (immutable dataclass).

### Class Signature

```python
from lionpride.session import MessageContent
from dataclasses import dataclass

@dataclass(slots=True)
class MessageContent:
    """Base class for message content variants (immutable)."""

    role: ClassVar[MessageRole] = MessageRole.UNSET

    @property
    def rendered(self) -> str | list[dict[str, Any]]: ...

    @property
    def chat_msg(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageContent: ...

    def with_updates(
        self,
        copy_containers: Literal["none", "shallow", "deep"] = "none",
        **updates: Any,
    ) -> MessageContent: ...
```

### Properties

#### `rendered`

Render content to string or list of content blocks (for multimodal messages).

**Returns:**

- str or list[dict]: Rendered content (implementation varies by subclass)

**Notes:**

This is an abstract property. Subclasses must implement their own rendering logic.

#### `chat_msg`

Format for chat API: `{"role": "...", "content": "..."}`.

**Returns:**

- dict[str, Any]: Chat format with role and rendered content

**Implementation:**

```python
{
    "role": self.role.value,
    "content": self.rendered
}
```

### Methods

#### `from_dict()`

Deserialize from dictionary (subclass-specific).

**Signature:**

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> MessageContent: ...
```

**Notes:**

Each subclass implements its own `from_dict()` method using field validation.

#### `with_updates()`

Create new instance with updated fields (functional update pattern).

**Signature:**

```python
def with_updates(
    self,
    copy_containers: Literal["none", "shallow", "deep"] = "none",
    **updates: Any,
) -> MessageContent: ...
```

**Parameters:**

- `copy_containers` ({'none', 'shallow', 'deep'}, default 'none'): How to copy mutable containers (list, dict, set)
  - `'none'`: Don't copy containers (reference same objects)
  - `'shallow'`: Shallow copy containers (copy container, reference elements)
  - `'deep'`: Deep copy containers (copy container and all nested elements)
- `**updates`: Fields to update

**Returns:**

- MessageContent: New instance with updates applied

**Examples:**

```python
>>> from lionpride.session import InstructionContent

>>> content = InstructionContent(
...     instruction="Original",
...     context=["item1", "item2"],
... )

# Update with shallow copy
>>> updated = content.with_updates(
...     copy_containers="shallow",
...     instruction="Modified",
... )
>>> updated.instruction
'Modified'
>>> updated.context
['item1', 'item2']

# Update context (deep copy to avoid mutation)
>>> updated = content.with_updates(
...     copy_containers="deep",
...     context=["item1", "item2", "item3"],
... )
>>> updated.context
['item1', 'item2', 'item3']
>>> content.context  # Original unchanged
['item1', 'item2']
```

**Use Cases:**

- Updating historical instructions in `prepare_messages_for_chat()`
- Creating message variants with modified fields
- Removing tool_schemas/response_model from historical messages

---

## SystemContent

System message with optional timestamp for model behavior configuration.

### Class Signature

```python
from lionpride.session import SystemContent
from dataclasses import dataclass

@dataclass(slots=True)
class SystemContent(MessageContent):
    """System message with optional timestamp."""

    role: ClassVar[MessageRole] = MessageRole.SYSTEM

    system_message: MaybeUnset[str] = Unset
    system_datetime: MaybeUnset[str | Literal[True]] = Unset
    datetime_factory: MaybeUnset[Callable[[], str]] = Unset

    @classmethod
    def create(
        cls,
        system_message: str | None = None,
        system_datetime: str | Literal[True] | None = None,
        datetime_factory: Callable[[], str] | None = None,
    ) -> SystemContent: ...
```

### Fields

**system_message** : str, optional

System instructions for model behavior.

**system_datetime** : str or True, optional

System timestamp. If True, uses current UTC time. If str, uses provided timestamp. Cannot be set with `datetime_factory`.

**datetime_factory** : Callable[[], str], optional

Function returning timestamp string. Called during rendering. Cannot be set with `system_datetime`.

### Properties

#### `rendered`

Render system message with optional timestamp.

**Returns:**

- str: Rendered system message

**Format:**

```
System Time: {timestamp}

{system_message}
```

**Examples:**

```python
>>> from lionpride.session import SystemContent

# System message only
>>> content = SystemContent.create(system_message="You are helpful")
>>> content.rendered
'You are helpful'

# With auto timestamp
>>> content = SystemContent.create(
...     system_message="You are helpful",
...     system_datetime=True,
... )
>>> content.rendered
'System Time: 2025-11-24T...\n\nYou are helpful'

# With custom timestamp
>>> content = SystemContent.create(
...     system_message="You are helpful",
...     system_datetime="2025-11-24T10:00:00Z",
... )
>>> content.rendered
'System Time: 2025-11-24T10:00:00Z\n\nYou are helpful'

# With datetime factory
>>> content = SystemContent.create(
...     system_message="You are helpful",
...     datetime_factory=lambda: "Custom time format",
... )
>>> content.rendered
'System Time: Custom time format\n\nYou are helpful'
```

### Validation

**Raises:**

- ValueError: If both `system_datetime` and `datetime_factory` are set

### Usage Patterns

```python
from lionpride.session import Message, SystemContent

# Basic system message
msg = Message(content=SystemContent.create(
    system_message="You are a helpful coding assistant"
))

# System message with timestamp
msg = Message(content=SystemContent.create(
    system_message="You are a helpful assistant",
    system_datetime=True,
))

# Custom timestamp format
from datetime import datetime
msg = Message(content=SystemContent.create(
    system_message="You are helpful",
    datetime_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"),
))
```

---

## InstructionContent

User instruction with structured outputs, tool schemas, and multimodal support.

### Class Signature

```python
from lionpride.session import InstructionContent
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass(slots=True)
class InstructionContent(MessageContent):
    """User instruction with structured outputs."""

    role: ClassVar[MessageRole] = MessageRole.USER

    instruction: MaybeUnset[str] = Unset
    context: MaybeUnset[list[Any]] = Unset
    tool_schemas: MaybeUnset[list[type[BaseModel] | dict[str, Any]]] = Unset
    response_model: MaybeUnset[type[BaseModel]] = Unset
    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset

    @classmethod
    def create(
        cls,
        instruction: str | None = None,
        context: list[Any] | None = None,
        tool_schemas: list[type[BaseModel] | dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
        images: list[str] | None = None,
        image_detail: Literal["low", "high", "auto"] | None = None,
    ) -> InstructionContent: ...
```

### Fields

**instruction** : str, optional

User instruction or query.

**context** : list[Any], optional

Additional context for instruction (past results, reference data, etc.).

**tool_schemas** : list[type[BaseModel] or dict], optional

Available tool/function schemas (Pydantic models or JSON schema dicts).

**response_model** : type[BaseModel], optional

Expected response schema for structured output.

**images** : list[str], optional

Image URLs for multimodal messages (http/https only, validated for security).

**image_detail** : {'low', 'high', 'auto'}, optional

Image detail level for vision models. Default: 'auto'.

### Properties

#### `rendered`

Render instruction with YAML context, TypeScript schemas, and JSON format instructions.

**Returns:**

- str or list[dict]: String for text-only messages, list of content blocks for multimodal messages

**Text Format:**

```yaml
Instruction: {instruction}

Context:
  - {context_item_1}
  - {context_item_2}

Tools:
  {tool_name}:
    # {tool_description}
    {typescript_schema}

Output Types:
  {response_model_typescript_schema}

ResponseFormat:
  **MUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT.**
  Example structure:
  ```json
  {...}
  ```

  Return ONLY valid JSON without markdown code blocks.

```

**Multimodal Format (when images present):**

```python
[
    {"type": "text", "text": "{rendered_text}"},
    {"type": "image_url", "image_url": {"url": "{url}", "detail": "{detail}"}},
    ...
]
```

**Examples:**

```python
>>> from lionpride.session import InstructionContent
>>> from pydantic import BaseModel

# Basic instruction
>>> content = InstructionContent.create(instruction="Explain AI")
>>> content.rendered
'Instruction: Explain AI'

# With context
>>> content = InstructionContent.create(
...     instruction="Analyze this",
...     context=["Data point 1", "Data point 2"],
... )
>>> content.rendered
'Instruction: Analyze this\n\nContext:\n  - Data point 1\n  - Data point 2'

# With response model
>>> class Analysis(BaseModel):
...     summary: str
...     score: float
>>> content = InstructionContent.create(
...     instruction="Analyze quarterly results",
...     response_model=Analysis,
... )
>>> content.rendered
'''
Instruction: Analyze quarterly results

Output Types:
  interface Analysis {
    summary: string;
    score: number;
  }

ResponseFormat:
  **MUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT.**
  Example structure:
  ```json
  {"summary": "...", "score": 0}
  ```

  Return ONLY valid JSON without markdown code blocks.
'''

# With images (multimodal)
>>>
>>> content = InstructionContent.create(
...     instruction="Describe this image",
...     images=["https://example.com/image.jpg"],
...     image_detail="high",
... )
>>> content.rendered
[
    {"type": "text", "text": "Instruction: Describe this image"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg", "detail": "high"}}
]

```

### Validation

**Image URL Security:**

Image URLs are validated to prevent security vulnerabilities:

- **Allowed**: `http://`, `https://` only
- **Rejected**: `file://` (local file access), `javascript:` (XSS), `data://` (DoS)

**Raises:**

- ValueError: If image URL uses disallowed scheme or is malformed

**Examples:**

```python
# Valid
>>> InstructionContent.create(images=["https://example.com/image.jpg"])

# Invalid (raises ValueError)
>>> InstructionContent.create(images=["file:///etc/passwd"])
ValueError: Image URL must use http:// or https:// scheme, got: file://
```

### Usage Patterns

```python
from lionpride.session import Message, InstructionContent
from pydantic import BaseModel

# Basic instruction
msg = Message(content=InstructionContent.create(
    instruction="Explain quantum computing"
))

# Instruction with context
msg = Message(content=InstructionContent.create(
    instruction="Summarize the discussion",
    context=["Point 1: ...", "Point 2: ...", "Point 3: ..."],
))

# Structured output
class Report(BaseModel):
    title: str
    sections: list[str]
    conclusion: str

msg = Message(content=InstructionContent.create(
    instruction="Generate a report on AI safety",
    response_model=Report,
))

# Tool schemas
class SearchParams(BaseModel):
    """Search for information."""
    query: str
    max_results: int = 10

msg = Message(content=InstructionContent.create(
    instruction="Find recent papers on transformers",
    tool_schemas=[SearchParams],
))

# Multimodal with images
msg = Message(content=InstructionContent.create(
    instruction="Describe what you see and identify any issues",
    images=[
        "https://example.com/diagram1.jpg",
        "https://example.com/diagram2.jpg",
    ],
    image_detail="high",
))
```

---

## AssistantResponseContent

Assistant text response (model-generated).

### Class Signature

```python
from lionpride.session import AssistantResponseContent
from dataclasses import dataclass

@dataclass(slots=True)
class AssistantResponseContent(MessageContent):
    """Assistant text response."""

    role: ClassVar[MessageRole] = MessageRole.ASSISTANT

    assistant_response: MaybeUnset[str] = Unset

    @classmethod
    def create(cls, assistant_response: str | None = None) -> AssistantResponseContent: ...
```

### Fields

**assistant_response** : str, optional

Assistant-generated text response.

### Properties

#### `rendered`

Render assistant response (returns text as-is).

**Returns:**

- str: Assistant response text (empty string if unset)

**Examples:**

```python
>>> from lionpride.session import AssistantResponseContent

>>> content = AssistantResponseContent.create(
...     assistant_response="The capital of France is Paris."
... )
>>> content.rendered
'The capital of France is Paris.'

# Empty response
>>> content = AssistantResponseContent.create()
>>> content.rendered
''
```

### Usage Patterns

```python
from lionpride.session import Message, AssistantResponseContent

# Basic response
msg = Message(content=AssistantResponseContent.create(
    assistant_response="I'd be happy to help with that!"
))

# Multi-paragraph response
msg = Message(content=AssistantResponseContent.create(
    assistant_response=(
        "Python is a high-level programming language.\n\n"
        "Key features include:\n"
        "- Simple syntax\n"
        "- Dynamic typing\n"
        "- Rich ecosystem"
    )
))
```

---

## ActionRequestContent

Function/tool call request from assistant.

### Class Signature

```python
from lionpride.session import ActionRequestContent
from dataclasses import dataclass

@dataclass(slots=True)
class ActionRequestContent(MessageContent):
    """Action/function call request."""

    role: ClassVar[MessageRole] = MessageRole.ASSISTANT

    function: MaybeUnset[str] = Unset
    arguments: MaybeUnset[dict[str, Any]] = Unset

    @classmethod
    def create(
        cls,
        function: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> ActionRequestContent: ...
```

### Fields

**function** : str, optional

Function/tool name to call.

**arguments** : dict[str, Any], optional

Function arguments (key-value pairs).

### Properties

#### `rendered`

Render action request as YAML.

**Returns:**

- str: YAML representation of function call

**Format:**

```yaml
function: {function_name}
arguments:
  {arg1}: {value1}
  {arg2}: {value2}
```

**Examples:**

```python
>>> from lionpride.session import ActionRequestContent

>>> content = ActionRequestContent.create(
...     function="search",
...     arguments={"query": "quantum computing", "max_results": 5},
... )
>>> content.rendered
'function: search\narguments:\n  query: quantum computing\n  max_results: 5'

# Empty arguments
>>> content = ActionRequestContent.create(function="get_time")
>>> content.rendered
'function: get_time\narguments: {}'
```

### Usage Patterns

```python
from lionpride.session import Message, ActionRequestContent

# Tool call request
msg = Message(content=ActionRequestContent.create(
    function="search_web",
    arguments={
        "query": "latest AI research",
        "max_results": 10,
        "date_range": "last_month",
    },
))

# Function call with no arguments
msg = Message(content=ActionRequestContent.create(
    function="get_current_time"
))
```

---

## ActionResponseContent

Function/tool call response with success tracking.

### Class Signature

```python
from lionpride.session import ActionResponseContent
from dataclasses import dataclass

@dataclass(slots=True)
class ActionResponseContent(MessageContent):
    """Function call response."""

    role: ClassVar[MessageRole] = MessageRole.TOOL

    request_id: MaybeUnset[str] = Unset
    result: MaybeUnset[Any] = Unset
    error: MaybeUnset[str] = Unset

    @property
    def success(self) -> bool: ...

    @classmethod
    def create(
        cls,
        request_id: str | None = None,
        result: Any | None = None,
        error: str | None = None,
    ) -> ActionResponseContent: ...
```

### Fields

**request_id** : str, optional

Request identifier (for correlating request/response).

**result** : Any, optional

Function execution result (if successful).

**error** : str, optional

Error message (if failed).

### Properties

#### `success`

Check if action succeeded.

**Returns:**

- bool: True if error is unset, False otherwise

**Examples:**

```python
>>> from lionpride.session import ActionResponseContent

>>> content = ActionResponseContent.create(result={"data": 42})
>>> content.success
True

>>> content = ActionResponseContent.create(error="Connection timeout")
>>> content.success
False
```

#### `rendered`

Render action response as YAML.

**Returns:**

- str: YAML representation of result or error

**Format (success):**

```yaml
success: true
request_id: {request_id}
result:
  {result_yaml}
```

**Format (error):**

```yaml
success: false
request_id: {request_id}
error: {error_message}
```

**Examples:**

```python
>>> from lionpride.session import ActionResponseContent

# Success
>>> content = ActionResponseContent.create(
...     request_id="req_123",
...     result={"papers": ["Paper 1", "Paper 2"]},
... )
>>> content.rendered
'success: true\nrequest_id: req_123\nresult:\n  papers:\n    - Paper 1\n    - Paper 2'

# Error
>>> content = ActionResponseContent.create(
...     request_id="req_123",
...     error="API rate limit exceeded",
... )
>>> content.rendered
'success: false\nrequest_id: req_123\nerror: API rate limit exceeded'
```

### Usage Patterns

```python
from lionpride.session import Message, ActionResponseContent

# Successful tool result
msg = Message(content=ActionResponseContent.create(
    request_id="search_123",
    result={
        "papers": [
            {"title": "Attention Is All You Need", "year": 2017},
            {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2018},
        ]
    },
))

# Failed tool call
msg = Message(content=ActionResponseContent.create(
    request_id="search_123",
    error="Search API returned 500: Internal Server Error",
))
```

---

## Common Patterns

### Structured Output Workflow

```python
from lionpride.session import Session, Message, InstructionContent
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    key_findings: list[str]
    confidence: float

session = Session()
branch = session.create_branch(name="analysis")

# Instruction with response model
instruction = Message(content=InstructionContent.create(
    instruction="Analyze quarterly sales data",
    context=["Q3 Sales: $1.5M", "Q3 Expenses: $800K"],
    response_model=Analysis,
))
session.add_message(instruction, branches=branch)

# LLM generates JSON response matching schema
# response = await llm_api.generate(messages=[instruction.chat_msg])
# analysis = Analysis.model_validate_json(response)
```

### Tool Calling Workflow

```python
from lionpride.session import (
    Session, Message, InstructionContent, ActionRequestContent, ActionResponseContent
)
from pydantic import BaseModel

class SearchParams(BaseModel):
    """Search for information."""
    query: str
    max_results: int = 10

session = Session()
branch = session.create_branch(name="tool_workflow")

# 1. User instruction with available tools
instruction = Message(content=InstructionContent.create(
    instruction="Find recent papers about transformers",
    tool_schemas=[SearchParams],
))
session.add_message(instruction, branches=branch)

# 2. LLM requests tool call
tool_request = Message(content=ActionRequestContent.create(
    function="search",
    arguments={"query": "transformers", "max_results": 5},
))
session.add_message(tool_request, branches=branch)

# 3. Execute tool and return result
tool_result = Message(content=ActionResponseContent.create(
    result=["Paper 1", "Paper 2", "Paper 3"],
))
session.add_message(tool_result, branches=branch)

# 4. LLM uses result to respond
# prepare_messages_for_chat() automatically embeds tool_result into context
```

### Multimodal Interaction

```python
from lionpride.session import Session, Message, InstructionContent

session = Session()
branch = session.create_branch(name="vision")

# Instruction with images
msg = Message(content=InstructionContent.create(
    instruction="Analyze these charts and identify trends",
    context=["Sales data from Q1-Q4 2024"],
    images=[
        "https://example.com/charts/q1.jpg",
        "https://example.com/charts/q2.jpg",
        "https://example.com/charts/q3.jpg",
        "https://example.com/charts/q4.jpg",
    ],
    image_detail="high",
))
session.add_message(msg, branches=branch)

# Message.rendered returns list of content blocks for vision API
print(msg.rendered)
# [
#     {"type": "text", "text": "Instruction: ..."},
#     {"type": "image_url", "image_url": {"url": "...", "detail": "high"}},
#     ...
# ]
```

### Updating Historical Messages

```python
from lionpride.session import prepare_messages_for_chat, InstructionContent

# Historical instruction with tool_schemas (not needed for past messages)
old_instruction = InstructionContent.create(
    instruction="Search for papers",
    tool_schemas=[SearchParams],  # Remove for historical messages
)

# Functional update (immutable)
cleaned = old_instruction.with_updates(
    copy_containers="deep",
    tool_schemas=None,  # Remove schemas
)

print(cleaned.tool_schemas)  # Unset (removed)
print(old_instruction.tool_schemas)  # [SearchParams] (unchanged)
```

## Design Rationale

### Why Discriminated Union?

Type safety + flexibility:

1. **Type Safety**: Pydantic validates fields per content type
2. **Auto-Role Assignment**: Content type determines message role
3. **Custom Rendering**: Each type has specific rendering logic
4. **Extensibility**: Easy to add new content types

### Why Immutable Dataclasses?

MessageContent instances are immutable (`slots=True`, no `__setattr__`):

1. **Safety**: Prevents accidental mutations affecting multiple branches
2. **Functional Updates**: `with_updates()` encourages functional programming patterns
3. **Performance**: Slots reduce memory overhead
4. **Hash Stability**: Immutability enables reliable hashing (future feature)

### Why MaybeUnset Fields?

`Unset` sentinel enables:

1. **Optional Fields**: Distinguish between "not set" and "set to None"
2. **Conditional Rendering**: Only render fields that are set
3. **Compact Serialization**: Omit unset fields from JSON

### Why TypeScript Schemas in Rendering?

TypeScript schemas (not JSON schema) for structured outputs:

1. **Readability**: More concise and readable than JSON schema
2. **LLM Familiarity**: Models trained on TypeScript understand it better
3. **Nesting**: Clear visualization of nested types

### Why Separate request_id in ActionResponseContent?

Correlating requests and responses:

1. **Async Tools**: Requests and responses may not be adjacent
2. **Multi-Tool**: Multiple tool calls in progress simultaneously
3. **Debugging**: Trace which response matches which request

## See Also

- **Related Classes**:
  - [Message](message.md): Container for MessageContent
  - [MessageRole](message.md#messagerole): Role enumeration

- **Module Overview**:
  - [session Overview](overview.md): Module-level documentation

## Examples

See [overview.md Examples](overview.md#examples) for comprehensive usage patterns.
