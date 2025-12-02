# Operations Module

> Layered LLM operations with capability-based security and structured output

## Overview

Composable hierarchy of LLM operations with explicit security controls.

**Architecture:**

```text
generate (L1) - stateless LLM call, no persistence
parse (L1)    - JSON extraction with fuzzy matching
communicate (L2) - generate + parse + validation + persistence
operate (L3)  - structured output + capability security + tool execution
react (L4)    - multi-turn ReAct reasoning loop
```

## Parameter Classes

### Hierarchy (Inheritance)

```text
GenerateParams      (standalone - LLM call params)
ParseParams         (standalone - extraction params)
ActParams           (standalone - tool execution params)
InterpretParams     (standalone - instruction refinement)

CommunicateParams   (has generate + parse)
        |
OperateParams       (inherits CommunicateParams, adds act fields)
        |
ReactParams         (inherits OperateParams, adds loop params)
```

**Access Patterns:**

```python
communicate.generate.instruction   # 1 level
operate.generate.instruction       # 1 level (inherited)
react.generate.instruction         # 1 level (inherited)
```

---

### GenerateParams

Stateless LLM call parameters.

```python
@dataclass(frozen=True, slots=True)
class GenerateParams(Params):
    instruction: str | Message        # Required
    imodel: iModel | str = None       # Falls back to session default
    context: dict[str, Any] = None    # Additional context
    images: list[str] = None          # Multimodal input
    request_model: type[BaseModel] = None  # Structured output schema
    return_as: Literal["text", "raw", "message", "calling"] = "calling"
    imodel_kwargs: dict[str, Any] = field(default_factory=dict)
```

**return_as**: `"calling"` (APICalling with metadata), `"text"`, `"raw"`, `"message"`

---

### ParseParams

JSON extraction from text with fuzzy key matching.

```python
@dataclass(frozen=True, slots=True)
class ParseParams(Params):
    text: str = None                    # Raw text to parse
    target_keys: list[str] = []         # Expected keys for fuzzy matching
    imodel: iModel | str = None         # Model for LLM reparse fallback
    similarity_threshold: float = 0.85  # Fuzzy match threshold
    handle_unmatched: HandleUnmatched = "force"
    max_retries: int = 3                # LLM reparse attempts (max 5)
```

**handle_unmatched**: `"ignore"`, `"raise"`, `"remove"`, `"fill"`, `"force"` (default)

---

### ActParams

Tool execution parameters.

```python
@dataclass(frozen=True, slots=True)
class ActParams(Params):
    tools: list[str] | bool = False  # True=all, list=specific, False=none
    tool_schemas: list[dict] = None  # Pre-computed schemas
    concurrent: bool = True
    timeout: float = None
```

---

### InterpretParams

Instruction refinement parameters.

```python
@dataclass(frozen=True, slots=True)
class InterpretParams(Params):
    text: str = None           # Raw instruction to refine
    imodel: iModel | str = None
    domain: str = "general"    # Domain hint
    style: str = "concise"     # Output style
    temperature: float = 0.1
```

---

### CommunicateParams

Stateful chat with optional structured output.

```python
@dataclass(frozen=True, slots=True)
class CommunicateParams(Params):
    generate: GenerateParams = None      # Required
    parse: ParseParams = None
    operable: Operable = None            # Structured output schema
    capabilities: set[str] = None        # REQUIRED when using structured output
    auto_fix: bool = True
    strict_validation: bool = True
```

---

### OperateParams

Structured output with optional tool execution. Inherits `CommunicateParams`.

```python
@dataclass(frozen=True)
class OperateParams(CommunicateParams):
    # Inherited: generate, parse, operable, capabilities, auto_fix, strict_validation

    # Tool execution
    tools: list[str] | bool = False
    tool_concurrent: bool = True
    tool_timeout: float = None

    # Operate-specific
    actions: bool = False         # Enable action_requests in output
    reason: bool = False          # Enable reasoning in output
    skip_validation: bool = False # Return raw text
    return_message: bool = False  # Return (result, message) tuple
```

---

### ReactParams

Multi-turn reasoning loops. Inherits `OperateParams`.

```python
@dataclass(frozen=True)
class ReactParams(OperateParams):
    # Inherited: all OperateParams fields

    max_steps: int = 10
    return_trace: bool = False          # Verbose logging
    intermediate_response_options: list[type[BaseModel]] | type[BaseModel] = None
    intermediate_listable: bool = False
    intermediate_nullable: bool = True
```

---

## Functions

### generate()

Stateless LLM call. No persistence, no validation.

```python
async def generate(
    session: Session,
    branch: Branch | str,
    params: GenerateParams,
) -> APICalling | str | dict | Message: ...
```

Returns type based on `params.return_as`. Raises `ConfigurationError` if no valid imodel.

```python
params = GenerateParams(instruction="What is 2 + 2?", return_as="text")
result = await generate(session, branch, params)  # "2 + 2 equals 4."
```

---

### parse()

Extract JSON from text with fuzzy key matching. Falls back to LLM reparse if needed.

```python
async def parse(session: Session, branch: Branch, params: ParseParams) -> dict[str, Any]: ...
```

```python
params = ParseParams(text='{"name": "Alice", "age": 30}', target_keys=["name", "age"])
result = await parse(session, branch, params)  # {"name": "Alice", "age": 30}
```

---

### communicate()

Stateful chat with optional structured output. Generate + Parse + Validate + Persist.

```python
async def communicate(
    session: Session, branch: Branch | str, params: CommunicateParams, validator: Validator = None
) -> str | Any: ...
```

**Two paths**: Text (no operable) returns `str`. IPU (with operable) returns validated model.

```python
# Text path
params = CommunicateParams(generate=GenerateParams(instruction="Hello!"))
result = await communicate(session, branch, params)  # str

# Structured output (requires capabilities)
params = CommunicateParams(
    generate=GenerateParams(instruction="Analyze: Python is great"),
    operable=Operable.from_model(Analysis),
    capabilities={"summary", "score"},
)
result = await communicate(session, branch, params)  # Analysis instance
```

---

### operate()

Structured output with optional tool execution. Primary high-level operation.

```python
async def operate(
    session: Session, branch: Branch | str, params: OperateParams, validator: Validator = None
) -> Any: ...
```

Returns validated model, `(result, message)` tuple if `return_message=True`, or `str` if `skip_validation=True`.

```python
class Sentiment(BaseModel):
    label: str
    confidence: float

branch = session.create_branch(capabilities={"label", "confidence"})

params = OperateParams(
    generate=GenerateParams(instruction="Analyze sentiment: I love this!", request_model=Sentiment),
    capabilities={"label", "confidence"},
)
result = await operate(session, branch, params)  # Sentiment instance
```

**With tools**: Set `actions=True` and ensure branch has `action_requests`, `action_responses` capabilities.

---

### act()

Execute tool calls from action_requests.

```python
async def act(
    action_requests: list[ActionRequest], session: Session, branch: Branch,
    *, max_concurrent: int = 10, retry_timeout: float = None
) -> list[ActionResponse]: ...
```

---

### execute_tools()

Execute tool calls from parsed response and update with results.

```python
async def execute_tools(
    parsed_response: Any, session: Session, branch: Branch, *, max_concurrent: int = 10
) -> tuple[Any, list[ActionResponse]]: ...
```

---

### has_action_requests()

```python
def has_action_requests(parsed_response: Any) -> bool: ...
```

---

### react()

Multi-turn ReAct reasoning loop. Runs operate() until `is_done=True` or `max_steps`.

```python
async def react(session: Session, branch: Branch | str, params: ReactParams) -> ReactResult: ...
```

**Required capabilities**: `reasoning`, `action_requests`, `is_done`, plus any intermediate options.

```python
branch = session.create_branch(
    capabilities={"reasoning", "action_requests", "is_done", "action_responses"},
    resources={"search", "gpt-4o-mini"},
)

params = ReactParams(
    generate=GenerateParams(instruction="Research AI trends"),
    capabilities=set(),
    actions=True,
    max_steps=5,
)
result = await react(session, branch, params)
```

---

### react_stream()

Streaming version of react. Yields steps as they complete.

```python
async def react_stream(session: Session, branch: Branch | str, params: ReactParams) -> AsyncGenerator[ReactStep, None]: ...
```

---

### interpret()

Refine user instructions for better LLM understanding.

```python
async def interpret(session: Session, branch: Branch, params: InterpretParams) -> str: ...
```

---

## Result Types

```python
class ReactStep(BaseModel):
    step: int                                    # 1-indexed
    reasoning: str | None
    actions_requested: list[ActionRequest]
    actions_executed: list[ActionResponse]
    intermediate_options: dict[str, Any] | None
    is_final: bool

class ReactResult(BaseModel):
    steps: list[ReactStep]
    total_steps: int
    completed: bool
    reason_stopped: str

class ReactStepResponse(BaseModel):  # LLM output schema
    reasoning: str | None
    action_requests: list[ActionRequest]
    is_done: bool
```

---

## Security Model

Every structured output operation requires explicit capability declaration:

```python
# Branch defines ALLOWED capabilities/resources
branch = session.create_branch(
    capabilities={"summary", "score", "action_requests", "action_responses"},
    resources={"gpt-4o-mini", "search_tool"},
)

# Operation declares NEEDED capabilities (must be subset of branch)
params = OperateParams(
    generate=GenerateParams(...),
    capabilities={"summary", "score"},
    actions=True,  # Adds action_requests, action_responses
)
```

**Gates**: Branch capabilities > Params capabilities. Resources checked for models/tools.

---

## Helper Functions

```python
# Build Operable for ReactStepResponse with optional intermediate options
def build_step_operable(
    intermediate_options: list[type[BaseModel]] | type[BaseModel] = None,
    *, intermediate_listable: bool = False, intermediate_nullable: bool = True
) -> Operable: ...

# Build Operable for intermediate response options (union types)
def build_intermediate_operable(
    options: list[type[BaseModel]] | type[BaseModel],
    *, listable: bool = False, nullable: bool = True
) -> Operable: ...
```

---

## Usage Patterns

### Pattern 1: Structured Output

```python
class Analysis(BaseModel):
    summary: str
    confidence: float

branch = session.create_branch(capabilities={"summary", "confidence"})
params = OperateParams(
    generate=GenerateParams(instruction="Analyze this text", request_model=Analysis),
    capabilities={"summary", "confidence"},
)
result = await operate(session, branch, params)
```

### Pattern 2: Tool Use

```python
tool = Tool(func_callable=lambda q: f"Results for: {q}")
session.services.register(tool)

branch = session.create_branch(
    capabilities={"answer", "action_requests", "action_responses"},
    resources={"gpt-4o-mini", "search"},
)
params = OperateParams(
    generate=GenerateParams(instruction="Search for Python 3.12"),
    capabilities={"answer"},
    actions=True,
)
result = await operate(session, branch, params)
```

### Pattern 3: ReAct Loop

```python
branch = session.create_branch(
    capabilities={"reasoning", "action_requests", "is_done", "action_responses"},
)
params = ReactParams(
    generate=GenerateParams(instruction="Research and summarize"),
    capabilities=set(),
    actions=True,
    max_steps=5,
)
result = await react(session, branch, params)
```

---

## Common Pitfalls

### 1. Missing Capabilities Declaration

```python
# WRONG - ValidationError
params = OperateParams(generate=GenerateParams(request_model=MyModel))

# CORRECT
params = OperateParams(
    generate=GenerateParams(request_model=MyModel),
    capabilities={"field1", "field2"},
)
```

### 2. Branch Capabilities Mismatch

```python
branch = session.create_branch(capabilities={"summary"})
# WRONG - AccessError (score not in branch)
params = OperateParams(capabilities={"summary", "score"})

# CORRECT - ensure branch has all needed capabilities
branch = session.create_branch(capabilities={"summary", "score"})
```

### 3. Actions Without Action Capabilities

```python
branch = session.create_branch(capabilities={"summary"})
# WRONG - actions=True adds action_requests, action_responses
params = OperateParams(capabilities={"summary"}, actions=True)

# CORRECT
branch = session.create_branch(capabilities={"summary", "action_requests", "action_responses"})
```

---

## See Also

- [Session](../session/session.md), [Message](../session/message.md), [iModel](../services/imodel.md), [Tool](../services/tool.md), [Operable](../types/operable.md)

---

## Design Rationale

**Layered Operations**: Each layer adds capabilities while keeping lower layers testable. generate (pure) -> parse (isolated) -> communicate (persistent) -> operate (secure) -> react (multi-turn).

**Explicit Capabilities**: Prevents accidental field exposure, privilege escalation, and silent failures. Missing capabilities raise immediately.

**Flat Inheritance**: `react.generate.instruction` not `react.operate.communicate.generate.instruction`. Reduces cognitive load.
