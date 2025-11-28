# CLAUDE.md - lionpride Codebase Guide

**Purpose**: Help AI agents navigate the lionpride codebase effectively.

**Repository**: [khive-ai/lionpride](https://github.com/khive-ai/lionpride)
**License**: Apache-2.0
**Copyright**: 2025 HaiyangLi (Ocean)

---

## Quick Start for AI Agents

```python
# Essential imports
from lionpride import Session, Branch, Message, iModel, ServiceRegistry
from lionpride.operations.operate import operate, OperateParams, CommunicateParams, GenerateParams

# Create session with model
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch(name="main")

# Conduct operation
result = await session.conduct("operate", branch, instruction="Your task here")
```

---

## Architecture Overview

```text
src/lionpride/
├── core/              # Foundation primitives (Element, Pile, Flow, Graph, Event)
├── session/           # Session, Branch, Message management
│   └── messages/      # Message content types and utilities
├── services/          # iModel, Tool, ServiceRegistry
│   ├── types/         # Backend, Endpoint, iModel, Tool definitions
│   ├── mcps/          # MCP (Model Context Protocol) integration
│   └── utilities/     # Rate limiting, resilience, token calculation
├── operations/        # Operation system
│   └── operate/       # Core operations (generate, parse, communicate, operate, react)
├── rules/             # Validation rules with auto-correction
├── types/             # Type system (Spec, Operable, Params)
│   └── spec_adapters/ # Pydantic field adaptation
├── lndl/              # LNDL (Lion Natural Description Language) parser
├── libs/              # Utility libraries
│   ├── concurrency/   # Async patterns, cancellation, task management
│   ├── schema_handlers/  # JSON schema, TypeScript conversion
│   └── string_handlers/  # JSON extraction, fuzzy matching
└── ln/                # Convenience functions (to_dict, to_list, json_dumps, etc.)
```

---

## Core Primitives (core/)

### Element

Base class for all entities. Provides UUID identity, creation timestamp, and metadata.

```python
from lionpride import Element

el = Element()
print(el.id)          # UUID4
print(el.created_at)  # datetime
print(el.metadata)    # dict[str, Any]
```

### Pile[T]

Type-safe collection with O(1) UUID lookup. Thread-safe.

```python
from lionpride import Pile, Element

pile = Pile(items=[el1, el2], item_type=Element)
pile[el.id]           # UUID lookup
pile[0]               # Index access
pile[lambda x: x.metadata.get("tag") == "test"]  # Filter
```

### Flow[E, P]

Composition of items (Pile[E]) and progressions (Pile[P]).

```python
from lionpride import Flow, Progression

flow = Flow(item_type=Message)
prog = Progression(name="main", order=[msg1.id, msg2.id])
flow.add_progression(prog)
flow.add_item(msg3, progressions=["main"])
```

### Graph

Directed graph with nodes and edges. Supports pathfinding and topological sort.

```python
from lionpride import Graph, Node, Edge

graph = Graph()
graph.add_node(node1)
edge = Edge(head=node1.id, tail=node2.id, label=["workflow"])
graph.add_edge(edge)
path = await graph.find_path(start=node1, end=node2)
```

### Event

Async execution with lifecycle tracking (PENDING → PROCESSING → COMPLETED/FAILED).

```python
from lionpride import Event

class MyEvent(Event):
    async def _invoke(self):
        return "result"

event = MyEvent(timeout=5.0)
result = await event.invoke()
```

---

## Session System (session/)

### Session

Central orchestrator for messages, branches, services, and operations.

```python
from lionpride import Session, iModel

model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(
    default_generate_model=model,
    default_parse_model=model,
)

# Create branches
branch = session.create_branch(
    name="analysis",
    capabilities={"InsightsModel"},  # Allowed output schemas
    resources={"gpt-4o-mini"},       # Allowed services
)

# Conduct operations
result = await session.conduct("operate", branch, instruction="Analyze this")
```

### Branch

Named progression of messages with access control.

```python
branch = session.create_branch(name="main")
branch.capabilities  # set[str] - allowed output types
branch.resources     # set[str] - allowed service names
```

### Message

Universal container with auto-derived role based on content type.

```python
from lionpride import Message
from lionpride.session.messages import InstructionContent, AssistantResponseContent

# User message
msg = Message(content={"instruction": "Do X"})
msg.role  # MessageRole.USER

# Assistant message
msg = Message(content={"assistant_response": "Done"})
msg.role  # MessageRole.ASSISTANT
```

---

## Services (services/)

### iModel

Wraps LLM backends with unified interface.

```python
from lionpride import iModel

model = iModel(
    provider="openai",           # or "anthropic", "gemini"
    model="gpt-4o-mini",
    name="my_model",             # Optional custom name
    api_key="...",               # Optional, uses env vars by default
)

# Direct invocation
response = await model.invoke(messages=[...])
```

### ServiceRegistry

Manages models and tools with O(1) name lookup.

```python
from lionpride import ServiceRegistry

registry = ServiceRegistry()
registry.register(model)
registry.get("model_name")      # Get by name
registry.has("model_name")      # Check existence
registry.list_names()           # All registered names
```

### Tool

Wraps callable functions for LLM tool use.

```python
from lionpride import Tool

def my_function(arg1: str, arg2: int) -> str:
    return f"{arg1}: {arg2}"

tool = Tool.from_callable(my_function)
result = await tool.invoke({"arg1": "test", "arg2": 42})
```

---

## Operations (operations/)

### Operation Hierarchy

```text
GenerateParams ──────────────┐
ParseParams ─────────────────┼─► CommunicateParams
                             │
CommunicateParams ───────────┤
ActParams ───────────────────┼─► OperateParams
                             │
OperateParams ───────────────┴─► ReactParams
```

### generate

Lowest-level LLM call. No message persistence.

```python
from lionpride.operations.operate import generate, GenerateParams

params = GenerateParams(
    imodel=model,
    instruction="Translate to French: Hello",
)
result = await generate(session, branch, params)
```

### parse

Extract structured data from text.

```python
from lionpride.operations.operate import parse, ParseParams

params = ParseParams(
    text="The answer is 42",
    request_model=AnswerModel,
)
result = await parse(session, branch, params)
```

### communicate

Generate + parse in one operation.

```python
from lionpride.operations.operate import communicate, CommunicateParams

params = CommunicateParams(
    generate=GenerateParams(instruction="..."),
    parse=ParseParams(request_model=MyModel),
)
result = await communicate(session, branch, params)
```

### operate

Full structured output with validation and auto-correction.

```python
from lionpride.operations.operate import operate, OperateParams

params = OperateParams(
    communicate=CommunicateParams(...),
    act=ActParams(tools=[my_tool]),  # Optional tool use
)
result = await operate(session, branch, params)
```

### react

Multi-turn operation loop (ReAct pattern).

```python
from lionpride.operations.operate import react, ReactParams

params = ReactParams(
    operate=OperateParams(...),
    max_iterations=5,
)
result = await react(session, branch, params)
```

---

## Validation (rules/)

### Rule System

Validation rules with auto-correction support.

```python
from lionpride.rules import NumberRule, StringRule, BooleanRule

# Number validation
rule = NumberRule(lower_bound=0, upper_bound=100)
result = await rule.validate(value=50)

# String validation with auto-fix
rule = StringRule(strip=True, to_lower=True, auto_fix=True)
result = await rule.validate(value="  HELLO  ")  # Returns "hello"
```

### Validator

Applies rules to Operable schemas.

```python
from lionpride.rules import Validator
from lionpride.types import Operable, Spec

validator = Validator()
result = await validator.validate_operable(data={"field": value}, operable=operable)
```

---

## Type System (types/)

### Spec

Field specification with validation, defaults, and constraints.

```python
from lionpride.types import Spec

spec = Spec(
    int,
    name="count",
    default=0,
    validator=lambda x: x >= 0,
    description="Item count",
)
```

### Operable

Collection of Specs that generates a Pydantic model.

```python
from lionpride.types import Operable, Spec

operable = Operable([
    Spec(str, name="name"),
    Spec(int, name="age", validator=lambda x: 0 <= x <= 150),
])

# Auto-generated Pydantic model
Model = operable.model
instance = Model(name="Alice", age=30)
```

---

## LNDL (lndl/)

Lion Natural Description Language - DSL for structured LLM output.

```python
from lionpride.lndl import parse_lndl, resolve_lndl

# Parse LNDL string
ast = parse_lndl("""
@name: str = "default"
@age: int
""")

# Resolve to values
result = resolve_lndl(ast, context={"name": "Alice", "age": 30})
```

---

## Common Patterns

### Pattern 1: Basic Chat

```python
from lionpride import Session, iModel

model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch()

result = await session.conduct("operate", branch, instruction="Hello!")
```

### Pattern 2: Structured Output

```python
from pydantic import BaseModel
from lionpride.operations.operate import operate, OperateParams, CommunicateParams, GenerateParams

class Analysis(BaseModel):
    summary: str
    score: float

params = OperateParams(
    communicate=CommunicateParams(
        generate=GenerateParams(
            instruction="Analyze this text",
            request_model=Analysis,
        )
    )
)

result = await operate(session, branch, params)  # Returns Analysis instance
```

### Pattern 3: Multi-Model

```python
from lionpride import Session, iModel, ServiceRegistry

registry = ServiceRegistry()
registry.register(iModel(provider="openai", model="gpt-4o", name="gpt4"))
registry.register(iModel(provider="anthropic", model="claude-3-5-sonnet", name="claude"))

session = Session(services=registry)
branch = session.create_branch(resources={"gpt4", "claude"})

# Use specific model
params = GenerateParams(imodel="claude", instruction="...")
```

### Pattern 4: Tool Use

```python
from lionpride import Tool
from lionpride.operations.operate import operate, OperateParams, ActParams

def search(query: str) -> str:
    return f"Results for: {query}"

tool = Tool.from_callable(search)
session.services.register(tool)

params = OperateParams(
    communicate=CommunicateParams(...),
    act=ActParams(tools=["search"]),
)

result = await operate(session, branch, params)
```

---

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=lionpride --cov-report=term-missing

# Specific module
uv run pytest tests/core/test_pile.py -v
```

---

## Key Files to Read

| File | Purpose |
|------|---------|
| `src/lionpride/__init__.py` | Public API exports |
| `src/lionpride/session/session.py` | Session and Branch |
| `src/lionpride/operations/operate/factory.py` | Main operate function |
| `src/lionpride/services/types/imodel.py` | iModel implementation |
| `src/lionpride/core/pile.py` | Pile collection |
| `src/lionpride/types/spec.py` | Spec and type system |

---

## Versioning

- **lionpride**: v1 primitives (this repo)
- **lionagi**: v0 production framework (origin)

See [AGENTS.md](AGENTS.md) for quick reference.
