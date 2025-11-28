# AGENTS.md - lionpride Quick Reference

**One-page reference for AI agents working with lionpride.**

---

## Essential Imports

```python
# Core
from lionpride import Session, Branch, Message, iModel, ServiceRegistry
from lionpride import Element, Pile, Flow, Graph, Node, Edge, Progression, Event

# Operations
from lionpride.operations.operate import (
    operate, react, communicate, generate, parse,
    OperateParams, ReactParams, CommunicateParams, GenerateParams, ParseParams, ActParams,
)

# Types
from lionpride.types import Spec, Operable, Params

# Rules
from lionpride.rules import Validator, NumberRule, StringRule, BooleanRule, ChoiceRule

# Work (declarative workflows)
from lionpride.work import Report, Form, flow_report
```

---

## Common Patterns

### 1. Basic Session Setup

```python
model = iModel(provider="openai", model="gpt-4o-mini")
session = Session(default_generate_model=model)
branch = session.create_branch(name="main")
```

### 2. Conduct Operation

```python
result = await session.conduct("operate", branch, instruction="Your task")
```

### 3. Structured Output

```python
from pydantic import BaseModel

class MyOutput(BaseModel):
    field1: str
    field2: int

params = OperateParams(
    communicate=CommunicateParams(
        generate=GenerateParams(
            instruction="...",
            request_model=MyOutput,
        )
    )
)
result = await operate(session, branch, params)
```

### 4. Multi-Model

```python
registry = ServiceRegistry()
registry.register(iModel(provider="openai", model="gpt-4o", name="gpt4"))
registry.register(iModel(provider="anthropic", model="claude-3-5-sonnet", name="claude"))
session = Session(services=registry)
branch = session.create_branch(resources={"gpt4", "claude"})
```

### 5. Tool Use

```python
from lionpride import Tool

tool = Tool.from_callable(my_function)
session.services.register(tool)

params = OperateParams(
    communicate=CommunicateParams(...),
    act=ActParams(tools=["my_function"]),
)
```

### 6. Declarative Workflows (Report/Form)

```python
from pydantic import BaseModel
from lionpride.work import Report, flow_report

class Analysis(BaseModel):
    summary: str
    score: float

class MyReport(Report):
    # Schema attributes (filled during execution)
    analysis: Analysis | None = None
    insights: str | None = None

    # Workflow definition
    assignment: str = "topic -> insights"
    form_assignments: list[str] = [
        "topic -> analysis",
        "analysis -> insights",
    ]

report = MyReport()
report.initialize(topic="AI adoption")
result = await flow_report(session, branch, report)
```

---

## Key Classes

| Class | Purpose | Import |
|-------|---------|--------|
| `Session` | Orchestrator | `from lionpride import Session` |
| `Branch` | Conversation thread | `from lionpride import Branch` |
| `Message` | Universal container | `from lionpride import Message` |
| `iModel` | LLM wrapper | `from lionpride import iModel` |
| `ServiceRegistry` | Service manager | `from lionpride import ServiceRegistry` |
| `Tool` | Function wrapper | `from lionpride import Tool` |
| `Pile` | O(1) collection | `from lionpride import Pile` |
| `Flow` | Items + progressions | `from lionpride import Flow` |
| `Spec` | Field spec | `from lionpride.types import Spec` |
| `Operable` | Schema generator | `from lionpride.types import Operable` |
| `Report` | Workflow namespace | `from lionpride.work import Report` |
| `Form` | Assignment data contract | `from lionpride.work import Form` |

---

## Operation Hierarchy

```text
GenerateParams → CommunicateParams → OperateParams → ReactParams
                 ParseParams ↗       ActParams ↗
```

---

## File Locations

| What | Where |
|------|-------|
| Session/Branch | `src/lionpride/session/session.py` |
| Operations | `src/lionpride/operations/operate/` |
| iModel | `src/lionpride/services/types/imodel.py` |
| Pile/Flow | `src/lionpride/core/` |
| Spec/Operable | `src/lionpride/types/` |
| Validation | `src/lionpride/rules/` |
| Report/Form | `src/lionpride/work/` |

---

## Testing

```bash
uv run pytest                              # All tests
uv run pytest --cov=lionpride             # With coverage
uv run pytest tests/core/ -v              # Core tests
```

---

See [CLAUDE.md](CLAUDE.md) for detailed documentation.
