# Work System - Declarative Workflow Orchestration

The work system provides capability-based workflow orchestration through declarative DSL.

## Core Concepts

- **Form**: Unit of work with assignment DSL declaring data flow and resources
- **Report**: Workflow orchestrator - subclass to define output schemas
- **FormResources**: Capability declarations specifying what APIs/tools each step can access
- **flow_report**: Executes report via compiled graph with automatic dependency resolution

## Assignment DSL

```text
[branch:] inputs -> outputs [| resources]
```

### Components

| Component | Required | Description |
|-----------|----------|-------------|
| `branch:` | No | Named branch prefix for this step |
| `inputs -> outputs` | Yes | Data flow declaration |
| `\| resources` | No | API and tool declarations |

### Resource Types

| Type | Description | Example |
|------|-------------|---------|
| `api` | Default model for all roles | `api:gpt4o` |
| `api_gen` | Model for generation | `api_gen:gpt5` |
| `api_parse` | Model for parsing/extraction | `api_parse:gpt4mini` |
| `api_interpret` | Model for react interpretation | `api_interpret:gemini` |
| `tool` | Tool access | `tool:search`, `tool:*` |

### Resource Resolution

**APIs**: `api_gen` → `api` → branch's only model → error if ambiguous

**Tools**: Explicit tools only, or `tool:*` for all branch tools. No tools by default (safe).

## Examples

### Simple (defaults)

```python
"context -> plan"
```

- Uses branch's only model (or error if multiple)
- No tools

### With branch prefix

```python
"orchestrator: context -> plan"
```

- Executes on branch named "orchestrator"

### With single API

```python
"context -> plan | api:gpt4mini"
```

- Uses gpt4mini for all API roles (gen, parse, interpret)

### With role-specific APIs

```python
"orchestrator: context -> plan | api_gen:gpt5, api_parse:gpt4mini"
```

- Generation: gpt5
- Parsing: gpt4mini
- Interpretation: falls back to gpt5 (api_gen)

### With tools

```python
"planner: context -> plan | api:gpt4o, tool:search, tool:calendar"
```

- Specific tools: search, calendar

```python
"executor: plan -> result | api:gpt4o, tool:*"
```

- All tools available to branch
