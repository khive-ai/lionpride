# Getting Structured Outputs with Spec and Operable

## Overview

This recipe demonstrates how to get typed, validated responses from LLMs using lionpride's Spec and Operable system. Two approaches: LNDL (fuzzy parsing with 75% token reduction) and JSON schema validation.

## Prerequisites

```bash
pip install lionpride pydantic
```

## The Code

### Example 1: Basic Structured Output (LNDL)

LNDL is lionpride's domain-specific language that reduces tokens by 75% compared to JSON schema for complex outputs:

```python
import asyncio
from lionpride import Session, Spec, Operable
from lionpride.services import iModel
from lionpride.operations import communicate

async def extract_person_info():
    """Extract structured person info using LNDL (Spec + Operable)"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    # Define structure using Spec
    person_spec = Operable(
        specs=[
            Spec(str, name="name", nullable=False),
            Spec(int, name="age", nullable=False),
            Spec(str, name="occupation", nullable=True),
            Spec(list[str], name="skills", listable=True, default_factory=list),
        ],
        name="Person"
    )

    branch = session.create_branch(name="extraction")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": (
                "Extract person information from this text: "
                "'Alice is a 32-year-old software engineer who knows Python, "
                "Rust, and TypeScript.'"
            ),
            "imodel": model.name,
            "operable": person_spec,  # Enable LNDL mode
            "return_as": "model",      # Return parsed dict
        }
    )

    print(f"Name: {result['name']}")
    print(f"Age: {result['age']}")
    print(f"Occupation: {result['occupation']}")
    print(f"Skills: {result['skills']}")

    return result

asyncio.run(extract_person_info())
```

### Example 2: Pydantic Models (JSON Schema)

For stricter validation, use Pydantic models with JSON schema:

```python
import asyncio
from pydantic import BaseModel, Field
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

class Person(BaseModel):
    """Person information"""
    name: str = Field(..., description="Person's full name")
    age: int = Field(..., ge=0, le=150, description="Person's age")
    occupation: str | None = Field(None, description="Person's job")
    skills: list[str] = Field(default_factory=list, description="List of skills")

async def extract_with_pydantic():
    """Extract structured person info using Pydantic"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    branch = session.create_branch(name="extraction")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": (
                "Extract person information from this text: "
                "'Bob is a 28-year-old data scientist proficient in Python, "
                "SQL, and machine learning.'"
            ),
            "imodel": model.name,
            "response_model": Person,  # Use Pydantic model
            "return_as": "model",       # Return validated instance
        }
    )

    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print(f"Skills: {result.skills}")
    print(f"\nType: {type(result)}")  # Pydantic model instance

    return result

asyncio.run(extract_with_pydantic())
```

### Example 3: Nested Structures (LNDL)

Spec supports nested structures for complex data:

```python
import asyncio
from lionpride import Session, Spec, Operable
from lionpride.services import iModel
from lionpride.operations import communicate

async def extract_company():
    """Extract nested company structure"""
    session = Session()
    model = iModel(provider="anthropic", endpoint="messages",
                   model="claude-3-5-sonnet-20241022", temperature=0)
    session.services.register(model)

    # Nested structure: Company has employees
    employee_spec = Operable(
        specs=[
            Spec(str, name="name"),
            Spec(str, name="role"),
            Spec(int, name="years_experience"),
        ],
        name="Employee"
    )

    company_spec = Operable(
        specs=[
            Spec(str, name="company_name"),
            Spec(str, name="industry"),
            Spec(int, name="founded_year"),
            Spec(list[dict], name="employees", listable=True),  # List of employees
        ],
        name="Company"
    )

    branch = session.create_branch(name="company-extraction")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": (
                "Extract company info: 'TechCorp is a software company founded in 2015. "
                "Key employees include Alice (CTO, 10 years experience) and Bob "
                "(Lead Engineer, 5 years experience).'"
            ),
            "imodel": model.name,
            "operable": company_spec,
            "return_as": "model",
        }
    )

    print(f"Company: {result['company_name']}")
    print(f"Industry: {result['industry']}")
    print(f"Founded: {result['founded_year']}")
    print(f"\nEmployees:")
    for emp in result['employees']:
        print(f"  - {emp['name']}: {emp['role']} ({emp['years_experience']} yrs)")

    return result

asyncio.run(extract_company())
```

### Example 4: Classification with Validation

Use validators in Spec for runtime validation:

```python
import asyncio
from lionpride import Session, Spec, Operable
from lionpride.services import iModel
from lionpride.operations import communicate

def validate_sentiment(value: str) -> str:
    """Validate sentiment is one of allowed values"""
    allowed = ["positive", "negative", "neutral"]
    if value.lower() not in allowed:
        raise ValueError(f"Sentiment must be one of {allowed}")
    return value.lower()

def validate_confidence(value: float) -> float:
    """Validate confidence is in range [0, 1]"""
    if not 0 <= value <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    return value

async def classify_sentiment():
    """Classify text sentiment with validation"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    sentiment_spec = Operable(
        specs=[
            Spec(str, name="sentiment", validator=validate_sentiment),
            Spec(float, name="confidence", validator=validate_confidence),
            Spec(str, name="reasoning", nullable=True),
        ],
        name="SentimentAnalysis"
    )

    branch = session.create_branch(name="classification")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": (
                "Analyze sentiment: 'This product exceeded my expectations! "
                "The quality is outstanding and delivery was fast.'"
            ),
            "imodel": model.name,
            "operable": sentiment_spec,
            "return_as": "model",
        }
    )

    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reasoning: {result['reasoning']}")

    return result

asyncio.run(classify_sentiment())
```

### Example 5: Batch Extraction

Extract multiple items from text:

```python
import asyncio
from pydantic import BaseModel
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import communicate

class Product(BaseModel):
    """Product information"""
    name: str
    price: float
    category: str

class ProductList(BaseModel):
    """List of products"""
    products: list[Product]

async def extract_products():
    """Extract list of products from text"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0)
    session.services.register(model)

    branch = session.create_branch(name="products")

    result = await communicate(
        session=session,
        branch=branch,
        parameters={
            "instruction": (
                "Extract all products: 'Our store offers: "
                "iPhone 15 for $999 (electronics), "
                "Nike Air Max for $150 (shoes), and "
                "The Great Gatsby for $12 (books).'"
            ),
            "imodel": model.name,
            "response_model": ProductList,
            "return_as": "model",
        }
    )

    print(f"Found {len(result.products)} products:\n")
    for product in result.products:
        print(f"  {product.name}: ${product.price:.2f} ({product.category})")

    return result

asyncio.run(extract_products())
```

## Expected Output

### Example 1 (LNDL)

```
Name: Alice
Age: 32
Occupation: software engineer
Skills: ['Python', 'Rust', 'TypeScript']
```

### Example 2 (Pydantic)

```
Name: Bob
Age: 28
Occupation: data scientist
Skills: ['Python', 'SQL', 'machine learning']

Type: <class '__main__.Person'>
```

### Example 3 (Nested)

```
Company: TechCorp
Industry: software
Founded: 2015

Employees:
  - Alice: CTO (10 yrs)
  - Bob: Lead Engineer (5 yrs)
```

### Example 4 (Classification)

```
Sentiment: positive
Confidence: 0.95
Reasoning: The text contains strong positive indicators like 'exceeded expectations',
'outstanding', and 'fast', with no negative elements.
```

### Example 5 (Batch)

```
Found 3 products:

  iPhone 15: $999.00 (electronics)
  Nike Air Max: $150.00 (shoes)
  The Great Gatsby: $12.00 (books)
```

## LNDL vs JSON Schema Comparison

### Token Efficiency

For a 20-field schema:

| Approach | Tokens | Reduction |
|----------|--------|-----------|
| JSON Schema | 800 | Baseline |
| LNDL (Operable) | ~200 | 75% |

### When to Use Each

**Use LNDL (Operable + Spec):**

- Complex schemas with many fields (>10)
- Token budget is critical
- Fuzzy parsing acceptable (handles model variations)
- Need validators and metadata

**Use JSON Schema (Pydantic):**

- Strict validation required
- Existing Pydantic models
- OpenAPI/JSON Schema compatibility needed
- Simpler schemas (<10 fields)

## Variations

### Optional Fields with Defaults

```python
user_spec = Operable(
    specs=[
        Spec(str, name="username", nullable=False),
        Spec(str, name="email", nullable=True, default="no-email@example.com"),
        Spec(bool, name="is_active", default=True),
        Spec(list[str], name="tags", default_factory=list),
    ],
    name="User"
)
```

### Enum-like Fields

```python
from lionpride.rules import ChoiceRule

status_spec = Operable(
    specs=[
        Spec(str, name="status", validator=ChoiceRule(["pending", "approved", "rejected"])),
        Spec(str, name="reason"),
    ],
    name="ApprovalStatus"
)
```

### Return Raw Dict vs Pydantic Instance

```python
# Return as dict
result = await communicate(
    ...,
    parameters={
        "operable": spec,
        "return_as": "model",  # Returns dict
    }
)

# Return as Pydantic instance
result = await communicate(
    ...,
    parameters={
        "response_model": MyModel,
        "return_as": "model",  # Returns MyModel instance
    }
)
```

## Common Pitfalls

1. **Mixing operable and response_model**

   ```python
   # ❌ Wrong - use one or the other
   result = await communicate(
       parameters={
           "operable": spec,
           "response_model": Model,  # Conflict!
       }
   )

   # ✅ Right - choose one approach
   result = await communicate(
       parameters={"operable": spec}  # LNDL
   )
   # OR
   result = await communicate(
       parameters={"response_model": Model}  # JSON schema
   )
   ```

2. **Forgetting return_as="model"**

   ```python
   # ❌ Wrong - returns text, not structured data
   result = await communicate(
       parameters={"operable": spec}
   )

   # ✅ Right - explicitly request model parsing
   result = await communicate(
       parameters={"operable": spec, "return_as": "model"}
   )
   ```

3. **Type mismatches in Spec**

   ```python
   # ❌ Wrong - base_type doesn't match
   Spec(str, name="age")  # Age should be int

   # ✅ Right
   Spec(int, name="age")
   ```

## Next Steps

- **Tool calling with structured outputs**: See [Tool Calling](tool_calling.md)
- **ReAct with typed responses**: See [Multi-Agent](multi_agent.md)
- **Validation strategies**: See [Error Handling](error_handling.md)

## See Also

- [API Reference: Spec](../api/types.md#spec)
- [API Reference: Operable](../api/types.md#operable)
- [LNDL Deep Dive](../user_guide/lndl.md)
- [Pydantic Integration](../integration/pydantic.md)
