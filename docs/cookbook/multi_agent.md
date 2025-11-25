# Multi-Agent Workflows with Builder and Flow

## Overview

This recipe demonstrates building multi-agent workflows using lionpride's Builder and Flow system. Create operation graphs where different agents collaborate, specialize in tasks, and execute in dependency-aware order.

## Prerequisites

```bash
pip install lionpride pydantic
```

## The Code

### Example 1: Simple Sequential Workflow

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import Builder, flow

async def research_write_edit():
    """Three-agent workflow: research → write → edit"""
    session = Session()

    # Create specialized models for each role
    researcher = iModel(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3,
        name="researcher"
    )

    writer = iModel(
        provider="anthropic",
        endpoint="messages",
        model="claude-3-5-haiku-20241022",
        temperature=0.7,
        name="writer"
    )

    editor = iModel(
        provider="openai",
        model="gpt-4o",
        temperature=0.2,
        name="editor"
    )

    for model in [researcher, writer, editor]:
        session.services.register(model)

    # Build operation graph
    builder = Builder(session=session)

    # Step 1: Research
    research_op = builder.communicate(
        instruction="Research key facts about quantum computing for a blog post",
        imodel="researcher",
        branch="research",
    )

    # Step 2: Write (depends on research)
    write_op = builder.communicate(
        instruction=(
            "Using the research provided, write a 300-word blog post "
            "about quantum computing for a general audience"
        ),
        imodel="writer",
        branch="writing",
        context_from=[research_op],  # Depends on research
    )

    # Step 3: Edit (depends on writing)
    edit_op = builder.communicate(
        instruction="Edit this blog post for clarity and engagement",
        imodel="editor",
        branch="editing",
        context_from=[write_op],  # Depends on writing
    )

    # Execute workflow
    results = await flow(builder.operations)

    print("=== Research ===")
    print(results[research_op.id].result)
    print("\n=== Draft ===")
    print(results[write_op.id].result)
    print("\n=== Final (Edited) ===")
    print(results[edit_op.id].result)

    return results

asyncio.run(research_write_edit())
```

### Example 2: Parallel Analysis with Synthesis

```python
import asyncio
from pydantic import BaseModel, Field
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import Builder, flow

class Analysis(BaseModel):
    """Analysis result"""
    perspective: str
    key_points: list[str]
    recommendation: str

async def multi_perspective_analysis():
    """Multiple agents analyze from different perspectives, then synthesize"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", temperature=0.7)
    session.services.register(model)

    topic = "Should our company adopt a 4-day work week?"

    builder = Builder(session=session)

    # Parallel analyses from different perspectives
    hr_analysis = builder.communicate(
        instruction=f"Analyze from HR perspective: {topic}",
        imodel=model.name,
        branch="hr",
        response_model=Analysis,
    )

    finance_analysis = builder.communicate(
        instruction=f"Analyze from financial perspective: {topic}",
        imodel=model.name,
        branch="finance",
        response_model=Analysis,
    )

    operations_analysis = builder.communicate(
        instruction=f"Analyze from operations perspective: {topic}",
        imodel=model.name,
        branch="operations",
        response_model=Analysis,
    )

    # Synthesis depends on all analyses
    synthesis = builder.communicate(
        instruction=(
            "Synthesize the HR, finance, and operations analyses into "
            "a balanced recommendation for leadership"
        ),
        imodel=model.name,
        branch="synthesis",
        context_from=[hr_analysis, finance_analysis, operations_analysis],
    )

    # Execute: parallel analyses, then synthesis
    results = await flow(builder.operations)

    print("=== HR Analysis ===")
    print(results[hr_analysis.id].result)
    print("\n=== Finance Analysis ===")
    print(results[finance_analysis.id].result)
    print("\n=== Operations Analysis ===")
    print(results[operations_analysis.id].result)
    print("\n=== Final Synthesis ===")
    print(results[synthesis.id].result)

    return results

asyncio.run(multi_perspective_analysis())
```

### Example 3: Tool-Using Agents in Workflow

```python
import asyncio
from lionpride import Session
from lionpride.services import iModel, Tool
from lionpride.operations import Builder, flow

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text"""
    # Simulated sentiment analysis
    if "great" in text.lower() or "excellent" in text.lower():
        return {"sentiment": "positive", "score": 0.9}
    elif "bad" in text.lower() or "poor" in text.lower():
        return {"sentiment": "negative", "score": 0.8}
    else:
        return {"sentiment": "neutral", "score": 0.5}

def get_product_reviews(product: str) -> list[str]:
    """Get reviews for a product"""
    reviews = {
        "laptop": [
            "Great performance and battery life!",
            "Screen is excellent but keyboard is mediocre",
            "Best laptop I've owned"
        ],
        "phone": [
            "Camera quality is poor",
            "Battery drains too quickly",
            "Good value for the price"
        ]
    }
    return reviews.get(product.lower(), [])

async def review_analysis_workflow():
    """Multi-agent workflow with tool usage"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o", temperature=0)
    session.services.register(model)

    # Register tools
    sentiment_tool = Tool.from_function(analyze_sentiment)
    reviews_tool = Tool.from_function(get_product_reviews)
    session.services.register(sentiment_tool)
    session.services.register(reviews_tool)

    builder = Builder(session=session)

    # Step 1: Fetch reviews
    fetch_op = builder.react(
        instruction="Get reviews for the laptop product",
        imodel=model.name,
        branch="fetch",
        tools=[reviews_tool.name],
        max_steps=2,
    )

    # Step 2: Analyze each review (depends on fetch)
    analyze_op = builder.react(
        instruction="Analyze the sentiment of each review",
        imodel=model.name,
        branch="analyze",
        tools=[sentiment_tool.name],
        context_from=[fetch_op],
        max_steps=5,
    )

    # Step 3: Generate summary (depends on analysis)
    summary_op = builder.communicate(
        instruction=(
            "Summarize the review analysis: what are customers saying? "
            "What should we improve?"
        ),
        imodel=model.name,
        branch="summary",
        context_from=[analyze_op],
    )

    # Execute workflow
    results = await flow(builder.operations)

    print("=== Reviews Fetched ===")
    print(results[fetch_op.id].result)
    print("\n=== Sentiment Analysis ===")
    print(results[analyze_op.id].result)
    print("\n=== Summary ===")
    print(results[summary_op.id].result)

    return results

asyncio.run(review_analysis_workflow())
```

### Example 4: Conditional Branching

```python
import asyncio
from pydantic import BaseModel
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import Builder, flow

class CodeReview(BaseModel):
    """Code review result"""
    has_issues: bool
    issues: list[str]
    severity: str  # "minor", "major", "critical"

async def code_review_workflow():
    """Conditional workflow based on review results"""
    session = Session()
    model = iModel(provider="openai", model="gpt-4o", temperature=0)
    session.services.register(model)

    code_to_review = """
def process_data(data):
    results = []
    for item in data:
        result = item * 2
        results.append(result)
    return results
    """

    builder = Builder(session=session)

    # Step 1: Initial review
    review_op = builder.communicate(
        instruction=f"Review this code for issues:\n{code_to_review}",
        imodel=model.name,
        branch="review",
        response_model=CodeReview,
    )

    # Execute initial review
    review_results = await flow([review_op])
    review = review_results[review_op.id].result

    print(f"=== Code Review ===")
    print(f"Has issues: {review.has_issues}")
    print(f"Severity: {review.severity}")
    print(f"Issues: {review.issues}")

    # Conditional: if critical issues, request fixes
    if review.has_issues and review.severity == "critical":
        print("\n=== Critical Issues Found - Requesting Fixes ===")

        fix_op = builder.communicate(
            instruction=f"Fix the critical issues:\n{code_to_review}",
            imodel=model.name,
            branch="fix",
            context_from=[review_op],
        )

        verify_op = builder.communicate(
            instruction="Verify the fixes are correct",
            imodel=model.name,
            branch="verify",
            context_from=[fix_op],
        )

        # Execute fix workflow
        fix_results = await flow([fix_op, verify_op])

        print("\n=== Fixed Code ===")
        print(fix_results[fix_op.id].result)
        print("\n=== Verification ===")
        print(fix_results[verify_op.id].result)

    else:
        print("\n=== No Critical Issues - Approved ===")

    return review

asyncio.run(code_review_workflow())
```

### Example 5: Iterative Refinement Loop

```python
import asyncio
from pydantic import BaseModel
from lionpride import Session
from lionpride.services import iModel
from lionpride.operations import Builder, flow

class Quality(BaseModel):
    """Quality assessment"""
    score: int  # 1-10
    feedback: str
    approved: bool

async def iterative_refinement():
    """Iterative refinement until quality threshold met"""
    session = Session()

    writer = iModel(provider="openai", model="gpt-4o-mini",
                    temperature=0.7, name="writer")
    critic = iModel(provider="anthropic", endpoint="messages",
                    model="claude-3-5-sonnet-20241022",
                    temperature=0.3, name="critic")

    session.services.register(writer)
    session.services.register(critic)

    topic = "The importance of code review"
    max_iterations = 3
    quality_threshold = 8

    current_draft = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print('='*60)

        builder = Builder(session=session)

        # Write or revise
        if current_draft is None:
            write_instruction = f"Write a technical blog post about: {topic}"
        else:
            write_instruction = (
                f"Revise this draft based on the feedback:\n{current_draft}"
            )

        write_op = builder.communicate(
            instruction=write_instruction,
            imodel="writer",
            branch=f"write-{iteration}",
        )

        # Critique
        critique_op = builder.communicate(
            instruction="Evaluate this blog post and provide a quality score (1-10)",
            imodel="critic",
            branch=f"critique-{iteration}",
            response_model=Quality,
            context_from=[write_op],
        )

        # Execute
        results = await flow([write_op, critique_op])

        current_draft = results[write_op.id].result
        quality = results[critique_op.id].result

        print(f"\nDraft (excerpt):\n{current_draft[:200]}...")
        print(f"\nQuality Score: {quality.score}/10")
        print(f"Feedback: {quality.feedback}")

        if quality.approved and quality.score >= quality_threshold:
            print(f"\n✓ Quality threshold met! Final score: {quality.score}/10")
            break
    else:
        print(f"\n⚠ Max iterations reached. Final score: {quality.score}/10")

    print(f"\n{'='*60}")
    print("FINAL DRAFT")
    print('='*60)
    print(current_draft)

    return current_draft

asyncio.run(iterative_refinement())
```

## Expected Output

### Example 1 (Sequential)

```
=== Research ===
Quantum computing leverages quantum mechanics principles like superposition
and entanglement. Key facts: qubits can be 0 and 1 simultaneously...

=== Draft ===
Imagine a computer that doesn't just process information as 0s and 1s,
but exists in multiple states at once...

=== Final (Edited) ===
[Polished version with improved clarity and engagement]
```

### Example 2 (Parallel)

```
=== HR Analysis ===
Perspective: Human Resources
Key Points: ['Improved work-life balance', 'Increased employee satisfaction'...]
Recommendation: Pilot program in one department

=== Finance Analysis ===
Perspective: Financial
Key Points: ['Potential productivity gains', 'Facility cost savings'...]
Recommendation: Trial with cost-benefit analysis

=== Operations Analysis ===
[Similar structure]

=== Final Synthesis ===
Based on comprehensive analysis across HR, finance, and operations,
I recommend a phased 6-month pilot program...
```

### Example 5 (Iterative)

```
============================================================
Iteration 1
============================================================

Draft (excerpt):
Code review is an essential practice in software development...

Quality Score: 6/10
Feedback: Good introduction but lacks specific examples and concrete benefits

============================================================
Iteration 2
============================================================

Draft (excerpt):
Code review is more than just finding bugs—it's a cornerstone of...

Quality Score: 9/10
Feedback: Excellent improvement! Clear examples and actionable insights

✓ Quality threshold met! Final score: 9/10
```

## Key Concepts

### Builder Pattern

```python
builder = Builder(session=session)

# Add operations
op1 = builder.communicate(...)
op2 = builder.react(...)

# Operations stored in builder.operations
# Execute with flow()
results = await flow(builder.operations)
```

### Dependencies

```python
# Sequential: op2 depends on op1
op1 = builder.communicate(...)
op2 = builder.communicate(
    context_from=[op1],  # Waits for op1
)

# Parallel: op2 and op3 run simultaneously
op2 = builder.communicate(...)
op3 = builder.communicate(...)

# Diamond: op4 waits for both op2 and op3
op4 = builder.communicate(
    context_from=[op2, op3],
)
```

### Execution Order

```
flow() automatically determines execution order based on dependencies:

No dependencies → Execute in parallel
Has dependencies → Execute after dependencies complete
```

## Variations

### Custom Operation Types

```python
# Mix different operation types
builder.communicate(...)  # Stateful chat
builder.generate(...)     # Stateless generation
builder.react(...)        # Tool-calling loop
builder.operate(...)      # Generic operation
```

### Error Handling

```python
results = await flow(builder.operations)

for op_id, result in results.items():
    if result.execution.status != "completed":
        print(f"Operation {op_id} failed: {result.execution.error}")
```

### Streaming Results

```python
from lionpride.operations import flow_stream

async for op_id, chunk in flow_stream(builder.operations):
    print(f"Operation {op_id}: {chunk}")
```

## Common Pitfalls

1. **Circular dependencies**

   ```python
   # ❌ Wrong - creates cycle
   op1 = builder.communicate(context_from=[op2])
   op2 = builder.communicate(context_from=[op1])  # Deadlock!

   # ✅ Right - linear dependency
   op1 = builder.communicate()
   op2 = builder.communicate(context_from=[op1])
   ```

2. **Forgetting to await flow()**

   ```python
   # ❌ Wrong - returns coroutine, doesn't execute
   results = flow(builder.operations)

   # ✅ Right - await execution
   results = await flow(builder.operations)
   ```

3. **Reusing builder incorrectly**

   ```python
   # ❌ Wrong - operations accumulate
   builder = Builder(session=session)
   builder.communicate(...)
   await flow(builder.operations)  # 1 operation
   builder.communicate(...)
   await flow(builder.operations)  # Now 2 operations!

   # ✅ Right - new builder per workflow
   builder1 = Builder(session=session)
   builder1.communicate(...)
   await flow(builder1.operations)

   builder2 = Builder(session=session)
   builder2.communicate(...)
   await flow(builder2.operations)
   ```

## Next Steps

- **Streaming workflows**: See [Streaming](streaming.md)
- **Error handling in workflows**: See [Error Handling](error_handling.md)
- **Production deployment**: See [Deployment Guide](../user_guide/deployment.md)

## See Also

- [API Reference: Builder](../api/operations.md#builder)
- [API Reference: flow()](../api/operations.md#flow)
- [Workflow Patterns](../patterns/workflow.md)
- [Advanced Orchestration](../user_guide/orchestration.md)
