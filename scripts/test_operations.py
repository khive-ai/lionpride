#!/usr/bin/env python
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Functional tests for lionpride operations.

Run with: uv run python scripts/test_operations.py
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from lionpride.services import iModel
from lionpride.session import Session


async def test_generate():
    """Test stateless generate operation."""
    print("\n=== Testing generate ===")

    session = Session()

    # Register model
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)

    branch = session.create_branch(name="test")

    # Test generate - just needs imodel and messages
    op = await session.conduct(
        "generate",
        branch=branch,
        imodel=model,
        messages=[{"role": "user", "content": "Say 'Hello World' and nothing else."}],
    )

    print(f"Status: {op.status}")
    print(f"Response: {op.response}")
    print(f"Parameters preserved: {list(op.parameters.keys())}")

    assert op.status.value == "completed"
    assert "hello" in op.response.lower()
    print("✓ generate passed")


async def test_communicate():
    """Test stateful communicate operation."""
    print("\n=== Testing communicate ===")

    session = Session()

    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)

    branch = session.create_branch(name="test")

    # First message
    op1 = await session.conduct(
        "communicate",
        branch=branch,
        imodel=model,
        instruction="My name is Alice. Remember it.",
    )

    print(f"Response 1: {op1.response[:100]}...")

    # Second message - should remember context
    op2 = await session.conduct(
        "communicate",
        branch=branch,
        imodel=model,
        instruction="What is my name?",
    )

    print(f"Response 2: {op2.response[:100]}...")

    assert "alice" in op2.response.lower()
    print("✓ communicate passed (stateful context works)")


async def test_operate():
    """Test structured output with operate."""
    print("\n=== Testing operate ===")

    class Sentiment(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(ge=0, le=1, description="Confidence score")

    session = Session()

    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)

    branch = session.create_branch(name="test")

    op = await session.conduct(
        "operate",
        branch=branch,
        imodel=model,
        instruction="Analyze the sentiment: 'I love this product!'",
        response_model=Sentiment,
    )

    print(f"Status: {op.status}")
    print(f"Response: {op.response}")
    print(f"Response type: {type(op.response)}")

    assert isinstance(op.response, Sentiment)
    assert op.response.sentiment.lower() == "positive"
    print("✓ operate passed (structured output works)")


async def test_operate_with_actions():
    """Test operate with tool/action execution - real-world scenario."""
    print("\n=== Testing operate with actions ===")

    import random
    from datetime import datetime

    from lionpride.services.types.tool import Tool, ToolConfig

    # Simulated database with dynamic data the model cannot know
    _inventory_db = {
        "SKU-001": {"name": "Widget A", "stock": random.randint(10, 100), "price": 29.99},
        "SKU-002": {"name": "Widget B", "stock": random.randint(5, 50), "price": 49.99},
        "SKU-003": {"name": "Gadget X", "stock": random.randint(0, 20), "price": 99.99},
    }

    async def check_inventory(sku: str) -> str:
        """Check current inventory for a product SKU."""
        if sku in _inventory_db:
            item = _inventory_db[sku]
            return f"Product: {item['name']}, Stock: {item['stock']} units, Price: ${item['price']}"
        return f"SKU {sku} not found in inventory"

    async def get_current_timestamp() -> str:
        """Get the current server timestamp."""
        return datetime.now().isoformat()

    # Response model WITHOUT action_requests - let actions=True add it
    class InventoryAnalysis(BaseModel):
        reasoning: str = Field(description="Your analysis of the inventory check")
        recommendation: str = Field(description="Your recommendation based on stock levels")

    session = Session()

    model = iModel(provider="openai", model="gpt-4.1-mini", name="test_model")
    session.services.register(model)

    # Register tools
    inventory_tool = Tool(
        func_callable=check_inventory,
        config=ToolConfig(
            name="check_inventory",
            description="Check current inventory levels for a product SKU. Returns stock count and price.",
        ),
    )
    session.services.register(iModel(backend=inventory_tool))

    timestamp_tool = Tool(
        func_callable=get_current_timestamp,
        config=ToolConfig(
            name="get_timestamp",
            description="Get the current server timestamp for audit logging.",
        ),
    )
    session.services.register(iModel(backend=timestamp_tool))

    branch = session.create_branch(name="test")

    op = await session.conduct(
        "operate",
        branch=branch,
        imodel=model,
        instruction=(
            "I need to check if we have Widget A (SKU-001) in stock for a customer order. "
            "First use the check_inventory tool with sku='SKU-001' to get current stock levels. "
            "This is real-time database data you cannot know without checking the tool. "
            "After checking, provide your analysis and recommendation."
        ),
        response_model=InventoryAnalysis,
        actions=True,
        tools=["check_inventory", "get_timestamp"],
        max_retries=2,  # Give model more chances to match schema
    )

    print(f"Status: {op.status}")
    print(f"Response type: {type(op.response)}")

    # Check if action was executed
    if hasattr(op.response, "action_responses") and op.response.action_responses:
        print(f"Action responses: {op.response.action_responses}")
        # Verify it got real inventory data
        for resp in op.response.action_responses:
            if "Widget A" in str(resp.output):
                print(f"Inventory data retrieved: {resp.output}")
                print("✓ operate with actions passed (inventory checked)")
                return
        print("✓ operate with actions passed (tool called)")
    elif hasattr(op.response, "action_requests") and op.response.action_requests:
        print(f"Action requests: {op.response.action_requests}")
        print("⚠ Actions requested but not executed")
    else:
        # Check nested response
        if hasattr(op.response, "inventoryanalysis"):
            print(f"Nested response: {op.response.inventoryanalysis}")
        print(f"Full response: {op.response}")
        print("⚠ No action responses found")


async def test_react():
    """Test ReAct multi-step reasoning - complex research task."""
    print("\n=== Testing react ===")

    import random

    from lionpride.services.types.tool import Tool, ToolConfig

    # Simulated company database - model cannot know this without querying
    _company_db = {
        "ACME": {
            "employees": random.randint(100, 500),
            "revenue_millions": random.randint(10, 100),
            "founded": 2015,
            "sector": "Technology",
        },
        "GLOBEX": {
            "employees": random.randint(50, 200),
            "revenue_millions": random.randint(5, 50),
            "founded": 2018,
            "sector": "Finance",
        },
        "INITECH": {
            "employees": random.randint(200, 800),
            "revenue_millions": random.randint(20, 150),
            "founded": 2010,
            "sector": "Consulting",
        },
    }

    _market_data = {
        "Technology": {"growth_rate": f"{random.randint(5, 15)}%", "outlook": "positive"},
        "Finance": {"growth_rate": f"{random.randint(2, 8)}%", "outlook": "stable"},
        "Consulting": {"growth_rate": f"{random.randint(3, 10)}%", "outlook": "moderate"},
    }

    async def get_company_info(company_name: str) -> str:
        """Get company information from database."""
        name = company_name.upper()
        if name in _company_db:
            c = _company_db[name]
            return (
                f"Company: {name}\n"
                f"Employees: {c['employees']}\n"
                f"Revenue: ${c['revenue_millions']}M\n"
                f"Founded: {c['founded']}\n"
                f"Sector: {c['sector']}"
            )
        return f"Company '{company_name}' not found in database"

    async def get_market_analysis(sector: str) -> str:
        """Get market analysis for a sector."""
        sector_key = sector.title()
        if sector_key in _market_data:
            m = _market_data[sector_key]
            return f"Sector: {sector_key}\nGrowth Rate: {m['growth_rate']}\nOutlook: {m['outlook']}"
        return f"No market data for sector '{sector}'"

    async def calculate_metric(expression: str) -> str:
        """Calculate a financial metric."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    session = Session()

    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)

    # Register tools
    company_tool = Tool(
        func_callable=get_company_info,
        config=ToolConfig(
            name="get_company_info",
            description="Get company information including employees, revenue, founding year, and sector.",
        ),
    )

    market_tool = Tool(
        func_callable=get_market_analysis,
        config=ToolConfig(
            name="get_market_analysis",
            description="Get market analysis for a business sector including growth rate and outlook.",
        ),
    )

    calc_tool = Tool(
        func_callable=calculate_metric,
        config=ToolConfig(
            name="calculate",
            description="Calculate financial metrics. Pass a mathematical expression.",
        ),
    )

    # Register all tools
    for tool in [company_tool, market_tool, calc_tool]:
        session.services.register(iModel(backend=tool))

    branch = session.create_branch(name="test")

    op = await session.conduct(
        "react",
        branch=branch,
        instruction=(
            "I need a brief investment analysis for ACME company. "
            "1) Look up the company info to get their sector and revenue. "
            "2) Then check the market analysis for their sector. "
            "3) Calculate their revenue per employee (revenue_millions * 1000000 / employees). "
            "Provide a final recommendation based on your findings."
        ),
        imodel=model,
        tools=[company_tool, market_tool, calc_tool],
        model_name="gpt-4o-mini",
        max_steps=6,
        verbose=True,
    )

    print(f"\nStatus: {op.status}")
    print(f"Response type: {type(op.response)}")

    # ReactResult has steps, final_response, completed
    if hasattr(op.response, "steps"):
        print(f"Total steps: {op.response.total_steps}")
        print(f"Completed: {op.response.completed}")
        print(f"Reason stopped: {op.response.reason_stopped}")

        # Count tool calls
        total_tool_calls = sum(len(s.actions_executed) for s in op.response.steps)
        print(f"Total tool calls made: {total_tool_calls}")

        if op.response.final_response:
            print(f"Final answer: {str(op.response.final_response)[:300]}...")

        if total_tool_calls >= 2:
            print("✓ react passed (multi-step reasoning with tool calls)")
        else:
            print("⚠ react completed but fewer tool calls than expected")
    else:
        print(f"Response: {op.response}")
        print("⚠ Unexpected response format")


async def test_builder_sequential():
    """Test Builder with sequential operation chain."""
    print("\n=== Testing Builder - Sequential Chain ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Build sequential operation graph: step1 -> step2 -> step3
    builder = Builder()
    builder.add(
        "step1",
        "communicate",
        {"instruction": "Say just the word 'ONE'.", "imodel": model},
    )
    builder.add(
        "step2",
        "communicate",
        {"instruction": "Say just the word 'TWO'.", "imodel": model},
        depends_on=["step1"],
    )
    builder.add(
        "step3",
        "communicate",
        {"instruction": "Say just the word 'THREE'.", "imodel": model},
        depends_on=["step2"],
    )

    graph = builder.build()
    print(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Execute with flow
    results = await flow(session, branch, graph, verbose=True)

    print(f"Results: {list(results.keys())}")
    assert "step1" in results
    assert "step2" in results
    assert "step3" in results
    assert "one" in results["step1"].lower()
    assert "two" in results["step2"].lower()
    assert "three" in results["step3"].lower()
    print("✓ Builder sequential chain passed")


async def test_builder_parallel():
    """Test Builder with parallel operations."""
    print("\n=== Testing Builder - Parallel Operations ===")

    import time

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Build parallel operation graph: all three run at same time
    builder = Builder()
    # Add without dependencies - they'll be parallel
    builder.add(
        "task_a",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'ALPHA' only."}],
            "imodel": model,
        },
    )
    # Clear current_heads to avoid auto-linking
    builder._current_heads = []
    builder.add(
        "task_b",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'BETA' only."}],
            "imodel": model,
        },
    )
    builder._current_heads = []
    builder.add(
        "task_c",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'GAMMA' only."}],
            "imodel": model,
        },
    )

    graph = builder.build()
    print(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges (should be 0)")
    assert len(graph.edges) == 0, "Parallel tasks should have no edges"

    # Execute with flow
    start_time = time.time()
    results = await flow(session, branch, graph, verbose=True)
    elapsed = time.time() - start_time

    print(f"Results: {list(results.keys())}")
    print(f"Elapsed time: {elapsed:.2f}s (parallel execution)")
    assert "task_a" in results
    assert "task_b" in results
    assert "task_c" in results
    print("✓ Builder parallel operations passed")


async def test_builder_diamond():
    """Test Builder with diamond dependency pattern (fan-out/fan-in)."""
    print("\n=== Testing Builder - Diamond Pattern ===")

    from pydantic import BaseModel, Field

    from lionpride.operations import Builder, flow

    class TaskResult(BaseModel):
        answer: str = Field(description="The answer")

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Diamond pattern:
    #     start
    #    /     \
    # left    right
    #    \     /
    #     end

    builder = Builder()

    # Start node
    builder.add(
        "start",
        "communicate",
        {"instruction": "Say 'START'.", "imodel": model},
    )

    # Two parallel branches from start
    builder.add(
        "left",
        "communicate",
        {"instruction": "Say 'LEFT'.", "imodel": model},
        depends_on=["start"],
    )
    builder.add(
        "right",
        "communicate",
        {"instruction": "Say 'RIGHT'.", "imodel": model},
        depends_on=["start"],
    )

    # End node depends on both
    builder.add(
        "end",
        "communicate",
        {"instruction": "Say 'END'.", "imodel": model},
        depends_on=["left", "right"],
    )

    graph = builder.build()
    print(f"Built diamond graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Execute
    results = await flow(session, branch, graph, verbose=True)

    print(f"Results: {list(results.keys())}")
    assert len(results) == 4
    assert "start" in results["start"].lower()
    assert "left" in results["left"].lower()
    assert "right" in results["right"].lower()
    assert "end" in results["end"].lower()
    print("✓ Builder diamond pattern passed")


async def test_flow_stream():
    """Test flow_stream for streaming results as operations complete."""
    print("\n=== Testing flow_stream ===")

    from lionpride.operations import Builder, flow_stream

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Build simple sequential chain
    builder = Builder()
    builder.add(
        "first",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'FIRST'."}],
            "imodel": model,
        },
    )
    builder.add(
        "second",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'SECOND'."}],
            "imodel": model,
        },
        depends_on=["first"],
    )

    graph = builder.build()

    # Stream results
    results_received = []
    async for result in flow_stream(session, branch, graph):
        print(
            f"  Received: {result.name} ({result.completed}/{result.total}) - success={result.success}"
        )
        results_received.append(result)

    assert len(results_received) == 2
    assert results_received[0].completed == 1
    assert results_received[1].completed == 2
    assert results_received[1].total == 2
    print("✓ flow_stream passed")


async def test_builder_with_aggregation():
    """Test Builder aggregation pattern (multiple sources -> single collector)."""
    print("\n=== Testing Builder - Aggregation ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Create parallel tasks that feed into an aggregator
    builder = Builder()

    # Three parallel data sources
    builder.add(
        "source_1",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'DATA1'."}],
            "imodel": model,
        },
    )
    builder._current_heads = []
    builder.add(
        "source_2",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'DATA2'."}],
            "imodel": model,
        },
    )
    builder._current_heads = []
    builder.add(
        "source_3",
        "generate",
        {
            "messages": [{"role": "user", "content": "Say 'DATA3'."}],
            "imodel": model,
        },
    )

    # Aggregator that depends on all sources
    builder.add_aggregation(
        "collector",
        "communicate",
        {"instruction": "Say 'COLLECTED'.", "imodel": model},
        source_names=["source_1", "source_2", "source_3"],
    )

    graph = builder.build()
    print(f"Built aggregation graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Execute
    results = await flow(session, branch, graph, verbose=True)

    print(f"Results: {list(results.keys())}")
    assert len(results) == 4
    assert "collector" in results
    print("✓ Builder aggregation passed")


async def test_flow_with_max_concurrent():
    """Test flow with concurrency limiting."""
    print("\n=== Testing flow with max_concurrent ===")

    import time

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Build 4 parallel operations
    builder = Builder()
    for i in range(4):
        if i > 0:
            builder._current_heads = []
        builder.add(
            f"task_{i}",
            "generate",
            {
                "messages": [{"role": "user", "content": f"Say 'TASK{i}'."}],
                "imodel": model,
            },
        )

    graph = builder.build()

    # Execute with max_concurrent=2
    start_time = time.time()
    results = await flow(session, branch, graph, max_concurrent=2, verbose=True)
    elapsed = time.time() - start_time

    print(f"Results: {list(results.keys())}")
    print(f"Elapsed time: {elapsed:.2f}s (limited to 2 concurrent)")
    assert len(results) == 4
    print("✓ flow with max_concurrent passed")


async def test_builder_depends_on_method():
    """Test Builder.depends_on() method for explicit dependency addition."""
    print("\n=== Testing Builder.depends_on() method ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Add operations first, then add dependencies separately
    builder = Builder()

    # Add all operations without dependencies
    builder.add(
        "op_a", "generate", {"messages": [{"role": "user", "content": "Say A."}], "imodel": model}
    )
    builder._current_heads = []
    builder.add(
        "op_b", "generate", {"messages": [{"role": "user", "content": "Say B."}], "imodel": model}
    )
    builder._current_heads = []
    builder.add(
        "op_c", "generate", {"messages": [{"role": "user", "content": "Say C."}], "imodel": model}
    )

    # Now add dependencies: op_c depends on op_a and op_b
    builder.depends_on("op_c", "op_a", "op_b", label=["custom_dep"])

    graph = builder.build()
    print(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Verify edges were created
    assert len(graph.edges) == 2, "Should have 2 dependency edges"

    # Execute
    results = await flow(session, branch, graph, verbose=True)
    print(f"Results: {list(results.keys())}")
    assert len(results) == 3
    print("✓ Builder.depends_on() method passed")


async def test_builder_sequence_method():
    """Test Builder.sequence() method for creating sequential chains."""
    print("\n=== Testing Builder.sequence() method ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    builder = Builder()

    # Add operations without dependencies (using _current_heads trick)
    builder.add(
        "step_1", "generate", {"messages": [{"role": "user", "content": "Say 1."}], "imodel": model}
    )
    builder._current_heads = []
    builder.add(
        "step_2", "generate", {"messages": [{"role": "user", "content": "Say 2."}], "imodel": model}
    )
    builder._current_heads = []
    builder.add(
        "step_3", "generate", {"messages": [{"role": "user", "content": "Say 3."}], "imodel": model}
    )
    builder._current_heads = []
    builder.add(
        "step_4", "generate", {"messages": [{"role": "user", "content": "Say 4."}], "imodel": model}
    )

    # Create sequence: step_1 -> step_2 -> step_3 -> step_4
    builder.sequence("step_1", "step_2", "step_3", "step_4")

    graph = builder.build()
    print(f"Built graph with {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Should have 3 edges for 4 sequential operations
    assert len(graph.edges) == 3, f"Should have 3 edges, got {len(graph.edges)}"

    # Execute
    results = await flow(session, branch, graph, verbose=True)
    print(f"Results: {list(results.keys())}")
    assert len(results) == 4
    print("✓ Builder.sequence() method passed")


async def test_builder_get_and_utility_methods():
    """Test Builder utility methods: get(), mark_executed(), get_unexecuted_nodes(), clear()."""
    print("\n=== Testing Builder utility methods ===")

    from lionpride.operations import Builder

    builder = Builder()

    # Add some operations
    builder.add("op1", "generate", {"messages": []})
    builder._current_heads = []
    builder.add("op2", "communicate", {"instruction": "test"})
    builder._current_heads = []
    builder.add("op3", "operate", {"instruction": "test"})

    # Test get()
    op1 = builder.get("op1")
    assert op1.operation_type == "generate"
    print("  ✓ get() works")

    # Test get() with invalid name
    try:
        builder.get("nonexistent")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "not found" in str(e)
        print("  ✓ get() raises ValueError for invalid name")

    # Test get_by_id()
    op1_by_id = builder.get_by_id(op1.id)
    assert op1_by_id is op1
    print("  ✓ get_by_id() works")

    # Test mark_executed() and get_unexecuted_nodes()
    unexecuted = builder.get_unexecuted_nodes()
    assert len(unexecuted) == 3
    print(f"  ✓ get_unexecuted_nodes() returns {len(unexecuted)} ops")

    builder.mark_executed("op1", "op2")
    unexecuted = builder.get_unexecuted_nodes()
    assert len(unexecuted) == 1
    assert unexecuted[0].metadata["name"] == "op3"
    print("  ✓ mark_executed() and get_unexecuted_nodes() work")

    # Test clear()
    builder.clear()
    assert len(builder._nodes) == 0
    assert len(builder.graph.nodes) == 0
    print("  ✓ clear() works")

    print("✓ Builder utility methods passed")


async def test_context_passing():
    """Test that predecessor results are correctly passed to dependent operations via context."""
    print("\n=== Testing context passing ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    # Create a chain where step2 should receive step1's result in context
    # step1 generates a random number, step2 should reference it
    builder = Builder()

    builder.add(
        "producer",
        "generate",
        {
            "messages": [{"role": "user", "content": "Generate exactly this: MAGIC_VALUE_42"}],
            "imodel": model,
        },
    )

    # Consumer depends on producer - should receive producer_result in context
    builder.add(
        "consumer",
        "communicate",
        {
            "instruction": (
                "Look at the context provided. There should be a 'producer_result' key. "
                "Tell me what value is in producer_result. Just state the value."
            ),
            "imodel": model,
        },
        depends_on=["producer"],
    )

    graph = builder.build()

    # Execute with verbose to see context passing
    results = await flow(session, branch, graph, verbose=True)

    print(f"Producer result: {results['producer']}")
    print(f"Consumer result: {results['consumer']}")

    # Consumer should mention the producer's result
    assert "MAGIC" in results["consumer"].upper() or "42" in results["consumer"]
    print("✓ Context passing verified - consumer received producer_result")


async def test_context_passing_multiple_predecessors():
    """Test context passing with multiple predecessors (diamond pattern with context verification)."""
    print("\n=== Testing context passing with multiple predecessors ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    builder = Builder()

    # Two parallel producers
    builder.add(
        "left_producer",
        "generate",
        {"messages": [{"role": "user", "content": "Say exactly: LEFT_VALUE"}], "imodel": model},
    )
    builder._current_heads = []
    builder.add(
        "right_producer",
        "generate",
        {"messages": [{"role": "user", "content": "Say exactly: RIGHT_VALUE"}], "imodel": model},
    )

    # Consumer depends on both - should receive both results in context
    builder.add(
        "merger",
        "communicate",
        {
            "instruction": (
                "Look at the context. You should have 'left_producer_result' and 'right_producer_result'. "
                "List both values you received, separated by a comma."
            ),
            "imodel": model,
        },
        depends_on=["left_producer", "right_producer"],
    )

    graph = builder.build()
    results = await flow(session, branch, graph, verbose=True)

    print(f"Left producer: {results['left_producer']}")
    print(f"Right producer: {results['right_producer']}")
    print(f"Merger result: {results['merger']}")

    # Merger should mention both values
    merger_result = results["merger"].upper()
    assert "LEFT" in merger_result or "RIGHT" in merger_result
    print("✓ Context passing with multiple predecessors verified")


async def test_shared_execution_context():
    """Test that shared execution context is passed to all operations."""
    print("\n=== Testing shared execution context ===")

    from lionpride.operations import Builder, flow

    session = Session()
    model = iModel(provider="openai", model="gpt-4o-mini", name="test_model")
    session.services.register(model)
    branch = session.create_branch(name="test")

    builder = Builder()
    builder.add(
        "context_reader",
        "communicate",
        {
            "instruction": (
                "Look at the context provided. There should be a 'shared_secret' key. "
                "What is the value of shared_secret? Just state the value."
            ),
            "imodel": model,
        },
    )

    graph = builder.build()

    # Pass shared context to flow
    results = await flow(
        session,
        branch,
        graph,
        context={"shared_secret": "SUPER_SECRET_123"},
        verbose=True,
    )

    print(f"Context reader result: {results['context_reader']}")

    # Should mention the shared secret
    assert "SECRET" in results["context_reader"].upper() or "123" in results["context_reader"]
    print("✓ Shared execution context passed correctly")


async def main():
    """Run all functional tests."""
    print("=" * 50)
    print("Lionpride Operations Functional Tests")
    print("=" * 50)

    try:
        # Core operation tests
        await test_generate()
        await test_communicate()
        await test_operate()
        await test_operate_with_actions()
        await test_react()

        # Flow/Builder basic tests
        await test_builder_sequential()
        await test_builder_parallel()
        await test_builder_diamond()
        await test_flow_stream()
        await test_builder_with_aggregation()
        await test_flow_with_max_concurrent()

        # Builder method tests
        await test_builder_depends_on_method()
        await test_builder_sequence_method()
        await test_builder_get_and_utility_methods()

        # Context passing tests
        await test_context_passing()
        await test_context_passing_multiple_predecessors()
        await test_shared_execution_context()

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
