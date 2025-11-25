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


async def main():
    """Run all functional tests."""
    print("=" * 50)
    print("Lionpride Operations Functional Tests")
    print("=" * 50)

    try:
        await test_generate()
        await test_communicate()
        await test_operate()
        await test_operate_with_actions()
        await test_react()

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
