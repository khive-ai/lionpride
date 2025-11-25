#!/usr/bin/env python
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Functional test for Builder and Flow patterns with IPU.

Tests:
- OperationGraphBuilder fluent API
- Sequential and parallel operation execution
- Context inheritance between operations
- Aggregation operations
- DependencyAwareExecutor with concurrency control
- IPU-based execution

Run with:
    uv run python scripts/functional_test/test_builder_flow.py
"""

import asyncio
import os
import time

from dotenv import load_dotenv


async def test_sequential_flow():
    """Test sequential operations with context inheritance using IPU."""
    from lionpride.ipu import IPU
    from lionpride.operations import Builder, flow
    from lionpride.services import iModel
    from lionpride.session import Message, Session, SystemContent

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    model = iModel(provider="openai")
    session.services.register(model, update=True)
    ipu.register_session(session)

    branch = session.create_branch(
        name="sequential_test",
        system=Message(
            content=SystemContent(
                system_message="You are a helpful assistant. Keep responses brief."
            )
        ),
    )

    builder = Builder()

    # Sequential chain: question -> followup
    builder.add(
        "question",
        "communicate",
        {
            "instruction": "What is the capital of Japan? Just name the city.",
            "imodel": model,
            "model": "gpt-4.1-mini",
        },
    )

    builder.add(
        "followup",
        "communicate",
        {
            "instruction": "What is a famous landmark in that city?",
            "imodel": model,
            "model": "gpt-4.1-mini",
        },
        depends_on=["question"],
    )

    graph = builder.build()
    assert len(list(graph.nodes)) == 2, "Should have 2 operations"
    assert len(graph.edges) == 1, "Should have 1 edge"

    # Execute via flow() with IPU
    results = await flow(session, branch, graph, ipu)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert "question" in results, "Missing 'question' result"
    assert "followup" in results, "Missing 'followup' result"

    print(f"  question: {results['question']}")
    print(f"  followup: {results['followup']}")
    print("  PASSED")
    return True


async def test_parallel_flow():
    """Test parallel operations with max_concurrent using IPU."""
    from lionpride.ipu import IPU
    from lionpride.operations import Builder, flow
    from lionpride.services import iModel
    from lionpride.session import Message, Session, SystemContent

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    model = iModel(provider="openai")
    session.services.register(model, update=True)
    ipu.register_session(session)

    branch = session.create_branch(
        name="parallel_test",
        system=Message(content=SystemContent(system_message="You are a helpful assistant.")),
    )

    builder = Builder()

    # Add first task
    builder.add(
        "task_a",
        "generate",
        {
            "imodel": model,
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Name one fruit"}],
        },
    )

    # Clear current heads for parallel execution
    builder._current_heads = []

    # Add second task (parallel)
    builder.add(
        "task_b",
        "generate",
        {
            "imodel": model,
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Name one vegetable"}],
        },
    )

    graph = builder.build()
    assert len(list(graph.nodes)) == 2, "Should have 2 operations"
    assert len(graph.edges) == 0, "Should have 0 edges (parallel)"

    start = time.time()
    results = await flow(session, branch, graph, ipu, max_concurrent=2)
    elapsed = time.time() - start

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"  task_a: {results['task_a']}")
    print(f"  task_b: {results['task_b']}")
    print(f"  elapsed: {elapsed:.2f}s")
    print("  PASSED")
    return True


async def test_aggregation_flow():
    """Test aggregation with multiple sources using IPU."""
    from lionpride.ipu import IPU
    from lionpride.operations import Builder, flow
    from lionpride.services import iModel
    from lionpride.session import Message, Session, SystemContent

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()
    model = iModel(provider="openai")
    session.services.register(model, update=True)
    ipu.register_session(session)

    branch = session.create_branch(
        name="aggregation_test",
        system=Message(
            content=SystemContent(
                system_message="You are a helpful assistant. Keep responses brief."
            )
        ),
    )

    builder = Builder()

    # Three parallel sources
    builder.add(
        "item1",
        "generate",
        {
            "imodel": model,
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Name one animal"}],
        },
    )

    builder._current_heads = []

    builder.add(
        "item2",
        "generate",
        {
            "imodel": model,
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Name one color"}],
        },
    )

    builder._current_heads = []

    builder.add(
        "item3",
        "generate",
        {
            "imodel": model,
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Name one number"}],
        },
    )

    # Aggregation depends on all three
    builder.add_aggregation(
        "combine",
        "communicate",
        {
            "instruction": "Make a silly sentence combining the items above.",
            "imodel": model,
            "model": "gpt-4.1-mini",
        },
        source_names=["item1", "item2", "item3"],
    )

    graph = builder.build()
    assert len(list(graph.nodes)) == 4, "Should have 4 operations"
    assert len(graph.edges) == 3, "Should have 3 edges (aggregation)"

    results = await flow(session, branch, graph, ipu, max_concurrent=3)

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    assert "combine" in results, "Missing aggregation result"

    print(f"  item1: {results['item1']}")
    print(f"  item2: {results['item2']}")
    print(f"  item3: {results['item3']}")
    print(f"  combine: {str(results['combine'])[:100]}...")
    print("  PASSED")
    return True


async def main():
    """Run all functional tests."""
    # Load environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(project_root, ".env")

    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    print("=" * 60)
    print("Builder/Flow Functional Tests (IPU Pattern)")
    print("=" * 60)

    print("\n1. Sequential Flow Test:")
    await test_sequential_flow()

    print("\n2. Parallel Flow Test:")
    await test_parallel_flow()

    print("\n3. Aggregation Flow Test:")
    await test_aggregation_flow()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
