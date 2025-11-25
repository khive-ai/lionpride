#!/usr/bin/env python3
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Functional test for multi-level operation graph with IPU.

Pattern: 1 → 3 → 6 → 2 → 1 (multi-level execution with aggregation)

Tests:
- Complex dependency graph (13 nodes)
- Multiple levels of operations
- Parallel execution at each level
- Aggregation from multiple sources
- IPU-validated execution

Run with:
    uv run python scripts/functional_test/test_operation_graph.py
"""

import asyncio
import os
import time

from dotenv import load_dotenv


async def test_multilevel_graph():
    """Test multi-level execution pattern (1→3→6→2→1) using IPU pattern."""
    from lionpride.ipu import IPU
    from lionpride.operations import Builder, flow
    from lionpride.services import iModel
    from lionpride.session import Message, Session, SystemContent

    # Load environment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(project_root, ".env")

    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    print("Testing Multi-Level Execution (1->3->6->2->1)")
    print("=" * 60)

    # Setup IPU + Session pattern
    ipu = IPU()
    session = Session()

    # Register model
    model = iModel(provider="openai")
    session.services.register(model, update=True)

    # Register session with IPU
    ipu.register_session(session)

    # Create main branch with system message
    branch = session.create_branch(
        name="main",
        system=Message(
            content=SystemContent(
                system_message="You are a helpful assistant. Keep responses concise (1-2 sentences)."
            )
        ),
    )
    print(f"\nCreated branch: {branch.name}")

    # Show registered operations
    print(f"Registered operations: {list(session.operations.list_names())}")

    # BUILD THE COMPLETE GRAPH STRUCTURE
    # Using 'communicate' operation which accepts instruction + imodel
    builder = Builder()

    # Common params for all operations
    def make_params(instruction: str) -> dict:
        return {
            "instruction": instruction,
            "imodel": model,
        }

    # Level 1: Single root (1 node)
    builder.add(
        "analyze_topics",
        "communicate",
        make_params("Identify 3 major areas of computer science in a brief list"),
    )

    # Level 2: Three branches (3 nodes) - all depend on root
    builder.add(
        "explore_algorithms",
        "communicate",
        make_params("Briefly describe algorithms and data structures"),
        depends_on=["analyze_topics"],
    )

    builder.add(
        "explore_systems",
        "communicate",
        make_params("Briefly describe computer systems"),
        depends_on=["analyze_topics"],
    )

    builder.add(
        "explore_ai",
        "communicate",
        make_params("Briefly describe artificial intelligence"),
        depends_on=["analyze_topics"],
    )

    # Level 3: Each branch splits into 2 (6 nodes total)
    # Branch 1 expansions
    builder.add(
        "algo_theory",
        "communicate",
        make_params("Explain computational complexity in one sentence"),
        depends_on=["explore_algorithms"],
    )

    builder.add(
        "algo_practice",
        "communicate",
        make_params("Name one practical sorting algorithm"),
        depends_on=["explore_algorithms"],
    )

    # Branch 2 expansions
    builder.add(
        "operating_systems",
        "communicate",
        make_params("Explain what an OS does in one sentence"),
        depends_on=["explore_systems"],
    )

    builder.add(
        "networks",
        "communicate",
        make_params("Explain what TCP/IP is in one sentence"),
        depends_on=["explore_systems"],
    )

    # Branch 3 expansions
    builder.add(
        "machine_learning",
        "communicate",
        make_params("Explain what ML is in one sentence"),
        depends_on=["explore_ai"],
    )

    builder.add(
        "nlp",
        "communicate",
        make_params("Explain what NLP is in one sentence"),
        depends_on=["explore_ai"],
    )

    # Level 4: Two aggregation points (2 nodes)
    builder.add_aggregation(
        "theory_synthesis",
        "communicate",
        make_params("Summarize the theoretical foundations discussed"),
        source_names=["algo_theory", "operating_systems", "machine_learning"],
    )

    builder.add_aggregation(
        "practice_synthesis",
        "communicate",
        make_params("Summarize the practical applications discussed"),
        source_names=["algo_practice", "networks", "nlp"],
    )

    # Level 5: Final synthesis (1 node)
    builder.add_aggregation(
        "final_synthesis",
        "communicate",
        make_params("Create a one-paragraph overview of computer science"),
        source_names=["theory_synthesis", "practice_synthesis"],
    )

    # Build and validate graph
    graph = builder.build()

    print("\nGraph Structure:")
    print(f"  Total nodes: {len(list(graph.nodes))}")
    print(f"  Total edges: {len(graph.edges)}")
    print("  DAG validated: OK")

    # Show builder state
    print(f"\nBuilder State: {builder}")

    # Execute the graph via IPU
    print("\nExecuting graph with automatic parallelization...")
    start_time = time.time()

    try:
        results = await flow(
            session=session,
            branch=branch,
            graph=graph,
            ipu=ipu,
            max_concurrent=10,
            stop_on_error=False,
            verbose=True,
        )
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    elapsed = time.time() - start_time
    print(f"\nExecution completed in {elapsed:.2f}s!")
    print(f"  Operations executed: {len(results)}")

    # Show level statistics
    print("\nLevel Statistics:")
    print("  Level 1 (root): 1 node")
    print("  Level 2 (branches): 3 nodes")
    print("  Level 3 (expansions): 6 nodes")
    print("  Level 4 (synthesis): 2 nodes")
    print("  Level 5 (final): 1 node")
    print("  Total: 13 nodes")

    # Show sample results
    print("\nSample Results:")
    for name in ["analyze_topics", "explore_algorithms", "algo_theory"]:
        if name in results:
            result_str = str(results[name])[:150]
            print(f"\n[{name}]:")
            print(f"  {result_str}...")

    # Show final synthesis
    if "final_synthesis" in results:
        print("\nFINAL SYNTHESIS:")
        print("=" * 60)
        print(results["final_synthesis"])
        print("=" * 60)

    # Verify all operations completed
    expected_ops = [
        "analyze_topics",
        "explore_algorithms",
        "explore_systems",
        "explore_ai",
        "algo_theory",
        "algo_practice",
        "operating_systems",
        "networks",
        "machine_learning",
        "nlp",
        "theory_synthesis",
        "practice_synthesis",
        "final_synthesis",
    ]

    missing = [op for op in expected_ops if op not in results]
    if missing:
        print(f"\nMissing operations: {missing}")
        return False
    else:
        print("\nAll 13 operations completed successfully!")

    print("\nMULTI-LEVEL GRAPH TEST PASSED!")
    return True


if __name__ == "__main__":
    import sys

    success = asyncio.run(test_multilevel_graph())
    sys.exit(0 if success else 1)
