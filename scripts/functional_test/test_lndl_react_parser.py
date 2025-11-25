#!/usr/bin/env python3
"""Test LNDL React Extension - Parser.

Validates that the parser correctly handles React constructs.
"""

from lionpride.lndl.ast import (
    Branch,
    Condition,
    Final,
    Loop,
    ReactAction,
    ReactBlock,
    Thought,
)
from lionpride.lndl.lexer import Lexer
from lionpride.lndl.parser import Parser


def test_simple_react_block():
    """Test parsing a simple React block with thought and action."""
    text = """
<react search_task>
  <thought t1>I need to search for information</thought>
  <action search>search_tool(query="AI safety")</action>
  <final>OUT{result: [obs_search], reasoning: [t1]}</final>
</react>
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    # Should have one react block
    assert program.react_blocks is not None, "Expected react_blocks to be set"
    assert len(program.react_blocks) == 1, (
        f"Expected 1 react block, got {len(program.react_blocks)}"
    )

    react_block = program.react_blocks[0]
    assert isinstance(react_block, ReactBlock)
    assert react_block.name == "search_task"

    # Check body
    body = react_block.body
    print(f"Body has {len(body)} steps")

    # First step: Thought
    thought = body[0]
    assert isinstance(thought, Thought), f"Expected Thought, got {type(thought)}"
    assert thought.alias == "t1"
    assert "search for information" in thought.content

    # Second step: Action
    action = body[1]
    assert isinstance(action, ReactAction), f"Expected ReactAction, got {type(action)}"
    assert action.alias == "search"
    assert action.function == "search_tool"
    assert action.arguments.get("query") == "AI safety"

    # Third step: Final
    final = body[2]
    assert isinstance(final, Final), f"Expected Final, got {type(final)}"
    assert final.out_block.fields.get("result") == ["obs_search"]
    assert final.out_block.fields.get("reasoning") == ["t1"]

    print("✓ Simple React block parsed correctly")
    return True


def test_react_with_branch():
    """Test parsing React block with if/else branching."""
    text = """
<react fact_check>
  <thought>Let me verify this claim</thought>
  <action wiki>search_tool(source="wikipedia", query="test")</action>

  <if {obs_wiki.found == true}>
    <thought>Found Wikipedia data</thought>
    <final>OUT{verified: true, source: [obs_wiki]}</final>
  </if>
  <else>
    <thought>No Wikipedia results</thought>
    <final>OUT{verified: false, reason: "not found"}</final>
  </else>
</react>
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    assert program.react_blocks is not None
    react_block = program.react_blocks[0]
    assert react_block.name == "fact_check"

    body = react_block.body
    print(f"Branch test body has {len(body)} steps")

    # Find the branch
    branch = None
    for step in body:
        if isinstance(step, Branch):
            branch = step
            break

    assert branch is not None, "Expected to find a Branch"

    # Check condition
    assert isinstance(branch.condition, Condition)
    assert branch.condition.left == "obs_wiki.found"
    assert branch.condition.operator == "=="
    assert branch.condition.right is True

    # Check if body
    assert len(branch.if_body) >= 1, "Expected at least 1 step in if_body"
    assert isinstance(branch.if_body[0], Thought)

    # Check else body
    assert branch.else_body is not None
    assert len(branch.else_body) >= 1, "Expected at least 1 step in else_body"
    assert isinstance(branch.else_body[0], Thought)

    print("✓ React block with branching parsed correctly")
    return True


def test_react_with_loop():
    """Test parsing React block with loop construct."""
    text = """
<react iterative_search>
  <thought>Starting iterative search</thought>

  <loop until {obs_check.complete == true}>
    <action search>search_tool(query="test")</action>
    <action check>check_completeness()</action>
  </loop>

  <final>OUT{results: [obs_search]}</final>
</react>
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    assert program.react_blocks is not None
    react_block = program.react_blocks[0]
    assert react_block.name == "iterative_search"

    body = react_block.body
    print(f"Loop test body has {len(body)} steps")

    # Find the loop
    loop = None
    for step in body:
        if isinstance(step, Loop):
            loop = step
            break

    assert loop is not None, "Expected to find a Loop"

    # Check loop properties
    assert loop.loop_type == "until"
    assert isinstance(loop.condition, Condition)
    assert loop.condition.left == "obs_check.complete"
    assert loop.condition.operator == "=="
    assert loop.condition.right is True

    # Check loop body
    assert len(loop.body) >= 2, f"Expected at least 2 steps in loop body, got {len(loop.body)}"

    print("✓ React block with loop parsed correctly")
    return True


def test_react_times_loop():
    """Test parsing React block with 'times' loop."""
    text = """
<react repeat_task>
  <loop times 3>
    <thought>Iteration</thought>
    <action do>do_something()</action>
  </loop>
  <final>OUT{done: true}</final>
</react>
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    assert program.react_blocks is not None
    react_block = program.react_blocks[0]

    # Find the loop
    loop = None
    for step in react_block.body:
        if isinstance(step, Loop):
            loop = step
            break

    assert loop is not None, "Expected to find a Loop"
    assert loop.loop_type == "times"
    assert loop.condition == 3  # Integer, not Condition

    print("✓ React block with times loop parsed correctly")
    return True


def test_thought_without_alias():
    """Test parsing thought without alias."""
    text = """
<react test>
  <thought>Anonymous thought content</thought>
  <final>OUT{done: true}</final>
</react>
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    assert program.react_blocks is not None
    react_block = program.react_blocks[0]

    thought = react_block.body[0]
    assert isinstance(thought, Thought)
    assert thought.alias is None
    assert "Anonymous thought" in thought.content

    print("✓ Thought without alias parsed correctly")
    return True


def test_react_without_name():
    """Test parsing React block without name."""
    text = """
<react>
  <thought>Test</thought>
  <final>OUT{done: true}</final>
</react>
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    assert program.react_blocks is not None
    react_block = program.react_blocks[0]
    assert react_block.name is None

    print("✓ React block without name parsed correctly")
    return True


def test_backward_compatibility():
    """Test that existing LNDL (non-React) still works."""
    text = """
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar reasoning>Some intermediate reasoning</lvar>
<lact Report.summary s>generate_summary(prompt="test")</lact>
OUT{title: [t], summary: [s], reasoning: [reasoning]}
"""
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, source_text=text)
    program = parser.parse()

    # Should have lvars and lacts, no react blocks
    assert len(program.lvars) == 2
    assert len(program.lacts) == 1
    assert program.out_block is not None
    assert program.react_blocks is None

    print("✓ Backward compatibility maintained")
    return True


def main():
    """Run all parser tests."""
    print("=" * 60)
    print("LNDL React Extension - Parser Tests")
    print("=" * 60)
    print()

    tests = [
        ("Simple React block", test_simple_react_block),
        ("React with branch", test_react_with_branch),
        ("React with until loop", test_react_with_loop),
        ("React with times loop", test_react_times_loop),
        ("Thought without alias", test_thought_without_alias),
        ("React without name", test_react_without_name),
        ("Backward compatibility", test_backward_compatibility),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} returned False")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed with error: {e}")
            import traceback

            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
