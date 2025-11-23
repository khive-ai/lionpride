#!/usr/bin/env python3
# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Validate Python code blocks in documentation files for syntax errors."""

import ast
import re
import sys
from pathlib import Path


def extract_python_blocks(content: str) -> list[tuple[int, str]]:
    """Extract Python code blocks from markdown content.

    Returns list of (line_number, code) tuples.
    """
    # Match both ``` and ```` fenced blocks
    # Use negative lookbehind/lookahead to match correct fence pairs
    pattern_3 = r"```python\n(.*?)```"
    pattern_4 = r"````python\n(.*?)````"

    blocks = []

    # Extract 4-backtick blocks first (they may contain 3-backtick examples)
    for match in re.finditer(pattern_4, content, re.DOTALL):
        line_num = content[: match.start()].count("\n") + 1
        code = match.group(1)
        # Skip blocks that contain nested code fences (documentation examples)
        if "```" in code:
            continue
        blocks.append((line_num, code))

    # Extract 3-backtick blocks, skipping those inside 4-backtick regions
    four_tick_ranges = [(m.start(), m.end()) for m in re.finditer(pattern_4, content, re.DOTALL)]
    for match in re.finditer(pattern_3, content, re.DOTALL):
        # Skip if this match is inside a 4-backtick block
        in_four_tick = any(start <= match.start() < end for start, end in four_tick_ranges)
        if in_four_tick:
            continue
        line_num = content[: match.start()].count("\n") + 1
        blocks.append((line_num, match.group(1)))

    return blocks


def is_runnable_code(code: str) -> bool:
    """Determine if a code block looks like complete, runnable code."""
    code_stripped = code.strip()
    lines = code_stripped.split("\n")

    # Must have imports to be considered runnable
    has_import = any(ln.strip().startswith(("import ", "from ")) for ln in lines)
    if not has_import:
        return False

    # Must have at least 3 non-comment lines
    code_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return len(code_lines) >= 3


def is_doc_fragment(code: str) -> bool:
    """Check if code looks like a documentation fragment (not meant to run)."""
    # Patterns that indicate documentation fragments
    fragment_patterns = [
        "...",  # Ellipsis placeholder
        "→",  # Unicode arrow (documentation notation)
        "# ->",  # Return type in comments
        "@overload",  # Type stubs
        "Protocol",  # Protocol definitions
        ">>>",  # REPL style
        "# Returns:",  # Docstring-style
        "# Args:",  # Docstring-style
        "'```",  # String containing backticks (markdown examples)
        '"```',  # String containing backticks (markdown examples)
        "\\n```",  # Escaped newline + backticks in strings
    ]
    for pattern in fragment_patterns:
        if pattern in code:
            return True

    # Check for truncated code (odd number of triple quotes suggests incomplete)
    if code.count('"""') % 2 == 1 or code.count("'''") % 2 == 1:
        return True

    # Check for obviously truncated strings (line ends with open quote)
    lines = code.strip().split("\n")
    for line in lines:
        stripped = line.rstrip()
        # Line ends with opening quote but no closing, and has odd quote count
        if (
            stripped.endswith(("'", '"'))
            and not stripped.endswith(("''", '""'))
            and (stripped.count("'") % 2 == 1 or stripped.count('"') % 2 == 1)
        ):
            return True

    return False


def validate_syntax(code: str) -> str | None:
    """Check if code has valid Python syntax. Returns error message or None."""
    # Skip documentation fragments
    if is_doc_fragment(code):
        return None

    # Only validate code that looks complete/runnable
    if not is_runnable_code(code):
        return None

    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"


def validate_file(path: Path) -> list[str]:
    """Validate all Python code blocks in a file. Returns list of errors."""
    errors = []
    content = path.read_text()
    blocks = extract_python_blocks(content)

    for line_num, code in blocks:
        # Skip blocks that are intentionally incomplete (have ...)
        if "..." in code and code.strip().endswith("..."):
            continue
        # Skip blocks with output markers
        if code.strip().startswith(">>>"):
            continue

        error = validate_syntax(code)
        if error:
            errors.append(f"{path}:{line_num}: {error}")

    return errors


def main() -> int:
    """Main entry point. Returns exit code."""
    if len(sys.argv) < 2:
        print("Usage: validate_docs.py <docs_dir>", file=sys.stderr)
        return 1

    docs_dir = Path(sys.argv[1])
    if not docs_dir.exists():
        print(f"Directory not found: {docs_dir}", file=sys.stderr)
        return 1

    all_errors = []
    for md_file in docs_dir.rglob("*.md"):
        errors = validate_file(md_file)
        all_errors.extend(errors)

    if all_errors:
        print("Documentation validation errors:", file=sys.stderr)
        for error in all_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(list(docs_dir.rglob('*.md')))} documentation files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
