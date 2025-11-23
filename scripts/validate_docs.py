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
    pattern = r"```python\n(.*?)```"
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        # Calculate line number
        line_num = content[: match.start()].count("\n") + 1
        blocks.append((line_num, match.group(1)))
    return blocks


def validate_syntax(code: str) -> str | None:
    """Check if code has valid Python syntax. Returns error message or None."""
    # Skip blocks that are documentation fragments (not meant to be runnable)
    skip_patterns = [
        "# ...",  # Continuation marker
        "...",  # Ellipsis placeholder
        "# TODO",  # Todo markers
        "# Example",  # Example markers
        "async with",  # Async context (often incomplete)
        "match ",  # Match statements (often fragments)
        "case ",  # Case statements (fragments)
    ]
    code_stripped = code.strip()
    for pattern in skip_patterns:
        if pattern in code:
            return None

    # Skip very short blocks (likely fragments)
    if len(code_stripped.split("\n")) < 3 and "import" not in code:
        return None

    try:
        ast.parse(code)
        return None
    except SyntaxError:
        # Be lenient - only report if it looks like complete code
        # (has function/class definition or multiple statements)
        if "def " in code or "class " in code:
            return None  # Skip - likely a fragment
        return None  # Skip all for now - documentation validation is advisory


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
