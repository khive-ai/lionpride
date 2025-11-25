#!/usr/bin/env python3
"""
OUTPUT Token Measurement - Measures response format efficiency.

This measures how many tokens the LLM uses to RESPOND with structured data,
comparing JSON output vs LNDL output format.

The question: For the same data, how many tokens does each output format use?

Usage:
    uv run python scripts/test_output_token_measurement.py
"""

import json
import sys

sys.path.insert(0, "src")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


# =============================================================================
# Test Data - Same semantic content, different output formats
# =============================================================================

# Level 1: Simple (flat structure)
SIMPLE_DATA = {
    "name": "Alice Chen",
    "age": 32,
    "email": "alice@example.com",
    "occupation": "Software Engineer",
    "is_active": True,
}

# Level 2: Medium (single nesting, list)
MEDIUM_DATA = {
    "name": "Bob Smith",
    "age": 28,
    "email": "bob@example.com",
    "phone": "+1-555-0123",
    "occupation": "Data Scientist",
    "department": "Analytics",
    "salary": 125000.00,
    "address": {
        "street": "123 Main St",
        "city": "San Francisco",
        "country": "USA",
        "postal_code": "94102",
    },
    "skills": ["Python", "SQL", "Machine Learning", "Statistics"],
    "is_active": True,
}

# Level 3: Complex (deep nesting, multiple lists)
COMPLEX_DATA = {
    "first_name": "Carol",
    "last_name": "Johnson",
    "age": 35,
    "gender": "female",
    "nationality": "Canadian",
    "contact": {
        "email": "carol@techcorp.com",
        "phone": "+1-555-0199",
        "linkedin": "linkedin.com/in/caroljohnson",
    },
    "address": {
        "street": "456 Oak Avenue",
        "city": "Toronto",
        "country": "Canada",
        "postal_code": "M5V 2H1",
    },
    "current_employment": {
        "company": "TechCorp Inc",
        "position": "VP Engineering",
        "start_date": "2021-03-15",
        "end_date": None,
        "department": "Engineering",
        "manager": "David Lee",
    },
    "previous_employments": [
        {
            "company": "StartupXYZ",
            "position": "Senior Engineer",
            "start_date": "2018-01-10",
            "end_date": "2021-03-01",
            "department": "Backend",
            "manager": "Eve Wilson",
        },
        {
            "company": "BigTech Co",
            "position": "Software Engineer",
            "start_date": "2015-06-01",
            "end_date": "2017-12-15",
            "department": "Platform",
            "manager": "Frank Brown",
        },
    ],
    "education": [
        {
            "institution": "University of Toronto",
            "degree": "Master of Science",
            "field": "Computer Science",
            "year": 2015,
        },
        {
            "institution": "McGill University",
            "degree": "Bachelor of Science",
            "field": "Mathematics",
            "year": 2013,
        },
    ],
    "skills": ["Python", "Go", "Kubernetes", "System Design", "Leadership"],
    "certifications": ["AWS Solutions Architect", "Google Cloud Professional"],
    "languages": ["English", "French", "Mandarin"],
    "is_active": True,
    "created_at": "2021-03-15T09:00:00Z",
    "updated_at": "2024-01-10T14:30:00Z",
    "tags": ["engineering", "leadership", "senior"],
    "notes": "Key hire for engineering expansion",
}

# Level 4: Very Complex (20+ fields with deep nesting)
VERY_COMPLEX_DATA = {
    **COMPLEX_DATA,
    "projects": [
        {
            "name": "Platform Redesign",
            "status": "completed",
            "team_size": 12,
            "budget": 500000,
            "milestones": ["Design", "MVP", "Beta", "Launch"],
        },
        {
            "name": "ML Pipeline",
            "status": "in_progress",
            "team_size": 8,
            "budget": 300000,
            "milestones": ["Research", "Prototype", "Production"],
        },
    ],
    "performance_reviews": [
        {"year": 2023, "rating": 4.8, "feedback": "Exceptional leadership"},
        {"year": 2022, "rating": 4.5, "feedback": "Strong technical growth"},
    ],
    "preferences": {
        "work_style": "hybrid",
        "communication": "async",
        "timezone": "EST",
        "tools": ["Slack", "Linear", "GitHub"],
    },
}


# =============================================================================
# Output Format Generators
# =============================================================================


def to_json(data: dict) -> str:
    """Standard JSON output (what most LLMs produce)."""
    return json.dumps(data, indent=2)


def to_json_compact(data: dict) -> str:
    """Compact JSON (no whitespace)."""
    return json.dumps(data, separators=(",", ":"))


def to_lndl(data: dict, indent: int = 0) -> str:
    """LNDL output format - compact key:value notation.

    Rules:
    - No quotes around keys
    - No quotes around simple string values (unless contains special chars)
    - Lists use [item1, item2] syntax
    - Nested objects use indentation
    - Wrapped in OUT{field:[value]}
    """

    def format_value(v, depth=0):
        if v is None:
            return "null"
        elif isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, str):
            # Quote only if contains special characters
            if any(c in v for c in [",", ":", "[", "]", "{", "}", "\n", '"']):
                return f'"{v}"'
            return v
        elif isinstance(v, list):
            if not v:
                return "[]"
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                # Simple list - inline
                items = [format_value(x) for x in v]
                return f"[{', '.join(items)}]"
            else:
                # Complex list - one per line
                items = [format_value(x, depth + 1) for x in v]
                indent_str = "  " * (depth + 1)
                return (
                    "[\n"
                    + ",\n".join(f"{indent_str}{item}" for item in items)
                    + "\n"
                    + "  " * depth
                    + "]"
                )
        elif isinstance(v, dict):
            if not v:
                return "{}"
            indent_str = "  " * (depth + 1)
            pairs = []
            for k, val in v.items():
                formatted = format_value(val, depth + 1)
                pairs.append(f"{indent_str}{k}:{formatted}")
            return "{\n" + "\n".join(pairs) + "\n" + "  " * depth + "}"
        return str(v)

    # Format as OUT{first_field:[formatted_data]}
    if data:
        first_key = next(iter(data.keys()))
        formatted = format_value(data)
        return f"OUT{{{first_key}:{formatted}}}"
    return "OUT{}"


def to_yaml_like(data: dict, indent: int = 0) -> str:
    """YAML-like output (another compact alternative)."""

    def format_value(v, depth=0):
        indent_str = "  " * depth
        if v is None:
            return "null"
        elif isinstance(v, bool):
            return "true" if v else "false"
        elif isinstance(v, (int, float)):
            return str(v)
        elif isinstance(v, str):
            if "\n" in v or ":" in v:
                return f'"{v}"'
            return v
        elif isinstance(v, list):
            if not v:
                return "[]"
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                items = [format_value(x) for x in v]
                return f"[{', '.join(items)}]"
            else:
                lines = []
                for item in v:
                    lines.append(f"\n{indent_str}- {format_value(item, depth + 1)}")
                return "".join(lines)
        elif isinstance(v, dict):
            if not v:
                return "{}"
            lines = []
            for k, val in v.items():
                formatted = format_value(val, depth + 1)
                if isinstance(val, dict) and val:
                    lines.append(f"\n{indent_str}{k}:{formatted}")
                else:
                    lines.append(f"\n{indent_str}{k}: {formatted}")
            return "".join(lines)
        return str(v)

    result = format_value(data)
    return result.strip()


# =============================================================================
# Main Test
# =============================================================================


def measure_output(name: str, data: dict, fields: int, nesting: str) -> dict:
    """Measure output tokens for different formats."""
    json_output = to_json(data)
    json_compact = to_json_compact(data)
    lndl_output = to_lndl(data)
    yaml_output = to_yaml_like(data)

    json_tokens = count_tokens(json_output)
    compact_tokens = count_tokens(json_compact)
    lndl_tokens = count_tokens(lndl_output)
    yaml_tokens = count_tokens(yaml_output)

    return {
        "name": name,
        "fields": fields,
        "nesting": nesting,
        "json_tokens": json_tokens,
        "json_compact_tokens": compact_tokens,
        "lndl_tokens": lndl_tokens,
        "yaml_tokens": yaml_tokens,
        "json_to_lndl": (1 - lndl_tokens / json_tokens) * 100,
        "compact_to_lndl": (1 - lndl_tokens / compact_tokens) * 100,
        "outputs": {
            "json": json_output,
            "json_compact": json_compact,
            "lndl": lndl_output,
            "yaml": yaml_output,
        },
    }


def main():
    print("=" * 80)
    print("OUTPUT TOKEN MEASUREMENT - Response Format Efficiency")
    print("=" * 80)
    print()
    print("Measuring how many tokens the LLM uses to OUTPUT structured data")
    print("(Same semantic content, different output formats)")
    print()

    try:
        import tiktoken

        print("Using tiktoken for accurate token counting (gpt-4o)")
    except ImportError:
        print("WARNING: tiktoken not installed, using approximation")
    print()

    results = [
        measure_output("Simple", SIMPLE_DATA, 5, "flat"),
        measure_output("Medium", MEDIUM_DATA, 10, "1 level"),
        measure_output("Complex", COMPLEX_DATA, 20, "2 levels"),
        measure_output("Very Complex", VERY_COMPLEX_DATA, 30, "3 levels"),
    ]

    # Print results table
    print("-" * 80)
    print(
        f"{'Complexity':<15} {'Fields':>6} {'JSON':>8} {'Compact':>8} {'LNDL':>8} {'YAML':>8} {'JSON→LNDL':>10}"
    )
    print("-" * 80)

    for r in results:
        reduction = f"{r['json_to_lndl']:.1f}%"
        print(
            f"{r['name']:<15} {r['fields']:>6} {r['json_tokens']:>8} {r['json_compact_tokens']:>8} {r['lndl_tokens']:>8} {r['yaml_tokens']:>8} {reduction:>10}"
        )

    print("-" * 80)
    print()

    # Averages
    avg_json_lndl = sum(r["json_to_lndl"] for r in results) / len(results)
    avg_compact_lndl = sum(r["compact_to_lndl"] for r in results) / len(results)

    print(f"Average reduction (JSON → LNDL): {avg_json_lndl:.1f}%")
    print(f"Average reduction (Compact JSON → LNDL): {avg_compact_lndl:.1f}%")
    print()

    # Show example outputs for Medium complexity
    print("=" * 80)
    print("EXAMPLE OUTPUT COMPARISON (Medium Complexity)")
    print("=" * 80)

    medium = results[1]

    print("\n--- JSON OUTPUT ---")
    print(f"Tokens: {medium['json_tokens']}")
    print("-" * 40)
    print(medium["outputs"]["json"][:800])
    if len(medium["outputs"]["json"]) > 800:
        print("...")

    print("\n--- COMPACT JSON OUTPUT ---")
    print(f"Tokens: {medium['json_compact_tokens']}")
    print("-" * 40)
    print(medium["outputs"]["json_compact"][:500])
    if len(medium["outputs"]["json_compact"]) > 500:
        print("...")

    print("\n--- LNDL OUTPUT ---")
    print(f"Tokens: {medium['lndl_tokens']}")
    print("-" * 40)
    print(medium["outputs"]["lndl"])

    print("\n--- YAML-LIKE OUTPUT ---")
    print(f"Tokens: {medium['yaml_tokens']}")
    print("-" * 40)
    print(medium["outputs"]["yaml"][:500])
    if len(medium["outputs"]["yaml"]) > 500:
        print("...")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"""
OUTPUT Token Efficiency (how many tokens LLM uses to respond):

| Complexity     | JSON | Compact | LNDL | Reduction |
|----------------|------|---------|------|-----------|
| Simple (5)     | {results[0]["json_tokens"]:>4} | {results[0]["json_compact_tokens"]:>7} | {results[0]["lndl_tokens"]:>4} | {results[0]["json_to_lndl"]:.1f}% |
| Medium (10)    | {results[1]["json_tokens"]:>4} | {results[1]["json_compact_tokens"]:>7} | {results[1]["lndl_tokens"]:>4} | {results[1]["json_to_lndl"]:.1f}% |
| Complex (20)   | {results[2]["json_tokens"]:>4} | {results[2]["json_compact_tokens"]:>7} | {results[2]["lndl_tokens"]:>4} | {results[2]["json_to_lndl"]:.1f}% |
| Very Complex   | {results[3]["json_tokens"]:>4} | {results[3]["json_compact_tokens"]:>7} | {results[3]["lndl_tokens"]:>4} | {results[3]["json_to_lndl"]:.1f}% |

LNDL output saves tokens by:
- Removing quotes around keys
- Removing quotes around simple string values
- Using compact nested syntax
- Using OUT{{field:[data]}} wrapper

Average reduction vs formatted JSON: {avg_json_lndl:.1f}%
Average reduction vs compact JSON: {avg_compact_lndl:.1f}%
""")

    return results


if __name__ == "__main__":
    main()
