from typing import Any
from urllib.parse import urlparse

import orjson
from pydantic import BaseModel

from lionpride.libs.schema_handlers import (
    breakdown_pydantic_annotation,
    minimal_yaml,
    typescript_schema,
)


def _validate_image_url(url: str) -> None:
    """Validate image URL to prevent security vulnerabilities.

    Security checks:
        - Reject file:// URLs (local file access)
        - Reject javascript: URLs (XSS attacks)
        - Reject data:// URLs (DoS via large embedded images)
        - Only allow http:// and https:// schemes
        - Validate URL format

    Args:
        url: URL to validate

    Raises:
        ValueError: If URL is invalid or uses disallowed scheme
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"Image URL must be non-empty string, got: {type(url).__name__}")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Malformed image URL '{url}': {e}") from e

    # Only allow http and https schemes
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Image URL must use http:// or https:// scheme, got: {parsed.scheme}://"
            f"\nRejected URL: {url}"
            f"\nReason: Disallowed schemes (file://, javascript://, data://) pose "
            f"security risks (local file access, XSS, DoS)"
        )

    # Ensure netloc (domain) is present for http/https
    if not parsed.netloc:
        raise ValueError(f"Image URL missing domain: {url}")


def _format_task(task_data: dict) -> str:
    text = "## Task\n"
    text += minimal_yaml(task_data)
    return text


def _format_model_schema(request_model: type[BaseModel]) -> str:
    model_schema = request_model.model_json_schema()
    model_schema = ""
    if defs := model_schema.get("$defs"):
        for def_name, def_schema in defs.items():
            if def_ts := typescript_schema(def_schema):
                model_schema += f"\n{def_name}:\n" + "\n".join(
                    f"  {line}" for line in def_ts.split("\n")
                )
    return model_schema


def _format_json_response_structure(request_model: type[BaseModel]) -> str:
    json_schema = "\n\n## ResponseFormat\n"
    json_schema += "```json\n"
    json_schema += orjson.dumps(breakdown_pydantic_annotation(request_model)).decode()
    json_schema += "\n```\nMUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT. Return ONLY valid JSON without markdown code blocks.\n"
    return json_schema


def _create_lndl_format_section(self, model_schema: dict[str, Any]) -> str:
    """Create LNDL format response section.

    Explains LNDL format without being prescriptive about aliases.
    Model can choose any aliases - fuzzy matching will handle it.
    """
    model_name = model_schema.get("title", "Response")
    properties = model_schema.get("properties", {})
    field_names = list(properties.keys())
    spec_name = model_name.lower()

    # Build field descriptions with types
    field_descriptions = []
    for field in field_names:
        field_schema = properties.get(field, {})
        field_type = field_schema.get("type", "string")
        if field_type == "array":
            field_type = "array"
        elif field_type == "integer":
            field_type = "integer"
        elif field_type == "number":
            field_type = "number"
        elif field_type == "boolean":
            field_type = "boolean"
        else:
            field_type = "string"
        field_descriptions.append(f"{field}: {field_type}")

    return (
        f"ResponseFormat: **USE LNDL FORMAT**\n\n"
        f"Model: {model_name}\n"
        f"Fields: {', '.join(field_descriptions)}\n\n"
        f"LNDL Format:\n"
        f"1. Declare each field with: <lvar {model_name}.fieldname your_alias>value</lvar>\n"
        f"2. Choose any alias you want (short names work well)\n"
        f"3. Reference your aliases in OUT block: OUT{{{spec_name}: [alias1, alias2, ...]}}\n\n"
        f"Example pattern (you choose the aliases):\n"
        f"<lvar {model_name}.{field_names[0]} a>...</lvar>\n"
        f"<lvar {model_name}.{field_names[1] if len(field_names) > 1 else 'field2'} b>...</lvar>\n"
        f"OUT{{{spec_name}: [a, b, ...]}}\n\n"
        f"Notes:\n"
        f'- Arrays: use JSON syntax like ["item1", "item2"]\n'
        f"- Numbers: use plain numbers like 42 or 3.14\n"
        f"- Booleans: use true or false"
    )


def _create_example_from_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
    """Create simple example dict from JSON schema (lionagi pattern)."""
    example = {}
    properties = schema.get("properties", {})

    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type")
        if field_type == "string":
            example[field_name] = "..."
        elif field_type == "number" or field_type == "integer":
            example[field_name] = 0
        elif field_type == "boolean":
            example[field_name] = False
        elif field_type == "array":
            items_schema = field_schema.get("items", {})
            items_type = items_schema.get("type")
            if items_type == "string":
                example[field_name] = ["..."]
            elif items_type == "object":
                example[field_name] = [self._create_example_from_schema(items_schema)]
            else:
                example[field_name] = []
        elif field_type == "object":
            example[field_name] = self._create_example_from_schema(field_schema)
        else:
            example[field_name] = None

    return example
