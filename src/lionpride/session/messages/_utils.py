import textwrap
from urllib.parse import urlparse

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


def _clean_multiline_strings(data: dict) -> dict:
    """Clean multiline strings for YAML block scalars (| not |-)."""
    cleaned = {}
    for k, v in data.items():
        if isinstance(v, str) and "\n" in v:
            # Strip trailing whitespace from each line, ensure ends with newline for "|"
            lines = "\n".join(line.rstrip() for line in v.split("\n"))
            cleaned[k] = lines if lines.endswith("\n") else lines + "\n"
        elif isinstance(v, list):
            cleaned[k] = [
                (_clean_multiline(item) if isinstance(item, str) and "\n" in item else item)
                for item in v
            ]
        elif isinstance(v, dict):
            cleaned[k] = _clean_multiline_strings(v)
        else:
            cleaned[k] = v
    return cleaned


def _clean_multiline(s: str) -> str:
    """Clean a multiline string: strip line trailing whitespace, ensure final newline."""
    lines = "\n".join(line.rstrip() for line in s.split("\n"))
    return lines if lines.endswith("\n") else lines + "\n"


def _format_task(task_data: dict) -> str:
    text = "## Task\n"
    text += minimal_yaml(_clean_multiline_strings(task_data))
    return text


def _format_model_schema(request_model: type[BaseModel]) -> str:
    model_schema = request_model.model_json_schema()
    schema_text = ""
    if defs := model_schema.get("$defs"):
        for def_name, def_schema in defs.items():
            if def_ts := typescript_schema(def_schema):
                schema_text += f"\n{def_name}:\n" + textwrap.indent(def_ts, "  ")
    return schema_text


def _format_json_response_structure(request_model: type[BaseModel]) -> str:
    """Format response structure with Python types (unquoted)."""
    schema = breakdown_pydantic_annotation(request_model)
    json_schema = "\n\n## ResponseFormat\n"
    json_schema += "```json\n"
    json_schema += _format_schema_pretty(schema, indent=0)
    json_schema += "\n```\nMUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT. Return ONLY valid JSON without markdown code blocks.\n"
    return json_schema


def _format_schema_pretty(schema: dict, indent: int = 0) -> str:
    """Format schema dict with unquoted Python type values."""
    lines = ["{"]
    items = list(schema.items())
    for i, (key, value) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(value, dict):
            nested = _format_schema_pretty(value, indent + 4)
            lines.append(f'{" " * (indent + 4)}"{key}": {nested}{comma}')
        elif isinstance(value, list) and value:
            if isinstance(value[0], dict):
                nested = _format_schema_pretty(value[0], indent + 4)
                lines.append(f'{" " * (indent + 4)}"{key}": [{nested}]{comma}')
            else:
                lines.append(f'{" " * (indent + 4)}"{key}": [{value[0]}]{comma}')
        else:
            lines.append(f'{" " * (indent + 4)}"{key}": {value}{comma}')
    lines.append(f"{' ' * indent}}}")
    return "\n".join(lines)
