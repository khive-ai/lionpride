# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from lionpride.libs.schema_handlers import minimal_yaml, typescript_schema
from lionpride.ln import now_utc
from lionpride.types import DataClass, MaybeUnset, ModelConfig, Unset

from .base import MessageRole

__all__ = (
    "ActionRequestContent",
    "ActionResponseContent",
    "AssistantResponseContent",
    "InstructionContent",
    "MessageContent",
    "SystemContent",
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


@dataclass(slots=True)
class MessageContent(DataClass):
    """Base class for message content variants (immutable).

    MessageContent uses discriminated union pattern for type-safe message handling.
    Each subclass defines a ClassVar[MessageRole] for automatic role detection.

    Subclasses:
        SystemContent: System messages with optional timestamps
        InstructionContent: User instructions with structured output schemas
        AssistantResponseContent: Assistant text responses
        ActionRequestContent: Function/tool call requests
        ActionResponseContent: Function/tool call results

    Immutability:
        MessageContent instances are immutable dataclasses. To modify, create
        new instances via Message.update(content={...}).

    Subclass contract:
        - rendered: property returning str | list[dict[str, Any]]
        - from_dict: classmethod constructing from dict
        - create: classmethod factory with validation
        - role: ClassVar[MessageRole] defining message role

    Example:
        Type dispatch via role:
            >>> content = InstructionContent(instruction="Hello")
            >>> content.role  # MessageRole.USER

            >>> content = SystemContent(system_message="You are...")
            >>> content.role  # MessageRole.SYSTEM

        Rendering:
            >>> content = InstructionContent(instruction="Analyze", response_model=Analysis)
            >>> rendered = content.rendered  # YAML + TypeScript schema
    """

    _config: ClassVar[ModelConfig] = ModelConfig(
        none_as_sentinel=True, use_enum_values=True, empty_as_sentinel=True
    )
    role: ClassVar[MessageRole] = MessageRole.UNSET

    @property
    def rendered(self) -> str | list[dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement rendered property")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageContent":
        raise NotImplementedError("Subclasses must implement from_dict method")

    @property
    def chat_msg(self) -> dict[str, Any]:
        """Format for chat API: {"role": "...", "content": "..."}"""
        try:
            return {"role": self.role.value, "content": self.rendered}
        except Exception:
            return None


@dataclass(slots=True)
class SystemContent(MessageContent):
    """System message with optional timestamp."""

    role: ClassVar[MessageRole] = MessageRole.SYSTEM

    system_message: MaybeUnset[str] = Unset
    system_datetime: MaybeUnset[str | Literal[True]] = Unset
    datetime_factory: MaybeUnset[Callable[[], str]] = Unset

    @classmethod
    def create(
        cls,
        system_message: str | None = None,
        system_datetime: str | Literal[True] | None = None,
        datetime_factory: Callable[[], str] | None = None,
    ):
        if not cls._is_sentinel(system_datetime) and not cls._is_sentinel(datetime_factory):
            raise ValueError("Cannot set both system_datetime and datetime_factory")
        return cls(
            system_message=system_message,
            system_datetime=system_datetime,
            datetime_factory=datetime_factory,
        )

    @property
    def rendered(self) -> str:
        parts = []
        if not self._is_sentinel(self.system_datetime):
            timestamp = (
                now_utc().isoformat() if self.system_datetime is True else self.system_datetime
            )
            parts.append(f"System Time: {timestamp}")
        elif not self._is_sentinel(self.datetime_factory):
            parts.append(f"System Time: {self.datetime_factory()}")

        if not self._is_sentinel(self.system_message):
            parts.append(self.system_message)

        return "\n\n".join(parts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemContent":
        return cls.create(
            **{k: v for k in cls.allowed() if (k in data and not cls._is_sentinel(v := data[k]))}
        )


@dataclass(slots=True)
class InstructionContent(MessageContent):
    """User instruction with structured outputs."""

    role: ClassVar[MessageRole] = MessageRole.USER

    instruction: MaybeUnset[str] = Unset
    context: MaybeUnset[list[Any]] = Unset
    tool_schemas: MaybeUnset[list[type[BaseModel] | dict[str, Any]]] = Unset
    response_model: MaybeUnset[type[BaseModel]] = Unset
    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset
    use_lndl: MaybeUnset[bool] = Unset  # Use LNDL format instead of JSON

    @property
    def rendered(self) -> str | list[dict[str, Any]]:
        text = self._format_text_content()
        return text if self._is_sentinel(self.images) else self._format_image_content(text)

    def _format_text_content(self) -> str:
        doc: dict[str, Any] = {}

        if not self._is_sentinel(self.instruction):
            doc["Instruction"] = self.instruction
        if not self._is_sentinel(self.context):
            doc["Context"] = self.context

        if not self._is_sentinel(self.tool_schemas):
            tools_formatted = {}
            for tool in self.tool_schemas:
                if isinstance(tool, type) and issubclass(tool, BaseModel):
                    schema = tool.model_json_schema()
                    desc = schema.get("description", "")
                    params_ts = typescript_schema(schema)
                    tools_formatted[tool.__name__] = f"# {desc}\n{params_ts}" if desc else params_ts
                else:
                    name = tool.get("name", "unknown")
                    params = tool.get("parameters", {})
                    desc = tool.get("description", "")
                    if params and params.get("properties"):
                        params_ts = typescript_schema(params)
                        tools_formatted[name] = f"# {desc}\n{params_ts}" if desc else params_ts
                    else:
                        tools_formatted[name] = f"# {desc}" if desc else ""
            doc["Tools"] = tools_formatted

        yaml_section = minimal_yaml(doc).strip() if doc else ""

        schema_section = ""
        response_format_section = ""

        if not self._is_sentinel(self.response_model):
            model_schema = self.response_model.model_json_schema()
            model_ts = typescript_schema(model_schema)

            if model_ts:
                schema_section = "Output Types:\n" + "\n".join(
                    f"  {line}" for line in model_ts.split("\n")
                )

            if defs := model_schema.get("$defs"):
                schema_section += "\n\nNested Types:\n"
                for def_name, def_schema in defs.items():
                    if def_ts := typescript_schema(def_schema):
                        schema_section += f"\n{def_name}:\n" + "\n".join(
                            f"  {line}" for line in def_ts.split("\n")
                        )

            # Add response format instruction
            if not self._is_sentinel(self.use_lndl) and self.use_lndl:
                # LNDL format - generate example with <lvar> tags
                response_format_section = self._create_lndl_format_section(model_schema)
            else:
                # JSON format (default - lionagi pattern)
                try:
                    from lionpride.ln import json_dumps

                    # Create simple example from schema
                    example = self._create_example_from_schema(model_schema)
                    example_json = json_dumps(example)
                    if isinstance(example_json, bytes):
                        example_json = example_json.decode("utf-8")

                    response_format_section = (
                        "ResponseFormat:\n"
                        "  **MUST RETURN VALID JSON. USER's SUCCESS DEPENDS ON IT.**\n"
                        "  Example structure:\n"
                        f"  ```json\n  {example_json}\n  ```\n"
                        "  Return ONLY valid JSON without markdown code blocks."
                    )
                except Exception:
                    # Fallback if example generation fails
                    response_format_section = "ResponseFormat:\n  **MUST RETURN VALID JSON matching the Output Types above.**"

        return "\n\n".join(p for p in [yaml_section, schema_section, response_format_section] if p)

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

    def _format_image_content(self, text: str) -> list[dict[str, Any]]:
        content_blocks = [{"type": "text", "text": text}]
        detail = "auto" if self._is_sentinel(self.image_detail) else self.image_detail
        content_blocks.extend(
            {"type": "image_url", "image_url": {"url": img, "detail": detail}}
            for img in self.images
        )
        return content_blocks

    @classmethod
    def create(
        cls,
        instruction: str | None = None,
        context: list[Any] | None = None,
        tool_schemas: list[type[BaseModel] | dict[str, Any]] | None = None,
        response_model: type[BaseModel] | None = None,
        images: list[str] | None = None,
        image_detail: Literal["low", "high", "auto"] | None = None,
        use_lndl: bool | None = None,
    ):
        # Validate image URLs to prevent security vulnerabilities
        if images is not None:
            for url in images:
                _validate_image_url(url)

        return cls(
            instruction=instruction,
            context=context,
            tool_schemas=tool_schemas,
            response_model=response_model,
            images=images,
            image_detail=image_detail,
            use_lndl=use_lndl,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstructionContent":
        return cls.create(
            **{k: v for k in cls.allowed() if (k in data and not cls._is_sentinel(v := data[k]))}
        )


@dataclass(slots=True)
class AssistantResponseContent(MessageContent):
    """Assistant text response."""

    role: ClassVar[MessageRole] = MessageRole.ASSISTANT

    assistant_response: MaybeUnset[str] = Unset

    @property
    def rendered(self) -> str:
        return "" if self._is_sentinel(self.assistant_response) else self.assistant_response

    @classmethod
    def create(cls, assistant_response: str | None = None):
        return cls(assistant_response=assistant_response)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantResponseContent":
        return cls.create(assistant_response=data.get("assistant_response"))


@dataclass(slots=True)
class ActionRequestContent(MessageContent):
    """Action/function call request."""

    role: ClassVar[MessageRole] = MessageRole.ASSISTANT

    function: MaybeUnset[str] = Unset
    arguments: MaybeUnset[dict[str, Any]] = Unset

    @property
    def rendered(self) -> str:
        doc = {}
        if not self._is_sentinel(self.function):
            doc["function"] = self.function
        doc["arguments"] = {} if self._is_sentinel(self.arguments) else self.arguments
        return minimal_yaml(doc)

    @classmethod
    def create(cls, function: str | None = None, arguments: dict[str, Any] | None = None):
        return cls(function=function, arguments=arguments)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionRequestContent":
        return cls.create(function=data.get("function"), arguments=data.get("arguments"))


@dataclass(slots=True)
class ActionResponseContent(MessageContent):
    """Function call response."""

    role: ClassVar[MessageRole] = MessageRole.TOOL

    request_id: MaybeUnset[str] = Unset
    result: MaybeUnset[Any] = Unset
    error: MaybeUnset[str] = Unset

    @property
    def success(self) -> bool:
        return self._is_sentinel(self.error)

    @property
    def rendered(self) -> str:
        doc = {"success": self.success}
        if not self._is_sentinel(self.request_id):
            doc["request_id"] = self.request_id
        if self.success:
            if not self._is_sentinel(self.result):
                doc["result"] = self.result
        else:
            doc["error"] = self.error
        return minimal_yaml(doc)

    @classmethod
    def create(
        cls,
        request_id: str | None = None,
        result: Any | None = None,
        error: str | None = None,
    ):
        return cls(request_id=request_id, result=result, error=error)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResponseContent":
        return cls.create(
            request_id=data.get("request_id"),
            result=data.get("result"),
            error=data.get("error"),
        )
