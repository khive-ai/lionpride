# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, JsonValue

from lionpride.libs.schema_handlers import is_pydantic_model, minimal_yaml
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


@dataclass(slots=True)
class MessageContent(DataClass):
    _config: ClassVar[ModelConfig] = ModelConfig(
        none_as_sentinel=True,
        use_enum_values=True,
        empty_as_sentinel=True,
    )
    role: ClassVar[MessageRole] = MessageRole.UNSET

    def render(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement render method")

    def to_chat(self, *args, **kwargs) -> dict[str, Any]:
        """Format for chat API: {"role": "...", "content": "..."}"""
        try:
            return {"role": self.role.value, "content": self.render(*args, **kwargs)}
        except Exception:
            return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageContent":
        raise NotImplementedError("Subclasses must implement from_dict method")


@dataclass(slots=True)
class SystemContent(MessageContent):
    """System message with optional timestamp."""

    role: ClassVar[MessageRole] = MessageRole.SYSTEM

    system_message: MaybeUnset[str] = Unset
    system_datetime: MaybeUnset[str | Literal[True]] = Unset
    datetime_factory: MaybeUnset[Callable[[], str]] = Unset

    def render(self, *_args, **_kwargs) -> str:
        parts = []
        if not self._is_sentinel(self.system_datetime):
            timestamp = (
                now_utc().isoformat(timespec="seconds")
                if self.system_datetime is True
                else self.system_datetime
            )
            parts.append(f"System Time: {timestamp}")
        elif not self._is_sentinel(self.datetime_factory):
            parts.append(f"System Time: {self.datetime_factory()}")

        if not self._is_sentinel(self.system_message):
            parts.append(self.system_message)

        return "\n\n".join(parts)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemContent":
        return cls.create(
            **{k: v for k in cls.allowed() if (k in data and not cls._is_sentinel(v := data[k]))}
        )


@dataclass(slots=True)
class InstructionContent(MessageContent):
    """User instruction with structured outputs."""

    role: ClassVar[MessageRole] = MessageRole.USER

    instruction: MaybeUnset[JsonValue] = Unset
    """Primary instruction for the LLM."""

    context: MaybeUnset[list[JsonValue]] = Unset
    """Additional context for the LLM."""

    tool_schemas: MaybeUnset[list[str]] = Unset
    """Schemas for tools the LLM can use. From tool.render"""

    request_model: MaybeUnset[type[BaseModel]] = Unset
    """Pydantic model defining the expected structured response from LLM"""

    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset

    def render(
        self,
        structure_format: Literal["json", "custom"] = "json",
        custom_renderer: "Callable[[type[BaseModel]], str] | None" = None,
    ) -> str | list[dict[str, Any]]:
        text = self._format_text_content(structure_format, custom_renderer)
        return text if self._is_sentinel(self.images) else self._format_image_content(text)

    def _format_text_content(
        self,
        structure_format: Literal["json", "custom"] = "json",
        custom_renderer: "Callable[[type[BaseModel]], str] | None" = None,
    ) -> str:
        from ._utils import _format_json_response_structure, _format_model_schema, _format_task

        task_data = {
            "Instruction": self.instruction,
            "Context": self.context,
            "Tools": self.tool_schemas,
        }
        text = _format_task({k: v for k, v in task_data.items() if not self._is_sentinel(v)})
        if not self._is_sentinel(self.request_model) and is_pydantic_model(self.request_model):
            text += _format_model_schema(self.request_model)
            if structure_format == "json":
                text += _format_json_response_structure(self.request_model)
            elif structure_format == "custom" and custom_renderer is not None:
                text += custom_renderer(self.request_model)
        return text.strip()

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
        instruction: JsonValue = None,
        context: list[Any] | None = None,
        tool_schemas: list[str] | None = None,
        request_model: type[BaseModel] | None = None,
        images: list[str] | None = None,
        image_detail: Literal["low", "high", "auto"] | None = None,
    ):
        # Validate image URLs to prevent security vulnerabilities
        if images is not None:
            from ._utils import _validate_image_url

            for url in images:
                _validate_image_url(url)

        return cls(
            instruction=instruction,
            context=context,
            tool_schemas=tool_schemas,
            request_model=request_model,
            images=images,
            image_detail=image_detail,
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

    def render(self, *_args, **_kwargs) -> str:
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

    def render(self, *_args, **_kwargs) -> str:
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

    def render(self, *_args, **_kwargs) -> str:
        doc = {"success": self.success}
        if not self._is_sentinel(self.request_id):
            doc["request_id"] = str(self.request_id)[:8]
        if self.success:
            if not self._is_sentinel(self.result):
                doc["result"] = self.result
        else:
            doc["error"] = self.error
        return minimal_yaml(doc)

    @property
    def success(self) -> bool:
        return self._is_sentinel(self.error)

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
