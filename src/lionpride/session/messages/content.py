# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from pydantic import BaseModel, JsonValue

from lionpride.libs.schema_handlers import is_pydantic_model, minimal_yaml
from lionpride.ln import now_utc
from lionpride.types import DataClass, MaybeUnset, ModelConfig, Unset

from .base import MessageRole

logger = logging.getLogger(__name__)

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

    def to_chat(self, *args, **kwargs) -> dict[str, Any] | None:
        """Format for chat API: {"role": "...", "content": "..."}"""
        try:
            return {"role": self.role.value, "content": self.render(*args, **kwargs)}
        except Exception as e:
            logger.debug(f"Failed to render message content for chat API: {type(e).__name__}: {e}")
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
        parts: list[str] = []
        if not self._is_sentinel(self.system_datetime):
            timestamp = (
                now_utc().isoformat(timespec="seconds")
                if self.system_datetime is True
                else cast(str, self.system_datetime)
            )
            parts.append(f"System Time: {timestamp}")
        elif not self._is_sentinel(self.datetime_factory):
            factory = cast(Callable[[], str], self.datetime_factory)
            parts.append(f"System Time: {factory()}")

        if not self._is_sentinel(self.system_message):
            parts.append(cast(str, self.system_message))

        return "\n\n".join(parts)

    @classmethod
    def create(
        cls,
        system_message: str | None = None,
        system_datetime: str | Literal[True] | None = None,
        datetime_factory: Callable[[], str] | None = None,
    ) -> "SystemContent":
        if not cls._is_sentinel(system_datetime) and not cls._is_sentinel(datetime_factory):
            raise ValueError("Cannot set both system_datetime and datetime_factory")
        return cls(
            system_message=Unset if system_message is None else system_message,
            system_datetime=Unset if system_datetime is None else system_datetime,
            datetime_factory=Unset if datetime_factory is None else datetime_factory,
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
        from ._utils import (
            _format_json_response_structure,
            _format_model_schema,
            _format_task,
        )

        task_data: dict[str, Any] = {
            "Instruction": self.instruction,
            "Context": self.context,
            "Tools": self.tool_schemas,
        }
        text = _format_task({k: v for k, v in task_data.items() if not self._is_sentinel(v)})
        if not self._is_sentinel(self.request_model) and is_pydantic_model(self.request_model):
            model = cast(type[BaseModel], self.request_model)
            text += _format_model_schema(model)
            if structure_format == "json":
                text += _format_json_response_structure(model)
            elif structure_format == "custom" and custom_renderer is not None:
                text += custom_renderer(model)
        return text.strip()

    def _format_image_content(self, text: str) -> list[dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        detail: str = (
            "auto" if self._is_sentinel(self.image_detail) else cast(str, self.image_detail)
        )
        images = cast(list[str], self.images)  # Safe: only called when images is not sentinel
        content_blocks.extend(
            {"type": "image_url", "image_url": {"url": img, "detail": detail}} for img in images
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
    ) -> "InstructionContent":
        # Validate image URLs to prevent security vulnerabilities
        if images is not None:
            from ._utils import _validate_image_url

            for url in images:
                _validate_image_url(url)

        return cls(
            instruction=Unset if instruction is None else instruction,
            context=Unset if context is None else context,
            tool_schemas=Unset if tool_schemas is None else tool_schemas,
            request_model=Unset if request_model is None else request_model,
            images=Unset if images is None else images,
            image_detail=Unset if image_detail is None else image_detail,
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
        return (
            "" if self._is_sentinel(self.assistant_response) else cast(str, self.assistant_response)
        )

    @classmethod
    def create(cls, assistant_response: str | None = None) -> "AssistantResponseContent":
        return cls(assistant_response=(Unset if assistant_response is None else assistant_response))

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
        doc: dict[str, Any] = {}
        if not self._is_sentinel(self.function):
            doc["function"] = cast(str, self.function)
        doc["arguments"] = (
            {} if self._is_sentinel(self.arguments) else cast(dict[str, Any], self.arguments)
        )
        return minimal_yaml(doc)

    @classmethod
    def create(
        cls, function: str | None = None, arguments: dict[str, Any] | None = None
    ) -> "ActionRequestContent":
        return cls(
            function=Unset if function is None else function,
            arguments=Unset if arguments is None else arguments,
        )

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
        doc: dict[str, Any] = {"success": self.success}
        if not self._is_sentinel(self.request_id):
            doc["request_id"] = str(cast(str, self.request_id))[:8]
        if self.success:
            if not self._is_sentinel(self.result):
                doc["result"] = self.result
        else:
            doc["error"] = cast(str, self.error)
        return minimal_yaml(doc)

    @property
    def success(self) -> bool:
        return self._is_sentinel(self.error)

    @classmethod
    def create(
        cls,
        request_id: str | None = None,
        result: Any = Unset,
        error: str | None = None,
    ) -> "ActionResponseContent":
        return cls(
            request_id=Unset if request_id is None else request_id,
            result=result,
            error=Unset if error is None else error,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResponseContent":
        return cls.create(
            request_id=data.get("request_id"),
            result=data.get("result"),
            error=data.get("error"),
        )
