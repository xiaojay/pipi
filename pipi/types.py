from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any

ContentPart = dict[str, Any]


def now_ms() -> int:
    return int(time() * 1000)


def text_part(text: str) -> ContentPart:
    return {"type": "text", "text": text}


def image_part(data: str, mime_type: str) -> ContentPart:
    return {"type": "image", "data": data, "mime_type": mime_type}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            arguments=dict(data.get("arguments", {})),
        )


@dataclass
class ChatMessage:
    role: str
    content: list[ContentPart]
    timestamp: int = field(default_factory=now_ms)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    error_message: str | None = None
    usage: dict[str, Any] | None = None

    def text(self) -> str:
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(data["role"]),
            content=[dict(part) for part in data.get("content", [])],
            timestamp=int(data.get("timestamp", now_ms())),
            tool_calls=[ToolCall.from_dict(item) for item in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            provider=data.get("provider"),
            model=data.get("model"),
            stop_reason=data.get("stop_reason"),
            error_message=data.get("error_message"),
            usage=dict(data["usage"]) if data.get("usage") else None,
        )

    @classmethod
    def user(cls, content: list[ContentPart]) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: list[ContentPart],
        *,
        tool_calls: list[ToolCall] | None = None,
        provider: str | None = None,
        model: str | None = None,
        stop_reason: str | None = None,
        error_message: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> "ChatMessage":
        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls or [],
            provider=provider,
            model=model,
            stop_reason=stop_reason,
            error_message=error_message,
            usage=usage,
        )

    @classmethod
    def tool_result(cls, tool_call_id: str, tool_name: str, text: str) -> "ChatMessage":
        return cls(
            role="toolResult",
            content=[text_part(text)],
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )


@dataclass
class ToolExecutionResult:
    content: list[ContentPart]
    details: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")
