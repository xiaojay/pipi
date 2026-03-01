from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .types import ChatMessage, ToolCall


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str | None
    usage: dict[str, Any] | None
    raw: dict[str, Any]


class OpenAICompatClient:
    def __init__(self, timeout: float = 300.0) -> None:
        self.timeout = timeout

    def complete(
        self,
        *,
        model: ModelConfig,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model.model,
            "messages": self._convert_messages(system_prompt, messages),
            "tools": tools,
            "tool_choice": "auto" if tools else "none",
        }
        if model.provider == "openai" and model.thinking_level != "off":
            effort = self._thinking_to_reasoning_effort(model.thinking_level)
            if effort:
                payload["reasoning_effort"] = effort
        try:
            raw = self._perform_request(model, payload)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            error.close()
            if self._should_retry_without_reasoning(error.code, detail, payload):
                retry_payload = dict(payload)
                retry_payload.pop("reasoning_effort", None)
                try:
                    raw = self._perform_request(model, retry_payload)
                except urllib.error.HTTPError as retry_error:
                    retry_detail = retry_error.read().decode("utf-8", errors="replace")
                    retry_error.close()
                    raise RuntimeError(f"LLM request failed with HTTP {retry_error.code}: {retry_detail}") from retry_error
                except urllib.error.URLError as retry_error:
                    raise RuntimeError(f"LLM request failed: {retry_error.reason}") from retry_error
            else:
                raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error
        return self._parse_response(raw)

    def _perform_request(self, model: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json",
            **model.headers,
        }
        request = urllib.request.Request(
            url=f"{model.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _should_retry_without_reasoning(self, status_code: int, detail: str, payload: dict[str, Any]) -> bool:
        if status_code != 400:
            return False
        if "reasoning_effort" not in payload:
            return False
        return "reasoning_effort" in detail

    def _convert_messages(self, system_prompt: str, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})
        for message in messages:
            if message.role == "user":
                converted.append({"role": "user", "content": self._convert_content(message.content)})
            elif message.role == "assistant":
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.text() or "",
                }
                if message.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
                converted.append(assistant_message)
            elif message.role == "toolResult":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.text(),
                    }
                )
        return converted

    def _convert_content(self, content: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if all(part.get("type") == "text" for part in content):
            return "".join(part.get("text", "") for part in content)
        converted: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "text":
                converted.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image":
                mime_type = part.get("mime_type", "image/png")
                data = part.get("data", "")
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data}"},
                    }
                )
        return converted

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response did not contain any choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            )
        else:
            text = ""
        tool_calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"raw": raw_arguments}
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )
        finish_reason = choices[0].get("finish_reason")
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            usage=raw.get("usage"),
            raw=raw,
        )

    def _thinking_to_reasoning_effort(self, thinking_level: str) -> str | None:
        mapping = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        }
        return mapping.get(thinking_level)
