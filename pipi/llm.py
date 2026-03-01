from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ModelConfig
from .types import ChatMessage, ToolCall


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str | None
    usage: dict[str, Any] | None
    raw: dict[str, Any]


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: ModelConfig,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


class _JSONHTTPClient:
    def __init__(self, timeout: float = 300.0) -> None:
        self.timeout = timeout

    def _post_json(self, *, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAICompatClient(_JSONHTTPClient):
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
            detail = _read_http_error_detail(error)
            if self._should_retry_without_reasoning(error.code, detail, payload):
                retry_payload = dict(payload)
                retry_payload.pop("reasoning_effort", None)
                try:
                    raw = self._perform_request(model, retry_payload)
                except urllib.error.HTTPError as retry_error:
                    retry_detail = _read_http_error_detail(retry_error)
                    raise RuntimeError(f"LLM request failed with HTTP {retry_error.code}: {retry_detail}") from retry_error
                except urllib.error.URLError as retry_error:
                    raise RuntimeError(f"LLM request failed: {retry_error.reason}") from retry_error
            else:
                raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error
        return self._parse_response(raw)

    def _perform_request(self, model: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json",
            **model.headers,
        }
        return self._post_json(
            url=f"{model.base_url}/chat/completions",
            headers=headers,
            payload=payload,
        )

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


class AnthropicClient(_JSONHTTPClient):
    DEFAULT_MAX_TOKENS = 4096

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
            "max_tokens": self.DEFAULT_MAX_TOKENS,
            "messages": self._convert_messages(messages),
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = {"type": "auto"}
        try:
            raw = self._perform_request(model, payload)
        except urllib.error.HTTPError as error:
            detail = _read_http_error_detail(error)
            raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error.reason}") from error
        return self._parse_response(raw)

    def _perform_request(self, model: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": model.api_key,
            "Content-Type": "application/json",
            **model.headers,
        }
        return self._post_json(
            url=f"{model.base_url}/messages",
            headers=headers,
            payload=payload,
        )

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if not pending_tool_results:
                return
            converted.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        for message in messages:
            if message.role == "toolResult":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": self._convert_tool_result_content(message.content),
                        "is_error": self._is_tool_error(message),
                    }
                )
                continue

            flush_tool_results()
            if message.role == "user":
                converted.append({"role": "user", "content": self._convert_content(message.content)})
            elif message.role == "assistant":
                converted.append({"role": "assistant", "content": self._convert_assistant_content(message)})

        flush_tool_results()
        return converted

    def _convert_content(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "text":
                converted.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image":
                converted.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("mime_type", "image/png"),
                            "data": part.get("data", ""),
                        },
                    }
                )
        return converted

    def _convert_assistant_content(self, message: ChatMessage) -> list[dict[str, Any]]:
        content = self._convert_content(message.content)
        for tool_call in message.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            )
        return content

    def _convert_tool_result_content(self, content: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if all(part.get("type") == "text" for part in content):
            return "".join(part.get("text", "") for part in content)
        return self._convert_content(content)

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function") or {}
            converted.append(
                {
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "input_schema": dict(function.get("parameters") or {}),
                }
            )
        return converted

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in raw.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
            elif item.get("type") == "tool_use":
                raw_input = item.get("input")
                arguments = raw_input if isinstance(raw_input, dict) else {}
                tool_calls.append(
                    ToolCall(
                        id=str(item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=arguments,
                    )
                )
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=str(raw.get("stop_reason") or "") or None,
            usage=_normalize_anthropic_usage(raw.get("usage")),
            raw=raw,
        )

    def _is_tool_error(self, message: ChatMessage) -> bool:
        text = message.text()
        return text.startswith("Unknown tool:") or text.startswith(f"Tool {message.tool_name} failed:")


def create_llm_client(model: ModelConfig, timeout: float = 300.0) -> LLMClient:
    if model.provider in {"anthropic", "claude"}:
        return AnthropicClient(timeout=timeout)
    return OpenAICompatClient(timeout=timeout)


def _read_http_error_detail(error: urllib.error.HTTPError) -> str:
    detail = error.read().decode("utf-8", errors="replace")
    error.close()
    return detail


def _normalize_anthropic_usage(raw_usage: Any) -> dict[str, Any] | None:
    if not isinstance(raw_usage, dict):
        return None
    usage = dict(raw_usage)
    input_tokens = _int_or_zero(usage.get("input_tokens"))
    output_tokens = _int_or_zero(usage.get("output_tokens"))
    usage.setdefault("prompt_tokens", input_tokens)
    usage.setdefault("completion_tokens", output_tokens)
    usage.setdefault("total_tokens", input_tokens + output_tokens)
    return usage


def _int_or_zero(value: Any) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
