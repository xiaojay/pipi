from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from pipi.config import ModelConfig, resolve_model_config
from pipi.llm import AnthropicClient, OpenAICompatClient, create_llm_client
from pipi.types import ChatMessage, ToolCall, text_part


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class LLMClientTests(unittest.TestCase):
    def test_retries_without_reasoning_effort_when_backend_rejects_it(self) -> None:
        client = OpenAICompatClient()
        model = ModelConfig(
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://example.test/v1",
            api_key="test-key",
            thinking_level="medium",
        )
        messages = [ChatMessage.user([text_part("1+1")])]
        seen_payloads: list[dict] = []

        def side_effect(request, timeout=0):  # type: ignore[no-untyped-def]
            payload = json.loads(request.data.decode("utf-8"))
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    400,
                    "Bad Request",
                    hdrs=None,
                    fp=io.BytesIO(
                        b'{"error":{"message":"Unrecognized request argument supplied: reasoning_effort"}}'
                    ),
                )
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "2"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        with patch("urllib.request.urlopen", side_effect=side_effect):
            response = client.complete(model=model, system_prompt="", messages=messages, tools=[])

        self.assertEqual(response.text, "2")
        self.assertEqual(len(seen_payloads), 2)
        self.assertIn("reasoning_effort", seen_payloads[0])
        self.assertNotIn("reasoning_effort", seen_payloads[1])

    def test_anthropic_client_uses_messages_api_format(self) -> None:
        client = AnthropicClient()
        model = ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            base_url="https://api.anthropic.test/v1",
            api_key="anthropic-key",
            thinking_level="medium",
            headers={"anthropic-version": "2023-06-01"},
        )
        messages = [
            ChatMessage.user([text_part("Check the weather")]),
            ChatMessage.assistant(
                [text_part("I'll check.")],
                tool_calls=[ToolCall(id="toolu_1", name="get_weather", arguments={"location": "Paris"})],
            ),
            ChatMessage.tool_result("toolu_1", "get_weather", "18C"),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        seen_payloads: list[dict] = []
        seen_headers: list[dict[str, str]] = []

        def side_effect(request, timeout=0):  # type: ignore[no-untyped-def]
            seen_payloads.append(json.loads(request.data.decode("utf-8")))
            seen_headers.append({key.lower(): value for key, value in request.header_items()})
            return _FakeResponse(
                {
                    "content": [{"type": "text", "text": "Done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            )

        with patch("urllib.request.urlopen", side_effect=side_effect):
            response = client.complete(
                model=model,
                system_prompt="You are helpful.",
                messages=messages,
                tools=tools,
            )

        self.assertEqual(response.text, "Done")
        self.assertEqual(len(seen_payloads), 1)
        self.assertEqual(seen_payloads[0]["system"], "You are helpful.")
        self.assertEqual(seen_payloads[0]["messages"][0]["role"], "user")
        self.assertEqual(seen_payloads[0]["messages"][1]["content"][1]["type"], "tool_use")
        self.assertEqual(seen_payloads[0]["messages"][2]["role"], "user")
        self.assertEqual(seen_payloads[0]["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(seen_payloads[0]["messages"][2]["content"][0]["tool_use_id"], "toolu_1")
        self.assertEqual(seen_payloads[0]["tools"][0]["input_schema"]["required"], ["location"])
        self.assertEqual(seen_payloads[0]["tool_choice"], {"type": "auto"})
        self.assertEqual(seen_headers[0]["x-api-key"], "anthropic-key")
        self.assertEqual(seen_headers[0]["anthropic-version"], "2023-06-01")

    def test_anthropic_client_parses_tool_use_blocks(self) -> None:
        client = AnthropicClient()
        model = ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            base_url="https://api.anthropic.test/v1",
            api_key="anthropic-key",
            thinking_level="medium",
            headers={"anthropic-version": "2023-06-01"},
        )
        messages = [ChatMessage.user([text_part("2+2")])]

        def side_effect(request, timeout=0):  # type: ignore[no-untyped-def]
            return _FakeResponse(
                {
                    "content": [
                        {"type": "text", "text": "Let me calculate."},
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "calculator",
                            "input": {"expression": "2+2"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 12, "output_tokens": 5},
                }
            )

        with patch("urllib.request.urlopen", side_effect=side_effect):
            response = client.complete(model=model, system_prompt="", messages=messages, tools=[])

        self.assertEqual(response.text, "Let me calculate.")
        self.assertEqual(response.stop_reason, "tool_use")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "calculator")
        self.assertEqual(response.tool_calls[0].arguments, {"expression": "2+2"})
        self.assertEqual(response.usage, {"input_tokens": 12, "output_tokens": 5, "prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})

    def test_factory_returns_anthropic_client_for_claude_provider(self) -> None:
        model = ModelConfig(
            provider="claude",
            model="claude-sonnet-4-5",
            base_url="https://api.anthropic.com/v1",
            api_key="anthropic-key",
        )

        client = create_llm_client(model)

        self.assertIsInstance(client, AnthropicClient)

    def test_resolve_model_config_uses_claude_default_model(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            model = resolve_model_config(
                provider="anthropic",
                model=None,
                base_url=None,
                api_key="anthropic-key",
                thinking_level="medium",
            )

        self.assertEqual(model.model, "claude-sonnet-4-5")


if __name__ == "__main__":
    unittest.main()
