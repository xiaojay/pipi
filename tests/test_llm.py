from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from pipi.config import ModelConfig
from pipi.llm import OpenAICompatClient
from pipi.types import ChatMessage, text_part


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


if __name__ == "__main__":
    unittest.main()
