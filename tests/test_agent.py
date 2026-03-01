from __future__ import annotations

import unittest

from pipi.agent import CodingAgent, ToolCallProgress, ToolResultProgress
from pipi.config import ModelConfig
from pipi.llm import LLMResponse
from pipi.session import SessionManager
from pipi.tools.base import ToolDefinition
from pipi.types import ToolCall, ToolExecutionResult, text_part


class _FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    def complete(self, **_: object) -> LLMResponse:
        if not self.responses:
            raise AssertionError("No fake responses left")
        return self.responses.pop(0)


class AgentTests(unittest.TestCase):
    def test_prompt_collects_tool_runs(self) -> None:
        tool = ToolDefinition(
            name="echo",
            description="Echo text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            execute=lambda arguments: ToolExecutionResult([text_part(f"echoed: {arguments['text']}")]),
        )
        llm_client = _FakeLLMClient(
            [
                LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
                    stop_reason="tool_calls",
                    usage=None,
                    raw={},
                ),
                LLMResponse(
                    text="done",
                    tool_calls=[],
                    stop_reason="stop",
                    usage=None,
                    raw={},
                ),
            ]
        )
        agent = CodingAgent(
            model=ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                base_url="https://example.test/v1",
                api_key="test-key",
            ),
            session_manager=SessionManager.in_memory(),
            llm_client=llm_client,
            tools={"echo": tool},
        )

        result = agent.prompt_text("use a tool")

        self.assertEqual(result.tool_rounds, 1)
        self.assertEqual(len(result.tool_runs), 1)
        self.assertEqual(result.tool_runs[0].name, "echo")
        self.assertEqual(result.tool_runs[0].arguments, {"text": "hello"})
        self.assertEqual(result.tool_runs[0].output_text, "echoed: hello")
        self.assertFalse(result.tool_runs[0].is_error)
        self.assertEqual(result.assistant_message.text(), "done")

    def test_prompt_emits_tool_progress_as_each_step_completes(self) -> None:
        tool = ToolDefinition(
            name="echo",
            description="Echo text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            execute=lambda arguments: ToolExecutionResult([text_part(f"echoed: {arguments['text']}")]),
        )
        llm_client = _FakeLLMClient(
            [
                LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
                    stop_reason="tool_calls",
                    usage=None,
                    raw={},
                ),
                LLMResponse(
                    text="done",
                    tool_calls=[],
                    stop_reason="stop",
                    usage=None,
                    raw={},
                ),
            ]
        )
        agent = CodingAgent(
            model=ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                base_url="https://example.test/v1",
                api_key="test-key",
            ),
            session_manager=SessionManager.in_memory(),
            llm_client=llm_client,
            tools={"echo": tool},
        )
        seen_events: list[str] = []

        def on_progress(event: ToolCallProgress | ToolResultProgress) -> None:
            if isinstance(event, ToolCallProgress):
                seen_events.append(f"call:{event.name}:{event.arguments['text']}")
            else:
                seen_events.append(f"result:{event.tool_run.name}:{event.tool_run.output_text}")

        agent.prompt_text("use a tool", progress_handler=on_progress)

        self.assertEqual(seen_events, ["call:echo:hello", "result:echo:echoed: hello"])

    def test_missing_tool_is_reported_as_error(self) -> None:
        llm_client = _FakeLLMClient(
            [
                LLMResponse(
                    text="",
                    tool_calls=[ToolCall(id="call_1", name="missing", arguments={})],
                    stop_reason="tool_calls",
                    usage=None,
                    raw={},
                ),
                LLMResponse(
                    text="fallback",
                    tool_calls=[],
                    stop_reason="stop",
                    usage=None,
                    raw={},
                ),
            ]
        )
        agent = CodingAgent(
            model=ModelConfig(
                provider="openai",
                model="gpt-4o-mini",
                base_url="https://example.test/v1",
                api_key="test-key",
            ),
            session_manager=SessionManager.in_memory(),
            llm_client=llm_client,
            tools={},
        )

        result = agent.prompt_text("use a missing tool")

        self.assertEqual(len(result.tool_runs), 1)
        self.assertTrue(result.tool_runs[0].is_error)
        self.assertIn("Unknown tool: missing", result.tool_runs[0].output_text)
        self.assertEqual(result.tool_runs[0].message.error_message, "Unknown tool: missing")


if __name__ == "__main__":
    unittest.main()
