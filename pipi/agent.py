from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .llm import LLMResponse, OpenAICompatClient
from .session import SessionManager
from .types import ChatMessage, ContentPart, ToolExecutionResult, text_part
from .tools import ToolDefinition, build_builtin_tools


@dataclass
class AgentResult:
    assistant_message: ChatMessage
    tool_rounds: int


class CodingAgent:
    def __init__(
        self,
        *,
        model: ModelConfig,
        session_manager: SessionManager,
        llm_client: OpenAICompatClient | None = None,
        system_prompt: str | None = None,
        tools: dict[str, ToolDefinition] | None = None,
        max_tool_rounds: int = 12,
    ) -> None:
        self.model = model
        self.session_manager = session_manager
        self.llm_client = llm_client or OpenAICompatClient()
        self.system_prompt = system_prompt or self.default_system_prompt()
        self.max_tool_rounds = max_tool_rounds
        self.tools = tools or build_builtin_tools(self.session_manager.cwd)
        context = self.session_manager.build_session_context()
        if not context.model:
            self.session_manager.append_model_change(model.provider, model.model)
        if context.thinking_level == "off" and model.thinking_level != "off":
            self.session_manager.append_thinking_level_change(model.thinking_level)

    @staticmethod
    def default_system_prompt() -> str:
        return (
            "You are a pragmatic coding agent working in a local repository. "
            "Use tools when needed, keep edits precise, and explain concrete outcomes."
        )

    def prompt(self, content: list[ContentPart]) -> AgentResult:
        user_message = ChatMessage.user(content)
        self.session_manager.append_message(user_message)
        tool_rounds = 0
        while True:
            context = self.session_manager.build_session_context()
            response = self.llm_client.complete(
                model=self.model,
                system_prompt=self.system_prompt,
                messages=context.messages,
                tools=[tool.to_openai_tool() for tool in self.tools.values()],
            )
            assistant_message = self._response_to_message(response)
            self.session_manager.append_message(assistant_message)
            if not response.tool_calls:
                return AgentResult(assistant_message=assistant_message, tool_rounds=tool_rounds)
            tool_rounds += 1
            if tool_rounds > self.max_tool_rounds:
                raise RuntimeError(f"Exceeded maximum tool rounds ({self.max_tool_rounds})")
            for tool_call in response.tool_calls:
                tool_message = self._execute_tool_call(tool_call.name, tool_call.id, tool_call.arguments)
                self.session_manager.append_message(tool_message)

    def prompt_text(self, text: str, *, extra_parts: list[ContentPart] | None = None) -> AgentResult:
        parts = [text_part(text)]
        if extra_parts:
            parts.extend(extra_parts)
        return self.prompt(parts)

    def new_session(self) -> None:
        self.session_manager.new_session()
        self.session_manager.append_model_change(self.model.provider, self.model.model)
        self.session_manager.append_thinking_level_change(self.model.thinking_level)

    def set_model(self, model: ModelConfig) -> None:
        self.model = model
        self.session_manager.append_model_change(model.provider, model.model)
        self.session_manager.append_thinking_level_change(model.thinking_level)

    def _response_to_message(self, response: LLMResponse) -> ChatMessage:
        content = [text_part(response.text)] if response.text else []
        return ChatMessage.assistant(
            content=content,
            tool_calls=response.tool_calls,
            provider=self.model.provider,
            model=self.model.model,
            stop_reason=response.stop_reason,
            usage=response.usage,
        )

    def _execute_tool_call(self, name: str, tool_call_id: str, arguments: dict[str, Any]) -> ChatMessage:
        tool = self.tools.get(name)
        if not tool:
            return ChatMessage.tool_result(tool_call_id, name, f"Unknown tool: {name}")
        try:
            result = tool.execute(arguments)
        except Exception as error:
            return ChatMessage.tool_result(tool_call_id, name, f"Tool {name} failed: {error}")
        return ChatMessage.tool_result(tool_call_id, name, self._tool_result_to_text(result))

    def _tool_result_to_text(self, result: ToolExecutionResult) -> str:
        text = result.text
        if result.details:
            return f"{text}\n\nDetails:\n{result.details}"
        return text
