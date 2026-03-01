from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from .config import ModelConfig
from .llm import LLMClient, LLMResponse, create_llm_client
from .session import SessionManager
from .types import ChatMessage, ContentPart, ToolExecutionResult, text_part
from .tools import ToolDefinition, build_builtin_tools


@dataclass
class AgentResult:
    assistant_message: ChatMessage
    tool_rounds: int
    tool_runs: list["ToolRun"]


@dataclass
class ToolRun:
    name: str
    arguments: dict[str, Any]
    message: ChatMessage
    is_error: bool = False

    @property
    def output_text(self) -> str:
        return self.message.text()


@dataclass(frozen=True)
class ToolCallProgress:
    name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class ToolResultProgress:
    tool_run: ToolRun


AgentProgressEvent: TypeAlias = ToolCallProgress | ToolResultProgress
AgentProgressHandler: TypeAlias = Callable[[AgentProgressEvent], None]


class CodingAgent:
    def __init__(
        self,
        *,
        model: ModelConfig,
        session_manager: SessionManager,
        llm_client: LLMClient | None = None,
        system_prompt: str | None = None,
        tools: dict[str, ToolDefinition] | None = None,
        max_tool_rounds: int = 12,
    ) -> None:
        self.model = model
        self.session_manager = session_manager
        self.llm_client = llm_client or create_llm_client(model)
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

    def prompt(
        self,
        content: list[ContentPart],
        *,
        progress_handler: AgentProgressHandler | None = None,
    ) -> AgentResult:
        user_message = ChatMessage.user(content)
        self.session_manager.append_message(user_message)
        tool_rounds = 0
        tool_runs: list[ToolRun] = []
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
                return AgentResult(assistant_message=assistant_message, tool_rounds=tool_rounds, tool_runs=tool_runs)
            tool_rounds += 1
            if tool_rounds > self.max_tool_rounds:
                raise RuntimeError(f"Exceeded maximum tool rounds ({self.max_tool_rounds})")
            for tool_call in response.tool_calls:
                if progress_handler:
                    progress_handler(
                        ToolCallProgress(
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            tool_call_id=tool_call.id,
                        )
                    )
                tool_run = self._execute_tool_call(tool_call.name, tool_call.id, tool_call.arguments)
                tool_runs.append(tool_run)
                self.session_manager.append_message(tool_run.message)
                if progress_handler:
                    progress_handler(ToolResultProgress(tool_run=tool_run))

    def prompt_text(
        self,
        text: str,
        *,
        extra_parts: list[ContentPart] | None = None,
        progress_handler: AgentProgressHandler | None = None,
    ) -> AgentResult:
        parts = [text_part(text)]
        if extra_parts:
            parts.extend(extra_parts)
        return self.prompt(parts, progress_handler=progress_handler)

    def new_session(self) -> None:
        self.session_manager.new_session()
        self.session_manager.append_model_change(self.model.provider, self.model.model)
        self.session_manager.append_thinking_level_change(self.model.thinking_level)

    def set_model(self, model: ModelConfig) -> None:
        self.model = model
        self.llm_client = create_llm_client(model)
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

    def _execute_tool_call(self, name: str, tool_call_id: str, arguments: dict[str, Any]) -> ToolRun:
        tool = self.tools.get(name)
        if not tool:
            error_text = f"Unknown tool: {name}"
            return ToolRun(
                name=name,
                arguments=arguments,
                message=ChatMessage(
                    role="toolResult",
                    content=[text_part(error_text)],
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    error_message=error_text,
                ),
                is_error=True,
            )
        try:
            result = tool.execute(arguments)
        except Exception as error:
            error_text = f"Tool {name} failed: {error}"
            return ToolRun(
                name=name,
                arguments=arguments,
                message=ChatMessage(
                    role="toolResult",
                    content=[text_part(error_text)],
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    error_message=error_text,
                ),
                is_error=True,
            )
        return ToolRun(
            name=name,
            arguments=arguments,
            message=ChatMessage.tool_result(tool_call_id, name, self._tool_result_to_text(result)),
        )

    def _tool_result_to_text(self, result: ToolExecutionResult) -> str:
        text = result.text
        if result.details:
            return f"{text}\n\nDetails:\n{result.details}"
        return text
