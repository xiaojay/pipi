from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..types import ToolExecutionResult


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], ToolExecutionResult]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
