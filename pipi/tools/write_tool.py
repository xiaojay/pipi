from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .path_utils import resolve_to_cwd


def create_write_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        path = str(arguments["path"])
        content = str(arguments["content"])
        absolute_path = resolve_to_cwd(path, cwd)
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(content, encoding="utf-8")
        return ToolExecutionResult(content=[text_part(f"Successfully wrote {len(content)} bytes to {path}")])

    return ToolDefinition(
        name="write",
        description="Write content to a file. Creates parent directories automatically.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
        execute=execute,
    )
