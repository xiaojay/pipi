from __future__ import annotations

from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 500


def create_ls_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        path = str(arguments.get("path") or ".")
        limit = int(arguments.get("limit") or DEFAULT_LIMIT)
        absolute_path = resolve_to_cwd(path, cwd)
        if not absolute_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not absolute_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        entries = sorted(absolute_path.iterdir(), key=lambda item: item.name.lower())
        results = []
        entry_limit_reached = len(entries) > limit
        for entry in entries[:limit]:
            results.append(entry.name + ("/" if entry.is_dir() else ""))
        if not results:
            return ToolExecutionResult(content=[text_part("(empty directory)")])
        truncation = truncate_head("\n".join(results), max_lines=10**9, max_bytes=DEFAULT_MAX_BYTES)
        output = truncation.content
        notices = []
        details = {}
        if entry_limit_reached:
            notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
            details["entry_limit_reached"] = limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation.__dict__
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return ToolExecutionResult(content=[text_part(output)], details=details or None)

    return ToolDefinition(
        name="ls",
        description="List directory contents, including dotfiles.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list"},
                "limit": {"type": "integer", "description": "Maximum number of entries"},
            },
        },
        execute=execute,
    )
