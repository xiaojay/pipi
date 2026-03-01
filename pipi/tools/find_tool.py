from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, format_size, truncate_head

DEFAULT_LIMIT = 1000


def create_find_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        pattern = str(arguments["pattern"])
        path = str(arguments.get("path") or ".")
        limit = int(arguments.get("limit") or DEFAULT_LIMIT)
        search_path = resolve_to_cwd(path, cwd)
        if not search_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        results: list[str] = []
        for candidate in search_path.rglob("*"):
            relative = candidate.relative_to(search_path).as_posix()
            if relative.startswith(".git/") or relative.startswith("node_modules/"):
                continue
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(candidate.name, pattern):
                suffix = "/" if candidate.is_dir() else ""
                results.append(relative + suffix)
                if len(results) >= limit:
                    break
        if not results:
            return ToolExecutionResult(content=[text_part("No files found matching pattern")])
        truncation = truncate_head("\n".join(results), max_lines=10**9, max_bytes=DEFAULT_MAX_BYTES)
        output = truncation.content
        notices = []
        details = {}
        if len(results) >= limit:
            notices.append(f"{limit} results limit reached. Use limit={limit * 2} for more")
            details["result_limit_reached"] = limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation.__dict__
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return ToolExecutionResult(content=[text_part(output)], details=details or None)

    return ToolDefinition(
        name="find",
        description="Search for files by glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match"},
                "path": {"type": "string", "description": "Directory to search"},
                "limit": {"type": "integer", "description": "Maximum number of results"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )
