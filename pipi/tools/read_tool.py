from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .path_utils import resolve_read_path
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_head


def create_read_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        path = str(arguments["path"])
        offset = int(arguments["offset"]) if "offset" in arguments and arguments["offset"] is not None else None
        limit = int(arguments["limit"]) if "limit" in arguments and arguments["limit"] is not None else None
        absolute_path = resolve_read_path(path, cwd)
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        mime_type, _ = mimetypes.guess_type(str(absolute_path))
        if mime_type and mime_type.startswith("image/"):
            return ToolExecutionResult(
                content=[
                    text_part(
                        f"Read image file [{mime_type}] at {path}. "
                        "This Python port does not send image tool results back to the model; "
                        "attach the image with @file in the user prompt instead."
                    )
                ]
            )
        content = absolute_path.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        total_lines = len(lines)
        start_line = max(0, (offset or 1) - 1)
        if start_line >= len(lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(lines)} lines total)")
        if limit is not None:
            selected = "\n".join(lines[start_line : start_line + limit])
            user_limited = min(limit, total_lines - start_line)
        else:
            selected = "\n".join(lines[start_line:])
            user_limited = None
        truncation = truncate_head(selected)
        start_display = start_line + 1
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(lines[start_line].encode("utf-8")))
            output = (
                f"[Line {start_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            return ToolExecutionResult(content=[text_part(output)], details={"truncation": truncation.__dict__})
        output = truncation.content
        details = None
        if truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output += (
                "\n\n"
                f"[Showing lines {start_display}-{end_display} of {total_lines}. "
                f"Use offset={next_offset} to continue.]"
            )
            details = {"truncation": truncation.__dict__}
        elif user_limited is not None and start_line + user_limited < total_lines:
            next_offset = start_line + user_limited + 1
            remaining = total_lines - (start_line + user_limited)
            output += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        return ToolExecutionResult(content=[text_part(output)], details=details)

    return ToolDefinition(
        name="read",
        description=(
            f"Read the contents of a file. Text output is truncated to {DEFAULT_MAX_LINES} lines or "
            f"{DEFAULT_MAX_BYTES // 1024}KB. Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "1-indexed line offset"},
                "limit": {"type": "integer", "description": "Maximum number of lines to return"},
            },
            "required": ["path"],
        },
        execute=execute,
    )
