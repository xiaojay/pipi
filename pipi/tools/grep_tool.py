from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .path_utils import resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, format_size, truncate_head, truncate_line

DEFAULT_LIMIT = 100


def create_grep_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        pattern = str(arguments["pattern"])
        path = str(arguments.get("path") or ".")
        glob_pattern = arguments.get("glob")
        ignore_case = bool(arguments.get("ignoreCase"))
        literal = bool(arguments.get("literal"))
        context = max(0, int(arguments.get("context") or 0))
        limit = max(1, int(arguments.get("limit") or DEFAULT_LIMIT))
        search_path = resolve_to_cwd(path, cwd)
        if not search_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if search_path.is_file():
            files = [search_path]
            base_dir = search_path.parent
        else:
            files = [candidate for candidate in search_path.rglob("*") if candidate.is_file()]
            base_dir = search_path
        regex_flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(re.escape(pattern) if literal else pattern, regex_flags)
        output_lines: list[str] = []
        match_count = 0
        lines_truncated = False
        for file_path in files:
            relative = file_path.relative_to(base_dir).as_posix() if file_path != base_dir else file_path.name
            if relative.startswith(".git/") or relative.startswith("node_modules/"):
                continue
            if glob_pattern and not (fnmatch.fnmatch(relative, str(glob_pattern)) or fnmatch.fnmatch(file_path.name, str(glob_pattern))):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
            except OSError:
                continue
            for index, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                match_count += 1
                start = max(1, index - context)
                end = min(len(lines), index + context)
                for current in range(start, end + 1):
                    rendered, was_truncated = truncate_line(lines[current - 1], GREP_MAX_LINE_LENGTH)
                    lines_truncated = lines_truncated or was_truncated
                    prefix = ":" if current == index else "-"
                    output_lines.append(f"{relative}{prefix}{current}: {rendered}")
                if match_count >= limit:
                    break
            if match_count >= limit:
                break
        if match_count == 0:
            return ToolExecutionResult(content=[text_part("No matches found")])
        truncation = truncate_head("\n".join(output_lines), max_lines=10**9, max_bytes=DEFAULT_MAX_BYTES)
        output = truncation.content
        notices = []
        details = {}
        if match_count >= limit:
            notices.append(f"{limit} matches limit reached. Use limit={limit * 2} for more")
            details["match_limit_reached"] = limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation.__dict__
        if lines_truncated:
            notices.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read for full lines")
            details["lines_truncated"] = True
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"
        return ToolExecutionResult(content=[text_part(output)], details=details or None)

    return ToolDefinition(
        name="grep",
        description="Search file contents for a regex or literal string.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "glob": {"type": "string", "description": "Optional glob file filter"},
                "ignoreCase": {"type": "boolean", "description": "Case-insensitive search"},
                "literal": {"type": "boolean", "description": "Treat pattern as a literal string"},
                "context": {"type": "integer", "description": "Lines of context before and after each match"},
                "limit": {"type": "integer", "description": "Maximum number of matches"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )
