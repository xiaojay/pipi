from __future__ import annotations

from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .edit_diff import (
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .path_utils import resolve_to_cwd


def create_edit_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        path = str(arguments["path"])
        old_text = str(arguments["oldText"])
        new_text = str(arguments["newText"])
        absolute_path = resolve_to_cwd(path, cwd)
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raw_content = absolute_path.read_text(encoding="utf-8", errors="replace")
        bom, content = strip_bom(raw_content)
        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        normalized_old = normalize_to_lf(old_text)
        normalized_new = normalize_to_lf(new_text)
        found, index, match_length, _, replacement_content = fuzzy_find_text(normalized_content, normalized_old)
        if not found:
            raise ValueError(
                f"Could not find the exact text in {path}. The old text must match exactly including whitespace."
            )
        fuzzy_content = normalize_for_fuzzy_match(normalized_content)
        fuzzy_old = normalize_for_fuzzy_match(normalized_old)
        occurrences = fuzzy_content.count(fuzzy_old)
        if occurrences > 1:
            raise ValueError(
                f"Found {occurrences} occurrences of the text in {path}. Please provide more context to make it unique."
            )
        new_content = replacement_content[:index] + normalized_new + replacement_content[index + match_length :]
        if replacement_content == new_content:
            raise ValueError(f"No changes made to {path}. The replacement produced identical content.")
        final_content = bom + restore_line_endings(new_content, original_ending)
        absolute_path.write_text(final_content, encoding="utf-8")
        diff, first_changed_line = generate_diff_string(replacement_content, new_content)
        return ToolExecutionResult(
            content=[text_part(f"Successfully replaced text in {path}.")],
            details={"diff": diff, "first_changed_line": first_changed_line},
        )

    return ToolDefinition(
        name="edit",
        description="Edit a file by replacing exact text with new text.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "oldText": {"type": "string", "description": "Exact text to replace"},
                "newText": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "oldText", "newText"],
        },
        execute=execute,
    )
