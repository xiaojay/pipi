from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .tools.path_utils import resolve_read_path
from .tools.truncate import format_size, truncate_head
from .types import ContentPart, image_part, text_part


def process_file_arguments(paths: list[str], cwd: str | Path) -> list[ContentPart]:
    parts: list[ContentPart] = []
    for raw_path in paths:
        resolved = resolve_read_path(raw_path, cwd)
        mime_type, _ = mimetypes.guess_type(str(resolved))
        if mime_type and mime_type.startswith("image/"):
            parts.append(
                text_part(
                    f"Attached image file: {raw_path}\n"
                    "The next content block contains the image data as an attachment."
                )
            )
            parts.append(image_part(base64.b64encode(resolved.read_bytes()).decode("ascii"), mime_type))
            continue
        content = resolved.read_text(encoding="utf-8", errors="replace")
        truncation = truncate_head(content)
        snippet = truncation.content
        if truncation.truncated:
            snippet += (
                "\n\n"
                f"[File argument truncated to {format_size(truncation.output_bytes)}. "
                f"Original size: {format_size(truncation.total_bytes)}.]"
            )
        parts.append(
            text_part(
                f"Attached file: {raw_path}\n"
                "```text\n"
                f"{snippet}\n"
                "```"
            )
        )
    return parts
