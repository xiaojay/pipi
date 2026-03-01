from __future__ import annotations

import os
import subprocess
import tempfile
from collections import deque
from typing import Any

from ..types import ToolExecutionResult, text_part
from .base import ToolDefinition
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_tail


def create_bash_tool(cwd: str) -> ToolDefinition:
    def execute(arguments: dict[str, Any]) -> ToolExecutionResult:
        command = str(arguments["command"])
        timeout = float(arguments["timeout"]) if "timeout" in arguments and arguments["timeout"] is not None else None
        shell = os.environ.get("SHELL", "/bin/bash")
        process = subprocess.Popen(
            [shell, "-lc", command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        temp_file = None
        buffered_chunks: deque[bytes] = deque()
        buffered_bytes = 0
        total_bytes = 0
        max_buffer_bytes = DEFAULT_MAX_BYTES * 2
        try:
            while True:
                chunk = process.stdout.read1(4096) if process.stdout else b""
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > DEFAULT_MAX_BYTES and temp_file is None:
                    temp_handle = tempfile.NamedTemporaryFile(prefix="pi-bash-", suffix=".log", delete=False)
                    temp_file = temp_handle.name
                    for buffered in buffered_chunks:
                        temp_handle.write(buffered)
                    temp_handle.write(chunk)
                    temp_handle.flush()
                    temp_handle.close()
                elif temp_file:
                    with open(temp_file, "ab") as handle:
                        handle.write(chunk)
                buffered_chunks.append(chunk)
                buffered_bytes += len(chunk)
                while buffered_bytes > max_buffer_bytes and len(buffered_chunks) > 1:
                    buffered_bytes -= len(buffered_chunks.popleft())
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            raise RuntimeError(f"Command timed out after {timeout:g} seconds") from error
        finally:
            if process.stdout:
                process.stdout.close()
        output = b"".join(buffered_chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(output)
        result_text = truncation.content or "(no output)"
        details = None
        if truncation.truncated:
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            result_text += (
                "\n\n"
                f"[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_file}]"
            )
            details = {"truncation": truncation.__dict__, "full_output_path": temp_file}
        if return_code != 0:
            raise RuntimeError(f"{result_text}\n\nCommand exited with code {return_code}")
        return ToolExecutionResult(content=[text_part(result_text)], details=details)

    return ToolDefinition(
        name="bash",
        description=(
            f"Execute a shell command in the current working directory. "
            f"Returns the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB of output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
                "timeout": {"type": "number", "description": "Optional timeout in seconds"},
            },
            "required": ["command"],
        },
        execute=execute,
    )
