from __future__ import annotations

import io
import unittest

from pipi.agent import ToolCallProgress, ToolResultProgress, ToolRun
from pipi.cli import _build_progress_handler, _summarize_tool_output
from pipi.types import ChatMessage, text_part


class CLITests(unittest.TestCase):
    def test_summarize_tool_output_truncates_lines(self) -> None:
        summary = _summarize_tool_output("line 1\nline 2\nline 3\nline 4\n")

        self.assertEqual(summary, "line 1 | line 2 | line 3 ...")

    def test_progress_handler_shows_tool_and_result(self) -> None:
        stream = io.StringIO()
        progress_handler = _build_progress_handler(stream)
        tool_run = ToolRun(
            name="read",
            arguments={"path": "README.md"},
            message=ChatMessage.tool_result("call_1", "read", "first line\nsecond line\nthird line\nfourth line"),
        )

        progress_handler(ToolCallProgress(name="read", arguments={"path": "README.md"}, tool_call_id="call_1"))
        progress_handler(ToolResultProgress(tool_run=tool_run))

        output = stream.getvalue()
        self.assertIn('[tool] read {"path": "README.md"}', output)
        self.assertIn("[tool result] first line | second line | third line ...", output)

    def test_progress_handler_marks_errors(self) -> None:
        stream = io.StringIO()
        progress_handler = _build_progress_handler(stream)
        tool_run = ToolRun(
            name="bash",
            arguments={"command": "false"},
            message=ChatMessage(
                role="toolResult",
                content=[text_part("Tool bash failed: exit 1")],
                tool_call_id="call_1",
                tool_name="bash",
                error_message="Tool bash failed: exit 1",
            ),
            is_error=True,
        )

        progress_handler(ToolResultProgress(tool_run=tool_run))

        self.assertIn("[tool error] Tool bash failed: exit 1", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
