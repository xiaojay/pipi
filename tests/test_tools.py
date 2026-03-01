from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipi.tools.edit_tool import create_edit_tool
from pipi.tools.read_tool import create_read_tool
from pipi.tools.truncate import DEFAULT_MAX_BYTES, truncate_head, truncate_tail


class ToolTests(unittest.TestCase):
    def test_truncate_head_and_tail(self) -> None:
        content = "\n".join(f"line {index}" for index in range(3000))
        head = truncate_head(content)
        tail = truncate_tail(content)
        self.assertTrue(head.truncated)
        self.assertTrue(tail.truncated)
        self.assertLessEqual(head.output_bytes, DEFAULT_MAX_BYTES)
        self.assertLessEqual(tail.output_bytes, DEFAULT_MAX_BYTES)

    def test_read_tool_with_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("a\nb\nc\nd\n", encoding="utf-8")
            tool = create_read_tool(temp_dir)
            result = tool.execute({"path": "sample.txt", "offset": 2, "limit": 2})
            self.assertIn("b\nc", result.text)
            self.assertIn("offset=4", result.text)

    def test_edit_tool_replaces_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            tool = create_edit_tool(temp_dir)
            result = tool.execute({"path": "sample.txt", "oldText": "beta", "newText": "delta"})
            self.assertIn("Successfully replaced text", result.text)
            self.assertIn("delta", path.read_text(encoding="utf-8"))
            self.assertIn("delta", result.details["diff"])


if __name__ == "__main__":
    unittest.main()
