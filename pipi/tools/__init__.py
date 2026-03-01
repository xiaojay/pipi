from __future__ import annotations

from .bash_tool import create_bash_tool
from .base import ToolDefinition
from .edit_tool import create_edit_tool
from .find_tool import create_find_tool
from .grep_tool import create_grep_tool
from .ls_tool import create_ls_tool
from .read_tool import create_read_tool
from .write_tool import create_write_tool


def build_builtin_tools(cwd: str) -> dict[str, ToolDefinition]:
    tools = {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "grep": create_grep_tool(cwd),
        "find": create_find_tool(cwd),
        "ls": create_ls_tool(cwd),
    }
    return tools
