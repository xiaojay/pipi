from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from .agent import CodingAgent
from .config import resolve_model_config
from .file_args import process_file_arguments
from .session import SessionManager
from .types import ChatMessage, text_part


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipi",
        description="Python port of the pi coding-agent core.",
    )
    parser.add_argument("messages", nargs="*", help="Prompt text")
    parser.add_argument("--provider", default=None, help="Provider name (openai, openrouter, groq, local)")
    parser.add_argument("--model", default=None, help="Model id")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--thinking", default="medium", help="Thinking level metadata")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt")
    parser.add_argument("--print", "-p", action="store_true", help="Single-shot mode")
    parser.add_argument("--continue", "-c", dest="continue_session", action="store_true", help="Continue latest session")
    parser.add_argument("--session", default=None, help="Open a specific session file")
    parser.add_argument("--session-dir", default=None, help="Custom session directory")
    parser.add_argument("--no-session", action="store_true", help="Disable persistence")
    parser.add_argument("--tools", default=None, help="Comma-separated tool names to keep enabled")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    file_args = [token[1:] for token in unknown if token.startswith("@")]
    extra_args = [token for token in unknown if not token.startswith("@")]
    if extra_args:
        args.messages.extend(extra_args)
    model = resolve_model_config(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        thinking_level=args.thinking,
    )
    session_manager = _create_session_manager(args)
    agent = CodingAgent(
        model=model,
        session_manager=session_manager,
        system_prompt=args.system_prompt,
    )
    if args.tools:
        keep = {item.strip() for item in args.tools.split(",") if item.strip()}
        agent.tools = {name: tool for name, tool in agent.tools.items() if name in keep}
    if args.print or args.messages or file_args:
        initial_parts = process_file_arguments(file_args, session_manager.cwd) if file_args else []
        first_message = " ".join(args.messages).strip()
        result = agent.prompt_text(first_message or "Continue.", extra_parts=initial_parts)
        print(result.assistant_message.text())
        return 0
    return run_repl(agent)


def run_repl(agent: CodingAgent) -> int:
    print("pipi Python port. Type /help for commands, /quit to exit.")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        if raw.startswith("/"):
            if _handle_command(agent, raw):
                return 0
            continue
        if raw.startswith("!!"):
            _run_shell_command(agent, raw[2:].strip(), include_in_context=False)
            continue
        if raw.startswith("!"):
            _run_shell_command(agent, raw[1:].strip(), include_in_context=True)
            continue
        file_args, message = _extract_inline_file_args(raw)
        extra_parts = process_file_arguments(file_args, agent.session_manager.cwd) if file_args else []
        result = agent.prompt_text(message, extra_parts=extra_parts)
        text = result.assistant_message.text().strip()
        if text:
            print(text)


def _handle_command(agent: CodingAgent, raw: str) -> bool:
    parts = raw.split(maxsplit=1)
    command = parts[0]
    argument = parts[1] if len(parts) > 1 else ""
    if command in {"/quit", "/exit"}:
        return True
    if command == "/help":
        print("/help, /quit, /new, /session, /tools")
        print("Inline file references are supported in the REPL via @path/to/file.")
        print("Use !command to run bash and keep the output in context, !!command to skip context.")
        return False
    if command == "/new":
        agent.new_session()
        print("Started a new session.")
        return False
    if command == "/session":
        session_file = agent.session_manager.get_session_file()
        print(f"session_id: {agent.session_manager.get_session_id()}")
        print(f"session_file: {session_file if session_file else '(in-memory)'}")
        return False
    if command == "/tools":
        print(", ".join(sorted(agent.tools)))
        return False
    if command == "/model" and argument:
        model = resolve_model_config(
            provider=agent.model.provider,
            model=argument,
            base_url=agent.model.base_url,
            api_key=agent.model.api_key,
            thinking_level=agent.model.thinking_level,
        )
        agent.set_model(model)
        print(f"Switched to {model.provider}/{model.model}")
        return False
    print(f"Unknown command: {command}")
    return False


def _create_session_manager(args: argparse.Namespace) -> SessionManager:
    cwd = Path.cwd()
    if args.no_session:
        return SessionManager.in_memory(cwd)
    if args.session:
        return SessionManager.open(args.session, args.session_dir)
    if args.continue_session:
        return SessionManager.continue_recent(cwd, args.session_dir)
    return SessionManager.create(cwd, args.session_dir)


def _extract_inline_file_args(raw: str) -> tuple[list[str], str]:
    try:
        parts = shlex.split(raw)
    except ValueError:
        return [], raw
    files = [part[1:] for part in parts if part.startswith("@")]
    message_parts = [part for part in parts if not part.startswith("@")]
    return files, " ".join(message_parts)


def _run_shell_command(agent: CodingAgent, command: str, *, include_in_context: bool) -> None:
    if not command:
        print("Missing shell command.")
        return
    bash_tool = agent.tools.get("bash")
    if not bash_tool:
        print("bash tool is disabled.")
        return
    try:
        result = bash_tool.execute({"command": command})
    except Exception as error:
        print(error)
        return
    output = result.text.strip()
    if output:
        print(output)
    if include_in_context:
        message = ChatMessage.user(
            [
                text_part(
                    "Ran shell command:\n"
                    f"`{command}`\n\n"
                    "```text\n"
                    f"{output}\n"
                    "```"
                )
            ]
        )
        agent.session_manager.append_message(message)
