#!/usr/bin/env python3
"""Browse and view pipi sessions as HTML in your browser.

Usage:
    python3 scripts/view_sessions.py          # list recent sessions, pick one
    python3 scripts/view_sessions.py <path>   # view a specific .jsonl file
"""

import base64
import json
import os
import sys
import tempfile
import webbrowser
from datetime import datetime, timezone
from pathlib import Path


# ─── Session Discovery ────────────────────────────────────────────────────────

def get_sessions_dir() -> Path:
    return Path.home() / ".pi" / "agent" / "sessions"


def load_session_meta(jsonl_path: Path) -> dict | None:
    """Read the first few lines to extract session metadata."""
    header = None
    model = None
    first_user_msg = None

    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if header is None:
                    header = entry
                    continue

                etype = entry.get("type")
                if etype == "model_change" and model is None:
                    model = f"{entry.get('provider', '')} / {entry.get('modelId', '')}"

                if etype == "message" and first_user_msg is None:
                    msg = entry.get("message", {})
                    if msg.get("role") == "user":
                        for c in msg.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "text":
                                first_user_msg = c["text"][:120].replace("\n", " ")
                                break

                if model and first_user_msg:
                    break
    except OSError:
        return None

    if not header:
        return None

    ts_str = header.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_str)
    except Exception:
        ts = datetime.min.replace(tzinfo=timezone.utc)

    return {
        "path": jsonl_path,
        "session_id": header.get("id", ""),
        "timestamp": ts,
        "timestamp_str": ts_str,
        "cwd": header.get("cwd", ""),
        "model": model or "unknown",
        "preview": first_user_msg or "(empty session)",
    }


def find_recent_sessions(n: int = 10) -> list[dict]:
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []

    all_sessions = []
    for jsonl_file in sessions_dir.rglob("*.jsonl"):
        meta = load_session_meta(jsonl_file)
        if meta:
            all_sessions.append(meta)

    all_sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return all_sessions[:n]


# ─── JSONL Parsing ────────────────────────────────────────────────────────────

def parse_session_file(path: Path) -> tuple[dict, list]:
    """Return (session_header, ordered_branch_entries)."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not entries:
        return {}, []

    header = entries[0] if entries[0].get("type") == "session" else {}

    # Build id → entry map
    by_id: dict[str, dict] = {}
    for e in entries[1:]:
        eid = e.get("id")
        if eid:
            by_id[eid] = e

    if not by_id:
        return header, []

    # Find leaf (entry whose id is not anyone's parentId)
    has_children: set[str] = set()
    for e in by_id.values():
        pid = e.get("parentId")
        if pid:
            has_children.add(pid)

    leaves = [e for e in by_id.values() if e["id"] not in has_children]
    if not leaves:
        return header, list(by_id.values())

    # Take the most recently timestamped leaf
    leaf = max(leaves, key=lambda e: e.get("timestamp", ""))

    # Walk back to root
    chain: list[dict] = []
    current: dict | None = leaf
    visited: set[str] = set()
    while current and current["id"] not in visited:
        visited.add(current["id"])
        chain.append(current)
        pid = current.get("parentId")
        current = by_id.get(pid) if pid else None

    return header, list(reversed(chain))


# ─── HTML Generation ──────────────────────────────────────────────────────────

def he(text: str) -> str:
    """HTML-escape a string."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def fmt_ts(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_str).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        if isinstance(c, dict):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "image":
                parts.append("[image]")
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def render_tool_calls(tool_calls: list) -> str:
    if not tool_calls:
        return ""
    items = []
    for tc in tool_calls:
        name = he(tc.get("name", ""))
        try:
            args_raw = tc.get("arguments") or "{}"
            args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            args_pretty = json.dumps(args_obj, indent=2, ensure_ascii=False)
        except Exception:
            args_pretty = str(tc.get("arguments", ""))
        items.append(f"""
        <div class="tool-call">
          <details>
            <summary><span class="tc-icon">⚡</span> <code>{name}</code></summary>
            <pre class="tool-args">{he(args_pretty)}</pre>
          </details>
        </div>""")
    return "\n".join(items)


def render_message_content(content) -> str:
    text = content_to_text(content)
    encoded = b64(text)
    return f'<div class="md-content" data-md="{encoded}"></div>'


def build_messages_html(branch: list) -> str:
    parts = []
    msg_index = 0

    for entry in branch:
        etype = entry.get("type")

        if etype == "model_change":
            provider = he(entry.get("provider", ""))
            model_id = he(entry.get("modelId", ""))
            ts = he(fmt_ts(entry.get("timestamp", "")))
            parts.append(f"""
    <div class="sys-event">
      <span class="sys-icon">⚙</span>
      <span>Model set to <strong>{provider} / {model_id}</strong></span>
      <span class="sys-time">{ts}</span>
    </div>""")

        elif etype == "thinking_level_change":
            level = he(entry.get("thinkingLevel", ""))
            parts.append(f"""
    <div class="sys-event">
      <span class="sys-icon">🧠</span>
      <span>Thinking level: <strong>{level}</strong></span>
    </div>""")

        elif etype == "message":
            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", [])
            tool_calls = msg.get("tool_calls") or []
            ts = he(fmt_ts(entry.get("timestamp", "")))
            msg_index += 1

            if role == "user":
                parts.append(f"""
    <div class="message user-msg">
      <div class="msg-meta">
        <span class="badge badge-user">User</span>
        <span class="msg-time">{ts}</span>
      </div>
      <div class="msg-body">
        {render_message_content(content)}
      </div>
    </div>""")

            elif role == "assistant":
                model_label = he(msg.get("model") or "")
                usage = msg.get("usage") or {}
                usage_html = ""
                if usage:
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)
                    usage_html = f'<span class="usage">↑{pt} ↓{ct} tok</span>'

                tool_calls_html = render_tool_calls(tool_calls)
                content_html = render_message_content(content)

                parts.append(f"""
    <div class="message asst-msg">
      <div class="msg-meta">
        <span class="badge badge-asst">Assistant</span>
        {f'<span class="model-tag">{model_label}</span>' if model_label else ''}
        <span class="msg-time">{ts}</span>
        {usage_html}
      </div>
      <div class="msg-body">
        {content_html}
        {tool_calls_html}
      </div>
    </div>""")

            elif role == "toolResult":
                tool_name = he(msg.get("tool_name") or "")
                is_error = bool(msg.get("error_message"))
                icon = "✗" if is_error else "✓"
                cls = "tool-result error-result" if is_error else "tool-result"
                result_text = content_to_text(content)

                parts.append(f"""
    <div class="{cls}">
      <details>
        <summary>
          <span class="tr-icon">{icon}</span>
          Tool result: <code>{tool_name}</code>
        </summary>
        <pre class="tool-output">{he(result_text)}</pre>
      </details>
    </div>""")

    return "\n".join(parts)


CSS = """
  :root {
    --bg: #f5f5f7;
    --surface: #ffffff;
    --border: #e2e8f0;
    --text: #1a202c;
    --text-muted: #718096;
    --user-bg: #ebf4ff;
    --user-border: #4299e1;
    --asst-bg: #ffffff;
    --asst-border: #e2e8f0;
    --tool-bg: #f7fafc;
    --tool-border: #cbd5e0;
    --err-bg: #fff5f5;
    --err-border: #fc8181;
    --badge-user: #3182ce;
    --badge-asst: #38a169;
    --accent: #4a5568;
    --header-bg: #2d3748;
    --header-text: #f7fafc;
    --code-bg: #f0f4f8;
    --radius: 10px;
    --shadow: 0 1px 4px rgba(0,0,0,.08);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    font-size: 15px;
  }

  /* ── Header ── */
  .header {
    background: var(--header-bg);
    color: var(--header-text);
    padding: 20px 32px;
    position: sticky;
    top: 0;
    z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,.3);
  }
  .header h1 {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: .01em;
    margin-bottom: 6px;
  }
  .header h1 code {
    font-size: .9rem;
    opacity: .75;
    font-family: "SF Mono", "Fira Code", monospace;
  }
  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    font-size: .8rem;
    opacity: .7;
  }
  .meta-row span { display: flex; align-items: center; gap: 4px; }

  /* ── Main ── */
  main {
    max-width: 860px;
    margin: 0 auto;
    padding: 28px 20px 60px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ── System events ── */
  .sys-event {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: .78rem;
    color: var(--text-muted);
    padding: 4px 12px;
    border-left: 2px solid var(--border);
    margin-left: 8px;
  }
  .sys-icon { font-size: .9rem; }
  .sys-time { margin-left: auto; font-size: .72rem; }

  /* ── Messages ── */
  .message {
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .user-msg {
    background: var(--user-bg);
    border: 1px solid var(--user-border);
    border-left: 3px solid var(--user-border);
    margin-left: 40px;
  }
  .asst-msg {
    background: var(--asst-bg);
    border: 1px solid var(--asst-border);
    border-left: 3px solid var(--badge-asst);
    margin-right: 40px;
  }
  .msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    font-size: .78rem;
    background: rgba(0,0,0,.025);
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    color: #fff;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
  }
  .badge-user { background: var(--badge-user); }
  .badge-asst { background: var(--badge-asst); }
  .model-tag {
    color: var(--text-muted);
    font-size: .72rem;
    font-family: "SF Mono", "Fira Code", monospace;
  }
  .msg-time { color: var(--text-muted); font-size: .72rem; }
  .usage { margin-left: auto; color: var(--text-muted); font-size: .72rem; }

  .msg-body {
    padding: 14px 18px;
  }

  /* ── Markdown content ── */
  .md-content p { margin-bottom: .75em; }
  .md-content p:last-child { margin-bottom: 0; }
  .md-content h1,.md-content h2,.md-content h3 {
    margin: 1em 0 .4em;
    font-weight: 600;
    line-height: 1.3;
  }
  .md-content h1 { font-size: 1.3em; }
  .md-content h2 { font-size: 1.15em; }
  .md-content h3 { font-size: 1em; }
  .md-content ul, .md-content ol {
    margin: .5em 0 .75em 1.4em;
  }
  .md-content li { margin-bottom: .25em; }
  .md-content pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    overflow-x: auto;
    margin: .75em 0;
    font-size: .85em;
  }
  .md-content code {
    font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: .88em;
  }
  .md-content p > code {
    background: var(--code-bg);
    padding: .1em .35em;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .md-content blockquote {
    border-left: 3px solid var(--border);
    margin: .5em 0;
    padding: .25em 0 .25em 1em;
    color: var(--text-muted);
  }
  .md-content table {
    border-collapse: collapse;
    width: 100%;
    margin: .75em 0;
    font-size: .9em;
  }
  .md-content th, .md-content td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
  }
  .md-content th { background: var(--code-bg); font-weight: 600; }
  .md-content a { color: var(--badge-user); text-decoration: none; }
  .md-content a:hover { text-decoration: underline; }

  /* ── Tool calls ── */
  .tool-call {
    margin-top: 10px;
    border: 1px solid var(--tool-border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--tool-bg);
  }
  .tool-call details > summary {
    padding: 7px 12px;
    cursor: pointer;
    font-size: .82rem;
    font-weight: 500;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tool-call details > summary:hover { background: rgba(0,0,0,.04); }
  .tc-icon { font-size: .9rem; }
  .tool-args {
    padding: 10px 14px;
    font-size: .8rem;
    font-family: "SF Mono", "Fira Code", monospace;
    overflow-x: auto;
    border-top: 1px solid var(--tool-border);
    background: #fff;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── Tool results ── */
  .tool-result {
    border: 1px solid var(--tool-border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--tool-bg);
    box-shadow: var(--shadow);
  }
  .error-result {
    background: var(--err-bg);
    border-color: var(--err-border);
  }
  .tool-result details > summary {
    padding: 9px 14px;
    cursor: pointer;
    font-size: .82rem;
    font-weight: 500;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tool-result details > summary:hover { background: rgba(0,0,0,.04); }
  .tr-icon { font-size: .9rem; }
  .tool-output {
    padding: 10px 14px;
    font-size: .8rem;
    font-family: "SF Mono", "Fira Code", monospace;
    overflow-x: auto;
    border-top: 1px solid var(--tool-border);
    background: #fff;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
  }

  /* ── Highlight.js overrides ── */
  .hljs { background: transparent !important; padding: 0 !important; }

  @media (max-width: 600px) {
    .user-msg { margin-left: 0; }
    .asst-msg { margin-right: 0; }
    main { padding: 16px 12px 40px; }
  }
"""

JS = """
  // Decode base64 → UTF-8
  function b64decode(s) {
    const bytes = Uint8Array.from(atob(s), c => c.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  }

  document.querySelectorAll('.md-content[data-md]').forEach(el => {
    const raw = b64decode(el.dataset.md);
    el.innerHTML = marked.parse(raw);
  });

  // Syntax-highlight all code blocks rendered by marked
  document.querySelectorAll('.md-content pre code').forEach(el => {
    hljs.highlightElement(el);
  });

  // Also highlight standalone pre blocks (tool args, tool output)
  document.querySelectorAll('pre.tool-args, pre.tool-output').forEach(el => {
    hljs.highlightElement(el);
  });
"""


def generate_html(header: dict, branch: list, meta: dict) -> str:
    session_ts = he(fmt_ts(header.get("timestamp", "")))
    cwd = he(header.get("cwd", ""))
    session_id = he(header.get("id", "")[:8])
    model = he(meta.get("model", ""))
    messages_html = build_messages_html(branch)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Session {session_id}</title>
  <link rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <style>{CSS}</style>
</head>
<body>

  <div class="header">
    <h1>Session <code>{session_id}</code></h1>
    <div class="meta-row">
      <span>📁 {cwd}</span>
      <span>🕐 {session_ts}</span>
      <span>🤖 {model}</span>
    </div>
  </div>

  <main>
{messages_html}
  </main>

  <script>
{JS}
  </script>
</body>
</html>"""


# ─── CLI ──────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def print_session_list(sessions: list[dict]) -> None:
    print(f"\n{BOLD}Recent Sessions{RESET}  ({len(sessions)} found)\n")
    print(f"{'─'*72}")
    for i, s in enumerate(sessions, 1):
        ts_local = fmt_ts(s["timestamp_str"])
        cwd_short = s["cwd"].replace(str(Path.home()), "~")
        preview = s["preview"]
        if len(preview) > 80:
            preview = preview[:77] + "…"
        idx = f"{CYAN}{i:2}.{RESET}"
        print(f"{idx} {BOLD}{ts_local}{RESET}  {DIM}{cwd_short}{RESET}")
        print(f"     {GREEN}{s['model']}{RESET}")
        print(f"     {DIM}{preview}{RESET}")
        print()
    print(f"{'─'*72}")


def prompt_selection(n: int) -> int:
    while True:
        try:
            raw = input(f"\nSelect session [1–{n}] (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        try:
            choice = int(raw)
            if 1 <= choice <= n:
                return choice
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {n}.")


def open_html(html: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as f:
        f.write(html)
        tmp_path = f.name
    print(f"\n{DIM}Generated: {tmp_path}{RESET}")
    webbrowser.open(f"file://{tmp_path}")
    print(f"{GREEN}Opened in browser.{RESET}\n")


def main() -> None:
    # Direct file argument
    if len(sys.argv) == 2:
        path = Path(sys.argv[1]).expanduser().resolve()
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        meta = load_session_meta(path) or {"model": "unknown", "preview": ""}
        header, branch = parse_session_file(path)
        html = generate_html(header, branch, meta)
        open_html(html)
        return

    # Discover sessions
    sessions = find_recent_sessions(10)
    if not sessions:
        print(
            "No sessions found.\n"
            f"Expected sessions under: {get_sessions_dir()}"
        )
        sys.exit(1)

    print_session_list(sessions)
    choice = prompt_selection(len(sessions))
    selected = sessions[choice - 1]

    print(f"\n{DIM}Parsing {selected['path'].name}…{RESET}")
    header, branch = parse_session_file(selected["path"])
    html = generate_html(header, branch, selected)
    open_html(html)


if __name__ == "__main__":
    main()
