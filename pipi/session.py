from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_default_session_dir
from .types import ChatMessage, text_part

CURRENT_SESSION_VERSION = 3


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id(existing: set[str]) -> str:
    for _ in range(100):
        value = uuid.uuid4().hex[:8]
        if value not in existing:
            return value
    return uuid.uuid4().hex


@dataclass
class SessionContext:
    messages: list[ChatMessage]
    thinking_level: str
    model: dict[str, str] | None


class SessionManager:
    def __init__(self, cwd: str | Path, session_dir: Path, session_file: Path | None, persist: bool) -> None:
        self.cwd = str(Path(cwd).resolve())
        self.session_dir = session_dir
        self.persist = persist
        self.session_dir.mkdir(parents=True, exist_ok=True) if self.persist else None
        self.session_id = ""
        self.session_file = session_file
        self.file_entries: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.leaf_id: str | None = None
        if session_file:
            self.set_session_file(session_file)
        else:
            self.new_session()

    @classmethod
    def create(cls, cwd: str | Path, session_dir: str | Path | None = None) -> "SessionManager":
        resolved_dir = Path(session_dir).expanduser() if session_dir else get_default_session_dir(cwd)
        return cls(cwd, resolved_dir, None, True)

    @classmethod
    def open(cls, path: str | Path, session_dir: str | Path | None = None) -> "SessionManager":
        resolved_path = Path(path).expanduser().resolve()
        entries = cls._load_entries_from_file(resolved_path)
        header = next((entry for entry in entries if entry.get("type") == "session"), None)
        cwd = header.get("cwd") if header else str(Path.cwd())
        resolved_dir = Path(session_dir).expanduser() if session_dir else resolved_path.parent
        return cls(cwd, resolved_dir, resolved_path, True)

    @classmethod
    def continue_recent(cls, cwd: str | Path, session_dir: str | Path | None = None) -> "SessionManager":
        resolved_dir = Path(session_dir).expanduser() if session_dir else get_default_session_dir(cwd)
        most_recent = cls._find_most_recent_session(resolved_dir)
        if most_recent:
            return cls(cwd, resolved_dir, most_recent, True)
        return cls(cwd, resolved_dir, None, True)

    @classmethod
    def in_memory(cls, cwd: str | Path | None = None) -> "SessionManager":
        return cls(cwd or Path.cwd(), Path("."), None, False)

    @classmethod
    def list(cls, cwd: str | Path, session_dir: str | Path | None = None) -> list[dict[str, Any]]:
        resolved_dir = Path(session_dir).expanduser() if session_dir else get_default_session_dir(cwd)
        return cls._list_sessions_from_dir(resolved_dir)

    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        root = get_default_session_dir(Path.home()).parent
        sessions: list[dict[str, Any]] = []
        if not root.exists():
            return sessions
        for entry in root.iterdir():
            if entry.is_dir():
                sessions.extend(cls._list_sessions_from_dir(entry))
        sessions.sort(key=lambda item: item["modified"], reverse=True)
        return sessions

    def new_session(self, *, parent_session: str | None = None) -> Path | None:
        self.session_id = str(uuid.uuid4())
        timestamp = _iso_now()
        header = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self.session_id,
            "timestamp": timestamp,
            "cwd": self.cwd,
            "parentSession": parent_session,
        }
        self.file_entries = [header]
        self.by_id = {}
        self.leaf_id = None
        if self.persist:
            filename = timestamp.replace(":", "-").replace(".", "-") + f"_{self.session_id}.jsonl"
            self.session_file = self.session_dir / filename
            self._rewrite_file()
        else:
            self.session_file = None
        return self.session_file

    def set_session_file(self, session_file: str | Path) -> None:
        self.session_file = Path(session_file).expanduser().resolve()
        if self.session_file.exists():
            self.file_entries = self._load_entries_from_file(self.session_file)
            if not self.file_entries:
                self.new_session()
                if self.persist and self.session_file:
                    self._rewrite_file()
                return
            header = next((entry for entry in self.file_entries if entry.get("type") == "session"), None)
            self.session_id = header.get("id") if header else str(uuid.uuid4())
            self._build_index()
        else:
            self.new_session()
            self.session_file = Path(session_file).expanduser().resolve()
            if self.persist:
                self._rewrite_file()

    def get_session_id(self) -> str:
        return self.session_id

    def get_session_file(self) -> Path | None:
        return self.session_file

    def get_leaf_id(self) -> str | None:
        return self.leaf_id

    def get_entries(self) -> list[dict[str, Any]]:
        return [entry for entry in self.file_entries if entry.get("type") != "session"]

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        return self.by_id.get(entry_id)

    def get_branch(self, from_id: str | None = None) -> list[dict[str, Any]]:
        start_id = from_id if from_id is not None else self.leaf_id
        current = self.by_id.get(start_id) if start_id else None
        path: list[dict[str, Any]] = []
        while current:
            path.insert(0, current)
            parent_id = current.get("parentId")
            current = self.by_id.get(parent_id) if parent_id else None
        return path

    def branch(self, branch_from_id: str) -> None:
        if branch_from_id not in self.by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self.leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        self.leaf_id = None

    def append_message(self, message: ChatMessage) -> str:
        entry_id = self._append_entry(
            {
                "type": "message",
                "message": message.to_dict(),
            }
        )
        return entry_id

    def append_model_change(self, provider: str, model_id: str) -> str:
        return self._append_entry({"type": "model_change", "provider": provider, "modelId": model_id})

    def append_thinking_level_change(self, thinking_level: str) -> str:
        return self._append_entry({"type": "thinking_level_change", "thinkingLevel": thinking_level})

    def append_branch_summary(self, summary: str, from_id: str | None) -> str:
        return self._append_entry({"type": "branch_summary", "fromId": from_id or "root", "summary": summary})

    def append_session_info(self, name: str) -> str:
        return self._append_entry({"type": "session_info", "name": name.strip()})

    def build_session_context(self) -> SessionContext:
        entries = self.get_branch()
        messages: list[ChatMessage] = []
        thinking_level = "off"
        model: dict[str, str] | None = None
        compaction: dict[str, Any] | None = None
        for entry in entries:
            entry_type = entry["type"]
            if entry_type == "thinking_level_change":
                thinking_level = str(entry["thinkingLevel"])
            elif entry_type == "model_change":
                model = {"provider": str(entry["provider"]), "modelId": str(entry["modelId"])}
            elif entry_type == "message":
                message = ChatMessage.from_dict(entry["message"])
                if message.role == "assistant" and message.provider and message.model:
                    model = {"provider": message.provider, "modelId": message.model}
            elif entry_type == "compaction":
                compaction = entry

        def append_visible_message(entry: dict[str, Any]) -> None:
            entry_type = entry["type"]
            if entry_type == "message":
                messages.append(ChatMessage.from_dict(entry["message"]))
            elif entry_type == "branch_summary":
                messages.append(
                    ChatMessage.user(
                        [
                            text_part(
                                "The following is a summary of a branch that this conversation came back from:\n\n"
                                "<summary>\n"
                                f"{entry['summary']}\n"
                                "</summary>"
                            )
                        ]
                    )
                )

        if compaction:
            messages.append(
                ChatMessage.user(
                    [
                        text_part(
                            "The conversation history before this point was compacted into the following summary:\n\n"
                            "<summary>\n"
                            f"{compaction['summary']}\n"
                            "</summary>"
                        )
                    ]
                )
            )
            compaction_id = compaction["id"]
            compaction_index = next(index for index, item in enumerate(entries) if item["id"] == compaction_id)
            first_kept_id = compaction.get("firstKeptEntryId")
            include = False
            for entry in entries[:compaction_index]:
                if entry["id"] == first_kept_id:
                    include = True
                if include:
                    append_visible_message(entry)
            for entry in entries[compaction_index + 1 :]:
                append_visible_message(entry)
        else:
            for entry in entries:
                append_visible_message(entry)
        return SessionContext(messages=messages, thinking_level=thinking_level, model=model)

    def _append_entry(self, payload: dict[str, Any]) -> str:
        existing = set(self.by_id)
        entry_id = _generate_id(existing)
        entry = {
            "id": entry_id,
            "parentId": self.leaf_id,
            "timestamp": _iso_now(),
            **payload,
        }
        self.file_entries.append(entry)
        self.by_id[entry_id] = entry
        self.leaf_id = entry_id
        self._persist_entry(entry)
        return entry_id

    def _persist_entry(self, entry: dict[str, Any]) -> None:
        if not self.persist or not self.session_file:
            return
        with self.session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rewrite_file(self) -> None:
        if not self.persist or not self.session_file:
            return
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with self.session_file.open("w", encoding="utf-8") as handle:
            for entry in self.file_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _build_index(self) -> None:
        self.by_id = {}
        self.leaf_id = None
        for entry in self.file_entries:
            if entry.get("type") == "session":
                continue
            self.by_id[str(entry["id"])] = entry
            self.leaf_id = str(entry["id"])

    @staticmethod
    def _load_entries_from_file(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not entries:
            return []
        if entries[0].get("type") != "session":
            return []
        return entries

    @staticmethod
    def _find_most_recent_session(session_dir: Path) -> Path | None:
        if not session_dir.exists():
            return None
        files = [entry for entry in session_dir.glob("*.jsonl") if entry.is_file()]
        if not files:
            return None
        return max(files, key=lambda item: item.stat().st_mtime)

    @classmethod
    def _list_sessions_from_dir(cls, session_dir: Path) -> list[dict[str, Any]]:
        if not session_dir.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for file_path in session_dir.glob("*.jsonl"):
            try:
                entries = cls._load_entries_from_file(file_path)
                if not entries:
                    continue
                header = entries[0]
                messages = [
                    ChatMessage.from_dict(entry["message"])
                    for entry in entries[1:]
                    if entry.get("type") == "message"
                ]
                first_user = next((message.text() for message in messages if message.role == "user"), "")
                sessions.append(
                    {
                        "path": str(file_path),
                        "id": header["id"],
                        "cwd": header.get("cwd", ""),
                        "created": file_path.stat().st_ctime,
                        "modified": file_path.stat().st_mtime,
                        "message_count": len(messages),
                        "first_message": first_user,
                    }
                )
            except OSError:
                continue
        sessions.sort(key=lambda item: item["modified"], reverse=True)
        return sessions
