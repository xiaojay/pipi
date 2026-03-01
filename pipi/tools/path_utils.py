from __future__ import annotations

from pathlib import Path
import unicodedata

UNICODE_SPACES = {
    "\u00A0",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200A",
    "\u202F",
    "\u205F",
    "\u3000",
}
NARROW_NO_BREAK_SPACE = "\u202F"


def _normalize_unicode_spaces(value: str) -> str:
    return "".join(" " if char in UNICODE_SPACES else char for char in value)


def _normalize_at_prefix(value: str) -> str:
    return value[1:] if value.startswith("@") else value


def _try_macos_screenshot_path(file_path: Path) -> Path:
    return Path(str(file_path).replace(" AM.", f"{NARROW_NO_BREAK_SPACE}AM.").replace(" PM.", f"{NARROW_NO_BREAK_SPACE}PM."))


def _try_nfd_variant(file_path: Path) -> Path:
    return Path(unicodedata.normalize("NFD", str(file_path)))


def _try_curly_quote_variant(file_path: Path) -> Path:
    return Path(str(file_path).replace("'", "\u2019"))


def expand_path(file_path: str) -> Path:
    normalized = _normalize_unicode_spaces(_normalize_at_prefix(file_path))
    return Path(normalized).expanduser()


def resolve_to_cwd(file_path: str, cwd: str | Path) -> Path:
    expanded = expand_path(file_path)
    if expanded.is_absolute():
        return expanded
    return (Path(cwd) / expanded).resolve()


def resolve_read_path(file_path: str, cwd: str | Path) -> Path:
    resolved = resolve_to_cwd(file_path, cwd)
    if resolved.exists():
        return resolved
    variants = [
        _try_macos_screenshot_path(resolved),
        _try_curly_quote_variant(resolved),
    ]
    for variant in variants:
        if variant.exists():
            return variant
    return resolved
