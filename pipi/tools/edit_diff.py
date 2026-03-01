from __future__ import annotations

from difflib import SequenceMatcher, unified_diff


def detect_line_ending(content: str) -> str:
    crlf = content.find("\r\n")
    lf = content.find("\n")
    if lf == -1:
        return "\n"
    if crlf == -1:
        return "\n"
    return "\r\n" if crlf < lf else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u201F": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00A0": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200A": " ",
        "\u202F": " ",
        "\u205F": " ",
        "\u3000": " ",
    }
    normalized_lines = ["".join(replacements.get(char, char) for char in line).rstrip() for line in text.split("\n")]
    return "\n".join(normalized_lines)


def strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def fuzzy_find_text(content: str, old_text: str) -> tuple[bool, int, int, bool, str]:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return True, exact_index, len(old_text), False, content
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old)
    if fuzzy_index == -1:
        return False, -1, 0, False, content
    return True, fuzzy_index, len(fuzzy_old), True, fuzzy_content


def generate_diff_string(old_content: str, new_content: str) -> tuple[str, int | None]:
    diff_lines = list(
        unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=4,
        )
    )
    first_changed_line = None
    matcher = SequenceMatcher(a=old_content.splitlines(), b=new_content.splitlines())
    for opcode, _, _, j1, j2 in matcher.get_opcodes():
        if opcode != "equal":
            first_changed_line = j1 + 1 if j2 >= j1 else None
            break
    return "\n".join(diff_lines), first_changed_line
