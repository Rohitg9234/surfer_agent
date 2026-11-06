"""Shrink page text for the executor prompt. Full snapshot stays on the browser object."""
from __future__ import annotations

import re

_NAME = re.compile(r'"([^"]{81,})"')
_REF_LINE = re.compile(r"\[\d+\]")
_KEEP_PREFIX = re.compile(
    r"^(heading |frames? |iframe |block_page:|text |visible_text:|signed_out:|---|"
    r"form_dialog:|open_dialog:|form_field:|popup_title:)",
    re.I,
)
_CONTENT_ROLE = re.compile(r"\]\s+(article|listitem|paragraph|heading)\b", re.I)
_QUOTED = re.compile(r'"([^"]*)"')
_HEADING = re.compile(r"^(heading |\[[^\]]+\]\s+heading\b)", re.I)


def slim_snapshot(text: str, max_chars: int = 8000) -> str:
    visible: list[str] = []
    paragraphs: list[str] = []
    headings: list[str] = []
    other_content: list[str] = []
    chrome: list[str] = []
    in_visible = False
    for raw in text.splitlines():
        line = _NAME.sub(lambda m: '"' + m.group(1)[:77] + '..."', raw)
        stripped = line.strip()
        if stripped.startswith("visible_text:"):
            in_visible = True
            visible.append(line)
            continue
        if in_visible:
            if (
                _REF_LINE.search(line)
                or _KEEP_PREFIX.search(stripped)
                or stripped.startswith("URL:")
                or stripped.startswith("signed_out:")
                or stripped.startswith("form_dialog:")
                or stripped.startswith("block_page:")
                or stripped.startswith("frames")
            ):
                in_visible = False
            else:
                visible.append(line)
                continue
        if not (
            _REF_LINE.search(line)
            or _KEEP_PREFIX.search(stripped)
            or line.startswith("URL:")
        ):
            continue
        if _is_content_line(line):
            if _HEADING.search(stripped):
                headings.append(line)
            elif re.search(r"\]\s+(paragraph|article|listitem)\b|^text ", stripped, re.I):
                paragraphs.append(line)
            else:
                other_content.append(line)
        else:
            chrome.append(line)
    vis_join = "\n".join(visible)
    if len(vis_join) > 3600:
        visible = (vis_join[:3600].rstrip() + "\n…(visible_text truncated)").split("\n")
    chrome = _collapse_repeats(chrome)[:12]
    headings = headings[:18]
    other_content = other_content[:50]
    ordered = visible + paragraphs + headings + other_content
    if chrome:
        ordered += ["--- chrome ---"] + chrome
    out = "\n".join(ordered)
    if len(out) <= max_chars:
        return out
    return out[: max_chars - 20].rstrip() + "\n…(truncated)"


def _is_content_line(line: str) -> bool:
    stripped = line.strip()
    if _KEEP_PREFIX.search(stripped) or stripped.startswith("URL:"):
        return True
    if _CONTENT_ROLE.search(line):
        return True
    quoted = _QUOTED.search(line)
    return bool(quoted and len(quoted.group(1)) >= 48)


def _collapse_repeats(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        role = _role_key(lines[i])
        run = 1
        if role:
            while i + run < len(lines) and _role_key(lines[i + run]) == role:
                run += 1
        if run >= 5:
            out.extend(lines[i : i + 3])
            out.append(f"…{run - 3} similar {role} rows")
            i += run
        else:
            out.append(lines[i])
            i += 1
    return out


def _role_key(line: str) -> str:
    m = re.search(r"\]\s+(\S+)", line)
    return (m.group(1).lower() if m else "") or ""
