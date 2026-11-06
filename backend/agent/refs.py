"""Structured locators. The LLM still sees [1], [2]; we persist role/name/frame, never the integer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ARIA_REF = re.compile(r"\[ref=([^\]]+)\]")
_LINE = re.compile(
    r'^(?P<indent>\s*)-\s+(?P<role>\S+)\s+(?:"(?P<name>[^"]*)")?.*\[ref=(?P<aria>[^\]]+)\]',
    re.M,
)
_DOM_LINE = re.compile(r'^\[(\d+)\]\s+(\S+)\s+"(.*)"\s*$', re.M)


@dataclass
class ElementRef:
    n: int
    aria: str = ""
    role: str = ""
    name: str = ""
    nth: int = 0
    frame_path: str = ""
    ancestor_context: str = ""

    def descriptor(self) -> dict[str, str | int]:
        return {
            "role": self.role,
            "name": self.name,
            "nth": self.nth,
            "frame_path": self.frame_path,
            "ancestor_context": self.ancestor_context,
        }


def rewrite_aria_refs(snapshot: str) -> tuple[str, dict[int, str]]:
    """Turn Playwright `[ref=e5]` markers into sequential `[1]`, `[2]` for the LLM."""
    text, refs = parse_aria_snapshot(snapshot)
    return text, {n: e.aria for n, e in refs.items() if e.aria}


def parse_aria_snapshot(snapshot: str) -> tuple[str, dict[int, ElementRef]]:
    seen: list[str] = []
    known: set[str] = set()
    for raw in _ARIA_REF.findall(snapshot):
        if raw not in known:
            known.add(raw)
            seen.append(raw)
    imap = {raw: i + 1 for i, raw in enumerate(seen)}
    text = _ARIA_REF.sub(lambda m: f"[{imap[m.group(1)]}]", snapshot)

    refs: dict[int, ElementRef] = {}
    counts: dict[tuple[str, str, str], int] = {}
    frames: list[tuple[int, str]] = []
    ancestors: list[tuple[int, str]] = []
    for m in _LINE.finditer(snapshot):
        indent = len(m.group("indent") or "")
        role = (m.group("role") or "").lower()
        name = m.group("name") or ""
        aria = m.group("aria")
        n = imap.get(aria)
        if n is None:
            continue
        while frames and frames[-1][0] >= indent:
            frames.pop()
        while ancestors and ancestors[-1][0] >= indent:
            ancestors.pop()
        frame_path = "/".join(f[1] for f in frames)
        ancestor = " > ".join(a[1] for a in ancestors[-3:])
        key = (role, name.lower(), frame_path)
        nth = counts.get(key, 0)
        counts[key] = nth + 1
        refs[n] = ElementRef(
            n=n, aria=aria, role=role, name=name, nth=nth,
            frame_path=frame_path, ancestor_context=ancestor,
        )
        label = f'{role} "{name}"' if name else role
        ancestors.append((indent, label))
        if role in {"iframe", "frame"}:
            frames.append((indent, name or aria))
    return text, refs


def parse_dom_snapshot(text: str) -> dict[int, ElementRef]:
    refs: dict[int, ElementRef] = {}
    counts: dict[tuple[str, str], int] = {}
    for m in _DOM_LINE.finditer(text):
        n, role, name = int(m.group(1)), m.group(2).lower(), m.group(3)
        key = (role, name.lower())
        nth = counts.get(key, 0)
        counts[key] = nth + 1
        refs[n] = ElementRef(n=n, role=role, name=name, nth=nth)
    return refs


def strip_ref_args(args: dict, descriptors: dict[int, dict]) -> dict:
    out = dict(args)
    raw = out.pop("ref", None)
    if raw is None:
        return out
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return args
    desc = descriptors.get(n)
    if desc:
        out.update({k: v for k, v in desc.items() if v not in ("", None)})
    return out
