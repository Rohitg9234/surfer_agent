"""Markdown site/task playbooks (Firecrawl skill pattern, local files, no Firecrawl API)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


def _parse(text: str) -> tuple[dict[str, object], str]:
    meta: dict[str, object] = {}
    body = text
    if not text.startswith("---"):
        return meta, body.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return meta, body.strip()
    fm, body = parts[1], parts[2]
    key: str | None = None
    for raw in fm.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- ") and key == "domains":
            domains = meta.setdefault("domains", [])
            if isinstance(domains, list):
                domains.append(line[2:].strip())
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            key = k.strip()
            val = v.strip().strip('"')
            if key == "domains" and not val:
                meta["domains"] = []
            else:
                meta[key] = val
    return meta, body.strip()


@dataclass
class Playbook:
    name: str
    description: str
    body: str
    domains: list[str]


class PlaybookStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).parent / "playbooks"
        self.by_name: dict[str, Playbook] = {}
        self.by_domain: dict[str, Playbook] = {}
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.rglob("*.md")):
            meta, body = _parse(path.read_text(encoding="utf-8"))
            name = str(meta.get("name") or path.stem)
            pb = Playbook(
                name=name,
                description=str(meta.get("description") or ""),
                body=body,
                domains=[d.lower() for d in (meta.get("domains") or []) if isinstance(d, str)],
            )
            self.by_name[name] = pb
            for domain in pb.domains:
                self.by_domain[domain] = pb

    def catalog(self) -> str:
        if not self.by_name:
            return "(no playbooks)"
        lines = []
        for pb in self.by_name.values():
            extra = f" (sites: {', '.join(pb.domains)})" if pb.domains else ""
            lines.append(f"- {pb.name}: {pb.description}{extra}")
        return "\n".join(lines)

    def get(self, name: str) -> Playbook | None:
        return self.by_name.get(name)

    def for_url(self, url: str) -> Playbook | None:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return None
        if host in self.by_domain:
            return self.by_domain[host]
        stripped = host.removeprefix("www.")
        if stripped in self.by_domain:
            return self.by_domain[stripped]
        for domain, pb in self.by_domain.items():
            if host.endswith("." + domain) or stripped.endswith("." + domain):
                return pb
        return None
