"""Short-term (this conversation/run) and long-term (disk) memory."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .refs import strip_ref_args
from .types import Step, ToolCall

_TOKEN = re.compile(r"[a-z0-9]{3,}")
_SECRETISH = re.compile(
    r"password|card number|\bcvv\b|supersecretpassword|4111[\d ]{12}",
    re.I,
)

BUILTIN_TACTICS = [
    {
        "domain": "wikipedia.org",
        "pattern": "history",
        "advice": (
            "Wikipedia History: call search_page with History, quote the returned "
            "section text / dated facts, then note + done. Do not click TOC History "
            "links (they are often covered). Do not keep searching after section text appears."
        ),
    },
    {
        "domain": "wikipedia.org",
        "pattern": "extract",
        "advice": (
            "Wikipedia extract: use visible_text and search_page section text. "
            "read_element [1] is usually nav (Jump to content), not the article."
        ),
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(str(text).split())
    if _SECRETISH.search(text):
        text = _SECRETISH.sub("[redacted]", text)
    return text[:limit]


def trace_from_steps(steps: list[Step]) -> list[dict]:
    """Compact tool trace stored on the assistant message so the UI can reopen it."""
    out: list[dict] = []
    for step in steps[-40:]:
        detail = step.result.error or step.result.output or step.verdict.reason or ""
        args = {
            str(key): _clip(val, 180)
            for key, val in (step.call.args or {}).items()
            if val not in ("", None)
        }
        out.append({
            "index": step.index,
            "name": step.call.name,
            "args": args,
            "status": step.verdict.status.value,
            "detail": _clip(detail, 240),
        })
    return out


def _domain(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _new_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"conv-{stamp}-{uuid.uuid4().hex[:6]}"


class Conversation:
    """One ChatGPT-style thread. Follow-ups only see this thread, not other chats."""

    def __init__(self, conv_id: str | None = None, title: str = "", messages: list[dict] | None = None) -> None:
        self.id = conv_id or _new_id()
        self.title = title
        self.messages: list[dict] = list(messages or [])

    def add(self, role: str, content: str, *, url: str = "", trace: list | None = None) -> None:
        text = (content or "").strip()
        if not text:
            return
        if not self.title and role == "user":
            self.title = text.replace("\n", " ")[:72]
        item = {"role": role, "content": text[:4000], "url": url, "at": _now()}
        if trace:
            item["trace"] = trace
        self.messages.append(item)

    def transcript(self, *, last: int = 16) -> list[str]:
        lines = []
        for msg in self.messages[-last:]:
            who = "user" if msg.get("role") == "user" else "agent"
            body = (msg.get("content") or "").replace("\n", " ")
            if who == "agent" and body.lower().startswith("stuck."):
                body = (
                    "previous attempt got stuck after repeating a read. "
                    "Use the evidence already collected and call done; do not repeat that search."
                )
            cap = 400 if who == "user" else 700
            lines.append(f"  {who}: {body[:cap]}")
        return lines

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "messages": self.messages}

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(
            conv_id=str(data.get("id") or _new_id()),
            title=str(data.get("title") or ""),
            messages=list(data.get("messages") or []),
        )


class ConversationStore:
    """Saves/resumes one conversation at a time (like reopening a ChatGPT thread)."""

    def __init__(self, folder: str | Path | None = None) -> None:
        base = Path(__file__).resolve().parent.parent / "memory" / "conversations"
        self.folder = Path(folder) if folder else base
        self.folder.mkdir(parents=True, exist_ok=True)
        self._cursor = self.folder / "current.txt"

    def path_for(self, conv_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", conv_id)[:80]
        return self.folder / f"{safe}.json"

    def save(self, conv: Conversation) -> None:
        dest = self.path_for(conv.id)
        dest.write_text(json.dumps(conv.to_dict(), indent=2), encoding="utf-8")
        self._cursor.write_text(conv.id, encoding="utf-8")

    def load(self, conv_id: str) -> Conversation | None:
        path = self.path_for(conv_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return Conversation.from_dict(data) if isinstance(data, dict) else None

    def last(self) -> Conversation | None:
        if not self._cursor.is_file():
            return None
        return self.load(self._cursor.read_text(encoding="utf-8").strip())

    def resume_or_new(self, *, fresh: bool = False) -> Conversation:
        if not fresh:
            found = self.last()
            if found is not None:
                return found
        conv = Conversation()
        self.save(conv)
        return conv

    def delete(self, conv_id: str) -> None:
        path = self.path_for(conv_id)
        if path.is_file():
            path.unlink()
        if self._cursor.is_file() and self._cursor.read_text(encoding="utf-8").strip() == conv_id:
            self._cursor.unlink()


class LongTermMemory:
    """Persists facts, site tactics, and past episode summaries across process restarts."""

    def __init__(self, path: str | Path | None = None) -> None:
        env = os.getenv("AGENT_MEMORY_PATH", "").strip()
        if path is None and env:
            path = env
        self.path = Path(path) if path else None
        self.data: dict = {"facts": [], "tactics": [], "episodes": []}
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    for key in self.data:
                        if isinstance(loaded.get(key), list):
                            self.data[key] = loaded[key]
            except (OSError, json.JSONDecodeError):
                pass

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def remember_fact(self, text: str, *, url: str = "", topic: str = "", conversation_id: str = "") -> None:
        text = (text or "").strip()
        if not text or _SECRETISH.search(text):
            return
        entry = {
            "text": text[:1500], "url": url, "topic": topic[:200],
            "conversation_id": conversation_id, "at": _now(),
        }
        facts = self.data["facts"]
        if any(f.get("text") == entry["text"] and f.get("url") == url and f.get("conversation_id") == conversation_id for f in facts):
            return
        facts.append(entry)
        self.data["facts"] = facts[-80:]
        self.save()

    def remember_tactic(self, domain: str, advice: str, *, pattern: str = "") -> None:
        domain = (domain or "").lower()
        advice = (advice or "").strip()
        if not domain or not advice:
            return
        tactics = self.data["tactics"]
        if any(t.get("domain") == domain and t.get("advice") == advice for t in tactics):
            return
        tactics.append({"domain": domain, "pattern": pattern.lower(), "advice": advice[:400], "at": _now()})
        self.data["tactics"] = tactics[-60:]
        self.save()

    def recall(self, goal: str, url: str = "", *, conversation_id: str = "", limit: int = 8) -> list[str]:
        want = _tokens(goal) | _tokens(url)
        host = _domain(url)
        lines: list[tuple[int, str]] = []

        for tactic in BUILTIN_TACTICS + list(self.data.get("tactics") or []):
            domain = (tactic.get("domain") or "").lower()
            pattern = (tactic.get("pattern") or "").lower()
            advice = tactic.get("advice") or ""
            if not advice:
                continue
            score = 0
            if host and domain and (host == domain or host.endswith("." + domain)):
                score += 5
            if pattern and pattern in (goal or "").lower():
                score += 4
            if want & _tokens(domain + " " + pattern + " " + advice):
                score += 1
            if score:
                lines.append((score, f"tactic: {advice}"))

        for fact in self.data.get("facts") or []:
            if conversation_id and fact.get("conversation_id") != conversation_id:
                continue
            blob = f"{fact.get('topic', '')} {fact.get('url', '')} {fact.get('text', '')}"
            score = len(want & _tokens(blob))
            fact_host = _domain(fact.get("url") or "")
            if host and fact_host and (host == fact_host or host.endswith("." + fact_host)):
                score += 3
            if score >= 2:
                lines.append((score, f"fact: {fact.get('text', '')[:400]}"))

        for ep in self.data.get("episodes") or []:
            if conversation_id and ep.get("conversation_id") != conversation_id:
                continue
            blob = f"{ep.get('goal', '')} {ep.get('answer', '')} {ep.get('url', '')}"
            score = len(want & _tokens(blob))
            if score >= 3:
                flag = "ok" if ep.get("success") else "failed"
                lines.append((score, f"past {flag}: {ep.get('goal', '')[:80]} -> {(ep.get('answer') or '')[:220]}"))

        lines.sort(key=lambda x: -x[0])
        out: list[str] = []
        seen: set[str] = set()
        for _, text in lines:
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    def ingest_run(
        self, goal: str, *, success: bool, answer: str, url: str, steps: list[Step],
        conversation_id: str = "",
    ) -> None:
        if not _SECRETISH.search(answer or ""):
            self.data["episodes"].append({
                "goal": (goal or "")[:300],
                "success": success,
                "answer": (answer or "")[:800],
                "url": url,
                "conversation_id": conversation_id,
                "at": _now(),
            })
            self.data["episodes"] = self.data["episodes"][-40:]
        if success and answer:
            self.remember_fact(answer, url=url, topic=goal, conversation_id=conversation_id)
        domain = _domain(url)
        failed_history_click = False
        saw_section_text = False
        for step in steps:
            name = step.call.name
            args = step.call.args or {}
            out = (step.result.output or "") + " " + (step.result.error or "")
            label = str(args.get("text") or args.get("ref") or "")
            if "Section text" in (step.result.output or ""):
                saw_section_text = True
            if (
                name in {"click", "click_text"}
                and "history" in label.lower()
                and step.verdict.status.value in {"error", "blocked", "no_effect"}
            ):
                failed_history_click = True
            if "covered" in out.lower() and "history" in out.lower():
                failed_history_click = True
        if failed_history_click and domain.endswith("wikipedia.org"):
            self.remember_tactic(
                "wikipedia.org",
                "Do not click Wikipedia TOC History; search_page('History') and quote section text.",
                pattern="history",
            )
        if saw_section_text and not success and "history" in (goal or "").lower():
            self.remember_tactic(
                "wikipedia.org",
                "When search_page returns History section text, note four dated facts and call done immediately.",
                pattern="history",
            )
        self.save()


class Memory:
    """Short-term = this conversation thread. Long-term = saved thread + site tactics."""

    def __init__(
        self,
        history_window: int = 8,
        *,
        long_term: LongTermMemory | None = None,
        conversation: Conversation | None = None,
        store: ConversationStore | None = None,
        chroma: Any | None = None,
    ) -> None:
        self.notes: list[str] = []
        self.steps: list[Step] = []
        self.turns: list[tuple[str, str]] = []
        self.recalled: list[str] = []
        self.evidence: list[tuple[str, str]] = []
        self.history_window = history_window
        self.long_term = long_term if long_term is not None else LongTermMemory()
        self.conversation = conversation or Conversation()
        self.store = store
        self.chroma = chroma
        self.current_goal = ""
        self.last_url = ""
        self._hydrate_turns()

    def _hydrate_turns(self) -> None:
        self.turns = []
        pending_user = ""
        for msg in self.conversation.messages:
            if msg.get("role") == "user":
                pending_user = msg.get("content") or ""
            elif msg.get("role") == "assistant" and pending_user:
                self.turns.append((pending_user, msg.get("content") or ""))
                pending_user = ""

    def new_conversation(self) -> Conversation:
        self.conversation = Conversation()
        self.notes = []
        self.steps = []
        self.turns = []
        self.recalled = []
        if self.store:
            self.store.save(self.conversation)
        if self.chroma is not None:
            try:
                self.chroma.upsert_conversation(self.conversation)
            except Exception as exc:
                print(f"(chroma memory skipped: {exc})", flush=True)
        return self.conversation

    def add_note(self, text: str) -> None:
        text = text.strip()
        if text and text not in self.notes:
            self.notes.append(text)

    def add_evidence(self, source: str, text: str) -> None:
        """Keep read-tool output the executor can quote. History only stores a short preview."""
        raw = (text or "").strip()
        if len(raw) < 20:
            return
        if _SECRETISH.search(raw):
            raw = _SECRETISH.sub("[redacted]", raw)
        item = (source, raw[:2200])
        if self.evidence and self.evidence[-1] == item:
            return
        self.evidence.append(item)
        self.evidence = self.evidence[-4:]

    def evidence_block(self) -> str:
        if not self.evidence:
            return ""
        parts = [f"[{source}]\n{text}" for source, text in self.evidence[-3:]]
        return "\n\n".join(parts)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def begin_run(self, goal: str, url: str = "") -> None:
        self.current_goal = goal
        self.steps = []
        self.evidence = []
        self.recalled = self.long_term.recall(
            goal, url, conversation_id=self.conversation.id,
        )
        if self.chroma is not None:
            try:
                tail = [(msg.get("content") or "") for msg in self.conversation.messages[-8:]]
                for line in self.chroma.recall_lines(
                    goal, url=url, conversation_id=self.conversation.id, exclude=tail,
                ):
                    if line not in self.recalled:
                        self.recalled.append(line)
                self.recalled = self.recalled[:12]
            except Exception as exc:
                print(f"(chroma recall skipped: {exc})", flush=True)

    def end_run(self, goal: str, *, success: bool, answer: str, url: str = "") -> None:
        self.last_url = url
        self.conversation.add("user", goal, url=url)
        self.conversation.add("assistant", answer or "", url=url, trace=trace_from_steps(self.steps))
        self.turns.append((goal, (answer or "")[:1500]))
        self.turns = self.turns[-16:]
        if self.store:
            self.store.save(self.conversation)
        self.long_term.ingest_run(
            goal, success=success, answer=answer or "", url=url, steps=self.steps,
            conversation_id=self.conversation.id,
        )
        if self.chroma is not None:
            try:
                self.chroma.index_run(
                    self.conversation, goal=goal, success=success,
                    answer=answer or "", url=url or "",
                )
            except Exception as exc:
                print(f"(chroma memory skipped: {exc})", flush=True)

    def prompt_notes(self) -> list[str]:
        bits = list(self.notes)
        thread = self.conversation.transcript(last=16)
        if thread:
            bits.append(f"This conversation ({self.conversation.id}) — use only this thread:")
            bits.extend(thread)
        if self.recalled:
            bits.append("Memory for this conversation / site tactics:")
            bits.extend(f"  {line}" for line in self.recalled)
        return bits

    def plan_context(self) -> str:
        lines = self.prompt_notes()
        return "\n".join(lines) if lines else ""

    def history_summary(self) -> str:
        recent = self.steps[-self.history_window:]
        if not recent:
            return "(no actions yet)"
        lines = []
        for s in recent:
            blob = s.result.error or s.result.output or ""
            cap = 160
            if s is recent[-1] and s.call.name in {"search_page", "read_element", "list_fields", "fill_field"}:
                cap = 900
            detail = blob[:cap].replace("\n", " ")
            lines.append(f"#{s.index} {s.call.name}{s.call.args} -> {s.verdict.status.value}: {detail}")
        return "\n".join(lines)


class SkillStore:
    """Long-term. Remembers action sequences that completed a sub-goal on a site.

    Starter version: exact-match lookup, used only as a HINT to the executor
    (element numbers change between page loads, so we never replay blindly).
    Upgrade path: embedding lookup + Stagehand-style replay with self-healing.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        env = os.getenv("AGENT_SKILLS_PATH", "").strip()
        if path is None and env:
            path = env
        self.path = Path(path) if path else None
        self._data: dict[str, dict[str, list[dict]]] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}

    @staticmethod
    def domain_of(url: str) -> str:
        return urlparse(url).hostname or ""

    @staticmethod
    def _key(subgoal: str) -> str:
        return " ".join(subgoal.lower().split())

    def record(
        self,
        domain: str,
        subgoal: str,
        calls: list[ToolCall],
        descriptors: dict[int, dict] | None = None,
    ) -> None:
        if not domain or not calls:
            return
        desc = descriptors or {}
        self._data.setdefault(domain, {})[self._key(subgoal)] = [
            {"name": c.name, "args": strip_ref_args(c.args, desc)} for c in calls
        ]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def hint(self, domain: str, subgoal: str) -> str:
        calls = self._data.get(domain, {}).get(self._key(subgoal))
        if not calls:
            return ""
        seq = "; ".join(f"{c['name']}{c['args']}" for c in calls)
        return (
            f"On {domain}, this sub-goal was completed before with: {seq}. "
            "Replay by role/name from a fresh snapshot; do not reuse old [n] numbers."
        )
