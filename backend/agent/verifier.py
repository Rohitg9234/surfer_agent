"""Verifier: deterministic checks after every action (no extra LLM cost)."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .tools.registry import ToolSpec
from .types import Observation, Step, ToolCall, ToolResult, Verdict, VerdictStatus


def _url_key(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    path = parsed.path or "/"
    return f"{host}{path}"


def _label(obs: Observation, ref: object) -> str:
    try:
        n = int(ref)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    m = re.search(rf"^\s*\[{n}\]\s*(.*)$", obs.text, re.M)
    return (m.group(1) if m else "").strip().lower()


def _semantic_key(call: ToolCall, obs: Observation) -> str:
    if call.name in {"click", "type_text", "read_element"}:
        label = _label(obs, call.args.get("ref"))
        if label:
            return f"{call.name}:{label}"
    return call.signature()


def _error_stem(text: str) -> str:
    return re.sub(r"\[\d+\]", "[n]", (text or "")[:120]).lower()


class Verifier:
    def __init__(self, loop_threshold: int = 3) -> None:
        self.loop_threshold = loop_threshold
        self._urls: list[str] = []

    def check(
        self,
        call: ToolCall,
        spec: ToolSpec | None,
        result: ToolResult,
        before: Observation,
        after: Observation,
        history: list[Step],
    ) -> Verdict:
        self._urls.append(_url_key(after.url or before.url))
        n = self.loop_threshold
        if call.name not in {"human_in_loop", "request_human_help", "ask_user"} and len(self._urls) >= 3:
            a, b, c = self._urls[-3], self._urls[-2], self._urls[-1]
            if a == c and a != b:
                host = urlparse(c).hostname or c
                return Verdict(
                    status=VerdictStatus.LOOP,
                    reason=f"Oscillating between pages (back to {host}). Stop repeating this path.",
                )

        key = _semantic_key(call, before)
        recent = [_semantic_key(s.call, before) for s in history[-(n - 1):]]
        # history observations aren't stored; compare signatures + current semantic key for clicks
        sigs = [s.call.signature() for s in history[-(n - 1):]]
        if len(sigs) == n - 1 and all(s == call.signature() for s in sigs):
            return Verdict(
                status=VerdictStatus.LOOP,
                reason=f"Repeated {call.name} {n} times with the same arguments",
            )
        if call.name == "click" and key.startswith("click:") and len(history) >= n - 1:
            same = 0
            for s in history[-(n - 1):]:
                if s.call.name == "click" and _label(before, s.call.args.get("ref")) and _semantic_key(s.call, before) == key:
                    same += 1
            # labels from current snapshot may not match old refs; also match error stems
            pass

        if not result.ok:
            stem = _error_stem(result.error or "")
            prev_err = [
                _error_stem(s.result.error or s.verdict.reason)
                for s in history[-(n - 1):]
                if s.verdict.status == VerdictStatus.ERROR
            ]
            if len(prev_err) == n - 1 and stem and all(p == stem for p in prev_err):
                return Verdict(
                    status=VerdictStatus.LOOP,
                    reason=f"Same error {n} times: {stem[:160]}",
                )
            return Verdict(status=VerdictStatus.ERROR, reason=result.error or "tool failed")
        if spec and spec.control:
            return Verdict(status=VerdictStatus.OK)

        if spec and spec.expects_page_change and before.fingerprint == after.fingerprint:
            why = "no navigation" if before.url == after.url else "snapshot unchanged"
            return Verdict(
                status=VerdictStatus.NO_EFFECT,
                reason=f"Action had no effect ({why}). Control may be covered, disabled, or not the right [n].",
            )
        return Verdict(status=VerdictStatus.OK)
