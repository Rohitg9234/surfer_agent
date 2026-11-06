"""Starter policy gate used by the core before every action.

This is the minimum viable version. Your full Safety layer (injection
classifier, credential vault, audit log) should implement the same
`check(call, spec, obs) -> Decision` signature and replace this class.
It is deterministic code: it never asks an LLM for permission decisions.
"""
from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from .tools.registry import ToolSpec
from .types import Observation, Risk, ToolCall


class Decision(BaseModel):
    action: Literal["allow", "confirm", "block"]
    reason: str = ""


RISKY_LABEL = re.compile(
    r"\b(pay|purchase|buy now|place order|checkout|delete|remove account|"
    r"send (message|email|mail)|submit|transfer|subscribe|unsubscribe|"
    r"sign out|log out)\b",
    re.I,
)
CLICK_TEXT_RISKY = re.compile(
    r"\b(submit( application)?|send( message| email)?|pay( now)?|place order|"
    r"buy now|checkout|delete account|confirm (order|purchase|application))\b",
    re.I,
)
SECRET_LABEL = re.compile(
    r"password|card number|credit card|debit card|\bcvv\b|\bcvc\b|\bssn\b|security code",
    re.I,
)
CARD_VALUE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SENSITIVE_HOST = re.compile(
    r"(mail\.google|outlook\.live|outlook\.office|webmail|"
    r"accounts\.google|login\.microsoftonline|"
    r"onlinebanking|bankof|chase\.com|wellsfargo|paypal\.com|stripe\.com)",
    re.I,
)
_PASSWORD_CONTROL = re.compile(
    r"\[\d+\]\s+(textbox|input|combobox).{0,40}(password|passwd)",
    re.I,
)
_SEARCH_HOSTS = {
    "duckduckgo.com", "www.duckduckgo.com",
    "google.com", "www.google.com",
    "bing.com", "www.bing.com",
    "search.yahoo.com",
}
BROADCAST_SEND = re.compile(
    r"(send|email|mail).{0,80}(everyone|all (contacts|employees|recipients|staff)|entire (company|team|org))",
    re.I,
)
RESIGNATION_SEND = re.compile(
    r"(submit|send|email).{0,40}resignation|resignation email",
    re.I,
)
PAYMENT_GOAL = re.compile(
    r"\b(pay for|enter (a |my )?(credit |debit )?card|type .{0,20}card number|"
    r"checkout and pay|place an? order|buy (this|it) now)\b",
    re.I,
)


def element_label(obs: Observation, ref: int | None) -> str:
    if ref is None:
        return ""
    m = re.search(rf"^\s*\[{ref}\]\s*(.*)$", obs.text, re.M)
    return m.group(1) if m else ""


class Policy:
    def __init__(self, allowed_domains: list[str] | None = None, blocked_domains: list[str] | None = None):
        self.allowed = [d.lower() for d in (allowed_domains or [])]
        self.blocked = [d.lower() for d in (blocked_domains or [])]
        self._sensitive_host: str | None = None
        self.goal = ""

    @staticmethod
    def _matches(host: str, domains: list[str]) -> bool:
        return any(host == d or host.endswith("." + d) for d in domains)

    def observe(self, obs: Observation) -> None:
        host = (urlparse(obs.url).hostname or "").lower()
        if not host or self._is_search_host(host):
            return
        if SENSITIVE_HOST.search(obs.url) or _PASSWORD_CONTROL.search(obs.text or ""):
            self._sensitive_host = host

    @classmethod
    def _is_search_host(cls, host: str) -> bool:
        return host in _SEARCH_HOSTS

    def _goal_names_host(self, host: str) -> bool:
        """The user asked to open this site, so leaving Gmail/etc. is the task, not exfiltration."""
        goal = (self.goal or "").lower()
        host = (host or "").lower()
        if not goal or not host:
            return False
        if host in goal:
            return True
        parts = [p for p in host.split(".") if p and p not in {"www", "com", "org", "net", "co", "io"}]
        label = parts[-1] if parts else ""
        return len(label) >= 4 and label in goal

    def check_goal(self, goal: str) -> Decision:
        text = goal or ""
        if BROADCAST_SEND.search(text) or RESIGNATION_SEND.search(text):
            return Decision(
                action="block",
                reason="Refused: bulk send / resignation email is not allowed. The agent will not send mail.",
            )
        if PAYMENT_GOAL.search(text):
            return Decision(
                action="confirm",
                reason="This goal involves payment or card entry. Confirm to continue; the agent will still never type a card number.",
            )
        return Decision(action="allow")

    def check(self, call: ToolCall, spec: ToolSpec | None, obs: Observation) -> Decision:
        if spec is None:
            return Decision(action="block", reason=f"Unknown tool '{call.name}'")

        if call.name == "goto":
            parsed = urlparse(str(call.args.get("url", "")))
            host = (parsed.hostname or "").lower()
            if parsed.scheme not in ("http", "https") or not host:
                return Decision(action="block", reason="Only http(s) URLs are allowed")
            if self._matches(host, self.blocked):
                return Decision(action="block", reason=f"{host} is blocked")
            if self.allowed and not self._matches(host, self.allowed):
                return Decision(action="block", reason=f"{host} is not on the allowlist")
            if (
                self._sensitive_host
                and host != self._sensitive_host
                and not host.endswith("." + self._sensitive_host)
                and not self._goal_names_host(host)
            ):
                return Decision(
                    action="block",
                    reason=(
                        f"Off-domain navigation to {host} blocked after reading {self._sensitive_host}. "
                        "Allowed when the user goal names that site."
                    ),
                )

        if call.name == "screenshot" and SECRET_LABEL.search(obs.text):
            return Decision(action="block", reason="Screenshots are disabled on pages with password fields")

        label = element_label(obs, call.args.get("ref"))
        typed = str(call.args.get("text") or call.args.get("value") or "")
        if call.name in {"type_text", "fill_field"}:
            field = label or str(call.args.get("label") or "")
            user_supplied = bool(self.goal and typed and typed in self.goal)
            if CARD_VALUE.search(typed):
                return Decision(action="block", reason="Card numbers are not allowed")
            if SECRET_LABEL.search(field) and not user_supplied:
                return Decision(action="block", reason="Typing secrets is not allowed; use fill_credential")
        if call.name == "search":
            hosts = ("duckduckgo.com", "google.com")
            if self.allowed and not any(self._matches(h, self.allowed) for h in hosts):
                return Decision(action="block", reason="Web search is not on the allowlist")
            if all(self._matches(h, self.blocked) for h in hosts):
                return Decision(action="block", reason="Web search engines are blocked")

        if call.name == "click_text" and CLICK_TEXT_RISKY.search(str(call.args.get("text") or "")):
            return Decision(
                action="confirm",
                reason=(
                    f"Clicking {call.args.get('text')!r} on "
                    f"{urlparse(obs.url).hostname or obs.url} may submit or pay. "
                    "Allow only if the user explicitly approved submit."
                ),
            )

        confirm_detail = f" role/name={label!r}" if label else ""
        if spec.risk == Risk.HIGH:
            return Decision(
                action="confirm",
                reason=f"{call.name} is a high-risk action on {urlparse(obs.url).hostname or obs.url}{confirm_detail}",
            )
        if call.name == "click" and RISKY_LABEL.search(label):
            return Decision(
                action="confirm",
                reason=f"Clicking '{label}' on {urlparse(obs.url).hostname or obs.url} may have irreversible effects",
            )
        return Decision(action="allow")
