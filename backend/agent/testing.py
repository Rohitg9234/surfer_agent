"""Test doubles so you can run and test the loop without Playwright or an API key."""
from __future__ import annotations

from typing import Any

from .llm import LLMResponse
from .types import Observation, ToolCall


class ScriptedLLM:
    """Returns pre-written responses in order. Records every call for assertions."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


def plan(*subgoals: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name="submit_plan", args={"subgoals": list(subgoals)})])


def act(name: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)])


class MockBrowser:
    """Two tiny fake pages. Page text uses the same [ref] format Perception will produce."""

    PAGES = {
        "home": ('link-list\n[1] link "Pricing"\n[2] button "Pay now"\n[3] textbox "Search"',
                 {1: "pricing"}),
        "pricing": ('heading "Pricing"\ntext "Pro plan: $20/month"\n[1] link "Home"',
                    {1: "home"}),
    }

    def __init__(self) -> None:
        self.current = "home"
        self.typed: dict[int, str] = {}
        self.actions: list[str] = []

    async def observe(self) -> Observation:
        text, _ = self.PAGES[self.current]
        if self.typed:
            text += f"\ntyped: {self.typed}"
        return Observation(url=f"https://example.com/{self.current}", title=self.current, text=text)

    async def goto(self, url: str) -> None:
        self.actions.append(f"goto {url}")
        self.current = "pricing" if "pricing" in url else "home"

    async def click(self, ref: int) -> None:
        self.actions.append(f"click {ref}")
        self.current = self.PAGES[self.current][1].get(ref, self.current)

    async def click_text(self, text: str) -> None:
        self.actions.append(f"click_text {text}")

    async def fill_field(self, label: str, value: str) -> None:
        self.actions.append(f"fill_field {label}")
        self.typed[hash(label) & 0xFFFF] = value

    async def list_fields(self) -> str:
        self.actions.append("list_fields")
        return "Customer name | text\nEmail | email"

    async def type_text(self, ref: int, text: str, press_enter: bool = False) -> None:
        self.actions.append(f"type {ref}")
        self.typed[ref] = text

    async def scroll(self, direction: str, pages: float = 1.0) -> str:
        self.actions.append(f"scroll {direction} {pages}")
        return f"Scrolled {direction} ({pages} page); no new text"

    async def go_back(self) -> None:
        self.actions.append("back")
        self.current = "home"

    async def go_forward(self) -> None:
        self.actions.append("forward")

    async def reload(self) -> None:
        self.actions.append("reload")

    async def wait(self, seconds: float) -> None:
        self.actions.append("wait")

    async def hover(self, ref: int) -> None:
        self.actions.append(f"hover {ref}")

    async def select_option(self, ref: int, value: str) -> None:
        self.actions.append(f"select {ref} {value}")

    async def press_key(self, key: str, ref: int | None = None) -> None:
        self.actions.append(f"press {key}")

    async def read_element(self, ref: int) -> str:
        self.actions.append(f"read {ref}")
        return self.PAGES[self.current][0]

    async def double_click(self, ref: int) -> None:
        self.actions.append(f"dblclick {ref}")

    async def right_click(self, ref: int) -> None:
        self.actions.append(f"rightclick {ref}")

    async def wait_for_text(self, text: str, timeout_ms: int = 8000) -> None:
        self.actions.append(f"wait_text {text}")

    async def get_attribute(self, ref: int, name: str) -> str:
        self.actions.append(f"attr {ref} {name}")
        return ""

    async def check(self, ref: int) -> None:
        self.actions.append(f"check {ref}")

    async def uncheck(self, ref: int) -> None:
        self.actions.append(f"uncheck {ref}")

    async def screenshot(self, path: str) -> str:
        self.actions.append(f"screenshot {path}")
        return path

    async def new_tab(self, url: str | None = None) -> None:
        self.actions.append(f"new_tab {url or ''}")

    async def switch_tab(self, index: int) -> None:
        self.actions.append(f"switch_tab {index}")

    async def close_tab(self) -> None:
        self.actions.append("close_tab")

    async def list_tabs(self) -> str:
        self.actions.append("list_tabs")
        return "[1] home https://example.com/home (current)"

    async def adopt_focused_tab(self) -> str:
        self.actions.append("adopt_tab")
        return "Locked to this tab only: home https://example.com/home"

    async def scroll_to_text(self, text: str) -> None:
        self.actions.append(f"scroll_text {text}")

    async def search_page(self, pattern: str, max_results: int = 10) -> str:
        self.actions.append(f"search_page {pattern}")
        page = self.PAGES[self.current][0]
        if pattern.lower() in page.lower():
            return (
                f'Found matches for "{pattern}":\n'
                "Section text (quote from this and call done if it answers the user):\n"
                f"{page}"
            )
        return f'No matches for "{pattern}"'

    async def dropdown_options(self, ref: int) -> str:
        self.actions.append(f"dropdown {ref}")
        return "- one\n- two"

    async def drag(self, start_ref: int, end_ref: int) -> None:
        self.actions.append(f"drag {start_ref} {end_ref}")

    async def handle_dialog(self, action: str, text: str | None = None) -> None:
        self.actions.append(f"dialog {action}")

    async def save_pdf(self, path: str) -> str:
        self.actions.append(f"pdf {path}")
        return path

    async def wait_for_load(self, state: str = "networkidle", timeout_ms: int = 15000) -> None:
        self.actions.append(f"wait_load {state}")

    async def save_session(self, path: str) -> str:
        self.actions.append(f"save_session {path}")
        return path

    async def load_session(self, path: str) -> str:
        self.actions.append(f"load_session {path}")
        return path

    async def upload_file(self, ref: int | None, path: str) -> None:
        self.actions.append(f"upload {ref} {path}")

    async def bring_to_front(self) -> None:
        self.actions.append("bring_to_front")

    def invalidate_refs(self) -> None:
        self.actions.append("invalidate_refs")

    def descriptors(self) -> dict:
        return {1: {"role": "link", "name": "Pricing", "nth": 0, "frame_path": "", "ancestor_context": ""}}

    async def wait_for(self, *, url=None, text=None, gone=None, settled=False, timeout_ms=15000) -> str:
        self.actions.append(f"wait_for {text or url or gone or 'settled'}")
        return "ok"

    async def scroll_until(self, text: str | None = None, max_steps: int = 8) -> str:
        self.actions.append(f"scroll_until {text}")
        return "ok"

    async def click_download(self, ref: int, directory: str) -> str:
        self.actions.append(f"download {ref} {directory}")
        return f"{directory}/file.bin"

    async def stabilize(self) -> str:
        self.actions.append("stabilize")
        return ""

    async def recover_dynamics(self) -> str:
        self.actions.append("recover")
        return ""
