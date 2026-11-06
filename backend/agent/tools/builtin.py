"""Built-in tools.

=====================  ADD YOUR NEW TOOLS IN THIS FILE  =====================
(or in a new file inside agent/tools/ that you import from tools/__init__.py)

Template for a new tool:

    class MyArgs(BaseModel):
        selector: str = Field(description="What the model should pass, in plain words")

    @tool(
        name="my_tool",
        description="One clear sentence the LLM reads to decide when to use it.",
        params=MyArgs,
        risk=Risk.LOW,                 # SAFE | LOW | HIGH (HIGH always asks the user)
        expects_page_change=True,      # False for read-only tools
    )
    async def my_tool(args: MyArgs, ctx: ToolContext) -> str:
        await ctx.browser.something(args.selector)   # call the Browser interface
        return "short text result the agent will see"

Nothing else needs editing: the Executor, validation, Policy and Verifier all
read the registry. If your tool needs a new browser capability, add the method
to agent/interfaces.py:Browser and implement it in your Playwright class.
==============================================================================
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, HttpUrl

from ..types import Risk, ToolCall
from .registry import EXTRA_GROUPS, ToolContext, tool

Ref = Annotated[int, Field(ge=1, description="Element number from the page snapshot, e.g. 3 for [3]")]


# ---------------------------------------------------------------- navigation
class GotoArgs(BaseModel):
    url: HttpUrl = Field(description="Absolute URL starting with http:// or https://")


@tool(name="goto", description="Navigate the current tab to a URL.", params=GotoArgs)
async def goto(args: GotoArgs, ctx: ToolContext) -> str:
    url = str(args.url)
    await ctx.browser.goto(url)
    return f"Navigated to {url}"


class SearchWebArgs(BaseModel):
    query: str = Field(min_length=1, description="Search query")


_SERP_FAIL = re.compile(
    r"captcha|unusual traffic|verify you are human|just a moment|enable javascript|"
    r"before you continue to google|our systems have detected|did not match any documents|"
    r"no results found|sorry, no results",
    re.I,
)
_DDG_CHALLENGE = re.compile(
    r"bots use duckduckgo|squares containing a duck|select all squares",
    re.I,
)
_RESULT_LINK = re.compile(
    r'(?:\[\d+\]\s+link\s+"([^"]+)"|-\s+link\s+"([^"]+)"\s+\[\d+\])',
    re.I,
)
_CHROME_LINK = re.compile(
    r"^(all|images|videos|news|maps|search|settings|duck\.ai|homepage)$",
    re.I,
)


def _serp_unusable(url: str, text: str) -> bool:
    blob = f"{url}\n{text}"
    if "/sorry" in (url or "").lower() or _SERP_FAIL.search(blob):
        return True
    return len(_result_titles(text)) < 1


def _ddg_challenge(url: str, text: str) -> bool:
    return bool(_DDG_CHALLENGE.search(f"{url}\n{text}"))


def _result_titles(text: str, limit: int = 5) -> list[str]:
    out: list[str] = []
    for match in _RESULT_LINK.finditer(text or ""):
        title = next((part for part in match.groups() if part), "")
        title = " ".join(title.split())
        if len(title) < 8 or _CHROME_LINK.match(title):
            continue
        if title not in out:
            out.append(title)
        if len(out) >= limit:
            break
    return out


async def _watch_serp(ctx: ToolContext, seconds: float = 8.0):
    """DuckDuckGo draws result links a few seconds after navigation."""
    deadline = time.monotonic() + seconds
    obs = await ctx.browser.observe()
    while True:
        if _ddg_challenge(obs.url, obs.text):
            return obs, "captcha"
        if not _serp_unusable(obs.url, obs.text):
            return obs, "ok"
        if time.monotonic() >= deadline:
            return obs, "empty"
        try:
            await ctx.browser.wait(1)
        except Exception:
            return obs, "empty"
        obs = await ctx.browser.observe()


@tool(
    name="search",
    description="Web search on DuckDuckGo. Wait for the result links, then click one. If DuckDuckGo shows the duck-square challenge, call human_in_loop kind=captcha on that page. Do not open Google or the DuckDuckGo homepage yourself.",
    params=SearchWebArgs,
)
async def search(args: SearchWebArgs, ctx: ToolContext) -> str:
    q = quote_plus(args.query)
    await ctx.browser.goto(f"https://duckduckgo.com/?q={q}&ia=web")
    obs, state = await _watch_serp(ctx)
    if state == "captcha":
        return (
            "DuckDuckGo is showing its bot check (select the squares with a duck). "
            "Call human_in_loop kind=captcha and stay on this page. Do not open Google."
        )
    if state == "ok":
        titles = _result_titles(obs.text)
        listed = "; ".join(titles) if titles else "the result links in the snapshot"
        return (
            f"Opened DuckDuckGo search for {args.query!r}. "
            f"Click one of these results: {listed}."
        )
    await ctx.browser.goto(f"https://www.bing.com/search?q={q}")
    obs, state = await _watch_serp(ctx, seconds=4)
    if state == "ok":
        titles = _result_titles(obs.text)
        listed = "; ".join(titles) if titles else "the result links in the snapshot"
        return f"DuckDuckGo did not draw results. Opened Bing for {args.query!r}. Click one of: {listed}."
    return (
        "DuckDuckGo did not show result links. Stay on this search page and click a result if one is visible. "
        "If you see a duck-square or other bot check, call human_in_loop kind=captcha. Do not open Google."
    )


class NoArgs(BaseModel):
    pass


@tool(name="go_back", description="Go back to the previous page in history.", params=NoArgs)
async def go_back(args: NoArgs, ctx: ToolContext) -> str:
    await ctx.browser.go_back()
    return "Went back"


@tool(name="go_forward", description="Go forward to the next page in history.", params=NoArgs)
async def go_forward(args: NoArgs, ctx: ToolContext) -> str:
    await ctx.browser.go_forward()
    return "Went forward"


@tool(name="reload", description="Reload the current page.", params=NoArgs, risk=Risk.SAFE)
async def reload(args: NoArgs, ctx: ToolContext) -> str:
    await ctx.browser.reload()
    return "Reloaded"


# --------------------------------------------------------------- interaction
class ClickArgs(BaseModel):
    ref: Ref


@tool(name="click", description="Click an element by its snapshot number.", params=ClickArgs)
async def click(args: ClickArgs, ctx: ToolContext) -> str:
    note = await ctx.browser.click(args.ref)
    base = f"Clicked [{args.ref}]"
    if isinstance(note, str) and note.strip():
        return f"{base}. {note.strip()}"
    return base


@tool(name="double_click", description="Double-click an element by its snapshot number.", params=ClickArgs)
async def double_click(args: ClickArgs, ctx: ToolContext) -> str:
    await ctx.browser.double_click(args.ref)
    return f"Double-clicked [{args.ref}]"


@tool(name="right_click", description="Right-click an element to open its context menu.", params=ClickArgs)
async def right_click(args: ClickArgs, ctx: ToolContext) -> str:
    await ctx.browser.right_click(args.ref)
    return f"Right-clicked [{args.ref}]"


class TypeArgs(BaseModel):
    ref: Ref
    text: str = Field(min_length=1, description="Text to type")
    press_enter: bool = Field(default=False, description="Press Enter after typing")


@tool(name="type_text", description="Type text into a field by its snapshot number.", params=TypeArgs)
async def type_text(args: TypeArgs, ctx: ToolContext) -> str:
    await ctx.browser.type_text(args.ref, args.text, args.press_enter)
    return f"Typed into [{args.ref}]"


class HoverArgs(BaseModel):
    ref: Ref


@tool(name="hover", description="Hover the pointer over an element by its snapshot number.", params=HoverArgs)
async def hover(args: HoverArgs, ctx: ToolContext) -> str:
    await ctx.browser.hover(args.ref)
    return f"Hovered [{args.ref}]"


class SelectArgs(BaseModel):
    ref: Ref
    value: str = Field(min_length=1, description="Visible option label, or the option value")


@tool(name="select_option", description="Choose an option in a dropdown by snapshot number.", params=SelectArgs)
async def select_option(args: SelectArgs, ctx: ToolContext) -> str:
    await ctx.browser.select_option(args.ref, args.value)
    return f"Selected {args.value!r} on [{args.ref}]"


class PressKeyArgs(BaseModel):
    key: str = Field(min_length=1, description="Key to press, e.g. Enter, Tab, Escape, ArrowDown")
    ref: Ref | None = Field(default=None, description="Optional element to focus first")


@tool(name="press_key", description="Press a keyboard key, optionally on a snapshot element.", params=PressKeyArgs)
async def press_key(args: PressKeyArgs, ctx: ToolContext) -> str:
    await ctx.browser.press_key(args.key, args.ref)
    return f"Pressed {args.key}"


class ReadArgs(BaseModel):
    ref: Ref


@tool(
    name="read_element",
    description="Read the full text of one element by its snapshot number.",
    params=ReadArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def read_element(args: ReadArgs, ctx: ToolContext) -> str:
    return await ctx.browser.read_element(args.ref)


class ScrollArgs(BaseModel):
    direction: Literal["up", "down"] = "down"
    pages: float = Field(default=1.0, ge=0.25, le=10, description="Viewport heights to scroll (10 ≈ page end)")


@tool(
    name="scroll",
    description="Scroll the page or the inner panel under the cursor (job sidebars, feeds).",
    params=ScrollArgs,
    expects_page_change=False,
)
async def scroll(args: ScrollArgs, ctx: ToolContext) -> str:
    detail = await ctx.browser.scroll(args.direction, args.pages)
    if isinstance(detail, str) and detail.strip():
        return detail
    return f"Scrolled {args.direction} ({args.pages} page)"


class ScrollTextArgs(BaseModel):
    text: str = Field(min_length=1, description="Visible text to bring into view")


@tool(name="scroll_to_text", description="Scroll until the given visible text is in view.", params=ScrollTextArgs)
async def scroll_to_text(args: ScrollTextArgs, ctx: ToolContext) -> str:
    await ctx.browser.scroll_to_text(args.text)
    return f"Scrolled to {args.text!r}"


class WaitArgs(BaseModel):
    seconds: float = Field(default=2, ge=0, le=10, description="Seconds to wait (max 10)")


@tool(
    name="wait",
    description="Wait for the page to finish loading or updating.",
    params=WaitArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def wait(args: WaitArgs, ctx: ToolContext) -> str:
    await ctx.browser.wait(args.seconds)
    return f"Waited {args.seconds}s"


class WaitTextArgs(BaseModel):
    text: str = Field(min_length=1, description="Visible text that should appear")
    timeout_ms: int = Field(default=8000, ge=200, le=30_000, description="How long to wait in milliseconds")


@tool(
    name="wait_for_text",
    description="Wait until some visible text appears on the page.",
    params=WaitTextArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def wait_for_text(args: WaitTextArgs, ctx: ToolContext) -> str:
    await ctx.browser.wait_for_text(args.text, args.timeout_ms)
    return f"Saw {args.text!r}"


class WaitLoadArgs(BaseModel):
    state: Literal["load", "domcontentloaded", "networkidle"] = "networkidle"
    timeout_ms: int = Field(default=15000, ge=500, le=50_000)


@tool(
    name="wait_for_load",
    description="Wait until the page load event or network is idle (SPA / spinner).",
    params=WaitLoadArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def wait_for_load(args: WaitLoadArgs, ctx: ToolContext) -> str:
    await ctx.browser.wait_for_load(args.state, args.timeout_ms)
    return f"Page reached {args.state}"


class WaitForArgs(BaseModel):
    url: str | None = Field(default=None, description="URL glob, e.g. **/pricing")
    text: str | None = Field(default=None, description="Visible text that should appear")
    gone: str | None = Field(default=None, description="Visible text that should disappear (spinner)")
    settled: bool = Field(default=False, description="No DOM mutations for a short quiet period")
    timeout_ms: int = Field(default=15000, ge=500, le=50_000)


@tool(
    name="wait_for",
    description="Wait until URL matches, text appears or disappears, or the page is settled.",
    params=WaitForArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def wait_for(args: WaitForArgs, ctx: ToolContext) -> str:
    return await ctx.browser.wait_for(
        url=args.url, text=args.text, gone=args.gone, settled=args.settled, timeout_ms=args.timeout_ms,
    )


class ScrollUntilArgs(BaseModel):
    text: str | None = Field(default=None, description="Stop when this text is visible")
    max_steps: int = Field(default=8, ge=1, le=30)


@tool(
    name="scroll_until",
    description="Scroll down until text appears or the page stops changing (lazy lists). If the text is already visible, use click_text instead.",
    params=ScrollUntilArgs,
    expects_page_change=False,
)
async def scroll_until(args: ScrollUntilArgs, ctx: ToolContext) -> str:
    return await ctx.browser.scroll_until(args.text, args.max_steps)


class ClickTextArgs(BaseModel):
    text: str = Field(min_length=1, description="Visible button or link label, e.g. Easy Apply")


@tool(
    name="click_text",
    description="Click the first button or link with this visible label. Do not use for Submit/Send/Pay unless the user approved. Prefer this for Easy Apply / Next / cookie Accept.",
    params=ClickTextArgs,
)
async def click_text(args: ClickTextArgs, ctx: ToolContext) -> str:
    note = await ctx.browser.click_text(args.text)
    base = f"Clicked {args.text!r}"
    if isinstance(note, str) and note.strip():
        return f"{base}. {note.strip()}"
    return base


class ListFieldsArgs(BaseModel):
    pass


@tool(
    name="list_fields",
    description="List visible form controls as 'label | type'. Use this before filling, and to describe a form. Do not submit.",
    params=ListFieldsArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def list_fields(args: ListFieldsArgs, ctx: ToolContext) -> str:
    listed = await ctx.browser.list_fields()
    return listed or "No form fields visible"


class FillFieldArgs(BaseModel):
    label: str = Field(min_length=1, description="Visible label or placeholder, e.g. Phone, First name")
    value: str = Field(min_length=1, description="What to type into that field")


@tool(
    name="fill_field",
    description="Type into a form field by its label (works inside Easy Apply / modal dialogs).",
    params=FillFieldArgs,
)
async def fill_field(args: FillFieldArgs, ctx: ToolContext) -> str:
    await ctx.browser.fill_field(args.label, args.value)
    secret = re.search(r"password|card number|credit card|\bcvv\b|\bcvc\b", args.label, re.I)
    shown = "[redacted]" if secret else args.value
    if re.search(r"\bsearch\b", args.label, re.I):
        try:
            await ctx.browser.press_key("Enter")
        except Exception:
            pass
        return f"Filled {args.label!r} with {shown!r} and pressed Enter to run the search."
    return f"Filled {args.label!r} with {shown!r}. Do not submit unless the user typed submit."


class AttrArgs(BaseModel):
    ref: Ref
    name: str = Field(min_length=1, description="HTML attribute name, e.g. href, value, aria-label")


@tool(
    name="get_attribute",
    description="Read one HTML attribute from a snapshot element (href, value, aria-label, ...).",
    params=AttrArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def get_attribute(args: AttrArgs, ctx: ToolContext) -> str:
    return await ctx.browser.get_attribute(args.ref, args.name)


@tool(name="check", description="Check a checkbox or radio by snapshot number.", params=ClickArgs)
async def check(args: ClickArgs, ctx: ToolContext) -> str:
    await ctx.browser.check(args.ref)
    return f"Checked [{args.ref}]"


@tool(name="uncheck", description="Uncheck a checkbox by snapshot number.", params=ClickArgs)
async def uncheck(args: ClickArgs, ctx: ToolContext) -> str:
    await ctx.browser.uncheck(args.ref)
    return f"Unchecked [{args.ref}]"


class ScreenshotArgs(BaseModel):
    filename: str = Field(default="shot.png", min_length=1, description="File name under screenshots/")


@tool(
    name="screenshot",
    description="Save a PNG screenshot of the current page.",
    params=ScreenshotArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def screenshot(args: ScreenshotArgs, ctx: ToolContext) -> str:
    from pathlib import Path

    folder = Path("screenshots")
    folder.mkdir(exist_ok=True)
    path = str(folder / Path(args.filename).name)
    saved = await ctx.browser.screenshot(path)
    return f"Saved screenshot to {saved}"


class SearchPageArgs(BaseModel):
    pattern: str = Field(min_length=1, description="Text to find in the current page")
    max_results: int = Field(default=10, ge=1, le=50)


@tool(
    name="search_page",
    description="Find text on the current page. Snapshot match [n] can be used with read_element; page-text snippets cannot.",
    params=SearchPageArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def search_page(args: SearchPageArgs, ctx: ToolContext) -> str:
    return await ctx.browser.search_page(args.pattern, args.max_results)


@tool(
    name="dropdown_options",
    description="List the options of a <select> dropdown by snapshot number.",
    params=ClickArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def dropdown_options(args: ClickArgs, ctx: ToolContext) -> str:
    return await ctx.browser.dropdown_options(args.ref)


class DragArgs(BaseModel):
    start_ref: Ref
    end_ref: Ref


@tool(name="drag", description="Drag one snapshot element onto another.", params=DragArgs)
async def drag(args: DragArgs, ctx: ToolContext) -> str:
    await ctx.browser.drag(args.start_ref, args.end_ref)
    return f"Dragged [{args.start_ref}] to [{args.end_ref}]"


class DialogArgs(BaseModel):
    action: Literal["accept", "dismiss"] = "dismiss"
    text: str | None = Field(default=None, description="Optional text for prompt() dialogs")


@tool(
    name="handle_dialog",
    description="Set how the next JavaScript alert/confirm/prompt is handled.",
    params=DialogArgs,
)
async def handle_dialog(args: DialogArgs, ctx: ToolContext) -> str:
    await ctx.browser.handle_dialog(args.action, args.text)
    return f"Next dialog will be {args.action}ed"


class PdfArgs(BaseModel):
    filename: str = Field(default="page.pdf", min_length=1)


@tool(
    name="save_pdf",
    description="Save the current page as a PDF under screenshots/.",
    params=PdfArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def save_pdf(args: PdfArgs, ctx: ToolContext) -> str:
    from pathlib import Path

    folder = Path("screenshots")
    folder.mkdir(exist_ok=True)
    name = Path(args.filename).name
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = str(folder / name)
    saved = await ctx.browser.save_pdf(path)
    return f"Saved PDF to {saved}"


_SECRET_NAMES = {".env", ".env.local", "credentials.json", "id_rsa", "id_ed25519"}


def _local_file(path: str, *, must_exist: bool) -> Path:
    p = Path(path).expanduser().resolve()
    if p.name.lower() in _SECRET_NAMES or p.suffix.lower() in {".pem", ".key"}:
        raise ValueError("That path is not allowed")
    if must_exist and not p.is_file():
        raise FileNotFoundError(f"No file at {p}")
    return p


def _sandboxed(path: str, root: str, *, must_exist: bool) -> Path:
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    raw = Path(path).expanduser()
    p = (base / raw.name).resolve() if not raw.is_absolute() else raw.resolve()
    if base not in p.parents and p != base:
        raise ValueError(f"Path must be under {base}")
    return _local_file(str(p), must_exist=must_exist)


class SessionPathArgs(BaseModel):
    path: str = Field(default="sessions/state.json", min_length=1, description="Local JSON storage_state path")


@tool(
    name="save_session",
    description="Save cookies and localStorage to a local JSON file (values are not shown).",
    params=SessionPathArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def save_session(args: SessionPathArgs, ctx: ToolContext) -> str:
    dest = _local_file(args.path, must_exist=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    saved = await ctx.browser.save_session(str(dest))
    return f"Saved session to {saved} (cookie values omitted)"


@tool(
    name="load_session",
    description="Restore cookies and localStorage from a JSON file saved by save_session, then open a blank tab.",
    params=SessionPathArgs,
    risk=Risk.LOW,
    expects_page_change=True,
)
async def load_session(args: SessionPathArgs, ctx: ToolContext) -> str:
    src = _local_file(args.path, must_exist=True)
    loaded = await ctx.browser.load_session(str(src))
    return f"Loaded session from {loaded}. Navigate to the site next."


class UploadArgs(BaseModel):
    path: str = Field(min_length=1, description="Local file to attach to a file input")
    ref: int | None = Field(default=None, ge=1, description="File input or upload button number; omit if a chooser is already open")


@tool(
    name="upload_file",
    description="Attach a local file to a file input, or to an open file chooser if ref is omitted.",
    params=UploadArgs,
)
async def upload_file(args: UploadArgs, ctx: ToolContext) -> str:
    src = _sandboxed(args.path, "uploads", must_exist=True)
    await ctx.browser.upload_file(args.ref, str(src))
    if args.ref is None:
        return f"Attached {src.name} to the open file chooser"
    return f"Attached {src.name} to [{args.ref}]"


class DownloadArgs(BaseModel):
    ref: Ref
    directory: str = Field(default="downloads", min_length=1)


@tool(
    name="click_download",
    description="Click an element and save the resulting browser download.",
    params=DownloadArgs,
)
async def click_download(args: DownloadArgs, ctx: ToolContext) -> str:
    folder = Path("downloads").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    saved = await ctx.browser.click_download(args.ref, str(folder))
    return f"Downloaded to {saved}"


class CredentialArgs(BaseModel):
    ref: Ref
    name: str = Field(min_length=1, description="Credential id, e.g. password. Looked up as AGENT_CREDENTIAL_PASSWORD")
    press_enter: bool = False


@tool(
    name="fill_credential",
    description="Type a saved secret into a field. The secret is never shown to the model.",
    params=CredentialArgs,
    risk=Risk.HIGH,
)
async def fill_credential(args: CredentialArgs, ctx: ToolContext) -> str:
    if ctx.vault is None:
        raise RuntimeError("No credential vault is configured")
    secret = ctx.vault.get(args.name)
    await ctx.browser.type_text(args.ref, secret, args.press_enter)
    return f"Filled credential {args.name!r} into [{args.ref}]"


class TotpArgs(BaseModel):
    ref: Ref
    name: str = Field(default="totp", description="Vault id for the TOTP secret, e.g. totp -> AGENT_CREDENTIAL_TOTP")
    press_enter: bool = False


@tool(
    name="fill_totp",
    description="Type a one-time code from a vault TOTP secret. The secret and code never go to the model.",
    params=TotpArgs,
    risk=Risk.HIGH,
)
async def fill_totp(args: TotpArgs, ctx: ToolContext) -> str:
    if ctx.vault is None:
        raise RuntimeError("No credential vault is configured")
    code = ctx.vault.totp_code(args.name)
    await ctx.browser.type_text(args.ref, code, args.press_enter)
    return f"Filled TOTP into [{args.ref}]"


class NewTabArgs(BaseModel):
    url: HttpUrl | None = Field(default=None, description="Optional URL to open in the new tab")


@tool(name="new_tab", description="Open a new browser tab, optionally at a URL.", params=NewTabArgs)
async def new_tab(args: NewTabArgs, ctx: ToolContext) -> str:
    url = str(args.url) if args.url else None
    await ctx.browser.new_tab(url)
    return f"Opened new tab{f' at {url}' if url else ''}"


class SwitchTabArgs(BaseModel):
    index: int = Field(ge=1, description="1-based tab number from list_tabs")


@tool(name="switch_tab", description="Switch to a tab by its 1-based index from list_tabs.", params=SwitchTabArgs)
async def switch_tab(args: SwitchTabArgs, ctx: ToolContext) -> str:
    await ctx.browser.switch_tab(args.index)
    return f"Switched to tab {args.index}"


@tool(
    name="close_tab",
    description="Close the current tab. Fails if it is the last tab.",
    params=NoArgs,
)
async def close_tab(args: NoArgs, ctx: ToolContext) -> str:
    await ctx.browser.close_tab()
    return "Closed tab"


@tool(
    name="list_tabs",
    description="List open tabs with their index, title, and URL.",
    params=NoArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def list_tabs(args: NoArgs, ctx: ToolContext) -> str:
    return await ctx.browser.list_tabs()


# ------------------------------------------------------------ memory and user
class NoteArgs(BaseModel):
    text: str = Field(min_length=1, description="A fact worth remembering for later steps, e.g. a price or name")


@tool(
    name="note",
    description="Save a fact you extracted (prices, names, URLs) so you can use it in later steps.",
    params=NoteArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def note(args: NoteArgs, ctx: ToolContext) -> str:
    ctx.memory.add_note(args.text)
    return "Noted"


@tool(
    name="list_notes",
    description="Show facts saved with `note` during this run.",
    params=NoArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def list_notes(args: NoArgs, ctx: ToolContext) -> str:
    notes = ctx.memory.notes
    return "\n".join(notes) if notes else "(no notes yet)"


class AskArgs(BaseModel):
    question: str = Field(description="A short question for the user")


class HumanLoopArgs(BaseModel):
    question: str = Field(min_length=1, description="What to show the person in the dialog")
    kind: Literal["question", "form", "captcha", "login", "approval"] = Field(
        default="question",
        description="question/form = they type an answer; captcha/login = they solve it in the page; approval = Yes/No",
    )
    placeholder: str = Field(default="", description="Hint inside the text box, e.g. full name")


async def _prompt_human(
    ctx: ToolContext,
    question: str,
    *,
    kind: str = "question",
    placeholder: str = "",
) -> str:
    if hasattr(ctx.browser, "_hitl_open"):
        ctx.browser._hitl_open = True
    if hasattr(ctx.browser, "bring_to_front"):
        try:
            await ctx.browser.bring_to_front()
        except Exception:
            pass
    fn = getattr(ctx.browser, "ask_human", None)
    if callable(fn):
        reply = await fn(question, kind=kind, placeholder=placeholder)
    else:
        reply = await ctx.ask_user(question)
    if hasattr(ctx.browser, "invalidate_refs"):
        ctx.browser.invalidate_refs()
    return (reply or "").strip()


@tool(
    name="human_in_loop",
    description="Show a dialog on the Chrome tab for the human. Use for missing form answers, CAPTCHA/bot checks, login/2FA, or Yes/No approval.",
    params=HumanLoopArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def human_in_loop(args: HumanLoopArgs, ctx: ToolContext) -> str:
    reply = await _prompt_human(ctx, args.question, kind=args.kind, placeholder=args.placeholder)
    if args.kind == "approval":
        return f"Human approval: {reply or 'no'}"
    if args.kind in {"captcha", "login"}:
        if not reply:
            raise RuntimeError(
                f"No person confirmed the {args.kind}. Do not treat it as solved. "
                "Try a different site, or ask again only when someone is at the browser."
            )
        return f"Human finished {args.kind}. Note: {reply}"
    return f"Human answered: {reply or '(empty)'}"


@tool(
    name="ask_user",
    description="Ask the user a question in an on-page dialog (same as human_in_loop kind=question).",
    params=AskArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def ask_user(args: AskArgs, ctx: ToolContext) -> str:
    reply = await _prompt_human(ctx, args.question, kind="question")
    return reply or "(no answer)"


class HumanHelpArgs(BaseModel):
    reason: str = Field(min_length=1, description="What the person must do in the browser")
    mode: Literal["takeover", "approval"] = Field(
        default="takeover",
        description="takeover = CAPTCHA/2FA/login in the window; approval = confirm an irreversible step",
    )
    kind: Literal["login", "captcha", "2fa", "approval", "other"] = "other"
    resume_hint: str = Field(default="", description="What the agent should do after they finish")


@tool(
    name="request_human_help",
    description="Pause for a person via on-page dialog. mode=takeover for CAPTCHA/2FA/login; mode=approval for irreversible steps.",
    params=HumanHelpArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def request_human_help(args: HumanHelpArgs, ctx: ToolContext) -> str:
    extra = f" Then the agent will: {args.resume_hint}" if args.resume_hint else ""
    if args.mode == "approval" or args.kind == "approval":
        kind = "approval"
        prompt = f"Confirm this step: {args.reason}.{extra}"
    elif args.kind in {"captcha", "login", "2fa"}:
        kind = "login" if args.kind in {"login", "2fa"} else "captcha"
        prompt = f"{args.reason}.{extra}"
    else:
        kind = "question"
        prompt = f"{args.reason}.{extra}"
    reply = await _prompt_human(ctx, prompt, kind=kind)
    return f"Human finished mode={args.mode} ({args.kind}). Reply: {reply or 'ok'}"


# ------------------------------------------------------------- loop control
class SubgoalDoneArgs(BaseModel):
    summary: str = Field(description="What was accomplished and any key result")


@tool(
    name="subgoal_complete",
    description="Call when the CURRENT sub-goal is finished. Moves on to the next sub-goal.",
    params=SubgoalDoneArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
    control="subgoal",
)
async def subgoal_complete(args: SubgoalDoneArgs, ctx: ToolContext) -> str:
    return args.summary


class DoneArgs(BaseModel):
    answer: str = Field(description="The final answer or result for the user, with sources if relevant")


@tool(
    name="done",
    description="Call ONLY when the user's goal is fully done (e.g. application submitted). Never call done if fields are missing, a CAPTCHA is up, or you could not fill the form — use human_in_loop instead.",
    params=DoneArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
    control="task",
)
async def done(args: DoneArgs, ctx: ToolContext) -> str:
    return args.answer


class MoreToolsArgs(BaseModel):
    group: Literal["files", "tabs", "forms", "session", "playbooks"]


@tool(
    name="more_tools",
    description="List extra tools in a group. Then call them with call_extra. Groups: files, tabs, forms, session, playbooks.",
    params=MoreToolsArgs,
    risk=Risk.SAFE,
    expects_page_change=False,
)
async def more_tools(args: MoreToolsArgs, ctx: ToolContext) -> str:
    names = EXTRA_GROUPS[args.group]
    return f"Extra tools in {args.group}: {', '.join(names)}. Call call_extra with one of these names."


class CallExtraArgs(BaseModel):
    name: str = Field(min_length=1, description="Registered extra tool name from more_tools")
    arguments_json: str = Field(default="{}", description="JSON object of that tool's arguments")


@tool(
    name="call_extra",
    description="Run a non-core tool (upload, tabs, credentials, playbooks). Use more_tools first if unsure of the name.",
    params=CallExtraArgs,
    risk=Risk.LOW,
)
async def call_extra(args: CallExtraArgs, ctx: ToolContext) -> str:
    import json

    from .registry import default_registry

    raw = json.loads(args.arguments_json or "{}")
    if not isinstance(raw, dict):
        raise ValueError("arguments_json must be a JSON object")
    result = await default_registry.execute(ToolCall(name=args.name, args=raw), ctx)
    if not result.ok:
        raise RuntimeError(result.error or "extra tool failed")
    return result.output or ""
