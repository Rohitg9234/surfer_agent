"""Playwright-backed Browser: numbered snapshot + actions on [ref] elements.

Uses the Playwright *library* (Python async API), not the test runner, CLI, or MCP.
observe() prefers Playwright's AI aria snapshot (same approach as Playwright MCP),
and falls back to a DOM stamp when the accessibility tree is empty.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Literal

from .refs import ElementRef, parse_aria_snapshot, parse_dom_snapshot, rewrite_aria_refs
from .types import Observation

_SAFE_SEARCH_RE = re.compile(r"^[\w\s\\d\[\]\\.+*?{}^$|-]{1,80}$")
_AGENT_CHROME = ("Google Chrome for Testing", "Chromium")
# Everyday Chrome is com.google.Chrome and must never be parked or hidden.
_AGENT_BUNDLES = frozenset({
    "com.google.chrome.for.testing",
    "org.chromium.Chromium",
})
_objc_lock = threading.Lock()
_park_lock = threading.Lock()
_appkit: Any = None


def _is_agent_browser(name: str, bundle: str = "") -> bool:
    if bundle in _AGENT_BUNDLES:
        return True
    return name in _AGENT_CHROME


def _chrome_visibility_script(visible: bool, *, activate: bool = False) -> str:
    """AppleScript that shows or hides Playwright's Chrome only.

    Process names are literal strings. A `repeat with` variable is a list
    reference, and System Events then ignores the hide. `activate` is only
    for a captcha or human-in-the-loop handoff; a background show must not
    set frontmost, or the window steals focus.
    """
    blocks = []
    for name in _AGENT_CHROME:
        lines = []
        if visible:
            lines.append(f'set visible of process "{name}" to true')
            if activate:
                lines.append(f'set frontmost of process "{name}" to true')
        else:
            lines.append(f'set visible of process "{name}" to false')
        action = "\n          ".join(lines)
        blocks.append(
            f'if exists process "{name}" then\n'
            f"          {action}\n"
            f"        end if"
        )
    body = "\n        ".join(blocks)
    return f'tell application "System Events"\n        {body}\nend tell'


def _osascript(script: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return False, ""
    return result.returncode == 0, (result.stdout or "").strip()


def _load_appkit() -> bool:
    global _appkit
    if _appkit is False:
        return False
    if _appkit is not None:
        return True
    try:
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        _appkit = objc
        return True
    except Exception:
        _appkit = False
        return False


def _objc_msg(restype: Any, argtypes: list[Any], obj: Any, selector: bytes, *args: Any) -> Any:
    objc = _appkit
    msg = objc.objc_msgSend
    msg.restype = restype
    msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    sel = objc.sel_registerName(selector)
    return msg(obj, sel, *args)


def _nsstring(ptr: Any) -> str:
    if not ptr:
        return ""
    raw = _objc_msg(ctypes.c_char_p, [], ptr, b"UTF8String")
    return raw.decode() if raw else ""


def _appkit_frontmost() -> tuple[str, str] | None:
    """NSWorkspace frontmost app. System Events often names Chrome for Testing as Google Chrome."""
    if sys.platform != "darwin" or not _load_appkit():
        return None
    with _objc_lock:
        try:
            workspace = _objc_msg(
                ctypes.c_void_p, [], _appkit.objc_getClass(b"NSWorkspace"), b"sharedWorkspace",
            )
            app = _objc_msg(ctypes.c_void_p, [], workspace, b"frontmostApplication")
            if not app:
                return None
            return (
                _nsstring(_objc_msg(ctypes.c_void_p, [], app, b"localizedName")),
                _nsstring(_objc_msg(ctypes.c_void_p, [], app, b"bundleIdentifier")),
            )
        except Exception:
            return None


def _frontmost_identity() -> tuple[str, str]:
    found = _appkit_frontmost()
    if found and found[0]:
        return found
    ok, raw = _osascript(
        'tell application "System Events"\n'
        '  set p to first application process whose frontmost is true\n'
        '  set n to name of p\n'
        '  set b to ""\n'
        '  try\n'
        '    set b to bundle identifier of p\n'
        '  end try\n'
        '  return n & tab & b\n'
        'end tell'
    )
    if not ok or not raw:
        return "", ""
    name, _, bundle = raw.partition("\t")
    return name.strip(), bundle.strip()


def _frontmost_app() -> str:
    name, _bundle = _frontmost_identity()
    return name


def _activate_app(name: str) -> None:
    if not name or _is_agent_browser(name):
        return
    try:
        subprocess.run(["open", "-a", name], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        pass
    safe = name.replace("\\", "").replace('"', "")
    _osascript(f'tell application "{safe}" to activate')


def _set_agent_chrome_visible(visible: bool, activate: bool = False) -> bool:
    """Show or hide Playwright's own Chrome. Never touches the user's everyday Chrome."""
    ok, _ = _osascript(_chrome_visibility_script(visible, activate=activate))
    return ok


def _frontmost_command() -> str:
    pid = _frontmost_pid()
    if not pid:
        return ""
    try:
        listed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return ""
    return (listed.stdout or "").strip()


def _frontmost_pid() -> int:
    if sys.platform != "darwin" or not _load_appkit():
        return 0
    with _objc_lock:
        try:
            workspace = _objc_msg(
                ctypes.c_void_p, [], _appkit.objc_getClass(b"NSWorkspace"), b"sharedWorkspace",
            )
            app = _objc_msg(ctypes.c_void_p, [], workspace, b"frontmostApplication")
            if not app:
                return 0
            return int(_objc_msg(ctypes.c_int, [], app, b"processIdentifier") or 0)
        except Exception:
            return 0


def _park_agent_chrome(return_app: str, agent_profile: str = "") -> bool:
    """If the agent browser took focus, give it back. Do not hide the window.

    Hiding it makes macOS activate it again on the next navigation. The window
    is opened in the background; this only corrects a later steal. Captcha and
    human-in-the-loop call bring_to_front instead. Everyday Google Chrome is
    left alone unless its command line is the agent's own profile copy.
    """
    if not _park_lock.acquire(blocking=False):
        return True
    try:
        name, bundle = _frontmost_identity()
        agent_copy = (
            bool(agent_profile)
            and bundle.lower() == "com.google.chrome"
            and f"--user-data-dir={agent_profile}" in _frontmost_command()
        )
        if not _is_agent_browser(name, bundle) and not agent_copy:
            return True
        if return_app and not _is_agent_browser(return_app):
            _activate_app(return_app)
        return True
    finally:
        _park_lock.release()


def _background_chrome_command(
    app: str,
    profile: str,
    port: int,
    profile_directory: str = "",
) -> list[str]:
    """Launch Chrome without activating it.

    `open -g` is the macOS flag that keeps the new app in the background.
    Starting the binary directly, which Playwright's launch() does, makes
    macOS focus the new tab.
    """
    command = [
        "open", "-g", "-n", "-a", app, "--args",
        f"--user-data-dir={profile}",
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-popup-blocking",
    ]
    if profile_directory:
        command.append(f"--profile-directory={profile_directory}")
    return command


def _playwright_chrome_app(pw: Any) -> str:
    exe = Path(pw.chromium.executable_path)
    app = exe.parent.parent.parent
    if app.suffix != ".app" or not app.is_dir():
        raise RuntimeError(f"Could not find the Chrome app bundle for {exe}")
    return str(app)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _open_background_chrome(
    app: str,
    profile: str,
    port: int,
    profile_directory: str = "",
) -> None:
    subprocess.run(
        _background_chrome_command(app, profile, port, profile_directory),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _debug_chrome_matches(port: int, profile: str) -> bool:
    needle_port = f"--remote-debugging-port={port}"
    needle_dir = f"--user-data-dir={profile}"
    try:
        listed = subprocess.run(
            ["ps", "-ax", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return any(
        needle_port in line and needle_dir in line and "Helper" not in line
        for line in (listed.stdout or "").splitlines()
    )


def _cdp_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=0.4) as resp:
            return resp.status == 200
    except Exception:
        return False


def _quit_background_chrome(profile: str) -> None:
    if not profile:
        return
    needle = f"--user-data-dir={profile}"
    try:
        listed = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return
    for line in (listed.stdout or "").splitlines():
        if needle not in line or "Helper" in line:
            continue
        pid_text = line.strip().split(None, 1)[0]
        try:
            os.kill(int(pid_text), signal.SIGTERM)
        except Exception:
            pass


def _reveal_app(app: str) -> None:
    if not app:
        return
    try:
        subprocess.run(["open", "-a", app], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        pass


def _reveal_agent_window(app: str, profile: str) -> None:
    if profile and app.endswith("Google Chrome.app"):
        _reveal_profile(profile)
        return
    _reveal_app(app)


def _reveal_profile(profile: str) -> None:
    """Bring forward the agent's Chrome, not the user's other Chrome windows."""
    if not profile:
        return
    needle = f"--user-data-dir={profile}"
    try:
        listed = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return
    pid = 0
    for line in (listed.stdout or "").splitlines():
        if needle not in line or "Helper" in line:
            continue
        pid_text = line.strip().split(None, 1)[0]
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        break
    if not pid:
        return
    _osascript(
        'tell application "System Events"\n'
        f'  set frontmost of (first process whose unix id is {pid}) to true\n'
        'end tell'
    )


def _search_indexes(hay: str, needle: str) -> list[int]:
    if not needle:
        return []
    if any(ch in needle for ch in r"*+?[]{}()\\") and _SAFE_SEARCH_RE.match(needle):
        try:
            return [m.start() for m in re.finditer(needle, hay, re.I)][:20]
        except re.error:
            pass
    out: list[int] = []
    start = 0
    while len(out) < 20:
        idx = hay.find(needle, start)
        if idx < 0:
            break
        out.append(idx)
        start = idx + max(len(needle), 1)
    return out


async def _stdin_enter() -> str:
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()

    def _on_enter() -> None:
        try:
            line = sys.stdin.readline()
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
            return
        # A closed pipe is readable immediately and readline() is "". That is not the user pressing Enter.
        if line == "":
            if not fut.done():
                fut.set_exception(EOFError("stdin is closed"))
            return
        if not fut.done():
            fut.set_result(line)

    try:
        loop.add_reader(sys.stdin.fileno(), _on_enter)
    except Exception:
        return await asyncio.to_thread(input, "Press Enter when done… ")
    try:
        return await fut
    finally:
        try:
            loop.remove_reader(sys.stdin.fileno())
        except Exception:
            pass

_OVERLAY_NAME = re.compile(
    r"^(accept( all)?( cookies)?|i agree|allow all|got it|reject all|"
    r"only (necessary|essential)( cookies)?)$",
    re.I,
)

_DISMISS_JS = """() => {
  if (document.getElementById('__web_agent_hitl')) return 0;
  const closeRe = /^(close|dismiss|got it|ok|accept|agree|not now|no thanks|skip|maybe later|continue|i understand|allow all|reject all|essential)$/i;
  const roots = [...document.querySelectorAll(
    '[role="dialog"], [role="alertdialog"], [aria-modal="true"], [class*="modal"], [class*="cookie"], [id*="onetrust"], [id*="cookie"]'
  )];
  let n = 0;
  for (const root of roots.slice(0, 4)) {
    if (root.id === '__web_agent_hitl') continue;
    const style = window.getComputedStyle(root);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    const blob = (root.innerText || '').toLowerCase();
    const isForm = root.querySelector('input, textarea, select')
      || /easy apply|application|phone|resume|first name/.test(blob);
    if (isForm) continue;
    const btns = [...root.querySelectorAll('button, [role="button"], a')];
    const hit = btns.find((b) => closeRe.test((b.innerText || b.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim()));
    if (hit) { hit.click(); n += 1; }
  }
  return n;
}"""

_VISIBLE_CAPTCHA_JS = """() => {
  const dlg = document.querySelector('[role="dialog"], [aria-modal="true"]');
  const formDialog = !!(dlg && (
    dlg.querySelector('input, textarea, select')
    || /easy apply|phone|resume|first name|mobile phone/i.test(dlg.innerText || '')
  ));
  const challenge = [...document.querySelectorAll('iframe')].some((el) => {
    const src = (el.src || '').toLowerCase();
    const r = el.getBoundingClientRect();
    const s = window.getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden') return false;
    const big = r.width > 250 && r.height > 140 && r.bottom > 0 && r.top < window.innerHeight;
    const challengeSrc = /recaptcha\\/api2\\/bframe|hcaptcha.com\\/captcha|challenges.cloudflare/.test(src);
    return big && (challengeSrc || /recaptcha|hcaptcha/.test(src));
  });
  return { challenge, formDialog };
}"""

_HITL_JS = """async (arg) => {
  const ID = '__web_agent_hitl';
  document.getElementById(ID)?.remove();
  const el = (tag, css, text) => {
    const n = document.createElement(tag);
    n.style.cssText = css;
    if (text != null) n.textContent = text;
    return n;
  };
  const kind = arg.kind || 'question';
  const takeover = kind === 'captcha' || kind === 'login';
  const host = el('div', takeover
    ? 'position:fixed;left:0;right:0;top:0;margin:0;padding:0;border:0;width:100%;max-width:none;z-index:2147483647;pointer-events:none;font-family:ui-sans-serif,system-ui,sans-serif;background:transparent;'
    : 'position:fixed;inset:0;margin:0;padding:0;border:0;width:100%;height:100%;max-width:none;z-index:2147483647;font-family:ui-sans-serif,system-ui,sans-serif;');
  host.id = ID;
  host.setAttribute('popover', 'manual');
  const box = el('div', takeover
    ? 'pointer-events:auto;margin:8px 12px 0;background:#1d4ed8;color:#fff;border-radius:12px;padding:12px 16px;box-shadow:0 12px 40px rgba(0,0,0,.45);display:flex;gap:12px;align-items:center;flex-wrap:wrap;'
    : 'width:min(440px,92vw);background:#fff;color:#0f172a;border-radius:16px;padding:20px 16px;box-shadow:0 25px 50px rgba(0,0,0,.35);');
  const title = el('div', takeover ? 'flex:1;font-size:14px;font-weight:600;line-height:1.4;' : 'margin:0 0 8px;font-size:18px;font-weight:700;',
    takeover ? ((arg.body || arg.title || 'Solve the check, then confirm.') + '  Click I am done or press Enter.') : (arg.title || 'Agent needs you'));
  const p = el('div', 'margin:0 0 12px;font-size:14px;line-height:1.45;white-space:pre-wrap;', arg.body || '');
  const ta = document.createElement('textarea');
  ta.style.cssText = 'width:100%;min-height:88px;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:8px;padding:8px;font:inherit;margin-bottom:12px;';
  ta.placeholder = arg.placeholder || 'Type your answer';
  const row = el('div', takeover ? 'display:flex;gap:8px;' : 'display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;');
  if (takeover) {
    box.appendChild(title);
    box.appendChild(row);
    host.appendChild(box);
  } else {
    const mask = el('div', 'position:fixed;inset:0;background:rgba(15,23,42,.55);display:flex;align-items:center;justify-content:center;');
    const tag = el('div', 'display:inline-block;font-size:11px;letter-spacing:.04em;text-transform:uppercase;background:#dbeafe;color:#1e3a8a;border-radius:999px;padding:2px 8px;margin-bottom:8px;', kind);
    box.appendChild(tag);
    box.appendChild(title);
    box.appendChild(p);
    box.appendChild(ta);
    box.appendChild(row);
    mask.appendChild(box);
    host.appendChild(mask);
  }
  (document.documentElement || document.body).appendChild(host);
  try { host.showPopover(); } catch (e) {}
  return await new Promise((resolve) => {
    const finish = (value) => {
      window.__web_agent_hitl_finish = null;
      window.removeEventListener('keydown', onKey, true);
      try { host.hidePopover(); } catch (e) {}
      host.remove();
      resolve(String(value ?? ''));
    };
    window.__web_agent_hitl_finish = finish;
    const onKey = (e) => {
      if (e.key !== 'Enter' || e.shiftKey) return;
      if (!takeover && e.target === ta) return;
      e.preventDefault();
      finish(takeover ? 'done' : (ta.value.trim() || 'done'));
    };
    window.addEventListener('keydown', onKey, true);
    const addBtn = (label, bg, color, valueFn) => {
      const b = el('button', 'border:0;border-radius:8px;padding:8px 14px;font:inherit;font-weight:700;cursor:pointer;background:' + bg + ';color:' + color + ';', label);
      b.addEventListener('click', () => finish(valueFn()));
      row.appendChild(b);
    };
    if (kind === 'approval') {
      ta.style.display = 'none';
      addBtn('No', '#e2e8f0', '#0f172a', () => 'no');
      addBtn('Yes, allow', '#2563eb', '#fff', () => 'yes');
    } else if (takeover) {
      addBtn('I am done', '#fff', '#1d4ed8', () => 'done');
    } else {
      addBtn('Send to agent', '#2563eb', '#fff', () => ta.value.trim());
    }
  });
}"""

_SCROLL_INFO_JS = """() => ({
  h: document.documentElement.scrollHeight,
  ih: window.innerHeight,
  top: window.scrollY,
})"""

_NESTED_SCROLL_JS = """(dir) => {
  const sign = dir === 'up' ? -1 : 1;
  const x = Math.floor(window.innerWidth / 2);
  const y = Math.floor(window.innerHeight / 2);
  let n = document.elementFromPoint(x, y);
  while (n && n !== document.body && n !== document.documentElement) {
    const sh = n.scrollHeight || 0;
    const ch = n.clientHeight || 0;
    if (sh > ch + 40) {
      const before = n.scrollTop;
      n.scrollBy(0, sign * Math.max(ch * 0.9, 400));
      if (n.scrollTop !== before) return 'nested';
    }
    n = n.parentElement;
  }
  const root = document.scrollingElement || document.documentElement;
  const before = root.scrollTop;
  root.scrollBy(0, sign * Math.max(window.innerHeight * 0.9, 400));
  return root.scrollTop !== before ? 'window' : 'none';
}"""
_BOT_WALL = re.compile(
    r"captcha|recaptcha|hcaptcha|cloudflare|just a moment|access denied|"
    r"unusual traffic|verify you are human|attention required",
    re.I,
)

_CONTENT_JS = """() => {
  const sel = [
    'article', '[role="article"]', '[data-id*="urn:li:activity"]',
    '.feed-shared-update-v2', '[data-testid*="tweet"]', 'main [data-testid="cellInnerDiv"]',
  ].join(',');
  const bits = [];
  for (const el of [...document.querySelectorAll(sel)].slice(0, 15)) {
    const t = (el.innerText || '').replace(/\\s+/g, ' ').trim();
    if (t.length > 50) bits.push(t.slice(0, 500));
  }
  let blob = bits.join('\\n---\\n');
  const wiki = document.querySelector('#mw-content-text .mw-parser-output, #mw-content-text');
  if (wiki) {
    const introEl = wiki.querySelector('p');
    const intro = ((introEl || {}).innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 700);
    const paraLinks = introEl
      ? [...introEl.querySelectorAll('a')].map((a) => (a.innerText || '').replace(/\\s+/g, ' ').trim()).filter((t) => t && t.length < 60).slice(0, 8)
      : [];
    const parts = [];
    for (const h of [...wiki.querySelectorAll('h2')].slice(0, 10)) {
      const wrap = h.closest('.mw-heading') || h;
      let cur = wrap.nextElementSibling;
      let t = '';
      while (cur && t.length < 320) {
        const cls = String(cur.className || '');
        if (/^H2$/.test(cur.tagName) || cls.includes('mw-heading')) break;
        t += ' ' + (cur.innerText || '');
        cur = cur.nextElementSibling;
      }
      const title = (h.innerText || '').replace(/\\[edit\\]/gi, '').trim();
      if (title) parts.push(title + ': ' + t.replace(/\\s+/g, ' ').trim().slice(0, 320));
    }
    blob = (paraLinks.length ? ('first_paragraph_links: ' + paraLinks.join(' | ') + '\\n') : '')
      + (intro ? ('intro: ' + intro + '\\n') : '') + parts.join('\\n');
  } else if (blob.length < 80) {
    const main = document.querySelector(
      'main, [role="main"], article, #content, #firstHeading'
    ) || document.body;
    blob = (main && main.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 4000);
  }
  const alerts = [...document.querySelectorAll('#flash, [role="alert"], .flash.success, .flash.error')]
    .map((el) => (el.innerText || '').replace(/\\s+/g, ' ').trim())
    .filter((t) => t.length > 2)
    .slice(0, 3);
  if (alerts.length) blob = 'alerts: ' + alerts.join(' | ') + '\\n' + blob;
  if (!wiki) {
    const heads = [...document.querySelectorAll('main h2, main h3, article h2, #content h2, #content h3')]
      .map((h) => (h.innerText || '').replace(/\\s+/g, ' ').trim())
      .filter((t) => t && t.length < 90)
      .slice(0, 24);
    if (heads.length >= 2) blob = 'headings: ' + heads.join(' | ') + '\\n' + blob;
  }
  const signedOut = !!(
    document.querySelector('input[name="session_key"], input[id="username"], .sign-in-form, .auth-wall')
    || /\\/login|\\/uas\\/login|checkpoint\\/lg/.test(location.pathname)
  );
  return { blob, signedOut };
}"""

_STAMP_JS = """() => {
  const visible = (el) => {
    const s = window.getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return s.display !== "none" && s.visibility !== "hidden"
      && Number(s.opacity) !== 0 && r.width > 0 && r.height > 0;
  };
  const labelOf = (el) => {
    const raw = el.getAttribute("aria-label") || el.getAttribute("placeholder")
      || el.getAttribute("title") || el.getAttribute("alt")
      || (el.tagName === "INPUT" || el.tagName === "TEXTAREA" ? el.value : "")
      || el.innerText || el.getAttribute("href") || "";
    return raw.replace(/\\s+/g, " ").trim().slice(0, 80);
  };
  const roleOf = (el) => {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit;
    const tag = el.tagName;
    if (tag === "A") return "link";
    if (tag === "BUTTON") return "button";
    if (tag === "SELECT") return "combobox";
    if (tag === "TEXTAREA") return "textbox";
    if (tag === "INPUT") {
      const t = (el.type || "text").toLowerCase();
      if (t === "checkbox") return "checkbox";
      if (t === "radio") return "radio";
      if (t === "submit" || t === "button") return "button";
      return "textbox";
    }
    return tag.toLowerCase();
  };

  const clearRefs = (root) => {
    root.querySelectorAll("[data-agent-ref]").forEach((el) => el.removeAttribute("data-agent-ref"));
    root.querySelectorAll("*").forEach((el) => { if (el.shadowRoot) clearRefs(el.shadowRoot); });
  };
  clearRefs(document);
  const sel = [
    "a[href]", "button", "input", "textarea", "select", "summary",
    "[role=button]", "[role=link]", "[role=textbox]", "[role=checkbox]",
    "[role=radio]", "[role=combobox]", "[role=menuitem]", "[role=tab]",
    "[role=option]", "[contenteditable=true]",
  ].join(",");
  const collect = (root) => {
    const found = [...root.querySelectorAll(sel)];
    root.querySelectorAll("*").forEach((el) => {
      if (el.shadowRoot) found.push(...collect(el.shadowRoot));
    });
    return found;
  };
  const els = collect(document).filter(visible);
  const lines = [];
  for (const h of [...document.querySelectorAll("h1,h2,h3")].filter(visible).slice(0, 12)) {
    const t = (h.innerText || "").replace(/\\s+/g, " ").trim().slice(0, 120);
    if (t) lines.push(`heading "${t}"`);
  }
  els.forEach((el, i) => {
    const ref = i + 1;
    el.setAttribute("data-agent-ref", String(ref));
    lines.push(`[${ref}] ${roleOf(el)} "${labelOf(el)}"`);
  });
  const body = (document.body && document.body.innerText || "").replace(/\\s+/g, " ").trim().slice(0, 1500);
  if (body) lines.push(`text "${body}"`);
  return { url: location.href, title: document.title, text: lines.join("\\n") };
}"""


class PlaywrightBrowser:
    def __init__(
        self,
        *,
        headless: bool = False,
        slow_mo: float = 0,
        snapshot: Literal["aria", "dom"] = "aria",
        storage_state: str | None = None,
        user_data_dir: str | None = None,
        profile_directory: str | None = None,
        channel: str | None = None,
        cdp_url: str | None = None,
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.snapshot = snapshot
        self.storage_state = storage_state
        self.user_data_dir = user_data_dir
        self.profile_directory = profile_directory
        self.channel = channel
        self.cdp_url = cdp_url
        self._owns_browser = True
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._aria_refs: dict[int, str] = {}
        self._refs: dict[int, ElementRef] = {}
        self._dialog_action: tuple[str, str | None] = ("dismiss", None)
        self._pending_chooser: Any = None
        self._tab_locked = False
        self._seen_page_ids: set[int] = set()
        self._hitl_open = False
        self._return_app = ""
        self._user_focused = False
        self._rest_gen = 0
        self._rest_task: asyncio.Task | None = None
        self._bg_profile = ""
        self._bg_app = ""

    async def start(self) -> "PlaywrightBrowser":
        from playwright.async_api import async_playwright
        from playwright.async_api import Error as PlaywrightError

        await self._remember_front_app()
        self._pw = await async_playwright().start()
        try:
            if self.cdp_url:
                await self._connect_cdp()
            elif self.user_data_dir:
                await self._launch_persistent()
            else:
                await self._launch_fresh()
        except PlaywrightError as e:
            await self._discard()
            raise self._launch_error(e) from e
        except Exception:
            await self._discard()
            raise
        self._bind_context()
        self._bind_page(self._page)
        await self._install_overlay_handler(self._page)
        self._mark_pages_seen()
        if not self.cdp_url:
            await self._rest_in_background()
            self._schedule_rest()
        return self

    async def _connect_cdp(self) -> None:
        last: Exception | None = None
        for attempt in range(4):
            try:
                self._browser = await self._pw.chromium.connect_over_cdp(
                    self.cdp_url,
                    timeout=30_000,
                    is_local=True,
                    no_defaults=True,
                )
                last = None
                break
            except Exception as e:
                last = e
                msg = str(e)
                if "setDownloadBehavior" not in msg and "context management" not in msg.lower():
                    raise
                await asyncio.sleep(0.6 * (attempt + 1))
        if last is not None:
            raise last
        self._owns_browser = False
        if self._browser.contexts:
            self._context = self._browser.contexts[0]
        else:
            self._context = await self._browser.new_context()
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

    def _all_pages(self) -> list[Any]:
        pages: list[Any] = []
        if self._browser is not None:
            for ctx in self._browser.contexts:
                pages.extend(ctx.pages)
        elif self._context is not None:
            pages.extend(self._context.pages)
        return pages

    def _is_internal_url(self, url: str) -> bool:
        u = (url or "").lower()
        return u.startswith((
            "chrome://", "chrome-untrusted://", "devtools://", "edge://",
            "about:blank", "about:newtab",
        )) or u in ("", "about:blank")

    async def adopt_focused_tab(self) -> str:
        """Use a real tab in the debug Chrome. Terminal focus hides Chrome, so do not require visibilityState."""
        pages = [p for p in self._all_pages() if not p.is_closed()]
        http_pages = [p for p in pages if (p.url or "").startswith(("http://", "https://"))]
        chosen = None
        for page in http_pages:
            try:
                if await page.evaluate("document.visibilityState") == "visible":
                    chosen = page
                    break
            except Exception:
                continue
        if chosen is None and http_pages:
            chosen = http_pages[0]
        if chosen is None:
            chosen = next((p for p in pages if not self._is_internal_url(p.url)), None)
        if chosen is None:
            chosen = pages[0] if pages else None
        if chosen is None:
            if self._context is None:
                raise RuntimeError("No browser tab is available")
            chosen = await self._context.new_page()
        self._page = chosen
        self._context = chosen.context
        self._tab_locked = True
        self._mark_pages_seen()
        self._bind_page(self._page)
        try:
            await self._install_overlay_handler(self._page)
        except Exception:
            pass
        title = ""
        try:
            title = await self._page.title()
        except Exception:
            pass
        listing = await self.list_tabs()
        return (
            f"Locked to this tab only: {title} {self._page.url}\n{listing}"
        )

    async def _launch_persistent(self) -> None:
        Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "accept_downloads": True,
            "args": [],
        }
        if self.slow_mo:
            launch_kwargs["slow_mo"] = self.slow_mo
        if self.channel:
            launch_kwargs["channel"] = self.channel
        if self.profile_directory:
            launch_kwargs["args"] = [f"--profile-directory={self.profile_directory}"]
        if self.storage_state:
            launch_kwargs["storage_state"] = self.storage_state
        self._context = await self._pw.chromium.launch_persistent_context(
            self.user_data_dir, **launch_kwargs,
        )
        self._browser = self._context.browser
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

    def _launch_in_background(self) -> bool:
        """macOS focuses any Chrome Playwright starts itself. open -g does not."""
        return (
            sys.platform == "darwin"
            and not self.headless
            and not self.cdp_url
            and not self.storage_state
            and not self.channel
        )

    async def _launch_background_mac(self) -> None:
        from .chrome_profiles import prepare_signed_in_chrome

        user_chrome = await asyncio.to_thread(prepare_signed_in_chrome)
        port = 9334 if user_chrome else _free_port()
        already = bool(
            user_chrome and await asyncio.to_thread(_debug_chrome_matches, port, user_chrome[1])
        )
        if user_chrome and not already and await asyncio.to_thread(_cdp_ready, port):
            port = _free_port()
        if already:
            app, profile, _directory = user_chrome
            self._bg_app = app
            self._bg_profile = profile
        else:
            if user_chrome:
                app, profile, directory = user_chrome
            else:
                app = _playwright_chrome_app(self._pw)
                profile = tempfile.mkdtemp(prefix="web-agent-chrome-")
                directory = ""
            self._bg_app = app
            self._bg_profile = profile
            await asyncio.to_thread(_open_background_chrome, app, profile, port, directory)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if await asyncio.to_thread(_cdp_ready, port):
                break
            await asyncio.sleep(0.15)
        else:
            raise RuntimeError(
                "Chrome opened in the background but the debugging port never came up."
            )
        previous = self.cdp_url
        self.cdp_url = f"http://127.0.0.1:{port}"
        try:
            await self._connect_cdp()
        finally:
            self.cdp_url = previous
        # Quit only the Chrome this agent started (its own profile folder).
        # Leave cdp_url empty so a captcha can still bring this window forward.
        self._owns_browser = bool(self._bg_profile)

    async def _launch_fresh(self) -> None:
        if self._launch_in_background():
            await self._launch_background_mac()
            return
        launch_kwargs: dict[str, Any] = {"headless": self.headless}
        if self.slow_mo:
            launch_kwargs["slow_mo"] = self.slow_mo
        ctx_kwargs: dict[str, Any] = {"accept_downloads": True}
        if self.storage_state:
            ctx_kwargs["storage_state"] = self.storage_state
        self._browser = await self._pw.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context(**ctx_kwargs)
        self._page = await self._context.new_page()

    def _launch_error(self, e: Exception) -> RuntimeError:
        msg = str(e)
        low = msg.lower()
        if self.cdp_url:
            return RuntimeError(
                f"Could not connect to {self.cdp_url}. Start Chrome with "
                "--remote-debugging-port=9222 and leave it open.\n"
                f"Original error: {msg}"
            )
        if "singleton" in low or "already in use" in low:
            return RuntimeError(
                "That browser profile is already open. Quit Chrome/Edge completely "
                "(Cmd+Q), then run again — or choose option c to attach to a running "
                "Chrome with remote debugging.\n"
                f"Original error: {msg}"
            )
        return RuntimeError(
            "Chromium is not installed for this Python Playwright, or the chosen "
            "browser channel is missing. Run:\n"
            "  python -m playwright install chromium\n"
            f"Original error: {msg}"
        )

    def _alive(self) -> bool:
        page = self._page
        if page is None:
            return False
        try:
            if page.is_closed():
                return False
        except Exception:
            return False
        browser = self._browser
        context = getattr(page, "context", None)
        if browser is None and context is not None:
            browser = getattr(context, "browser", None)
        if browser is not None and hasattr(browser, "is_connected"):
            try:
                if not browser.is_connected():
                    return False
            except Exception:
                return False
        return True

    async def _discard(self) -> None:
        pw, context, browser, owns = self._pw, self._context, self._browser, self._owns_browser
        profile = self._bg_profile
        self._bg_profile = ""
        self._bg_app = ""
        self._browser = self._pw = self._context = self._page = None
        if owns:
            try:
                if context is not None:
                    await context.close()
                elif browser is not None:
                    await browser.close()
            except Exception:
                pass
        if pw is not None:
            try:
                await pw.stop()
            except Exception:
                pass
        if profile:
            await asyncio.to_thread(_quit_background_chrome, profile)

    async def ensure_open(self) -> None:
        """Reopen a window the user closed, instead of failing the next message."""
        if self._alive():
            return
        if self.cdp_url and self._pw is not None:
            raise RuntimeError(
                "The Chrome window closed. Leave that Chrome open and try again."
            )
        await self._discard()
        await self.start()

    async def close(self) -> None:
        await self._discard()

    async def __aenter__(self) -> "PlaywrightBrowser":
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _bind_context(self) -> None:
        if self._context is None:
            return
        self._context.on("page", self._on_popup)

    def _on_popup(self, page: Any) -> None:
        self._bind_page(page)

    def _bind_page(self, page: Any) -> None:
        page.on("dialog", self._apply_dialog)
        page.on("filechooser", self._remember_chooser)
        page.on("framenavigated", self._on_main_navigation)

    def _on_main_navigation(self, frame: Any) -> None:
        page = self._page
        if page is None or frame != page.main_frame or self._hitl_open:
            return
        self._schedule_rest()

    async def _install_overlay_handler(self, page: Any) -> None:
        try:
            await page.add_locator_handler(
                page.get_by_role("button", name=_OVERLAY_NAME),
                self._dismiss_overlay,
            )
        except Exception:
            return

    async def _dismiss_overlay(self) -> None:
        page = self._require_page()
        btn = page.get_by_role("button", name=_OVERLAY_NAME)
        if await btn.count():
            await btn.first.click(timeout=2000)

    async def stabilize(self) -> str:
        """Dismiss popups/overlays and wait until the DOM is briefly quiet."""
        notes: list[str] = []
        notes.extend(await self._adopt_new_pages())
        closed = await self._dismiss_overlays()
        if closed:
            notes.append(closed)
        try:
            await self._wait_settled(2500, network=True)
            notes.append("settled")
        except Exception:
            pass
        return "; ".join(notes)

    async def recover_dynamics(self) -> str:
        """One recovery pass after a no-effect action: overlay, Escape, then scroll if the page is tall."""
        bits: list[str] = []
        closed = await self._dismiss_overlays()
        if closed:
            bits.append(closed)
        page = self._require_page()
        try:
            await page.keyboard.press("Escape")
            bits.append("escape")
        except Exception:
            pass
        try:
            info = await page.evaluate(_SCROLL_INFO_JS)
            if info and info.get("h", 0) > (info.get("ih") or 900) * 1.3:
                await self.scroll("down", 1.0)
                bits.append("scrolled")
        except Exception:
            pass
        try:
            await self._wait_settled(1500, network=False)
        except Exception:
            pass
        return ", ".join(bits) or "no overlay found"

    def _mark_pages_seen(self) -> None:
        for page in self._all_pages():
            self._seen_page_ids.add(id(page))

    async def _adopt_new_pages(self) -> list[str]:
        """Follow a window the page just opened. Tabs that existed at lock time stay put."""
        notes: list[str] = []
        for page in self._all_pages():
            pid = id(page)
            if pid in self._seen_page_ids:
                continue
            self._seen_page_ids.add(pid)
            self._bind_page(page)
            try:
                await self._install_overlay_handler(page)
            except Exception:
                pass
            if page.is_closed():
                continue
            url = page.url or ""
            if not (url.startswith(("http://", "https://")) or url.startswith("about:")):
                continue
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=4000)
            except Exception:
                pass
            title = ""
            try:
                title = await page.title()
            except Exception:
                title = ""
            self._page = page
            self._context = page.context
            notes.append(f"popup_title: {title or '(empty)'} url={page.url or url}")
        return notes

    async def _wait_out_spinner(self) -> None:
        """If a loader appears just after a click, wait until it goes away."""
        page = self._require_page()
        try:
            await page.evaluate(
                """async () => {
                  const visible = () => [...document.querySelectorAll('#loading, [role="progressbar"]')].some((el) => {
                    const s = getComputedStyle(el);
                    const r = el.getBoundingClientRect();
                    return s.display !== 'none' && s.visibility !== 'hidden' && r.height > 2;
                  });
                  const start = Date.now();
                  while (Date.now() - start < 900 && !visible()) {
                    await new Promise((r) => setTimeout(r, 100));
                  }
                  const until = Date.now() + 12000;
                  while (visible() && Date.now() < until) {
                    await new Promise((r) => setTimeout(r, 150));
                  }
                }"""
            )
        except Exception:
            pass

    async def ask_human(
        self,
        question: str,
        *,
        kind: str = "question",
        placeholder: str = "",
        timeout_s: float = 600,
    ) -> str:
        """Show a dialog on the current Chrome tab and wait for an answer."""
        self._hitl_open = True
        self._cancel_rest()
        page = self._require_page()
        try:
            if sys.platform == "darwin" and not self.headless and not self.cdp_url:
                await asyncio.to_thread(_set_agent_chrome_visible, True, True)
                await asyncio.to_thread(_reveal_agent_window, self._bg_app, self._bg_profile)
            await page.bring_to_front()
        except Exception:
            pass
        titles = {
            "question": "Agent needs an answer",
            "form": "Fill this for the application",
            "captcha": "Bot check / CAPTCHA",
            "login": "Sign in, then continue",
            "approval": "Confirm this action",
        }
        bodies = {
            "captcha": "Solve the CAPTCHA on the page (the bar does not cover it). Then click I am done or press Enter.",
            "login": "Sign in or finish 2FA on the page. Then click I am done or press Enter.",
        }
        body = question.strip()
        extra = bodies.get(kind)
        if extra and extra not in body:
            body = f"{body}\n\n{extra}" if body else extra
        print(
            "\n[human-in-loop] Confirm when you are finished:\n"
            "  • Click  I am done  on the blue bar at the top of Chrome, or\n"
            "  • Press Enter in Chrome, or\n"
            "  • Press Enter in this terminal\n",
            flush=True,
        )
        self._hitl_open = True
        payload = {
            "kind": kind,
            "title": titles.get(kind, "Agent needs you"),
            "body": body,
            "placeholder": placeholder,
        }
        try:
            reply = await asyncio.wait_for(self._wait_hitl(page, payload), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                await page.evaluate(
                    "() => { window.__web_agent_hitl_finish && window.__web_agent_hitl_finish(''); "
                    "document.getElementById('__web_agent_hitl')?.remove(); }"
                )
            except Exception:
                pass
            reply = ""
        except Exception as e:
            print(f"(on-page dialog failed: {e})\nType the answer in this terminal instead.", flush=True)
            try:
                reply = await asyncio.to_thread(input, f"\n[AGENT ASKS] {body}\n> ")
            except Exception:
                reply = ""
        finally:
            self._hitl_open = False
            self._schedule_rest()
        return str(reply or "").strip()

    async def _wait_hitl(self, page: Any, payload: dict) -> str:
        eval_task = asyncio.create_task(page.evaluate(_HITL_JS, payload))
        enter_task = asyncio.create_task(_stdin_enter())
        done, pending = await asyncio.wait(
            {eval_task, enter_task}, return_when=asyncio.FIRST_COMPLETED,
        )
        if enter_task in done:
            try:
                line = enter_task.result()
            except Exception:
                line = None
            if not line:
                # No terminal on this process. Wait for the button on the page.
                try:
                    return str(await eval_task or "")
                except Exception:
                    raise
            try:
                await page.evaluate(
                    "() => window.__web_agent_hitl_finish && window.__web_agent_hitl_finish('done')"
                )
            except Exception:
                pass
            try:
                return str(await asyncio.wait_for(eval_task, timeout=2) or "done")
            except Exception:
                return "done"
        enter_task.cancel()
        try:
            await enter_task
        except (asyncio.CancelledError, Exception):
            pass
        return str(eval_task.result() or "")

    async def _dismiss_overlays(self) -> str:
        if self._hitl_open:
            return ""
        page = self._require_page()
        n = 0
        try:
            n = int(await page.evaluate(_DISMISS_JS) or 0)
        except Exception:
            n = 0
        try:
            btn = page.get_by_role("button", name=_OVERLAY_NAME)
            if await btn.count():
                await btn.first.click(timeout=2000)
                n += 1
        except Exception:
            pass
        return f"closed {n} overlay(s)" if n else ""

    def _remember_chooser(self, chooser: Any) -> None:
        self._pending_chooser = chooser

    async def _apply_dialog(self, dialog: Any) -> None:
        action, text = getattr(self, "_dialog_action", ("dismiss", None))
        if action == "accept":
            await dialog.accept(text or "")
        else:
            await dialog.dismiss()

    def _require_page(self) -> Any:
        if self._page is None:
            raise RuntimeError("No browser is open. Call goto or search first.")
        return self._page

    def invalidate_refs(self) -> None:
        self._aria_refs = {}
        self._refs = {}

    def descriptors(self) -> dict[int, dict]:
        return {n: e.descriptor() for n, e in self._refs.items()}

    async def _locator(self, ref: int) -> Any:
        page = self._require_page()
        spec = self._refs.get(ref)
        if spec and spec.aria:
            loc = page.get_by_ref(spec.aria)
            try:
                if await loc.count() > 0:
                    return loc.first
            except Exception:
                pass
            healed = await self._heal(page, spec)
            if healed is not None:
                return healed
            raise ValueError(f"No element [{ref}] on the current page. Call observe first.")
        loc = page.locator(f'[data-agent-ref="{ref}"]')
        if await loc.count() == 0:
            if spec:
                healed = await self._heal(page, spec)
                if healed is not None:
                    return healed
            raise ValueError(f"No element [{ref}] on the current page. Call observe first.")
        return loc.first

    async def _heal(self, page: Any, spec: ElementRef) -> Any | None:
        if not spec.role or spec.role in {"generic", "none"}:
            return None
        try:
            loc = page.get_by_role(spec.role, name=spec.name or None)
            n = await loc.count()
        except Exception:
            return None
        if n == 0:
            return None
        return loc.nth(min(spec.nth, n - 1))

    async def observe(self) -> Observation:
        if not self._alive():
            return Observation(
                url="",
                title="",
                text=(
                    "No browser is open. Call goto or search when the task needs a web page. "
                    "Leave the browser closed if you can answer without one."
                ),
            )
        try:
            return await self._observe_now()
        except Exception as exc:
            if "has been closed" not in str(exc).lower():
                raise
            self._page = None
            return Observation(
                url="",
                title="",
                text="The browser window closed. Call goto or search to open a page.",
            )

    async def _observe_now(self) -> Observation:
        await self._prepare_view()
        page = self._require_page()
        self.invalidate_refs()
        if self.snapshot == "aria":
            obs = await self._observe_aria(page)
            if obs is not None:
                return await self._enrich_if_thin(obs)
        return await self._enrich_if_thin(await self._observe_dom(page))

    async def _prepare_view(self) -> None:
        try:
            await self._adopt_new_pages()
            await self._dismiss_overlays()
            await self._wait_settled(1200, network=False)
        except Exception:
            pass

    async def _enrich_if_thin(self, obs: Observation) -> Observation:
        page = self._require_page()
        try:
            info = await page.evaluate(_SCROLL_INFO_JS)
        except Exception:
            return obs
        visible = obs.text.split("visible_text:", 1)[-1] if "visible_text:" in obs.text else ""
        tall = bool(info) and info.get("h", 0) > (info.get("ih") or 900) * 1.3
        if not (tall and len(visible.strip()) < 240):
            return obs
        try:
            await self.scroll("down", 1.2)
            await asyncio.sleep(0.35)
            extra = await self._with_visible_text(page, "")
        except Exception:
            return obs
        if extra.strip():
            return obs.model_copy(update={"text": obs.text + "\n(after auto-scroll)" + extra})
        return obs

    async def _decorate_async(self, page: Any, text: str) -> str:
        extra = self._frame_footer(page)
        info: dict = {}
        try:
            raw = await page.evaluate(_VISIBLE_CAPTCHA_JS)
            info = raw if isinstance(raw, dict) else {"challenge": bool(raw)}
        except Exception:
            info = {}
        form_open = bool(info.get("formDialog")) or text.startswith("open_dialog:")
        if form_open:
            extra += (
                "\nform_dialog: Easy Apply / application modal is open. "
                "fill_field the labels. Do NOT treat this as a CAPTCHA. "
                "human_in_loop kind=form only if a value is missing."
            )
        try:
            fields = await self.list_fields()
        except Exception:
            fields = ""
        if fields and not fields.startswith("No form"):
            extra += "\n" + "\n".join(
                f"form_field: {line}" for line in fields.splitlines()[:12]
            )
        if form_open:
            return text + extra
        challenge = bool(info.get("challenge"))
        if challenge and not form_open:
            extra += (
                "\nblock_page: visible CAPTCHA challenge. Call human_in_loop kind=captcha. "
                "Do not call done."
            )
        return text + extra

    async def _observe_aria(self, page: Any) -> Observation | None:
        prefix = ""
        try:
            dialog = page.get_by_role("dialog")
            if await dialog.count():
                snap = await dialog.last.aria_snapshot(mode="ai")
                prefix = (
                    "open_dialog: a modal is open (application form). "
                    "Use fill_field / click_text on this dialog. Do not close it.\n"
                )
            else:
                snap = await page.aria_snapshot(mode="ai")
        except Exception:
            try:
                snap = await page.aria_snapshot(mode="ai")
            except Exception:
                return None
        if not snap or not str(snap).strip():
            return None
        text, self._refs = parse_aria_snapshot(str(snap))
        self._aria_refs = {n: e.aria for n, e in self._refs.items() if e.aria}
        if not self._aria_refs:
            return None
        return Observation(
            url=page.url, title=await page.title(),
            text=await self._decorate_async(page, await self._with_visible_text(page, prefix + text)),
        )

    async def _observe_dom(self, page: Any) -> Observation:
        data = await page.evaluate(_STAMP_JS)
        text = data.get("text", "")
        self._refs = parse_dom_snapshot(text)
        return Observation(
            url=data.get("url", ""),
            title=data.get("title", ""),
            text=await self._decorate_async(page, await self._with_visible_text(page, text)),
        )

    async def _with_visible_text(self, page: Any, text: str) -> str:
        extra = ""
        try:
            data = await page.evaluate(_CONTENT_JS)
        except Exception:
            return text
        blob = (data.get("blob") or "").strip()
        if blob:
            extra += "\nvisible_text:\n" + blob
        if data.get("signedOut"):
            extra += "\nsigned_out: yes — this page looks like a login wall, not a logged-in feed."
        return text + extra

    def _frame_footer(self, page: Any) -> str:
        extras = []
        for i, frame in enumerate(page.frames):
            if frame == page.main_frame:
                continue
            extras.append(f'iframe {i} url={frame.url}')
        if not extras:
            return ""
        return "\nframes (act via snapshot refs if listed above; otherwise the UI is inside a frame):\n" + "\n".join(extras)

    async def goto(self, url: str) -> None:
        await self.ensure_open()
        page = self._require_page()
        await page.goto(url, wait_until="domcontentloaded")
        try:
            await page.wait_for_load_state("load", timeout=8000)
        except Exception:
            pass
        await self.stabilize()
        self._schedule_rest()

    def _ref_label(self, ref: int) -> str:
        spec = self._refs.get(ref)
        if spec is None:
            return f"[{ref}]"
        name = f' "{spec.name}"' if spec.name else ""
        return f"[{ref}] {spec.role}{name}"

    async def click(self, ref: int) -> None:
        label = self._ref_label(ref)
        loc = await self._locator(ref)
        try:
            await loc.scroll_into_view_if_needed(timeout=3000)
        except Exception:
            pass
        await self._dismiss_overlays()
        try:
            await loc.click(timeout=8000)
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            covered = "intercepts pointer events" in low or "subtree intercepts" in low
            if covered:
                await self._dismiss_overlays()
                try:
                    await loc.click(timeout=5000)
                except Exception:
                    raise RuntimeError(
                        f"{label} is covered by another element. Click Accept/Close on the overlay, "
                        "not this same [n]."
                    ) from e
            else:
                head = msg.split("\n")[0][:280]
                raise RuntimeError(
                    f"Could not click {label}: {head}. The page can be loaded and the control still not "
                    "actionable (hidden, offscreen, not a button, or a stale snapshot number). "
                    "Choose a different [n] or search_page; do not retry this ref."
                ) from e
        await self._wait_out_spinner()
        note = await self.stabilize()
        self._schedule_rest()
        return note

    async def double_click(self, ref: int) -> None:
        await (await self._locator(ref)).dblclick()

    async def right_click(self, ref: int) -> None:
        await (await self._locator(ref)).click(button="right")

    async def type_text(self, ref: int, text: str, press_enter: bool = False) -> None:
        loc = await self._locator(ref)
        try:
            await loc.fill(text)
        except Exception:
            await loc.click()
            await self._require_page().keyboard.type(text)
        if press_enter:
            await loc.press("Enter")
        await self.stabilize()
        self._schedule_rest()

    async def scroll(self, direction: str, pages: float = 1.0) -> str:
        page = self._require_page()

        async def length() -> int:
            try:
                return int(await page.evaluate(
                    "() => (document.body && document.body.innerText || '').length"
                ) or 0)
            except Exception:
                return 0

        before = await length()
        steps = max(int(round(pages)), 1)
        for _ in range(steps):
            moved = "none"
            try:
                moved = await page.evaluate(_NESTED_SCROLL_JS, direction)
            except Exception:
                moved = "none"
            if moved == "none":
                height = await page.evaluate("() => window.innerHeight") or 900
                delta = int(height * 0.9)
                await page.mouse.wheel(0, delta if direction == "down" else -delta)
            await asyncio.sleep(0.2)
        after = await length()
        grew = "new text appeared" if after > before + 40 else "no new text"
        return f"Scrolled {direction} ({pages} page); {grew} (text length {before} -> {after})"

    async def scroll_to_text(self, text: str) -> None:
        loc = self._locator_by_label(text)
        await loc.scroll_into_view_if_needed(timeout=5000)

    def _locator_by_label(self, text: str) -> Any:
        page = self._require_page()
        btn = page.get_by_role("button", name=text)
        link = page.get_by_role("link", name=text)
        raw = page.get_by_text(text, exact=False)
        return btn.or_(link).or_(raw).first

    async def search_page(self, pattern: str, max_results: int = 10) -> str:
        needles = [p.strip().lower() for p in pattern.split("|") if p.strip()] or [pattern.lower()]
        snap_hits: list[str] = []
        for n, spec in self._refs.items():
            blob = f"{spec.role} {spec.name}".lower()
            if any(nd in blob for nd in needles):
                name = (spec.name or "")[:140]
                snap_hits.append(f'[{n}] {spec.role} "{name}"')
            if len(snap_hits) >= max_results:
                break
        page = self._require_page()
        try:
            body = await page.evaluate(
                """() => {
                  const el = document.querySelector(
                    '#mw-content-text, main, article, [role="main"], #content'
                  ) || document.body;
                  return el.innerText || '';
                }"""
            )
        except Exception:
            body = await page.inner_text("body")
        hay = (body or "").lower()
        snippets: list[str] = []
        seen: set[int] = set()
        for needle in needles:
            for idx in _search_indexes(hay, needle):
                if len(snippets) >= max_results:
                    break
                if idx in seen:
                    continue
                seen.add(idx)
                ctx0 = max(0, idx - 80)
                ctx1 = min(len(body), idx + max(len(needle), 8) + 80)
                snippet = body[ctx0:ctx1].replace("\n", " ")
                snippets.append(f"  …{snippet}…")
        section = ""
        try:
            section = await page.evaluate(
                """(needle) => {
                  const n = String(needle || '').toLowerCase();
                  if (!n) return '';
                  const hs = [...document.querySelectorAll('h1,h2,h3,h4')];
                  const h = hs.find((el) => (el.innerText || '').toLowerCase().includes(n));
                  if (!h) return '';
                  const wrap = h.closest('.mw-heading');
                  let cur = (wrap || h).nextElementSibling;
                  let t = (h.innerText || '').replace(/\\[edit\\]/gi, '').trim();
                  while (cur && t.length < 1800) {
                    if (/^H[1-4]$/.test(cur.tagName) || (cur.className || '').includes('mw-heading')) break;
                    t += ' ' + (cur.innerText || '');
                    cur = cur.nextElementSibling;
                  }
                  return t.replace(/\\s+/g, ' ').trim();
                }""",
                needles[0],
            )
        except Exception:
            section = ""
        if not snap_hits and not snippets and not section:
            return f'No matches for "{pattern}"'
        parts = [f'Found matches for "{pattern}":']
        if section:
            parts.append(f"Lead: {section[:300]}")
            parts.append("Section text (quote from this and call done if it answers the user; do not click a TOC History link):")
            parts.append(section[:1800])
        if snap_hits:
            heading_hits = [s for s in snap_hits if " heading " in f" {s} "]
            other_hits = [s for s in snap_hits if s not in heading_hits]
            ordered = heading_hits + other_hits
            parts.append("Snapshot matches (use these [n] with read_element; TOC links may be covered):")
            parts.extend(ordered[:max_results])
        if snippets:
            parts.append("Page-text snippets (these numbers are NOT snapshot refs):")
            parts.extend(snippets)
        return "\n".join(parts)

    async def dropdown_options(self, ref: int) -> str:
        loc = await self._locator(ref)
        options = await loc.locator("option").all_inner_texts()
        if not options:
            return "No <option> elements on this control"
        return "\n".join(f"- {o.strip()}" for o in options if o.strip())

    async def drag(self, start_ref: int, end_ref: int) -> None:
        start = await self._locator(start_ref)
        end = await self._locator(end_ref)
        await start.drag_to(end)

    async def handle_dialog(self, action: str, text: str | None = None) -> None:
        self._dialog_action = (action, text)

    async def save_pdf(self, path: str) -> str:
        await self._require_page().pdf(path=path)
        return path

    async def go_back(self) -> None:
        await self._require_page().go_back(wait_until="domcontentloaded")

    async def go_forward(self) -> None:
        await self._require_page().go_forward(wait_until="domcontentloaded")

    async def reload(self) -> None:
        await self._require_page().reload(wait_until="domcontentloaded")

    async def wait(self, seconds: float) -> None:
        await self._require_page().wait_for_timeout(int(seconds * 1000))

    async def wait_for_text(self, text: str, timeout_ms: int = 8000) -> None:
        await self._require_page().get_by_text(text).first.wait_for(timeout=timeout_ms)

    async def wait_for_load(self, state: str = "networkidle", timeout_ms: int = 15000) -> None:
        await self._require_page().wait_for_load_state(state, timeout=timeout_ms)

    async def wait_for(
        self,
        *,
        url: str | None = None,
        text: str | None = None,
        gone: str | None = None,
        settled: bool = False,
        timeout_ms: int = 15000,
    ) -> str:
        page = self._require_page()
        parts: list[str] = []
        if url:
            await page.wait_for_url(url, timeout=timeout_ms)
            parts.append(f"url matched {url!r}")
        if text:
            await page.get_by_text(text).first.wait_for(timeout=timeout_ms)
            parts.append(f"saw {text!r}")
        if gone:
            await page.get_by_text(gone).first.wait_for(state="hidden", timeout=timeout_ms)
            parts.append(f"{gone!r} gone")
        if settled or not parts:
            await self._wait_settled(timeout_ms)
            parts.append("settled")
        return "; ".join(parts)

    async def _wait_settled(self, timeout_ms: int, network: bool = True) -> None:
        page = self._require_page()
        if network:
            try:
                await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 3000))
            except Exception:
                pass
        try:
            await asyncio.wait_for(
                page.evaluate(
                    """async (quietMs) => {
                      await new Promise((resolve) => {
                        let t = setTimeout(resolve, quietMs);
                        const obs = new MutationObserver(() => {
                          clearTimeout(t);
                          t = setTimeout(() => { obs.disconnect(); resolve(); }, quietMs);
                        });
                        obs.observe(document, {subtree: true, childList: true, attributes: true, characterData: true});
                      });
                    }""",
                    300,
                ),
                timeout=max(timeout_ms / 1000, 0.6),
            )
        except Exception:
            pass

    async def scroll_until(self, text: str | None = None, max_steps: int = 8) -> str:
        last = None
        for i in range(max_steps):
            if text:
                loc = self._locator_by_label(text)
                try:
                    if await loc.count():
                        await loc.scroll_into_view_if_needed(timeout=5000)
                        return f"Found {text!r} after {i} scroll(s)"
                except Exception as e:
                    if "strict mode" not in str(e).lower():
                        raise
                    return f"Found {text!r} after {i} scroll(s) (multiple matches; use click_text)"
            sig = await self._require_page().evaluate(
                "() => [document.body && document.body.innerText.length, document.body && document.body.scrollHeight]"
            )
            if sig == last and i > 0:
                return "Page stopped changing"
            last = sig
            await self.scroll("down", 1.0)
        return f"Gave up after {max_steps} scrolls"

    async def click_text(self, text: str) -> None:
        page = self._require_page()
        await self._dismiss_overlays()
        loc = None
        escaped = re.escape(text)
        candidates = [
            page.get_by_role("button", name=re.compile(rf"^\s*{escaped}\s*$", re.I)),
            page.locator('input[type="submit"]').filter(has_text=text),
            page.get_by_role("button", name=text),
            page.get_by_role("link", name=re.compile(rf"^\s*{escaped}\s*$", re.I)),
            page.get_by_role("link", name=text),
            page.get_by_text(text, exact=True),
        ]
        for cand in candidates:
            try:
                n = await cand.count()
            except Exception:
                n = 0
            if n:
                loc = cand.first
                break
        if loc is None:
            loc = self._locator_by_label(text)
        try:
            await loc.scroll_into_view_if_needed(timeout=4000)
        except Exception:
            pass
        try:
            await loc.click(timeout=8000)
        except Exception:
            await loc.click(timeout=3000, force=True)
        page = self._require_page()
        try:
            await page.get_by_role("dialog").last.wait_for(state="visible", timeout=5000)
        except Exception:
            pass
        await self._wait_out_spinner()
        note = await self.stabilize()
        self._schedule_rest()
        return note

    async def list_fields(self) -> str:
        page = self._require_page()
        try:
            rows = await page.evaluate(
                """() => {
                  const dlg = document.querySelector('[role="dialog"], [aria-modal="true"]');
                  const root = dlg || document;
                  const out = [];
                  const seen = new Set();
                  const els = [...root.querySelectorAll('input, textarea, select')];
                  for (const el of els) {
                    const type = (el.tagName === 'SELECT' ? 'select' : (el.type || 'text')).toLowerCase();
                    if (['hidden', 'submit', 'button', 'image', 'file'].includes(type)) continue;
                    const style = window.getComputedStyle(el);
                    const r = el.getBoundingClientRect();
                    if (style.display === 'none' || style.visibility === 'hidden' || r.width < 2) continue;
                    let label = '';
                    if (el.id) {
                      const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                      if (lab) label = lab.innerText || '';
                    }
                    if (!label && el.closest('label')) label = el.closest('label').innerText || '';
                    if (!label) label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('name') || '';
                    label = String(label).replace(/\\s+/g, ' ').trim().slice(0, 80);
                    const key = label + '|' + type;
                    if (!label || seen.has(key)) continue;
                    seen.add(key);
                    out.push(label + ' | ' + type);
                    if (out.length >= 30) break;
                  }
                  return out;
                }"""
            )
        except Exception as e:
            return f"No form fields visible ({e})"
        if not rows:
            return "No form fields visible"
        return "\n".join(str(r) for r in rows)

    async def fill_field(self, label: str, value: str) -> None:
        page = self._require_page()
        dialog = page.get_by_role("dialog")
        root = dialog.last if await dialog.count() else page
        loc = root.get_by_label(label, exact=False)
        if await loc.count() == 0:
            loc = root.get_by_placeholder(label, exact=False)
        if await loc.count() == 0:
            loc = root.get_by_role("textbox", name=re.compile(re.escape(label), re.I))
        if await loc.count() == 0:
            loc = page.get_by_label(label, exact=False)
        if await loc.count() == 0:
            raise RuntimeError(
                f"No field labeled {label!r}. Call human_in_loop kind=form if you need the value, "
                "or pick a [n] textbox from the snapshot."
            )
        await loc.first.fill(value)
        self._schedule_rest()

    async def save_session(self, path: str) -> str:
        if self._context is None:
            raise RuntimeError("PlaywrightBrowser.start() was not called")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await self._context.storage_state(path=str(dest))
        return str(dest)

    async def load_session(self, path: str) -> str:
        if self._browser is None:
            raise RuntimeError("PlaywrightBrowser.start() was not called")
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"No session file at {path}")
        if self._context:
            await self._context.close()
        self._context = await self._browser.new_context(
            storage_state=str(src), accept_downloads=True,
        )
        self._page = await self._context.new_page()
        self._bind_page(self._page)
        await self._install_overlay_handler(self._page)
        self.invalidate_refs()
        return str(src)

    async def upload_file(self, ref: int | None, path: str) -> None:
        if ref is not None:
            loc = await self._locator(ref)
            try:
                await loc.set_input_files(path)
                return
            except Exception:
                page = self._require_page()
                async with page.expect_file_chooser() as pending:
                    await loc.click()
                chooser = await pending.value
                await chooser.set_files(path)
                self._pending_chooser = None
                return
        if self._pending_chooser is None:
            raise ValueError("No file chooser is open; pass a file-input snapshot number")
        await self._pending_chooser.set_files(path)
        self._pending_chooser = None

    async def _remember_front_app(self) -> None:
        if self._return_app or self.headless or self.cdp_url or sys.platform != "darwin":
            return
        name, bundle = await asyncio.to_thread(_frontmost_identity)
        if name and not _is_agent_browser(name, bundle):
            self._return_app = name

    def _cancel_rest(self) -> int:
        self._rest_gen += 1
        task = self._rest_task
        if task is not None and not task.done():
            task.cancel()
        return self._rest_gen

    def _schedule_rest(self) -> None:
        """Watch for Chrome taking focus and send it back. Paused while a person is needed."""
        if self.headless or self.cdp_url or sys.platform != "darwin" or self._hitl_open:
            return
        self._user_focused = False
        task = self._rest_task
        if task is not None and not task.done():
            return

        async def guard() -> None:
            try:
                while True:
                    if not self._hitl_open and not self._user_focused:
                        await self._rest_in_background()
                    await asyncio.sleep(0.35)
            except asyncio.CancelledError:
                return

        try:
            self._rest_task = asyncio.get_running_loop().create_task(guard())
        except RuntimeError:
            return

    async def _rest_in_background(self) -> None:
        """Keep the agent window behind the chat. Focus is reserved for a person."""
        if self.headless or self.cdp_url or sys.platform != "darwin" or self._hitl_open or self._user_focused:
            return
        await asyncio.to_thread(_park_agent_chrome, self._return_app, self._bg_profile)

    async def bring_to_front(self) -> None:
        self._user_focused = True
        self._cancel_rest()
        if sys.platform == "darwin" and not self.headless and not self.cdp_url:
            await asyncio.to_thread(_set_agent_chrome_visible, True, True)
            await asyncio.to_thread(_reveal_agent_window, self._bg_app, self._bg_profile)
        await self._require_page().bring_to_front()

    async def current_location(self) -> dict:
        page = self._page
        if page is None:
            return {"open": False, "url": "", "title": ""}
        try:
            if page.is_closed():
                return {"open": False, "url": "", "title": ""}
        except Exception:
            return {"open": False, "url": "", "title": ""}
        url = ""
        title = ""
        try:
            url = page.url or ""
        except Exception:
            url = ""
        try:
            title = await page.title()
        except Exception:
            title = ""
        return {"open": True, "url": url, "title": title}

    async def click_download(self, ref: int, directory: str) -> str:
        page = self._require_page()
        loc = await self._locator(ref)
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        async with page.expect_download(timeout=30_000) as pending:
            await loc.click()
        download = await pending.value
        dest = folder / download.suggested_filename
        await download.save_as(str(dest))
        return str(dest)

    async def hover(self, ref: int) -> None:
        await (await self._locator(ref)).hover()

    async def select_option(self, ref: int, value: str) -> None:
        loc = await self._locator(ref)
        try:
            await loc.select_option(label=value)
        except Exception:
            await loc.select_option(value=value)

    async def press_key(self, key: str, ref: int | None = None) -> None:
        if ref is None:
            await self._require_page().keyboard.press(key)
            return
        await (await self._locator(ref)).press(key)

    async def read_element(self, ref: int) -> str:
        loc = await self._locator(ref)
        text = (await loc.inner_text()).strip()
        if text:
            return text
        try:
            return (await loc.input_value()).strip()
        except Exception:
            return ""

    async def get_attribute(self, ref: int, name: str) -> str:
        value = await (await self._locator(ref)).get_attribute(name)
        return value or ""

    async def check(self, ref: int) -> None:
        await (await self._locator(ref)).check()

    async def uncheck(self, ref: int) -> None:
        await (await self._locator(ref)).uncheck()

    async def screenshot(self, path: str) -> str:
        await self._require_page().screenshot(path=path, full_page=True)
        return path

    async def new_tab(self, url: str | None = None) -> None:
        await self.ensure_open()
        if self._tab_locked:
            if url:
                await self._require_page().goto(url, wait_until="domcontentloaded")
            self._schedule_rest()
            return
        if self._context is None:
            raise RuntimeError("PlaywrightBrowser.start() was not called")
        self._page = await self._context.new_page()
        self._bind_page(self._page)
        await self._install_overlay_handler(self._page)
        if url:
            await self._page.goto(url, wait_until="domcontentloaded")
        self._schedule_rest()

    async def switch_tab(self, index: int) -> None:
        pages = self._pages()
        if index < 1 or index > len(pages):
            raise ValueError(f"Tab {index} does not exist ({len(pages)} open)")
        self._page = pages[index - 1]

    async def close_tab(self) -> None:
        pages = self._pages()
        if len(pages) <= 1:
            raise ValueError("Cannot close the last tab")
        closing = self._require_page()
        await closing.close()
        self._page = self._pages()[-1]

    async def list_tabs(self) -> str:
        lines = []
        for i, page in enumerate(self._pages(), 1):
            mark = " (current)" if page is self._page else ""
            title = await page.title()
            lines.append(f"[{i}] {title} {page.url}{mark}")
        return "\n".join(lines) or "(no tabs)"

    def _pages(self) -> list[Any]:
        pages = self._all_pages()
        if pages:
            return pages
        if self._context is None:
            raise RuntimeError("PlaywrightBrowser.start() was not called")
        return list(self._context.pages)
