"""Discover local Chrome/Chromium/Edge/Brave profiles for persistent login."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass(frozen=True)
class BrowserProfile:
    browser: str
    channel: str | None
    user_data_dir: str
    directory: str
    label: str

    @property
    def display(self) -> str:
        return f"{self.browser} — {self.label} ({self.directory})"


def _chrome_roots() -> list[tuple[str, str | None, Path]]:
    home = Path.home()
    mac = home / "Library" / "Application Support"
    linux = home / ".config"
    win = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [
        ("Google Chrome", "chrome", mac / "Google" / "Chrome"),
        ("Chrome Canary", "chrome-canary", mac / "Google" / "Chrome Canary"),
        ("Chromium", None, mac / "Chromium"),
        ("Microsoft Edge", "msedge", mac / "Microsoft Edge"),
        ("Brave", "chrome", mac / "BraveSoftware" / "Brave-Browser"),
        ("Google Chrome", "chrome", linux / "google-chrome"),
        ("Chromium", None, linux / "chromium"),
        ("Microsoft Edge", "msedge", linux / "microsoft-edge"),
        ("Brave", "chrome", linux / "BraveSoftware" / "Brave-Browser"),
        ("Google Chrome", "chrome", win / "Google" / "Chrome" / "User Data"),
        ("Microsoft Edge", "msedge", win / "Microsoft" / "Edge" / "User Data"),
        ("Brave", "chrome", win / "BraveSoftware" / "Brave-Browser" / "User Data"),
    ]
    seen: set[Path] = set()
    out: list[tuple[str, str | None, Path]] = []
    for name, channel, path in candidates:
        if not path or not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((name, channel, resolved))
    return out


def _read_info_cache(user_data_dir: Path) -> dict[str, dict]:
    state = user_data_dir / "Local State"
    if not state.is_file():
        return {}
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cache = data.get("profile", {}).get("info_cache") or {}
    return cache if isinstance(cache, dict) else {}


def list_profiles() -> list[BrowserProfile]:
    found: list[BrowserProfile] = []
    for browser, channel, root in _chrome_roots():
        cache = _read_info_cache(root)
        dirs = sorted({*cache.keys(), "Default"})
        for directory in dirs:
            folder = root / directory
            if directory != "Default" and not folder.is_dir():
                continue
            info = cache.get(directory) or {}
            name = str(info.get("gaia_name") or info.get("name") or directory)
            email = str(info.get("user_name") or info.get("gaia_id") or "").strip()
            label = f"{name} <{email}>" if email and "@" in email else name
            found.append(BrowserProfile(
                browser=browser,
                channel=channel,
                user_data_dir=str(root),
                directory=directory,
                label=label,
            ))
    return found


def pick_profile_interactive() -> tuple[str | None, str | None, str | None, str | None]:
    """Return (user_data_dir, profile_directory, channel, cdp_url)."""
    profiles = list_profiles()
    print("\nWhere should the agent open the browser?")
    print("  0) Fresh Chromium (not logged in)")
    print("  c) Connect to Chrome already running with remote debugging (port 9222)")
    for i, p in enumerate(profiles, 1):
        print(f"  {i}) {p.display}")
    raw = input("Choose [0]: ").strip().lower() or "0"
    if raw == "c":
        url = os.getenv("AGENT_CDP_URL", "http://127.0.0.1:9222")
        print(f"Attaching via CDP at {url}")
        return None, None, None, url
    if raw == "0":
        return None, None, None, None
    try:
        idx = int(raw)
    except ValueError:
        return None, None, None, None
    if 1 <= idx <= len(profiles):
        p = profiles[idx - 1]
        print(
            f"Using {p.display}.\n"
            "Quit that browser completely first — Chromium locks a profile if it is already open."
        )
        return p.user_data_dir, p.directory, p.channel, None
    return None, None, None, None


def choose_chrome_profile(profiles: list[BrowserProfile]) -> BrowserProfile | None:
    """The signed-in Google Chrome profile, not Canary, Edge, or Chromium."""
    chrome = [
        p for p in profiles
        if p.browser == "Google Chrome" and "Canary" not in p.browser
    ]
    signed = [p for p in chrome if "@" in p.label]
    pool = signed or chrome
    if not pool:
        return None
    for profile in pool:
        if profile.directory == "Default":
            return profile
    return pool[0]


def prepare_signed_in_chrome() -> tuple[str, str, str] | None:
    """Google Chrome.app plus a copy of the signed-in profile.

    Chrome will not open a debugging port on the live profile in
    ~/Library/Application Support/Google/Chrome, and Chrome for Testing cannot
    read that profile's cookies. A copy opened by Google Chrome.app stays
    signed in and can run beside the everyday window.
    """
    app = Path("/Applications/Google Chrome.app")
    if not app.is_dir():
        return None
    chosen = choose_chrome_profile(list_profiles())
    src_root = Path(chosen.user_data_dir) if chosen else (
        Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    )
    directory = chosen.directory if chosen else "Default"
    src_profile = src_root / directory
    if not src_profile.is_dir():
        return None
    dest_root = Path.home() / ".web_agent" / "google-chrome"
    dest_profile = dest_root / directory
    if not (dest_profile / "Cookies").is_file():
        dest_root.mkdir(parents=True, exist_ok=True)
        ignore = shutil.ignore_patterns(
            "Cache", "Code Cache", "GPUCache", "GrShaderCache", "ShaderCache",
            "Crashpad", "Service Worker", "*.log", "OptimizationGuide*",
        )
        print(
            f"Copying {chosen.display if chosen else directory} into {dest_root} (one-time)…",
            flush=True,
        )
        try:
            shutil.copytree(src_profile, dest_profile, dirs_exist_ok=True, ignore=ignore)
        except Exception:
            shutil.rmtree(dest_profile, ignore_errors=True)
            raise
        src_state = src_root / "Local State"
        if src_state.is_file():
            shutil.copy2(src_state, dest_root / "Local State")
    return str(app), str(dest_root), directory


def chrome_binary() -> Path | None:
    for path in (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
    ):
        if path.is_file():
            return path
    return None


def cdp_is_up(url: str, timeout: float = 1.5) -> bool:
    try:
        with urlopen(url.rstrip("/") + "/json/version", timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, OSError, TimeoutError, ValueError):
        return False


def debug_user_data_dir() -> Path:
    path = Path.home() / ".web_agent" / "chrome-debug"
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_debug_profile() -> None:
    """Copy cookies/logins from real Chrome into the debug profile once."""
    dest_root = debug_user_data_dir()
    dest_default = dest_root / "Default"
    if (dest_default / "Cookies").is_file():
        return
    src_root = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    src_default = src_root / "Default"
    if not src_default.is_dir():
        return
    print("Copying Chrome logins into ~/.web_agent/chrome-debug (one-time)…", flush=True)
    ignore = shutil.ignore_patterns(
        "Cache", "Code Cache", "GPUCache", "GrShaderCache", "ShaderCache",
        "Crashpad", "Service Worker", "*.log", "OptimizationGuide*",
    )
    shutil.copytree(src_default, dest_default, dirs_exist_ok=True, ignore=ignore)
    src_state = src_root / "Local State"
    if src_state.is_file() and not (dest_root / "Local State").is_file():
        shutil.copy2(src_state, dest_root / "Local State")


def chrome_debug_command(port: int = 9222) -> list[str]:
    binary = chrome_binary()
    if binary is None:
        raise RuntimeError("Google Chrome.app was not found in /Applications.")
    seed_debug_profile()
    return [
        str(binary),
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={debug_user_data_dir()}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def _how_to_start_chrome(port: int) -> str:
    cmd = " ".join(
        f"'{p}'" if (" " in p or "*" in p) else p for p in chrome_debug_command(port)
    )
    return (
        f"Nothing is listening on 127.0.0.1:{port}.\n\n"
        "Chrome refuses remote debugging on the default profile. We use "
        f"{debug_user_data_dir()} instead (seeded from your Chrome logins once).\n\n"
        "Close any extra Chrome window titled around that folder, then run:\n"
        f"  {cmd}\n"
        "Leave that window open and retry --cdp. Your everyday Chrome can stay open.\n"
    )


def _start_debug_chrome(port: int) -> None:
    cmd = chrome_debug_command(port)
    print("Starting a separate Chrome (debug profile) on port", port, flush=True)
    log_path = Path.home() / ".web_agent" / "chrome-debug-launch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab", buffering=0)
    subprocess.Popen(
        cmd,
        stdout=log,
        stderr=log,
        start_new_session=True,
        close_fds=True,
    )


def ensure_cdp(url: str, *, wait_seconds: float = 30) -> str:
    """Return a CDP HTTP endpoint that is actually accepting connections."""
    if cdp_is_up(url):
        return url
    port = urlparse(url).port or 9222
    endpoint = f"http://127.0.0.1:{port}"
    if cdp_is_up(endpoint):
        return endpoint
    # Everyday Chrome can stay open. A second process with --user-data-dir is fine.
    _start_debug_chrome(port)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if cdp_is_up(endpoint):
            return endpoint
        time.sleep(0.4)
    raise RuntimeError(
        f"Started Chrome but port {port} never opened.\n" + _how_to_start_chrome(port)
    )


def is_cdp_endpoint(value: str | None) -> bool:
    v = (value or "").strip().lower()
    if v.startswith(("http://", "https://", "ws://", "wss://")):
        return True
    host, _, port = v.rpartition(":")
    return bool(host and port.isdigit())


def recover_cdp_goal(
    cdp: str | None,
    goal: str,
    default_goal: str,
) -> tuple[str | None, str]:
    """`--cdp "do this"` must not treat the task as a CDP URL."""
    if not cdp:
        return None, goal
    if is_cdp_endpoint(cdp):
        return cdp, goal
    if goal == default_goal:
        return "http://127.0.0.1:9222", cdp
    return "http://127.0.0.1:9222", f"{cdp} {goal}".strip()


def normalize_cdp_argv(argv: list[str]) -> list[str]:
    """Turn `--cdphttp://127.0.0.1:9222` or `--cdpOpen LinkedIn` into `--cdp` + rest."""
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--cdp") and arg != "--cdp":
            rest = arg[len("--cdp"):].lstrip("=")
            out.extend(["--cdp", rest])
        else:
            out.append(arg)
    return out
