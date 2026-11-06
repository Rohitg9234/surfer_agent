"""Run the agent core with a real LLM.

    set -a && source .env && set +a
    python -m examples.run_demo              # Playwright Chromium (headed)
    python -m examples.run_demo --mock       # fake pages, no browser install
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pathlib import Path

from agent import AgentConfig, AgentCore
from agent.chrome_profiles import (
    ensure_cdp,
    normalize_cdp_argv,
    pick_profile_interactive,
    recover_cdp_goal,
)
from agent.llm import llm_from_env
from agent.memory import ConversationStore, LongTermMemory, Memory, SkillStore
from agent.playwright_browser import PlaywrightBrowser
from agent.testing import MockBrowser
from agent.types import Step, ToolCall

_MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"


def _hitl_hooks(browser):
    async def cli_ask(question: str) -> str:
        fn = getattr(browser, "ask_human", None)
        if callable(fn):
            try:
                return await fn(question, kind="question")
            except Exception as e:
                print(f"(on-page dialog failed: {e})", flush=True)
        return await asyncio.to_thread(input, f"\n[AGENT ASKS] {question}\n> ")

    async def cli_confirm(call: ToolCall, reason: str) -> bool:
        fn = getattr(browser, "ask_human", None)
        prompt = f"{reason}\n\n{call.name}{call.args}"
        if callable(fn):
            try:
                answer = await fn(prompt, kind="approval")
                return answer.strip().lower() in {"y", "yes", "allow", "ok", "true"}
            except Exception as e:
                print(f"(on-page dialog failed: {e})", flush=True)
        answer = await asyncio.to_thread(
            input, f"\n[CONFIRM] {reason}\n  {call.name}{call.args}\n  Allow? (y/N) "
        )
        return answer.strip().lower() == "y"

    return cli_confirm, cli_ask


def show(step: Step) -> None:
    print(f"#{step.index} [{step.verdict.status.value}] {step.call.name}{step.call.args}")


def _chroma_store():
    try:
        from agent.chroma_store import ChromaStore, semantic_embedder
        embedder = semantic_embedder()
        store = ChromaStore(_MEMORY_DIR / "chroma", embedding=embedder)
        store.import_legacy(_MEMORY_DIR / "conversations", _MEMORY_DIR / "long_term.json")
        return store
    except Exception as exc:
        print(f"(chroma memory unavailable: {exc})", flush=True)
        return None


def _persistent_memory(*, fresh: bool = False) -> tuple[Memory, SkillStore]:
    store = ConversationStore(_MEMORY_DIR / "conversations")
    conv = store.resume_or_new(fresh=fresh)
    long_term = LongTermMemory(_MEMORY_DIR / "long_term.json")
    skills = SkillStore(_MEMORY_DIR / "skills.json")
    memory = Memory(
        long_term=long_term, conversation=conv, store=store, chroma=_chroma_store(),
    )
    return memory, skills


def _banner(memory: Memory) -> None:
    conv = memory.conversation
    print()
    print("=" * 62)
    print("  web_agent chat")
    print(f"  conversation: {conv.id}")
    if conv.title:
        print(f"  title: {conv.title}")
    print("  Same-thread memory only (like ChatGPT). Other chats are not mixed in.")
    print("  /new   start a new conversation")
    print("  /quit  exit")
    print("=" * 62)
    if conv.messages:
        print("  (resumed — last messages)")
        for line in conv.transcript(last=6):
            print(line)
        print("-" * 62)


def _print_agent(result) -> None:
    print()
    print(f"Agent: {result.answer}")
    print(f"  [{ 'ok' if result.success else 'failed' } | {result.stop_reason} | "
          f"{result.input_tokens} in / {result.output_tokens} out]")
    print()


async def _one_turn(core: AgentCore, text: str) -> None:
    result = await core.run(text)
    _print_agent(result)


async def _chat(browser, first: str, *, fresh: bool = False) -> None:
    config = AgentConfig.from_env()
    print(f"Using {config.provider}: planner={config.planner_model} executor={config.executor_model}")
    confirm, ask_user = _hitl_hooks(browser)
    memory, skills = _persistent_memory(fresh=fresh)
    core = AgentCore(
        browser, llm_from_env(), config,
        confirm=confirm, ask_user=ask_user, on_step=show,
        memory=memory, skills=skills,
    )
    _banner(memory)
    if first:
        print(f"You: {first}")
        await _one_turn(core, first)
    while True:
        raw = await asyncio.to_thread(input, "You: ")
        text = (raw or "").strip()
        if not text:
            continue
        low = text.lower()
        if low in {"/quit", "/exit", "quit", "exit"}:
            print("Saved this conversation. Bye.")
            return
        if low in {"/new", "/clear"}:
            memory.new_conversation()
            print()
            print(f"(new conversation {memory.conversation.id})")
            print()
            continue
        if low in {"/help"}:
            print("/new  new chat   /quit  exit")
            continue
        await _one_turn(core, text)


async def _run(browser, goal: str, *, chat: bool = True, fresh: bool = False) -> None:
    if chat:
        await _chat(browser, goal, fresh=fresh)
        return
    config = AgentConfig.from_env()
    print(f"Using {config.provider}: planner={config.planner_model} executor={config.executor_model}")
    confirm, ask_user = _hitl_hooks(browser)
    memory, skills = _persistent_memory(fresh=True)
    core = AgentCore(
        browser, llm_from_env(), config,
        confirm=confirm, ask_user=ask_user, on_step=show,
        memory=memory, skills=skills,
    )
    print(f"Goal: {goal}")
    result = await core.run(goal)
    print("\nSUCCESS:", result.success, "| reason:", result.stop_reason)
    print("ANSWER:", result.answer)
    print("TOKENS:", result.input_tokens, "in /", result.output_tokens, "out")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use MockBrowser instead of Playwright")
    parser.add_argument("--headless", action="store_true", help="Run Chromium without a window")
    parser.add_argument(
        "--snapshot",
        choices=("aria", "dom"),
        default="aria",
        help="aria = Playwright accessibility tree (default); dom = querySelector fallback",
    )
    parser.add_argument(
        "--storage-state",
        default=os.getenv("AGENT_STORAGE_STATE"),
        help="Playwright storage_state JSON (cookies/localStorage) to start with",
    )
    parser.add_argument(
        "--user-data-dir",
        default=os.getenv("AGENT_USER_DATA_DIR"),
        help="Chrome/Chromium user-data dir (parent of Default / Profile 1)",
    )
    parser.add_argument("--profile-directory", default=os.getenv("AGENT_PROFILE_DIRECTORY"),
                        help="Chrome profile folder name, e.g. Default or 'Profile 1'")
    parser.add_argument("--channel", default=os.getenv("AGENT_BROWSER_CHANNEL"),
                        help="Playwright channel, e.g. chrome or msedge")
    parser.add_argument(
        "--cdp",
        nargs="?",
        const="http://127.0.0.1:9222",
        default=os.getenv("AGENT_CDP_URL"),
        help="Attach to Chrome on port 9222. Optional value is a CDP URL; a task string is the goal.",
    )
    parser.add_argument("--fresh", action="store_true",
                        help="Skip the profile picker and use a blank Chromium")
    parser.add_argument(
        "--auto-lock",
        action="store_true",
        help="CDP: lock the current tab without waiting for Enter",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=True,
        help="Terminal chat (default). Follow-ups stay in this conversation.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one goal and exit (no chat prompt)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new conversation instead of resuming the last one",
    )
    default_goal = "Find the Pro plan price on the pricing page"
    parser.add_argument("goal", nargs="?", default="")
    args = parser.parse_args(normalize_cdp_argv(sys.argv[1:]))
    empty = ""
    args.cdp, args.goal = recover_cdp_goal(args.cdp, args.goal, empty)
    chat = not args.once
    if args.once and not args.goal:
        args.goal = default_goal
    if args.mock or os.getenv("AGENT_BROWSER", "").lower() == "mock":
        await _run(MockBrowser(), args.goal, chat=chat, fresh=args.new)
        return
    user_data_dir = args.user_data_dir
    profile_directory = args.profile_directory
    channel = args.channel
    cdp_url = args.cdp
    if not args.fresh and not user_data_dir and not cdp_url:
        user_data_dir, profile_directory, channel, cdp_url = await asyncio.to_thread(
            pick_profile_interactive
        )
    if cdp_url:
        cdp_url = await asyncio.to_thread(ensure_cdp, cdp_url)
    async with PlaywrightBrowser(
        headless=args.headless,
        snapshot=args.snapshot,
        storage_state=args.storage_state,
        user_data_dir=user_data_dir,
        profile_directory=profile_directory,
        channel=channel,
        cdp_url=cdp_url,
    ) as browser:
        if args.goal and args.once:
            print(f"Goal: {args.goal}")
        if cdp_url:
            print(await browser.list_tabs())
            if args.auto_lock:
                print(await browser.adopt_focused_tab())
            else:
                print(
                    "\nOpen or click the tab in the debug Chrome window we started "
                    "(not everyday Chrome). Everyday Chrome is not on port 9222.\n"
                    "Then come back here and press Enter."
                )
                await asyncio.to_thread(input, "Press Enter when that tab is ready… ")
                print(await browser.adopt_focused_tab())
            url = getattr(browser, "_page", None)
            current = url.url if url is not None else ""
            goal_has_url = "http://" in (args.goal or "") or "https://" in (args.goal or "")
            if current.startswith(("about:", "chrome://", "edge://")) and not (args.once or goal_has_url):
                print(
                    "That lock is still a blank tab. Open LinkedIn (or your site) "
                    "in the debug Chrome window, then press Enter again."
                )
                await asyncio.to_thread(input, "Press Enter to lock again… ")
                print(await browser.adopt_focused_tab())
            elif current.startswith(("about:", "chrome://", "edge://")):
                print("Starting from a blank tab. The task URL will be opened from here.")
        await _run(browser, args.goal, chat=chat, fresh=args.new)


if __name__ == "__main__":
    asyncio.run(main())
