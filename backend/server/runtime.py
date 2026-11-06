"""One browser, one Chroma store, and the threads the UI can resume."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from agent import AgentConfig, AgentCore
from agent.chroma_store import ChromaStore, HashEmbedding, semantic_embedder
from agent.llm import llm_from_env
from agent.memory import Conversation, ConversationStore, LongTermMemory, Memory, SkillStore
from agent.types import Step

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT.parent / "frontend"


def _memory_dir() -> Path:
    override = os.getenv("AGENT_MEMORY_DIR", "").strip()
    return Path(override) if override else ROOT / "memory"


class Runtime:
    def __init__(self, memory_dir: Path | None = None, *, embedder=None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else _memory_dir()
        self._embedder = embedder
        self.conversations: ConversationStore | None = None
        self.long_term: LongTermMemory | None = None
        self.skills: SkillStore | None = None
        self.chroma: ChromaStore | None = None
        self.embed_kind = "local"
        self.browser = None
        self.browser_mode = "idle"
        self.lock = asyncio.Lock()
        self.pending: dict[str, asyncio.Future] = {}
        self._opened = False
        self._turn: asyncio.Task | None = None
        self._events: asyncio.Queue | None = None
        self._steer: list[str] = []
        self._stop = False

    def open(self) -> None:
        if self._opened:
            return
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.conversations = ConversationStore(self.memory_dir / "conversations")
        self.long_term = LongTermMemory(self.memory_dir / "long_term.json")
        self.skills = SkillStore(self.memory_dir / "skills.json")
        self.chroma, self.embed_kind = self._open_chroma(self.memory_dir / "chroma")
        imported = self.chroma.import_legacy(
            self.memory_dir / "conversations",
            self.memory_dir / "long_term.json",
        )
        if imported:
            print(f"Indexed {imported} thread(s) into Chroma.", flush=True)
        self._opened = True

    def _open_chroma(self, path: Path) -> tuple[ChromaStore, str]:
        if self._embedder is not None:
            return ChromaStore(path, embedding=self._embedder), (
                "local" if isinstance(self._embedder, HashEmbedding) else "semantic"
            )
        errors: list[str] = []
        try:
            return ChromaStore(path, embedding=semantic_embedder()), "semantic"
        except Exception as exc:
            errors.append(f"semantic: {exc}")
        try:
            print("Semantic embeddings unavailable. Using the local Chroma index.", flush=True)
            return ChromaStore(path, embedding=HashEmbedding()), "local"
        except Exception as exc:
            errors.append(f"local: {exc}")
        raise RuntimeError(
            "Could not open Chroma at "
            f"{path}. Delete that folder and start again if the embedding mode changed. "
            + " | ".join(errors)
        )

    async def close(self) -> None:
        browser = self.browser
        self.browser = None
        if browser is not None and hasattr(browser, "close"):
            try:
                await browser.close()
            except Exception:
                pass
        self.browser_mode = "idle"

    def status(self) -> dict:
        cfg = AgentConfig.from_env()
        return {
            "browser": self.browser_mode,
            "busy": self.lock.locked(),
            "provider": cfg.provider,
            "planner_model": cfg.planner_model,
            "executor_model": cfg.executor_model,
            "embed_kind": self.embed_kind,
            "threads": len(self.chroma.list_sessions()) if self.chroma else 0,
            "page": self.browser_public(),
        }

    def browser_public(self) -> dict:
        browser = self.browser
        if browser is None:
            return {"open": False, "url": "", "title": ""}
        page = getattr(browser, "_page", None)
        if page is None:
            return {"open": False, "url": "", "title": ""}
        try:
            if page.is_closed():
                return {"open": False, "url": "", "title": ""}
            url = page.url or ""
        except Exception:
            return {"open": False, "url": "", "title": ""}
        return {"open": True, "url": url, "title": ""}

    async def browser_view(self) -> dict:
        info = self.browser_public()
        browser = self.browser
        if not info["open"] or browser is None:
            return info
        fn = getattr(browser, "current_location", None)
        if not callable(fn):
            return info
        try:
            located = await fn()
        except Exception:
            return info
        if isinstance(located, dict):
            info.update({k: located.get(k, info.get(k, "")) for k in ("open", "url", "title")})
        return info

    async def focus_browser(self) -> dict:
        info = await self.browser_view()
        if not info.get("open"):
            return info
        fn = getattr(self.browser, "bring_to_front", None)
        if callable(fn):
            try:
                await fn()
            except Exception:
                pass
        return info

    def stop(self) -> bool:
        task = self._turn
        if task is None or task.done():
            return False
        self._stop = True
        task.cancel()
        return True

    def steer(self, text: str) -> bool:
        task = self._turn
        if task is None or task.done() or self._stop:
            return False
        note = " ".join((text or "").split())[:2000]
        if not note:
            return False
        self._steer.append(note)
        if self._events is not None:
            self._events.put_nowait(_sse("steer", {"text": note}))
        return True

    def _take_steer(self) -> str:
        if self._stop:
            return "__stop__"
        if not self._steer:
            return ""
        notes = self._steer[:]
        self._steer.clear()
        return "\n".join(notes)

    def list_conversations(self) -> list[dict]:
        assert self.chroma is not None
        return self.chroma.list_sessions()

    def create_conversation(self) -> dict:
        assert self.conversations is not None and self.chroma is not None
        conv = Conversation()
        self.conversations.save(conv)
        self.chroma.upsert_conversation(conv)
        return self.conversation_view(conv.id)

    def rename_conversation(self, conv_id: str, title: str) -> dict | None:
        assert self.conversations is not None and self.chroma is not None
        title = " ".join((title or "").split())[:72]
        if not title:
            return None
        conv = self.conversations.load(conv_id)
        if conv is None:
            raw = self.chroma.load_conversation(conv_id)
            if raw is None:
                return None
            conv = Conversation.from_dict(raw)
        conv.title = title
        self.conversations.save(conv)
        self.chroma.rename(conv_id, title)
        return self.conversation_view(conv_id)

    def delete_conversation(self, conv_id: str) -> bool:
        assert self.conversations is not None and self.chroma is not None
        found = self.conversations.load(conv_id) or self.chroma.load_conversation(conv_id)
        if found is None:
            return False
        self.chroma.delete_conversation(conv_id)
        self.conversations.delete(conv_id)
        return True

    def conversation_view(self, conv_id: str) -> dict | None:
        assert self.conversations is not None and self.chroma is not None
        conv = self.conversations.load(conv_id)
        if conv is None:
            raw = self.chroma.load_conversation(conv_id)
            if raw is None:
                return None
            conv = Conversation.from_dict(raw)
        short = conv.messages[-16:]
        query = ""
        for msg in reversed(conv.messages):
            if msg.get("role") == "user":
                query = msg.get("content") or ""
                break
        exclude = [msg.get("content") or "" for msg in short]
        long_term = (
            self.chroma.recall(query, conversation_id=conv.id, exclude=exclude)
            if query else []
        )
        return {
            "id": conv.id,
            "title": conv.title or "New chat",
            "messages": conv.messages,
            "short_term": [
                {"role": msg.get("role") or "", "content": msg.get("content") or "", "at": msg.get("at") or ""}
                for msg in short
            ],
            "long_term": long_term,
        }

    def search(self, query: str) -> list[dict]:
        assert self.chroma is not None
        return self.chroma.search(query) if query.strip() else []

    def resolve(self, token: str, value: str) -> bool:
        fut = self.pending.get(token)
        if fut is None or fut.done():
            return False
        fut.set_result(value)
        return True

    async def stream(self, conv_id: str, text: str):
        if self.lock.locked():
            yield _sse("error", {"message": "The agent is already working on another turn."})
            return
        await self.lock.acquire()
        self._stop = False
        self._steer.clear()
        queue: asyncio.Queue = asyncio.Queue()
        self._events = queue
        task = asyncio.create_task(self._run_turn(conv_id, text, queue))
        self._turn = task
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            self._stop = True
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self._turn = None
            self._events = None
            if self.lock.locked():
                self.lock.release()

    async def _run_turn(self, conv_id: str, text: str, queue: asyncio.Queue) -> None:
        try:
            assert self.conversations is not None and self.chroma is not None
            conv = self.conversations.load(conv_id)
            if conv is None:
                raw = self.chroma.load_conversation(conv_id)
                if raw is None:
                    await queue.put(_sse("error", {"message": "That thread is gone."}))
                    return
                conv = Conversation.from_dict(raw)
            self._attach_browser()
            await queue.put(_sse("status", {"text": "Working"}))
            if self.browser_public().get("open"):
                await queue.put(_sse("browser", self.browser_public()))
            memory = Memory(
                long_term=self.long_term,
                conversation=conv,
                store=self.conversations,
                chroma=self.chroma,
            )
            loop = asyncio.get_running_loop()

            async def confirm(call, reason: str) -> bool:
                token = uuid.uuid4().hex
                fut: asyncio.Future = loop.create_future()
                self.pending[token] = fut
                await queue.put(_sse("confirm", {
                    "token": token,
                    "reason": reason,
                    "name": call.name,
                    "args": call.args,
                }))
                try:
                    value = await asyncio.wait_for(fut, 180)
                except asyncio.TimeoutError:
                    return False
                finally:
                    self.pending.pop(token, None)
                return str(value).strip().lower() in {"y", "yes", "allow", "ok", "true", "1"}

            async def ask(question: str) -> str:
                token = uuid.uuid4().hex
                fut = loop.create_future()
                self.pending[token] = fut
                await queue.put(_sse("ask", {"token": token, "question": question}))
                try:
                    value = await asyncio.wait_for(fut, 180)
                except asyncio.TimeoutError:
                    return "No answer. Continue with what you already have."
                finally:
                    self.pending.pop(token, None)
                return str(value).strip() or "No answer. Continue with what you already have."

            applied: list[str] = []

            def take_steer() -> str:
                note = self._take_steer()
                if note and note != "__stop__":
                    applied.append(note)
                return note

            def on_step(step: Step) -> None:
                detail = step.result.error or step.result.output or ""
                page = self.browser_public()
                queue.put_nowait(_sse("step", {
                    "index": step.index,
                    "name": step.call.name,
                    "args": step.call.args,
                    "status": step.verdict.status.value,
                    "detail": detail[:500],
                    "url": page.get("url") or "",
                }))
                if page.get("open"):
                    queue.put_nowait(_sse("browser", page))

            core = AgentCore(
                self.browser,
                llm_from_env(),
                AgentConfig.from_env(),
                confirm=confirm,
                ask_user=ask,
                on_step=on_step,
                memory=memory,
                skills=self.skills,
                steer=take_steer,
            )
            result = await core.run(text)
            if applied:
                self._remember_steers(conv, applied)
            packed = self.conversation_view(conv.id) or {}
            await queue.put(_sse("done", {
                "success": result.success,
                "answer": result.answer,
                "stop_reason": result.stop_reason,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "recalled": memory.recalled,
                "conversation": packed,
            }))
        except asyncio.CancelledError:
            try:
                packed = self._mark_stopped(conv_id, text) or {}
            except Exception:
                packed = self.conversation_view(conv_id) or {}
            await queue.put(_sse("done", {
                "success": False,
                "answer": "Stopped.",
                "stop_reason": "stopped",
                "conversation": packed,
            }))
        except Exception as exc:
            message = str(exc)
            if "has been closed" in message.lower():
                await self._drop_browser()
                message = "The browser window closed. Send the message again and a new window will open."
            await queue.put(_sse("error", {"message": message}))
        finally:
            await queue.put(None)

    async def _drop_browser(self) -> None:
        browser = self.browser
        self.browser = None
        self.browser_mode = "idle"
        if browser is not None and hasattr(browser, "close"):
            try:
                await browser.close()
            except Exception:
                pass

    def _remember_steers(self, conv: Conversation, notes: list[str]) -> None:
        if not notes or not conv.messages:
            return
        last = conv.messages[-1]
        if last.get("role") != "assistant":
            return
        trace = list(last.get("trace") or [])
        for note in notes:
            trace.append({
                "index": "you",
                "name": "steer",
                "args": {},
                "status": "ok",
                "detail": note[:500],
            })
        last["trace"] = trace
        if self.conversations is not None:
            self.conversations.save(conv)
        if self.chroma is not None:
            try:
                self.chroma.upsert_conversation(conv)
            except Exception:
                pass

    def _mark_stopped(self, conv_id: str, text: str) -> dict | None:
        assert self.conversations is not None
        conv = self.conversations.load(conv_id)
        if conv is None and self.chroma is not None:
            raw = self.chroma.load_conversation(conv_id)
            conv = Conversation.from_dict(raw) if raw else None
        if conv is None:
            return None
        goal = (text or "").strip()
        last = conv.messages[-1] if conv.messages else None
        prev = conv.messages[-2] if len(conv.messages) > 1 else None
        already = (
            last is not None
            and last.get("role") == "assistant"
            and prev is not None
            and prev.get("role") == "user"
            and (prev.get("content") or "") == goal
        )
        if not already:
            conv.add("user", goal)
            conv.add("assistant", "Stopped.")
            self.conversations.save(conv)
            if self.chroma is not None:
                try:
                    self.chroma.upsert_conversation(conv)
                except Exception:
                    pass
        return self.conversation_view(conv_id)

    def _attach_browser(self) -> None:
        """Keep a browser object ready. Chrome starts only when the agent opens a page."""
        if self.browser is not None:
            return
        if os.getenv("AGENT_BROWSER", "").lower() == "mock":
            from agent.testing import MockBrowser
            self.browser = MockBrowser()
            self.browser_mode = "mock"
            return
        from agent.playwright_browser import PlaywrightBrowser
        headless = os.getenv("AGENT_HEADLESS", "").lower() in {"1", "true", "yes"}
        cdp = os.getenv("AGENT_CDP_URL") or None
        self.browser = PlaywrightBrowser(headless=headless, cdp_url=cdp)
        self.browser_mode = "cdp" if cdp else ("headless" if headless else "window")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
