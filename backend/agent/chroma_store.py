"""Chroma-backed session history and long-term recall.

Short-term memory stays the recent turns of the open thread (exact text).
Chroma stores every thread so it can be listed and resumed, and it embeds
facts, earlier turns, and site tactics for semantic long-term recall.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from chromadb.utils.embedding_functions import register_embedding_function

_SECRETISH = re.compile(
    r"password|card number|\bcvv\b|supersecretpassword|4111[\d ]{12}",
    re.I,
)
_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp_from_id(conv_id: str) -> str:
    match = re.match(r"conv-(\d{8})-(\d{6})", conv_id or "")
    if not match:
        return ""
    try:
        parsed = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return ""
    return parsed.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")


def _meta(data: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
        elif isinstance(value, float):
            out[key] = value
        elif value is None:
            out[key] = ""
        else:
            out[key] = str(value)[:500]
    return out


@register_embedding_function
class HashEmbedding(EmbeddingFunction):
    """Tiny local embedder for tests and for when the ONNX model cannot download."""

    def __init__(self, dims: int = 48) -> None:
        self.dims = dims

    def __call__(self, input: Documents) -> Embeddings:
        vectors: list[list[float]] = []
        for text in input:
            vec = [0.0] * self.dims
            for token in _TOKEN.findall((text or "").lower()):
                vec[hash(token) % self.dims] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "web_agent_hash"

    def get_config(self) -> dict[str, Any]:
        return {"dims": self.dims}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbedding":
        return HashEmbedding(dims=int(config.get("dims") or 48))

    def default_space(self) -> str:
        return "cosine"


def semantic_embedder():
    """ONNX MiniLM, the model Chroma ships. Downloads once into the user cache."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    model = ONNXMiniLM_L6_V2.DOWNLOAD_PATH / "onnx" / "model.onnx"
    if not model.is_file():
        print("Downloading the Chroma embedding model (one time)…", flush=True)
    embedder = DefaultEmbeddingFunction()
    embedder(["web agent memory"])
    return embedder


class ChromaStore:
    def __init__(self, path: str | Path, *, embedding: EmbeddingFunction | None = None) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding if embedding is not None else HashEmbedding()
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.sessions = self._collection("sessions")
        self.messages = self._collection("messages")
        self.memories = self._collection("memories")

    def _collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_conversation(self, conv: Any, *, sessions_only: bool = False) -> None:
        data = conv.to_dict() if hasattr(conv, "to_dict") else dict(conv)
        conv_id = str(data.get("id") or "").strip()
        if not conv_id:
            return
        messages = [m for m in (data.get("messages") or []) if (m.get("content") or "").strip()]
        title = (data.get("title") or "").strip() or "New chat"
        preview = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and (msg.get("content") or "").strip():
                preview = (msg.get("content") or "").replace("\n", " ")[:160]
                break
        updated = ""
        if messages and messages[-1].get("at"):
            updated = str(messages[-1].get("at"))
        updated = updated or _stamp_from_id(conv_id) or _now()
        created = updated
        existing = self.sessions.get(ids=[conv_id], include=["metadatas"])
        metas = existing.get("metadatas") or []
        if metas and metas[0] and metas[0].get("created_at"):
            created = str(metas[0]["created_at"])
        self.sessions.upsert(
            ids=[conv_id],
            documents=[f"{title}\n{preview or title}"],
            metadatas=[_meta({
                "title": title,
                "preview": preview,
                "updated_at": updated,
                "created_at": created,
                "message_count": len(messages),
            })],
        )
        ids: list[str] = []
        docs: list[str] = []
        metas_out: list[dict] = []
        for index, msg in enumerate(messages):
            content = (msg.get("content") or "").strip()
            if not content or _SECRETISH.search(content):
                continue
            ids.append(f"{conv_id}:{index:04d}")
            docs.append(content[:4000])
            metas_out.append(_meta({
                "conversation_id": conv_id,
                "role": msg.get("role") or "assistant",
                "at": msg.get("at") or "",
                "url": msg.get("url") or "",
                "index": index,
                "title": title,
            }))
        if ids and not sessions_only:
            self.messages.upsert(ids=ids, documents=docs, metadatas=metas_out)

    def rename(self, conv_id: str, title: str) -> None:
        title = " ".join((title or "").split())[:72]
        if not conv_id or not title:
            return
        existing = self.sessions.get(ids=[conv_id], include=["metadatas"])
        if not existing.get("ids"):
            return
        meta = dict((existing.get("metadatas") or [{}])[0] or {})
        preview = str(meta.get("preview") or "")
        meta["title"] = title
        self.sessions.upsert(
            ids=[conv_id],
            documents=[f"{title}\n{preview or title}"],
            metadatas=[_meta(meta)],
        )
        found = self.messages.get(where={"conversation_id": conv_id}, include=["metadatas"])
        ids = found.get("ids") or []
        metas = []
        for item in found.get("metadatas") or []:
            updated = dict(item or {})
            updated["title"] = title
            metas.append(_meta(updated))
        if ids:
            self.messages.update(ids=ids, metadatas=metas)

    def list_sessions(self) -> list[dict]:
        got = self.sessions.get(include=["metadatas", "documents"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        rows = []
        for conv_id, meta in zip(ids, metas):
            meta = meta or {}
            rows.append({
                "id": conv_id,
                "title": meta.get("title") or "Thread",
                "preview": meta.get("preview") or "",
                "updated_at": meta.get("updated_at") or "",
                "created_at": meta.get("created_at") or "",
                "message_count": int(meta.get("message_count") or 0),
            })
        rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return rows

    def load_conversation(self, conv_id: str) -> dict | None:
        session = self.sessions.get(ids=[conv_id], include=["metadatas"])
        if not session.get("ids"):
            return None
        meta = (session.get("metadatas") or [{}])[0] or {}
        got = self.messages.get(where={"conversation_id": conv_id}, include=["documents", "metadatas"])
        rows = []
        for doc, item in zip(got.get("documents") or [], got.get("metadatas") or []):
            item = item or {}
            rows.append((int(item.get("index") or 0), {
                "role": item.get("role") or "assistant",
                "content": doc or "",
                "url": item.get("url") or "",
                "at": item.get("at") or "",
            }))
        rows.sort(key=lambda pair: pair[0])
        return {
            "id": conv_id,
            "title": meta.get("title") or "",
            "messages": [row for _, row in rows],
        }

    def delete_conversation(self, conv_id: str) -> None:
        self.sessions.delete(ids=[conv_id])
        self._delete_where(self.messages, {"conversation_id": conv_id})
        self._delete_where(self.memories, {"conversation_id": conv_id})

    def remember(
        self,
        text: str,
        *,
        kind: str,
        conversation_id: str = "",
        url: str = "",
        topic: str = "",
    ) -> None:
        text = (text or "").strip()
        if not text or _SECRETISH.search(text):
            return
        digest = hashlib.sha1(f"{kind}|{conversation_id}|{text}".encode()).hexdigest()[:20]
        self.memories.upsert(
            ids=[f"{kind}-{digest}"],
            documents=[text[:1500]],
            metadatas=[_meta({
                "kind": kind,
                "conversation_id": conversation_id,
                "url": url,
                "topic": topic[:200],
                "at": _now(),
            })],
        )

    def index_run(self, conv: Any, *, goal: str, success: bool, answer: str, url: str) -> None:
        self.upsert_conversation(conv)
        conv_id = conv.id if hasattr(conv, "id") else str(conv.get("id"))
        answer = (answer or "").strip()
        if success and answer:
            self.remember(answer, kind="fact", conversation_id=conv_id, url=url, topic=goal)
        if answer or goal:
            flag = "ok" if success else "failed"
            self.remember(
                f"{flag}: {(goal or '')[:180]} -> {answer[:500]}",
                kind="episode",
                conversation_id=conv_id,
                url=url,
                topic=goal,
            )

    def recall(
        self,
        query: str,
        *,
        conversation_id: str = "",
        url: str = "",
        exclude: list[str] | None = None,
        limit: int = 6,
        min_score: float = 0.12,
    ) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        skipped = {(item or "").strip() for item in (exclude or []) if item}
        found: list[dict] = []
        for hit in self._query(self.memories, query, 16):
            kind = str((hit.get("metadata") or {}).get("kind") or "")
            cid = str((hit.get("metadata") or {}).get("conversation_id") or "")
            if hit["score"] < min_score:
                continue
            if kind != "tactic" and conversation_id and cid and cid != conversation_id:
                continue
            if hit["text"].strip() in skipped:
                continue
            found.append({
                "text": hit["text"],
                "kind": kind or "memory",
                "score": hit["score"],
                "conversation_id": cid,
            })
        if conversation_id:
            for hit in self._query(self.messages, query, 8, where={"conversation_id": conversation_id}):
                if hit["score"] < min_score or hit["text"].strip() in skipped:
                    continue
                found.append({
                    "text": hit["text"],
                    "kind": "earlier",
                    "score": hit["score"],
                    "conversation_id": conversation_id,
                })
        found.sort(key=lambda item: -item["score"])
        deduped: list[dict] = []
        seen: set[str] = set()
        for item in found:
            key = item["text"].strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def recall_lines(
        self,
        query: str,
        *,
        conversation_id: str = "",
        url: str = "",
        exclude: list[str] | None = None,
        limit: int = 6,
    ) -> list[str]:
        labels = {
            "fact": "fact",
            "episode": "past",
            "tactic": "tactic",
            "earlier": "earlier in this thread",
        }
        lines = []
        for hit in self.recall(
            query, conversation_id=conversation_id, url=url, exclude=exclude, limit=limit,
        ):
            prefix = labels.get(hit["kind"], hit["kind"])
            lines.append(f"{prefix}: {hit['text'][:400]}")
        return lines

    def search(self, query: str, *, limit: int = 8) -> list[dict]:
        best: dict[str, dict] = {}
        for hit in self._query(self.messages, query, max(limit * 3, 12)):
            meta = hit.get("metadata") or {}
            conv_id = str(meta.get("conversation_id") or "")
            if not conv_id:
                continue
            row = best.get(conv_id)
            if row is None or hit["score"] > row["score"]:
                best[conv_id] = {
                    "id": conv_id,
                    "title": meta.get("title") or "Thread",
                    "preview": hit["text"][:160],
                    "score": round(hit["score"], 3),
                }
        ranked = sorted(best.values(), key=lambda item: -item["score"])
        return ranked[:limit]

    def import_legacy(self, conversations_dir: str | Path, long_term_path: str | Path | None = None) -> int:
        folder = Path(conversations_dir)
        imported = 0
        if folder.is_dir():
            known = set(self.sessions.get(include=[]).get("ids") or [])
            for path in sorted(folder.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or not data.get("id"):
                    continue
                message_count = len(data.get("messages") or [])
                stored = 0
                if data["id"] in known:
                    meta = (self.sessions.get(ids=[data["id"]], include=["metadatas"]).get("metadatas") or [{}])[0] or {}
                    stored = int(meta.get("message_count") or 0)
                if data["id"] in known and stored >= message_count:
                    self.upsert_conversation(data, sessions_only=True)
                    continue
                self.upsert_conversation(data)
                imported += 1
        if long_term_path:
            self._import_long_term(Path(long_term_path))
        return imported

    def _import_long_term(self, path: Path) -> None:
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for fact in data.get("facts") or []:
            if isinstance(fact, dict):
                self.remember(
                    fact.get("text") or "",
                    kind="fact",
                    conversation_id=str(fact.get("conversation_id") or ""),
                    url=str(fact.get("url") or ""),
                    topic=str(fact.get("topic") or ""),
                )
        for episode in data.get("episodes") or []:
            if not isinstance(episode, dict):
                continue
            flag = "ok" if episode.get("success") else "failed"
            self.remember(
                f"{flag}: {(episode.get('goal') or '')[:180]} -> {(episode.get('answer') or '')[:500]}",
                kind="episode",
                conversation_id=str(episode.get("conversation_id") or ""),
                url=str(episode.get("url") or ""),
                topic=str(episode.get("goal") or ""),
            )
        for tactic in data.get("tactics") or []:
            if isinstance(tactic, dict):
                self.remember(
                    tactic.get("advice") or "",
                    kind="tactic",
                    topic=str(tactic.get("domain") or ""),
                )

    def _query(self, collection, text: str, n: int, where: dict | None = None) -> list[dict]:
        try:
            count = collection.count()
        except Exception:
            return []
        if count <= 0 or n <= 0:
            return []
        kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results": min(n, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        try:
            raw = collection.query(**kwargs)
        except Exception:
            return []
        docs = (raw.get("documents") or [[]])[0] or []
        metas = (raw.get("metadatas") or [[]])[0] or []
        distances = (raw.get("distances") or [[]])[0] or []
        hits = []
        for doc, meta, distance in zip(docs, metas, distances):
            try:
                score = 1.0 - float(distance)
            except (TypeError, ValueError):
                score = 0.0
            hits.append({"text": doc or "", "metadata": meta or {}, "score": score})
        return hits

    @staticmethod
    def _delete_where(collection, where: dict) -> None:
        try:
            got = collection.get(where=where, include=[])
        except Exception:
            return
        ids = got.get("ids") or []
        if ids:
            collection.delete(ids=ids)
