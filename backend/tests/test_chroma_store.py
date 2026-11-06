from fastapi.testclient import TestClient

from agent.chroma_store import ChromaStore, HashEmbedding
from agent.memory import Conversation, ConversationStore, LongTermMemory, Memory
from server.app import create_app
from server.runtime import Runtime


def test_chroma_keeps_a_thread_and_recalls_only_that_thread(tmp_path):
    store = ChromaStore(tmp_path / "chroma", embedding=HashEmbedding())
    conv = Conversation(conv_id="conv-pricing")
    conv.add("user", "What is the Pro plan price on the pricing page?")
    conv.add("assistant", "The Pro plan is $20 per month.")
    store.upsert_conversation(conv)
    store.remember(
        "The Pro plan is $20 per month.",
        kind="fact",
        conversation_id=conv.id,
        url="https://example.com/pricing",
        topic="pricing",
    )
    store.remember(
        "Do not click the Wikipedia TOC History link; search the History section.",
        kind="tactic",
        topic="wikipedia.org history",
    )

    loaded = store.load_conversation(conv.id)
    assert loaded is not None
    assert loaded["messages"][1]["content"].startswith("The Pro plan")
    assert store.list_sessions()[0]["id"] == conv.id

    hits = store.recall("What is the Pro plan price?", conversation_id=conv.id)
    assert any("20" in hit["text"] for hit in hits)

    other = store.recall("What is the Pro plan price?", conversation_id="conv-other")
    assert not any(hit["kind"] == "fact" and "20" in hit["text"] for hit in other)

    tactics = store.recall("wikipedia history section", conversation_id="conv-other")
    assert any(hit["kind"] == "tactic" for hit in tactics)

    store.delete_conversation(conv.id)
    assert store.load_conversation(conv.id) is None
    assert store.list_sessions() == []


def test_memory_indexes_the_thread_in_chroma(tmp_path):
    chroma = ChromaStore(tmp_path / "chroma", embedding=HashEmbedding())
    conv = Conversation(conv_id="conv-a")
    memory = Memory(
        long_term=LongTermMemory(path=None),
        conversation=conv,
        store=ConversationStore(tmp_path / "conversations"),
        chroma=chroma,
    )
    memory.end_run(
        "Pro plan price",
        success=True,
        answer="Pro plan is $20 per month.",
        url="https://example.com/pricing",
    )
    assert chroma.list_sessions()[0]["id"] == "conv-a"
    memory.begin_run("what is the Pro plan price", "https://example.com/pricing")
    assert any("20" in line for line in memory.recalled)


def test_api_lists_and_resumes_threads(tmp_path):
    folder = tmp_path / "conversations"
    folder.mkdir()
    (folder / "conv-old.json").write_text(
        '{"id":"conv-old","title":"LinkedIn history","messages":[{"role":"user","content":"four dated facts","url":"","at":"2026-11-05T17:00:00+00:00"},{"role":"assistant","content":"Founded in 2002.","url":"","at":"2026-11-05T17:01:00+00:00"}]}',
        encoding="utf-8",
    )
    (tmp_path / "long_term.json").write_text(
        '{"facts":[{"text":"Four dated facts: founded in 2002.","url":"","topic":"dated facts","conversation_id":"conv-old"}],"tactics":[],"episodes":[]}',
        encoding="utf-8",
    )
    runtime = Runtime(tmp_path, embedder=HashEmbedding())
    with TestClient(create_app(runtime)) as client:
        listed = client.get("/api/conversations")
        assert listed.status_code == 200
        ids = [row["id"] for row in listed.json()["conversations"]]
        assert ids == ["conv-old"]

        opened = client.get("/api/conversations/conv-old")
        assert opened.status_code == 200
        body = opened.json()
        assert body["messages"][1]["content"] == "Founded in 2002."
        assert body["short_term"][0]["role"] == "user"
        assert any("2002" in item["text"] for item in body["long_term"])

        created = client.post("/api/conversations")
        assert created.status_code == 200
        new_id = created.json()["id"]
        assert client.get(f"/api/conversations/{new_id}").json()["messages"] == []

        renamed = client.patch(f"/api/conversations/{new_id}", json={"title": "Morning notes"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Morning notes"
        titles = {row["id"]: row["title"] for row in client.get("/api/conversations").json()["conversations"]}
        assert titles[new_id] == "Morning notes"

        assert client.delete(f"/api/conversations/{new_id}").status_code == 200
        assert client.get(f"/api/conversations/{new_id}").status_code == 404
        assert client.get("/api/conversations/conv-old").status_code == 200
