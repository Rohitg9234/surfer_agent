"""Chat API. The UI is the static frontend served from the same origin."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.runtime import FRONTEND, Runtime


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class DecisionIn(BaseModel):
    token: str
    value: str = ""


class TitleIn(BaseModel):
    title: str = Field(min_length=1, max_length=72)


class SteerIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "runtime", None) is None:
            app.state.runtime = Runtime()
        app.state.runtime.open()
        yield
        await app.state.runtime.close()

    app = FastAPI(title="Web Agent", lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/status")
    def status():
        return app.state.runtime.status()

    @app.get("/api/conversations")
    def list_conversations():
        return {"conversations": app.state.runtime.list_conversations()}

    @app.post("/api/conversations")
    def create_conversation():
        return app.state.runtime.create_conversation()

    @app.get("/api/conversations/{conv_id}")
    def get_conversation(conv_id: str):
        view = app.state.runtime.conversation_view(conv_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return view

    @app.patch("/api/conversations/{conv_id}")
    def rename_conversation(conv_id: str, body: TitleIn):
        view = app.state.runtime.rename_conversation(conv_id, body.title)
        if view is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        return view

    @app.delete("/api/conversations/{conv_id}")
    def delete_conversation(conv_id: str):
        if not app.state.runtime.delete_conversation(conv_id):
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"ok": True}

    @app.get("/api/search")
    def search(q: str = ""):
        return {"hits": app.state.runtime.search(q)}

    @app.post("/api/decide")
    def decide(body: DecisionIn):
        if not app.state.runtime.resolve(body.token, body.value):
            raise HTTPException(status_code=404, detail="Nothing is waiting for that answer")
        return {"ok": True}

    @app.post("/api/stop")
    def stop_run():
        if not app.state.runtime.stop():
            raise HTTPException(status_code=409, detail="Nothing is running")
        return {"ok": True}

    @app.post("/api/steer")
    def steer_run(body: SteerIn):
        if not app.state.runtime.steer(body.text):
            raise HTTPException(status_code=409, detail="The agent is not running")
        return {"ok": True}

    @app.post("/api/browser/focus")
    async def focus_browser():
        info = await app.state.runtime.focus_browser()
        if not info.get("open"):
            raise HTTPException(status_code=409, detail="No browser is open")
        return info

    @app.post("/api/conversations/{conv_id}/messages")
    async def send_message(conv_id: str, body: MessageIn):
        if app.state.runtime.conversation_view(conv_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found")

        async def events():
            async for chunk in app.state.runtime.stream(conv_id, body.text.strip()):
                yield chunk

        return StreamingResponse(events(), media_type="text/event-stream")

    if FRONTEND.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
    return app


app = create_app()
