"""python -m server  →  http://127.0.0.1:8787"""
from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
    os.chdir(Path(__file__).resolve().parent.parent)
    port = int(os.getenv("PORT", "8787"))
    print(f"Web Agent  →  http://127.0.0.1:{port}", flush=True)
    uvicorn.run("server.app:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
