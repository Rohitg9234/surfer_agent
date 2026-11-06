"""Local credential lookup. Secrets never go into the LLM prompt."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import struct
import time


def totp(secret: str, *, digits: int = 6, period: int = 30) -> str:
    key = secret.strip().replace(" ", "").upper()
    pad = "=" * ((8 - len(key) % 8) % 8)
    raw = base64.b32decode(key + pad, casefold=True)
    counter = int(time.time()) // period
    msg = struct.pack(">Q", counter)
    digest = hmac.new(raw, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10 ** digits)).zfill(digits)


class CredentialVault:
    """Values from an in-memory map, then AGENT_CREDENTIAL_<NAME> env vars."""

    def __init__(self, data: dict[str, str] | None = None) -> None:
        self._data = {k.lower(): v for k, v in (data or {}).items()}

    def get(self, name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise KeyError("Empty credential name")
        if key in self._data:
            return self._data[key]
        env = os.getenv(f"AGENT_CREDENTIAL_{key.upper()}", "").strip()
        if env:
            return env
        raise KeyError(
            f"No credential {name!r}. Set AGENT_CREDENTIAL_{key.upper()} or pass vault data."
        )

    def totp_code(self, name: str = "totp") -> str:
        return totp(self.get(name))
