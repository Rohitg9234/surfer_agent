"""Shared data types for the agent core."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Risk(str, Enum):
    SAFE = "safe"   # read-only / control actions
    LOW = "low"     # reversible page interaction
    HIGH = "high"   # irreversible or external effect -> always needs confirmation


class ToolCall(BaseModel):
    id: str = ""
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        return f"{self.name}:{sorted(self.args.items())}"


class ToolResult(BaseModel):
    ok: bool
    output: str = ""
    error: str | None = None


class Observation(BaseModel):
    """What the Perception layer returns. `text` is the numbered snapshot."""
    url: str = ""
    title: str = ""
    text: str = ""
    screenshot_b64: str | None = None

    @property
    def fingerprint(self) -> str:
        return hashlib.sha1(f"{self.url}\n{self.text}".encode()).hexdigest()


class VerdictStatus(str, Enum):
    OK = "ok"
    NO_EFFECT = "no_effect"
    ERROR = "error"
    LOOP = "loop"
    BLOCKED = "blocked"


class Verdict(BaseModel):
    status: VerdictStatus
    reason: str = ""


class Plan(BaseModel):
    subgoals: list[str]


class Step(BaseModel):
    index: int
    subgoal: str
    call: ToolCall
    result: ToolResult
    verdict: Verdict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


class RunResult(BaseModel):
    success: bool
    answer: str = ""
    stop_reason: str
    steps: list[Step] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
