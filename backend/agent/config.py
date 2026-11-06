"""Configuration and budgets. Models are read from env so you can swap them freely."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_LOADED = False


def load_project_env() -> None:
    """Copy keys from the project .env into os.environ if they are not already set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def grok_api_key() -> str | None:
    load_project_env()
    return _first_env("XAI_API_KEY", "GROK_API_KEY")


def anthropic_api_key() -> str | None:
    load_project_env()
    return _first_env("ANTHROPIC_API_KEY")


def llm_provider_from_env() -> str:
    """Prefer Grok when its key is set; otherwise Claude."""
    if grok_api_key():
        return "grok"
    return "anthropic"


@dataclass
class Budget:
    max_steps: int = 12
    max_steps_per_subgoal: int = 5
    max_replans: int = 0
    max_total_tokens: int = 40_000


_GROK_DEFAULTS = ("grok-4", "grok-4-fast")
_CLAUDE_DEFAULTS = ("claude-sonnet-5-5", "claude-haiku-5-5")


@dataclass
class AgentConfig:
    planner_model: str = "claude-sonnet-5-5"   # stronger: planning and replanning
    executor_model: str = "claude-haiku-5-5"   # cheaper: one action per step
    provider: str = "anthropic"
    max_page_chars: int = 3_000
    loop_threshold: int = 3
    max_consecutive_failures: int = 2
    budget: Budget = field(default_factory=Budget)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        load_project_env()
        provider = llm_provider_from_env()
        planner, executor = _GROK_DEFAULTS if provider == "grok" else _CLAUDE_DEFAULTS
        cfg = cls(provider=provider)
        cfg.planner_model = os.getenv("AGENT_PLANNER_MODEL", planner)
        cfg.executor_model = os.getenv("AGENT_EXECUTOR_MODEL", executor)
        cfg.max_page_chars = _int_env("AGENT_MAX_PAGE_CHARS", 8000)
        cfg.max_consecutive_failures = _int_env("AGENT_MAX_CONSEC_FAIL", 3)
        cfg.budget = Budget(
            max_steps=_int_env("AGENT_MAX_STEPS", 24),
            max_steps_per_subgoal=_int_env("AGENT_MAX_STEPS_PER_SUBGOAL", 8),
            max_replans=_int_env("AGENT_MAX_REPLANS", 1),
            max_total_tokens=_int_env("AGENT_MAX_TOKENS", 80_000),
        )
        return cfg


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
