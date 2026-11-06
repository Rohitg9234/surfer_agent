"""Tool registry: the single source of truth for what the agent can do.

Adding a tool = writing one decorated async function (see builtin.py).
The registry then automatically:
  1. exposes the tool to the Executor LLM (schema generated from the Pydantic model),
  2. validates arguments before anything touches the browser,
  3. tells the Policy layer the tool's risk tier,
  4. tells the Verifier whether the page is expected to change.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from ..types import Risk, ToolCall, ToolResult


@dataclass
class ToolContext:
    """What every tool handler receives."""
    browser: Any                                   # agent.interfaces.Browser
    memory: Any                                    # agent.memory.Memory
    ask_user: Callable[[str], Awaitable[str]]
    vault: Any = None                              # agent.vault.CredentialVault


Handler = Callable[[Any, ToolContext], Awaitable[str]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: type[BaseModel]
    handler: Handler
    risk: Risk = Risk.LOW
    expects_page_change: bool = True   # Verifier flags NO_EFFECT if page stays identical
    control: str | None = None         # "subgoal" | "task" for loop-control tools

    def to_llm_schema(self) -> dict[str, Any]:
        schema = self.params.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


class ToolRegistry:
    def __init__(self, action_timeout: float = 60) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.action_timeout = action_timeout

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_llm_schema() for t in self._tools.values()]

    def executor_schemas(self, names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Stable, cached prefix: always the same tools in the same order."""
        order = names or CORE_TOOL_NAMES
        out = []
        for name in order:
            spec = self._tools.get(name)
            if spec:
                out.append(spec.to_llm_schema())
        return out

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        spec = self.get(call.name)
        if spec is None:
            return ToolResult(ok=False, error=f"Unknown tool '{call.name}'. Available: {self.names()}")
        try:
            args = spec.params(**call.args)
        except ValidationError as e:
            return ToolResult(ok=False, error=f"Invalid arguments for {call.name}: {e.errors()[0]['msg']}")
        try:
            output = await asyncio.wait_for(spec.handler(args, ctx), timeout=self.action_timeout)
            return ToolResult(ok=True, output=output)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, error=f"{call.name} timed out after {self.action_timeout:.0f}s")
        except Exception as e:  # tool failures are data for the agent, not crashes
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")


# Fixed set shown to the executor so the prompt prefix can be cached.
CORE_TOOL_NAMES = (
    "goto",
    "search",
    "click",
    "click_text",
    "fill_field",
    "list_fields",
    "type_text",
    "scroll",
    "scroll_until",
    "wait_for",
    "search_page",
    "read_element",
    "note",
    "go_back",
    "human_in_loop",
    "request_human_help",
    "subgoal_complete",
    "done",
    "more_tools",
    "call_extra",
)

EXTRA_GROUPS = {
    "files": ("upload_file", "click_download", "screenshot", "save_pdf"),
    "tabs": ("new_tab", "switch_tab", "close_tab", "list_tabs"),
    "forms": ("select_option", "check", "uncheck", "fill_credential", "fill_totp", "press_key"),
    "session": ("save_session", "load_session"),
    "playbooks": ("list_playbooks", "load_playbook", "lookup_site_playbook"),
}

default_registry = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    params: type[BaseModel],
    risk: Risk = Risk.LOW,
    expects_page_change: bool = True,
    control: str | None = None,
    registry: ToolRegistry = default_registry,
) -> Callable[[Handler], Handler]:
    def decorator(fn: Handler) -> Handler:
        registry.register(ToolSpec(name, description, params, fn, risk, expects_page_change, control))
        return fn
    return decorator
