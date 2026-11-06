"""Thin LLM gateway. Everything in the core calls the model through here."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import anthropic_api_key, grok_api_key
from .types import ToolCall


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        force_tool: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept Anthropic `{name, input_schema}` tools and OpenAI function tools."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" or "function" in tool:
            converted.append(tool)
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or tool.get("parameters") or {
                    "type": "object",
                    "properties": {},
                },
            },
        })
    return converted


def _parse_openai_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    parsed = json.loads(raw)
    return dict(parsed) if isinstance(parsed, dict) else {}


class AnthropicLLM:
    """Single-turn calls. We deliberately do not keep a growing chat history:
    each step sends goal + compact history + the CURRENT page only, which keeps
    cost flat as runs get longer."""

    def __init__(self, api_key: str | None = None, max_retries: int = 3):
        from anthropic import AsyncAnthropic  # imported lazily so tests need no SDK

        self._client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        force_tool: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
        if tools:
            cached = [dict(t) for t in tools]
            if cached:
                cached[-1] = {**cached[-1], "cache_control": {"type": "ephemeral"}}
            kwargs["tools"] = cached
            kwargs["tool_choice"] = (
                {"type": "tool", "name": force_tool} if force_tool else {"type": "any"}
            )
        resp = await self._client.messages.create(**kwargs)

        text, calls = "", []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))
        return LLMResponse(text, calls, resp.usage.input_tokens, resp.usage.output_tokens)


class GrokLLM:
    """xAI Grok via the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        *,
        base_url: str | None = None,
        client: Any | None = None,
    ):
        if client is not None:
            self._client = client
            return
        from openai import AsyncOpenAI  # imported lazily so tests need no SDK

        self._client = AsyncOpenAI(
            api_key=api_key or grok_api_key(),
            base_url=base_url or "https://api.x.ai/v1",
            max_retries=max_retries,
        )

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        force_tool: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = (
                {"type": "function", "function": {"name": force_tool}}
                if force_tool
                else "required"
            )
        resp = await self._client.chat.completions.create(**kwargs)
        message = resp.choices[0].message
        text = message.content or ""
        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            fn = call.function
            calls.append(ToolCall(
                id=getattr(call, "id", "") or "",
                name=fn.name,
                args=_parse_openai_arguments(fn.arguments),
            ))
        usage = resp.usage
        return LLMResponse(
            text,
            calls,
            getattr(usage, "prompt_tokens", 0) or 0 if usage else 0,
            getattr(usage, "completion_tokens", 0) or 0 if usage else 0,
        )


def llm_from_env(*, max_retries: int = 3) -> AnthropicLLM | GrokLLM:
    """Use Grok when XAI_API_KEY or GROK_API_KEY is set; otherwise Claude."""
    grok_key = grok_api_key()
    if grok_key:
        return GrokLLM(api_key=grok_key, max_retries=max_retries)
    claude_key = anthropic_api_key()
    if claude_key:
        return AnthropicLLM(api_key=claude_key, max_retries=max_retries)
    raise RuntimeError(
        "No LLM API key found. Set XAI_API_KEY (or GROK_API_KEY) for Grok, "
        "or ANTHROPIC_API_KEY for Claude."
    )
