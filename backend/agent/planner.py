"""Planner: strong model that breaks a goal into sub-goals and replans on failure."""
from __future__ import annotations

from typing import Any

from .llm import LLMClient
from .types import Plan, Usage

PLAN_TOOL: dict[str, Any] = {
    "name": "submit_plan",
    "description": "Submit the ordered list of sub-goals.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subgoals": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
                "description": "Ordered, concrete browser sub-goals. The last one produces the final answer.",
            }
        },
        "required": ["subgoals"],
    },
}

SYSTEM = """You are the planner for a browser agent that can ONLY act inside a web browser.
Break the user's goal into 1-6 ordered sub-goals.
Rules:
- Each sub-goal is a concrete outcome reachable in a few browser actions
  (e.g. "Open the pricing page of example.com", "Read the Pro plan price").
- Do not include vague steps like "think" or "plan".
- The LAST sub-goal must produce the final answer for the user.
- Prefer fewer sub-goals. Do not invent requirements the user did not state.
- The last sub-goal must extract/report facts, not only "open the page".
- If the user said not to submit, plan fill-and-report only (no Submit click).
- Missing values, CAPTCHA, login: include a human_in_loop step.
- Never plan purchases, sending messages, or deleting data unless the user explicitly asked.
- Refuse bulk email, resignation-to-everyone, and card-entry as the plan outcome; those must stop.
Always answer by calling submit_plan."""


class PlannerError(RuntimeError):
    pass


class Planner:
    def __init__(self, llm: LLMClient, model: str) -> None:
        self.llm, self.model = llm, model

    async def plan(self, goal: str, usage: Usage, memory_context: str = "") -> Plan:
        user = f"USER GOAL:\n{goal}"
        if memory_context.strip():
            user += (
                "\n\nMEMORY (trusted; from prior runs / this conversation):\n"
                f"{memory_context}\n"
                "Use it. If facts already answer the goal, still open the page to confirm, "
                "then done. For Wikipedia History, prefer search_page + quote section text."
            )
        return await self._call(user, usage)

    async def replan(
        self,
        goal: str,
        completed: list[str],
        failed_subgoal: str,
        reason: str,
        notes: list[str],
        current_url: str,
        usage: Usage,
    ) -> Plan:
        user = (
            f"USER GOAL:\n{goal}\n\n"
            f"COMPLETED SUB-GOALS:\n{chr(10).join(completed) or '(none)'}\n\n"
            f"FAILED SUB-GOAL: {failed_subgoal}\nWHY IT FAILED: {reason}\n\n"
            f"NOTES / MEMORY:\n{chr(10).join(notes) or '(none)'}\n"
            f"CURRENT URL: {current_url}\n\n"
            "Produce a NEW plan for the REMAINING work only, using a different approach "
            "from the one that failed (for example a different site or search query)."
        )
        return await self._call(user, usage)

    async def _call(self, user: str, usage: Usage) -> Plan:
        resp = await self.llm.complete(
            model=self.model, system=SYSTEM, user=user,
            tools=[PLAN_TOOL], force_tool="submit_plan", max_tokens=600,
        )
        usage.add(resp.input_tokens, resp.output_tokens)
        for call in resp.tool_calls:
            if call.name == "submit_plan":
                goals = [g.strip() for g in call.args.get("subgoals", []) if str(g).strip()]
                if goals:
                    return Plan(subgoals=goals)
        raise PlannerError("Planner returned no usable plan")
