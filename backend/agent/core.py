"""AgentCore: the observe -> decide -> check -> act -> verify loop.

Flow per step:
  Perception (browser.observe)  ->  Executor picks ONE tool call
  -> Policy gate (allow / confirm / block)  ->  ToolRegistry.execute
  -> browser.observe again  ->  Verifier verdict  ->  Memory
Failure handling: repeated failures, loops, or a step cap per sub-goal trigger
the Planner to replan (up to a limit). Budgets stop runaway runs.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable

from .config import AgentConfig
from .executor import Executor, ExecutorError
from .interfaces import Browser
from .llm import LLMClient
from .memory import Memory, SkillStore
from .playbook_store import PlaybookStore
from .planner import Planner, PlannerError
from .policy import Policy
from .tools import ToolContext, ToolRegistry, default_registry
from .vault import CredentialVault
from .types import (Observation, RunResult, Step, ToolCall, ToolResult, Usage,
                    Verdict, VerdictStatus)
from .verifier import Verifier

ConfirmFn = Callable[[ToolCall, str], Awaitable[bool]]
AskFn = Callable[[str], Awaitable[str]]


async def deny_all(call: ToolCall, reason: str) -> bool:
    """Safe default: risky actions are refused unless you pass a real confirm function."""
    return False


async def no_user(question: str) -> str:
    return "No user is available to answer. Make a reasonable assumption or finish with what you have."


def _page_delta(before: Observation, after: Observation) -> str:
    bits = []
    if before.url != after.url:
        bits.append(f"url {before.url} -> {after.url}")
    if before.title != after.title:
        bits.append(f"title {before.title!r} -> {after.title!r}")
    if bits:
        return "changed: " + "; ".join(bits)
    return ""


class AgentCore:
    def __init__(
        self,
        browser: Browser,
        llm: LLMClient,
        config: AgentConfig | None = None,
        *,
        registry: ToolRegistry | None = None,
        policy: Policy | None = None,
        memory: Memory | None = None,
        skills: SkillStore | None = None,
        confirm: ConfirmFn = deny_all,
        ask_user: AskFn = no_user,
        on_step: Callable[[Step], None] | None = None,
        vault: CredentialVault | None = None,
        steer: Callable[[], str] | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.browser = browser
        self.registry = registry or default_registry
        self.policy = policy or Policy()
        self.memory = memory or Memory()
        self.skills = skills or SkillStore()
        self.playbooks = PlaybookStore()
        self.vault = vault or CredentialVault()
        self.confirm, self.ask_user, self.on_step = confirm, ask_user, on_step
        self.steer = steer
        self._user_notes: list[str] = []
        self.planner = Planner(llm, self.config.planner_model)
        self.executor = Executor(llm, self.config.executor_model, self.registry, self.config.max_page_chars)
        self.verifier = Verifier(self.config.loop_threshold)

    # ------------------------------------------------------------------ run
    async def run(self, goal: str) -> RunResult:
        self._closed = False
        try:
            return await self._run(goal)
        except asyncio.CancelledError:
            return self._finish_stopped(goal)

    def _finish_stopped(self, goal: str) -> RunResult:
        usage = getattr(self, "_usage", None)
        answer = "Stopped."
        if self._closed:
            return RunResult(
                success=False,
                answer=answer,
                stop_reason="stopped",
                steps=self.memory.steps,
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
            )
        self._closed = True
        url = ""
        try:
            page = getattr(self.browser, "_page", None)
            url = page.url if page is not None else ""
        except Exception:
            url = ""
        self.memory.end_run(goal, success=False, answer=answer, url=url or "")
        return RunResult(
            success=False,
            answer=answer,
            stop_reason="stopped",
            steps=self.memory.steps,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
        )

    def _pull_steer(self) -> str:
        fn = self.steer
        if not callable(fn):
            return ""
        try:
            return (fn() or "").strip()
        except Exception:
            return ""

    async def _run(self, goal: str) -> RunResult:
        cfg, budget = self.config, self.config.budget
        usage = Usage()
        self._usage = usage
        ctx = ToolContext(browser=self.browser, memory=self.memory, ask_user=self.ask_user, vault=self.vault)

        def finish(success: bool, reason: str, answer: str = "") -> RunResult:
            if self._closed:
                return RunResult(
                    success=success, answer=answer, stop_reason=reason,
                    steps=self.memory.steps, input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            self._closed = True
            if not success and not answer:
                answer = _stuck_summary(self.memory.steps)
            url = ""
            try:
                page = getattr(self.browser, "_page", None)
                url = page.url if page is not None else ""
            except Exception:
                url = ""
            self.memory.end_run(goal, success=success, answer=answer, url=url or "")
            return RunResult(success=success, answer=answer, stop_reason=reason,
                             steps=self.memory.steps, input_tokens=usage.input_tokens,
                             output_tokens=usage.output_tokens)

        self.policy.goal = goal
        goal_decision = self.policy.check_goal(goal)
        if goal_decision.action == "block":
            return finish(False, "policy_block", goal_decision.reason)
        if goal_decision.action == "confirm":
            allowed = await self.confirm(
                ToolCall(name="run", args={"goal": goal}),
                goal_decision.reason,
            )
            if not allowed:
                return finish(False, "user_declined", goal_decision.reason)

        obs = await self.browser.observe()
        self.policy.observe(obs)
        obs = await self._pause_if_bot_wall(obs)
        self.memory.begin_run(goal, obs.url)

        try:
            plan = await self.planner.plan(goal, usage, self.memory.plan_context())
        except PlannerError as e:
            return finish(False, f"planner_error: {e}")

        sg = sg_steps = consec_fail = replans = 0
        completed: list[str] = []
        sg_calls: list[ToolCall] = []
        feedback = ""

        while True:
            if len(self.memory.steps) >= budget.max_steps:
                return finish(False, "max_steps")
            if usage.total >= budget.max_total_tokens:
                return finish(False, "max_tokens")
            if sg >= len(plan.subgoals):
                return finish(False, "plan_exhausted")

            subgoal = plan.subgoals[sg]
            sg_steps += 1
            index = len(self.memory.steps) + 1
            hint = self.skills.hint(SkillStore.domain_of(obs.url), subgoal)
            recalled = "\n".join(self.memory.recalled)
            if recalled:
                hint = f"{hint}\n{recalled}".strip() if hint else recalled
            site_pb = self.playbooks.for_url(obs.url)
            if site_pb:
                extra = f"Site playbook `{site_pb.name}` is available; call load_playbook if needed."
                hint = f"{hint}\n{extra}".strip() if hint else extra
            guidance = self._pull_steer()
            if guidance == "__stop__":
                return finish(False, "stopped", "Stopped.")
            if guidance:
                self._user_notes.append(guidance)
                correction = (
                    "The user just interrupted. Follow this instruction before anything else:\n"
                    + guidance
                )
                feedback = f"{correction}\n\n{feedback}".strip() if feedback else correction
            if self._user_notes:
                block = (
                    "USER CORRECTIONS (follow these now; they override the current plan):\n"
                    + "\n".join(f"- {note}" for note in self._user_notes[-6:])
                )
                hint = f"{hint}\n{block}".strip() if hint else block

            # 1) DECIDE ------------------------------------------------------
            try:
                call = await self.executor.next_action(
                    goal=goal, subgoal=subgoal, completed=completed, obs=obs,
                    is_final=(sg == len(plan.subgoals) - 1), 
                    history=self.memory.history_summary(), notes=self.memory.prompt_notes(),
                    hint=hint,
                    feedback=feedback, usage=usage,
                    evidence=self.memory.evidence_block(),
                )
                call, spec = _expand_call(call, self.registry)
            except ExecutorError as e:
                step = Step(index=index, subgoal=subgoal, call=ToolCall(name="(none)"),
                            result=ToolResult(ok=False, error=str(e)),
                            verdict=Verdict(status=VerdictStatus.ERROR, reason=str(e)))
                after = obs
            else:
                # 2) POLICY GATE ---------------------------------------------
                decision = self.policy.check(call, spec, obs)
                blocked_reason = ""
                duplicate_read = _duplicate_read(call, self.memory.steps)
                missing_ref = _missing_ref(call, obs)
                if missing_ref:
                    blocked_reason = missing_ref
                elif duplicate_read and _single_extract(goal) and _evidence_rich(self.memory.evidence_block()):
                    drafted = await self._answer_from_evidence(goal, usage)
                    if _usable_answer(drafted):
                        return finish(True, "done_from_evidence", drafted)
                    blocked_reason = (
                        "Already retrieved this. Call done using EVIDENCE. Do not repeat the read.\n"
                        + self.memory.evidence_block()[:1500]
                    )
                elif decision.action == "block":
                    blocked_reason = decision.reason
                elif decision.action == "confirm" and not await self.confirm(call, decision.reason):
                    blocked_reason = f"User declined: {decision.reason}"

                # 3) ACT + 4) VERIFY -----------------------------------------
                if blocked_reason:
                    result = ToolResult(ok=False, error=f"Blocked: {blocked_reason}")
                    verdict = Verdict(status=VerdictStatus.BLOCKED, reason=blocked_reason)
                    after = obs
                else:
                    result = await self.registry.execute(call, ctx)
                    if call.name == "request_human_help" and hasattr(self.browser, "invalidate_refs"):
                        self.browser.invalidate_refs()
                    after = obs if (spec and spec.control) else await self.browser.observe()
                    if call.name not in {"human_in_loop", "request_human_help", "ask_user"}:
                        after = await self._pause_if_bot_wall(after)
                    if spec and not spec.control:
                        self.policy.observe(after)
                        delta = _page_delta(obs, after)
                        if delta and result.ok:
                            result = result.model_copy(
                                update={"output": (result.output + "\n" + delta).strip()}
                            )
                    if result.ok:
                        _remember_evidence(self.memory, call, result.output)
                    verdict = self.verifier.check(call, spec, result, obs, after, self.memory.steps)
                    if (
                        result.ok
                        and verdict.status == VerdictStatus.NO_EFFECT
                        and hasattr(self.browser, "recover_dynamics")
                        and call.name in {"click", "goto", "type_text", "scroll"}
                    ):
                        recovered = await self.browser.recover_dynamics()
                        after2 = await self.browser.observe()
                        if after2.fingerprint != after.fingerprint:
                            after = after2
                            self.policy.observe(after)
                            extra = f"Recovered dynamics: {recovered or 'page changed'}."
                            result = result.model_copy(
                                update={"output": (result.output + "\n" + extra).strip()}
                            )
                            verdict = Verdict(
                                status=VerdictStatus.OK,
                                reason="page changed after overlay/scroll recover",
                            )
                step = Step(index=index, subgoal=subgoal, call=call, result=result, verdict=verdict)

            self.memory.add_step(step)
            if self.on_step:
                self.on_step(step)

            # 5) LOOP CONTROL -------------------------------------------------
            spec = self.registry.get(step.call.name)
            if step.verdict.status == VerdictStatus.OK and spec and spec.control == "task":
                thin = _thin_answer(goal, step.result.output, after)
                if _incomplete_done(step.result.output, after) or thin:
                    if "block_page:" in (after.text or ""):
                        after = await self._pause_if_bot_wall(after, force=True)
                    feedback = thin or (
                        "You called done but the task is NOT finished (missing fields, CAPTCHA, "
                        "or you said you cannot complete it). Use human_in_loop and keep filling. "
                        "Call done only after the application is actually submitted."
                    )
                    consec_fail += 1
                    obs = after
                    continue
                return finish(True, "done", step.result.output)
            if step.verdict.status == VerdictStatus.OK and spec and spec.control == "subgoal":
                if _premature_subgoal(subgoal, step.result.output):
                    feedback = (
                        "subgoal_complete was too early: you only reported navigation. "
                        "Extract the requested facts from visible_text / search_page snippets "
                        "or a snapshot match [n], then call subgoal_complete or done."
                    )
                    consec_fail += 1
                    obs = after
                    continue
                completed.append(f"{subgoal} -> {step.result.output}")
                desc = self.browser.descriptors() if hasattr(self.browser, "descriptors") else {}
                self.skills.record(SkillStore.domain_of(obs.url), subgoal, sg_calls, desc)
                sg, sg_steps, consec_fail, sg_calls, feedback = sg + 1, 0, 0, [], ""
                if sg >= len(plan.subgoals):  # last sub-goal finished without calling `done`
                    return finish(True, "plan_completed", step.result.output)
                continue

            if step.verdict.status == VerdictStatus.OK:
                consec_fail, feedback = 0, ""
                sg_calls.append(step.call)
            else:
                consec_fail += 1
                if consec_fail == 1:
                    extra = (
                        " If a popup/cookie dialog is in the way, click Close/Accept. "
                        "If the page is taller than the snapshot, scroll or scroll_until. "
                        "Otherwise pick a different [n] or search_page."
                    )
                elif consec_fail == 2:
                    extra = " Change strategy: close overlays, scroll for hidden content, another locator, or search. Do not retry the same control."
                else:
                    extra = " Stop retrying. Call done with what you know, or request_human_help."
                feedback = (f"Your last action {step.call.name}{step.call.args} was "
                            f"{step.verdict.status.value.upper()}: {step.verdict.reason}.{extra}")
            obs = after

            stuck = (step.verdict.status == VerdictStatus.LOOP
                     or consec_fail >= cfg.max_consecutive_failures
                     or sg_steps >= budget.max_steps_per_subgoal)
            if stuck:
                if _single_extract(goal) and _evidence_rich(self.memory.evidence_block()):
                    drafted = await self._answer_from_evidence(goal, usage)
                    if _usable_answer(drafted):
                        return finish(True, "done_from_evidence", drafted)
                if budget.max_replans <= 0 or replans >= budget.max_replans:
                    return finish(False, "stuck")
                try:
                    plan = await self.planner.replan(
                        goal, completed, subgoal, step.verdict.reason or "no progress",
                        self.memory.prompt_notes(), obs.url, usage)
                except PlannerError as e:
                    return finish(False, f"planner_error: {e}")
                replans += 1
                sg = sg_steps = consec_fail = 0
                sg_calls, feedback = [], ""


    async def _pause_if_bot_wall(self, obs: Observation, force: bool = False) -> Observation:
        text = obs.text or ""
        if "form_dialog:" in text or "open_dialog:" in text:
            return obs
        captcha = "block_page:" in text
        if not force and not captcha:
            return obs
        fn = getattr(self.browser, "ask_human", None)
        if not callable(fn):
            return obs
        kind = "captcha" if captcha else "form"
        await fn(
            "The agent cannot finish this step alone. Complete the CAPTCHA or form fields "
            "in the page, then click I am done.",
            kind=kind,
        )
        return await self.browser.observe()

    async def _answer_from_evidence(self, goal: str, usage: Usage) -> str:
        try:
            return await self.executor.answer_from_evidence(goal, self.memory.evidence_block(), usage)
        except Exception:
            return ""


_NAV_ONLY = re.compile(
    r"^(navigated to|opened |page content is (now )?visible|successfully (opened|navigated)).{0,220}$",
    re.I,
)
_EXTRACTIVE = re.compile(
    r"\b(quote|summarize|extract|first |stars?|version|weather|headline|"
    r"history|skills?|points?|classes|install command|paragraph)\b",
    re.I,
)


def _premature_subgoal(subgoal: str, summary: str) -> bool:
    text = (summary or "").strip()
    if not text or not _NAV_ONLY.search(text):
        return False
    return bool(_EXTRACTIVE.search(subgoal or ""))


_INCOMPLETE_DONE = re.compile(
    r"\b(cannot|can't|unable|could not|failed to|no application form|"
    r"not extractable|recaptcha|captcha)\b",
    re.I,
)
_HONEST_MISSING = re.compile(
    r"not (visible|listed|shown|available) on the page|if visible|salary not listed",
    re.I,
)


def _thin_answer(goal: str, answer: str, obs: Observation) -> str:
    """Return feedback when done dropped facts that are already on the page."""
    g, a, page = goal or "", answer or "", obs.text or ""
    if re.search(r"install command", g, re.I) and not re.search(r"\bpip3?\b|conda install|npm install", a, re.I):
        if re.search(r"pip3? install", page, re.I):
            return "The install command is on the page. Include it in done (for example pip install ...)."
    if re.search(r"subsection", g, re.I):
        listed = set(re.findall(r"\d+\.\d+", a))
        available = set(re.findall(r"\d+\.\d+", page))
        if len(available) >= 4 and len(listed) < min(4, len(available)):
            return (
                "List every subsection title from the headings line, not just the first. "
                f"The page shows {len(available)} numbered subsections."
            )
    if re.search(r"flash message|login succeeded", g, re.I):
        alert = re.search(r"alerts:\s*([^\n]+)", page)
        if alert:
            words = [w for w in re.findall(r"[A-Za-z]{5,}", alert.group(1)) if w.lower() not in {"close", "alert"}]
            if words and not any(w.lower() in a.lower() for w in words[:4]):
                return f"Quote the flash/alert text from the page: {alert.group(1)[:180]}"
    if re.search(r"hello world", g, re.I) and "hello world" not in a.lower() and "hello world" in page.lower():
        return "Hello World is on the page. Quote it in done."
    return ""


def _incomplete_done(answer: str, obs: Observation) -> bool:
    text = obs.text or ""
    ans = answer or ""
    if "block_page:" in text:
        return True
    if _HONEST_MISSING.search(ans) and "block_page:" not in text:
        return False
    if "form_dialog:" in text or "open_dialog:" in text:
        return bool(_INCOMPLETE_DONE.search(ans) and "submitted" not in ans.lower())
    return bool(_INCOMPLETE_DONE.search(ans))


_READ_EVIDENCE = {"search_page", "read_element", "list_fields", "fill_field", "dropdown_options"}
_REF_ACTIONS = {
    "click", "type_text", "read_element", "double_click", "right_click",
    "hover", "check", "uncheck", "select_option", "get_attribute",
}
_EXTRACT_GOAL = re.compile(
    r"\b(tell me|list|quote|report|give me|summarize|which|how many|"
    r"stars|version|weather|headline|title|points|classes|facts|describe|"
    r"what you filled|install command)\b",
    re.I,
)
_NO_AUTO_DONE = re.compile(
    r"\b(easy apply|do not submit|don't submit|do not book|pay for|resignation|"
    r"enter (a |my )?card)\b",
    re.I,
)


def _remember_evidence(memory: Memory, call: ToolCall, output: str) -> None:
    text = output or ""
    if call.name in _READ_EVIDENCE or "popup_title:" in text or "new text appeared" in text:
        memory.add_evidence(call.name, text)


def _duplicate_read(call: ToolCall, steps: list) -> bool:
    if call.name not in _READ_EVIDENCE:
        return False
    sig = call.signature()
    return any(s.call.signature() == sig and s.result.ok for s in steps)


def _missing_ref(call: ToolCall, obs: Observation) -> str:
    if call.name not in _REF_ACTIONS:
        return ""
    ref = call.args.get("ref")
    if ref is None:
        return ""
    try:
        n = int(ref)
    except (TypeError, ValueError):
        return ""
    text = obs.text or ""
    # DOM snapshots start the line with [n]. Aria snapshots end it: - textbox "Search" [n]
    if re.search(rf"(?m)^\s*\[{n}\]\s", text) or re.search(rf"(?m)\[{n}\]\s*$", text):
        return ""
    links = re.findall(
        r'(?m)(?:^\[(\d+)\]\s+(?:link|button|textbox)\s+"([^"]{1,60})"'
        r'|-\s+(?:link|button|textbox)\s+"([^"]{1,60})"\s+\[(\d+)\])',
        text,
    )
    named = []
    for a, b, c, d in links:
        num, name = (a, b) if a else (d, c)
        if num and name:
            named.append(f"[{num}] {name}")
    sample = ", ".join(named[:6])
    hint = f" Click one of these instead: {sample}." if sample else " Use click_text with a visible link name."
    return (
        f"[{n}] is not in the current snapshot.{hint} "
        "Do not invent snapshot numbers."
    )


def _single_extract(goal: str) -> bool:
    text = goal or ""
    if _NO_AUTO_DONE.search(text) or not _EXTRACT_GOAL.search(text):
        return False
    if text.lower().count("http") > 1 or text.count("\n") >= 2:
        return False
    return True


def _evidence_rich(text: str) -> bool:
    body = (text or "").strip()
    if len(body) < 80:
        return False
    low = body.lower()
    if "no matches" in low and "section text" not in low and "filled " not in low:
        return False
    return True


def _usable_answer(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    return not t.upper().startswith("INSUFFICIENT")


def _expand_call(call: ToolCall, registry: ToolRegistry) -> tuple[ToolCall, object]:
    if call.name != "call_extra":
        return call, registry.get(call.name)
    name = str(call.args.get("name") or "")
    raw = call.args.get("arguments_json") or "{}"
    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(args, dict):
        args = {}
    inner = ToolCall(name=name, args=args)
    return inner, registry.get(name)


def _stuck_summary(steps: list) -> str:
    if not steps:
        return "Stuck before any action."
    lines = ["Stuck. Here is what I tried:"]
    for s in steps[-12:]:
        why = (s.verdict.reason or s.result.error or s.result.output or "")[:100].replace("\n", " ")
        lines.append(f"#{s.index} {s.call.name}{s.call.args} -> {s.verdict.status.value}: {why}")
    return "\n".join(lines)
