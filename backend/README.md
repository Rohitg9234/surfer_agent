# Browser agent: agent core

The Planner / Executor / Verifier / Memory loop, plus the tool registry and a starter policy gate.
It depends only on a `Browser` interface, so Playwright and Perception can be built separately.

## Run

From this `backend/` directory (the virtualenv stays in the repo root):

    ../venv/bin/pip install -r requirements.txt
    ../venv/bin/python -m pytest -q          # no API key needed (scripted LLM + mock browser)
    ../venv/bin/python -m server             # chat UI at http://127.0.0.1:8787
    export XAI_API_KEY=...                   # preferred: Grok (GROK_API_KEY also works)
    # or: export ANTHROPIC_API_KEY=...       # fallback: Claude, if no Grok key is set
    ../venv/bin/python -m playwright install chromium   # one-time: local browsers, no API key
    ../venv/bin/python -m examples.run_demo             # real LLM + Playwright Chromium
    ../venv/bin/python -m examples.run_demo --mock      # real LLM against the mock browser
    ../venv/bin/python -m examples.run_demo --cdp --auto-lock "…"   # terminal chat
    ../venv/bin/python -m examples.run_demo --cdp --auto-lock --new  # start a new chat thread
    ../venv/bin/python -m examples.run_demo --once "…"              # one-shot, no chat prompt

The chat UI lists every saved thread. Open one to continue it. Short-term memory is the recent turns of that thread. Long-term memory is semantic recall from a local Chroma database (`memory/chroma`): facts, earlier lines from the same thread, and site tactics.

`AGENT_BROWSER=mock` runs the UI against the fake browser. `AGENT_HEADLESS=1` hides the Chromium window. `AGENT_CDP_URL` attaches to a Chrome you already started.

## Files

    agent/
      core.py          AgentCore.run(): the loop (start reading here)
      planner.py       goal -> sub-goals, and replanning when stuck
      executor.py      one tool call per step; wraps page text as untrusted
      verifier.py      deterministic OK / NO_EFFECT / ERROR / LOOP checks
      memory.py        per-run notes+history, and SkillStore for learned workflows
      policy.py        allow / confirm / block gate (starter version of your Safety layer)
      interfaces.py    Browser protocol you implement with Playwright
      llm.py           LLM gateway (Grok or Anthropic, chosen from env keys)
      config.py        models and budgets
      tools/
        registry.py    @tool decorator and ToolRegistry (rarely edited)
        builtin.py     <-- ADD YOUR TOOLS HERE
      testing.py       ScriptedLLM and MockBrowser

## Adding a tool (3 steps)

1. In `agent/tools/builtin.py` (or a new file you import in `tools/__init__.py`), write a Pydantic
   args model and a function with the `@tool(...)` decorator.
2. If it needs a new browser capability, add the method to `Browser` in `interfaces.py` and
   implement it in your Playwright class.
3. Nothing else. The Executor, argument validation, Policy gate and Verifier all read the registry.

Choosing the decorator flags:

- `risk=Risk.SAFE`  read-only or control actions
- `risk=Risk.LOW`   reversible interaction (default)
- `risk=Risk.HIGH`  irreversible or external effect: always asks the user first
- `expects_page_change=False` for read-only tools, so the Verifier does not flag NO_EFFECT

## Where each future layer plugs in

| Layer you build next | Plug-in point |
|---|---|
| Playwright browser + Perception | `agent/playwright_browser.py` implements `interfaces.Browser`; `observe()` stamps numbered `[ref]` attributes |
| Full Safety layer | Pass `policy=` with a class exposing `check(call, spec, obs) -> Decision` |
| Human confirmation UI | Pass `confirm=` (async). The default denies every risky action |
| Credential vault | Add a `fill_credential(ref, credential_id)` tool; the vault injects the secret in the browser, never in LLM context |
| Observability | Pass `on_step=` to log each `Step`; later wrap `llm.complete` for token traces |
| Other model providers | Implement `LLMClient.complete()` in `llm.py` |
| Benchmarks | Call `AgentCore(...).run(task_goal)` per task and score `RunResult` |
