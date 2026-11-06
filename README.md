# Surfer Agent

Surfer Agent browses the web for you and answers in the chat.

You describe a research task in natural language, just as you would brief a human researcher. The agent autonomously explores the internet, searches across websites and sources, navigates pages, gathers information, cross-references findings, and analyzes data to deliver comprehensive, evidence-backed research. Built primarily for deep research, it can handle complex, multi-step investigations from start to finish with minimal human intervention. It operates as a fully automated agent, while also supporting human-in-the-loop workflows whenever clarification, approval, or human judgment is needed. Every finding is grounded in actual sources, with citations and references, ensuring the final output is accurate, transparent, and verifiable.

A few things you can hand it:

- “Open the Python docs page for lists and give me the subsection titles.”
- “Search for today’s weather in Bangalore and cite the page you used.”
- “On this Wikipedia article, read the History section and give me four dated facts.”
- “Open the job page, tell me the title and company, and do not click Apply.”
- “Fill the name and phone on this form, then stop. Do not submit.”

If the page is behind a login, a CAPTCHA, or a missing value only you know, it stops and asks you on the page instead of guessing. Pay, send, and delete wait for an explicit yes. You can stop a run mid-way, or type a correction while it is working and it will follow that on the next step.

The same window is the product. The sidebar is the list of past tasks. Open one and the next message continues that thread, including facts it already collected. Expand **Steps** under a reply to see each page it opened and each click it made.

It runs on your machine. The chat UI and the agent are one process at [http://127.0.0.1:8787](http://127.0.0.1:8787).

## What you get while it works

- **A thread you can resume.** Each task stays in the sidebar. Coming back later continues that conversation, not a blank session.
- **Memory of this thread and of the site.** The recent turns stay in front of the agent. Older facts and tactics that worked on a site are recalled by meaning from a local [Chroma](https://www.trychroma.com/) database.
- **A trace of the browsing.** Tool calls, arguments, and outcomes stream into **Steps** as they happen.
- **Your hand on the wheel.** **Stop** ends the turn. **Steer** sends a correction without waiting for it to finish. Chrome stays behind the chat until you must act: the blue bar, a CAPTCHA, or a login.

## How a turn runs

```text
You
 └─ thread + short-term window + Chroma recall
      └─ Planner        goal → ordered sub-goals
           └─ Executor  one tool call, from the numbered page snapshot
                └─ Policy     allow, confirm, or block
                     └─ Browser    Playwright, only when a page is required
                          └─ Verifier   ok, no effect, error, or loop
                               └─ Memory  thread, facts, site tactics
```

Page text is wrapped as untrusted data. Instructions that appear inside a page do not override the task.

Chrome is not opened when you press Send. The agent opens it on `goto` or `search`. After that, the window stays behind the chat. It comes forward only when a person has to finish a step on the page.

## Requirements

- Python 3.11
- Chromium for Playwright, installed once with the command below
- An API key for Grok (`XAI_API_KEY` or `GROK_API_KEY`) or Claude (`ANTHROPIC_API_KEY`)

Grok is used when its key is set. Claude is the fallback.

## Quick start

From the repository root:

```bash
python3.11 -m venv venv
./venv/bin/pip install -r backend/requirements.txt
./venv/bin/python -m playwright install chromium
```

Put the key in `backend/.env` (this file stays on your machine):

```bash
XAI_API_KEY=your-key
# or, if you are not using Grok:
# ANTHROPIC_API_KEY=your-key
```

Start the server:

```bash
cd backend
../venv/bin/python -m server
```

Then open [http://127.0.0.1:8787](http://127.0.0.1:8787). The same process serves the API and the UI in `frontend/`.

## Using the product

| Surface | What it is for |
| --- | --- |
| Left sidebar | Thread history. Search it, or open a thread to continue it. |
| Title | Rename the open thread. The name is saved with the thread. |
| Steps | Live tool trace. The browser link appears only after a window exists. |
| Composer | Send a task. During a run, **Steer** sends a correction and **Stop** ends the turn. |
| Memory | Short-term window for this thread, and the Chroma hits beside it. |

## Configuration

Keys are read from the environment, then from `backend/.env` when a variable is unset.

| Variable | Default | Role |
| --- | --- | --- |
| `XAI_API_KEY` or `GROK_API_KEY` | unset | Use Grok. Preferred when set. |
| `ANTHROPIC_API_KEY` | unset | Use Claude when no Grok key is set. |
| `AGENT_PLANNER_MODEL` | `grok-4` or `claude-sonnet-5-5` | Model that writes the plan. |
| `AGENT_EXECUTOR_MODEL` | `grok-4-fast` or `claude-haiku-5-5` | Model that picks one action per step. |
| `AGENT_HEADLESS` | off | Set to `1` to hide the Chromium window. |
| `AGENT_BROWSER` | Playwright | Set to `mock` to run the UI without a browser. |
| `AGENT_CDP_URL` | unset | Attach to a Chrome you already started, for example `http://127.0.0.1:9222`. |
| `AGENT_MEMORY_DIR` | `backend/memory` | Where threads and the Chroma index are stored. |
| `PORT` | `8787` | HTTP port. |

Step and token budgets (`AGENT_MAX_STEPS`, `AGENT_MAX_STEPS_PER_SUBGOAL`, `AGENT_MAX_REPLANS`, `AGENT_MAX_TOKENS`) live in `backend/agent/config.py`.

## Repository

```text
frontend/          Chat UI. Served by the API process.
backend/
  server/          FastAPI app and the session runtime.
  agent/           Planner, executor, verifier, policy, memory, tools.
  examples/        Terminal runner, including a mock-browser mode.
  memory/          Saved threads. The Chroma index is created here.
venv/              Python environment. Kept at the repo root so its paths stay valid.
```

`backend/README.md` is the guide for the agent loop: where to add a tool, how the policy gate works, and how to run the terminal demo.

## Develop

From `backend/`, with no API key:

```bash
../venv/bin/python -m pytest -q
```

The tests use a scripted model and a mock browser. A real Chromium session is the server path above, or:

```bash
../venv/bin/python -m examples.run_demo --mock
../venv/bin/python -m examples.run_demo
```

## Data

Threads are JSON files under `backend/memory/conversations/`. Chroma stores embeddings beside them in `backend/memory/chroma/`. Deleting a thread in the UI removes it from both. Do not commit `.env` or the memory directory if they contain keys or private page text.
