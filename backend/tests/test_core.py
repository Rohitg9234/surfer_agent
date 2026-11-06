import asyncio
from types import SimpleNamespace

import pytest

from agent import AgentConfig, AgentCore
from agent.config import llm_provider_from_env
from agent.llm import GrokLLM, LLMResponse, llm_from_env, _to_openai_tools
from agent.playwright_browser import PlaywrightBrowser, rewrite_aria_refs
from agent.refs import parse_aria_snapshot, strip_ref_args
from agent.policy import Policy
from agent.types import Observation
from agent.vault import totp
from agent.tools import ToolContext, default_registry
from agent.memory import Memory
from agent.vault import CredentialVault
from agent.testing import MockBrowser, ScriptedLLM, act, plan
from agent.types import ToolCall, VerdictStatus


def run(core, goal="find the pro price"):
    return asyncio.run(core.run(goal))


def test_happy_path():
    llm = ScriptedLLM([
        plan("Open the pricing page", "Report the Pro price"),
        act("click", ref=1),
        act("subgoal_complete", summary="On pricing page"),
        act("note", text="Pro plan is $20/month"),
        act("done", answer="$20/month"),
    ])
    browser = MockBrowser()
    result = run(AgentCore(browser, llm, AgentConfig()))
    assert result.success and result.answer == "$20/month"
    assert result.stop_reason == "done"
    assert browser.actions == ["click 1"]
    assert [s.verdict.status for s in result.steps] == [VerdictStatus.OK] * 4


def test_premature_subgoal_complete_keeps_going():
    llm = ScriptedLLM([
        plan("Quote the first paragraph of the article"),
        act(
            "subgoal_complete",
            summary="Navigated to https://en.wikipedia.org successfully, page content is visible.",
        ),
        act("done", answer="Python is a high-level language."),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig()))
    assert result.success and "Python" in result.answer
    assert result.steps[0].call.name == "subgoal_complete"
    assert result.stop_reason == "done"


def test_broadcast_email_goal_is_blocked():
    llm = ScriptedLLM([plan("Open Gmail"), act("done", answer="sent")])
    result = run(
        AgentCore(MockBrowser(), llm, AgentConfig()),
        goal="Submit my resignation email in Gmail to everyone.",
    )
    assert not result.success
    assert result.stop_reason == "policy_block"
    assert llm.calls == []


def test_payment_goal_is_denied_without_confirm():
    llm = ScriptedLLM([plan("Open checkout"), act("done", answer="paid")])
    result = run(
        AgentCore(MockBrowser(), llm, AgentConfig()),
        goal="Pay for something / enter card.",
    )
    assert not result.success
    assert result.stop_reason == "user_declined"
    assert llm.calls == []


def test_honest_missing_field_done_succeeds():
    llm = ScriptedLLM([
        plan("Report stars and latest release if visible"),
        act("done", answer="Stars: 99.4k. Latest release name: not visible on the page."),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig()))
    assert result.success
    assert "99.4k" in result.answer


def test_incomplete_done_is_not_success():
    llm = ScriptedLLM([
        plan("Apply for the job"),
        act("done", answer="Cannot fill the form, recaptcha iframe only"),
        act("done", answer="Application submitted"),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig()))
    assert result.success
    assert "submitted" in result.answer.lower()
    assert any("Cannot fill" in (s.result.output or "") for s in result.steps)


def test_repeat_read_answers_from_evidence_without_a_second_fetch():
    class HistBrowser(MockBrowser):
        async def search_page(self, pattern: str, max_results: int = 10) -> str:
            self.actions.append(f"search_page {pattern}")
            return (
                'Found matches for "History":\n'
                "Section text (quote from this and call done):\n"
                "LinkedIn was launched in May 2003. It filed for an IPO in January 2011. "
                "Microsoft announced it would acquire LinkedIn in June 2016. "
                "LinkedIn shut down its China localized app in 2021."
            )

    llm = ScriptedLLM([
        plan("Give me 4 dated facts from History"),
        act("search_page", pattern="History"),
        act("search_page", pattern="History"),
        LLMResponse(text="May 2003 launch; January 2011 IPO; June 2016 Microsoft; 2021 China app shutdown."),
    ])
    browser = HistBrowser()
    result = run(
        AgentCore(browser, llm, AgentConfig()),
        goal="Open https://en.wikipedia.org/wiki/LinkedIn and give me 4 dated facts.",
    )
    assert result.success
    assert result.stop_reason == "done_from_evidence"
    assert "2003" in result.answer and "2016" in result.answer
    assert browser.actions.count("search_page History") == 1
    assert "Section text" in llm.calls[-1]["user"]


def test_thin_subsection_answer_is_not_accepted():
    class Docs(MockBrowser):
        async def observe(self) -> Observation:
            return Observation(
                url="https://docs.python.org/3/tutorial/datastructures.html",
                title="Data Structures",
                text=(
                    "headings: 5.1. More on Lists | 5.2. Using Lists as Stacks | "
                    "5.3. Using Lists as Queues | 5.4. Nested List Comprehensions\n"
                    '[1] link "Next"'
                ),
            )

    llm = ScriptedLLM([
        plan("List the subsection titles"),
        act("done", answer="5.1. More on Lists"),
        act("done", answer="5.1 More on Lists, 5.2 Using Lists as Stacks, 5.3 Using Lists as Queues, 5.4 Nested List Comprehensions"),
    ])
    result = run(
        AgentCore(Docs(), llm, AgentConfig()),
        goal="List the subsection titles on the Data Structures page.",
    )
    assert result.success
    assert "5.4" in result.answer
    assert "subsection" in (result.steps[0].verdict.reason or "").lower() or result.steps[1].call.name == "done"


def test_stale_ref_never_reaches_the_browser():
    llm = ScriptedLLM([
        plan("Open the pricing page"),
        act("click", ref=99),
        act("done", answer="stopped before a bad click"),
    ])
    browser = MockBrowser()
    result = run(AgentCore(browser, llm, AgentConfig()))
    assert "click 99" not in browser.actions
    assert result.steps[0].verdict.status == VerdictStatus.BLOCKED
    assert result.success and "stopped" in result.answer


def test_executor_prompt_keeps_search_evidence():
    llm = ScriptedLLM([
        plan("Quote the history section"),
        act("search_page", pattern="Pricing"),
        act("done", answer="Pro plan"),
    ])
    result = run(
        AgentCore(MockBrowser(), llm, AgentConfig()),
        goal="Tell me the pricing line.",
    )
    assert result.success
    evidence_prompt = llm.calls[2]["user"]
    assert "EVIDENCE ALREADY COLLECTED" in evidence_prompt
    assert "Section text" in evidence_prompt


def test_risky_click_is_denied_by_default():
    llm = ScriptedLLM([plan("Try paying"), act("click", ref=2), act("done", answer="stopped")])
    browser = MockBrowser()
    result = run(AgentCore(browser, llm, AgentConfig()))
    assert result.steps[0].verdict.status == VerdictStatus.BLOCKED
    assert browser.actions == []          # the click never reached the browser


def test_risky_click_runs_when_user_confirms():
    async def yes(call, reason):
        return True
    llm = ScriptedLLM([plan("Pay"), act("click", ref=2), act("done", answer="paid")])
    browser = MockBrowser()
    run(AgentCore(browser, llm, AgentConfig(), confirm=yes))
    assert "click 2" in browser.actions


def test_loop_stops_without_expensive_replan():
    llm = ScriptedLLM([
        plan("Find price"),
        act("scroll", direction="down"), act("scroll", direction="down"), act("scroll", direction="down"),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig()))
    assert not result.success
    assert result.stop_reason == "stuck"
    assert "tried" in result.answer.lower()
    assert sum(1 for c in llm.calls if c.get("force_tool") == "submit_plan") == 1
    assert len(result.steps) <= 3


def test_unknown_tool_is_blocked():
    llm = ScriptedLLM([
        plan("x"),
        act("type_text", ref=3, text="hunter2"),   # [3] is a normal textbox -> allowed
        act("teleport"),                            # unknown tool -> blocked
        act("done", answer="d"),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig()))
    assert result.steps[0].verdict.status == VerdictStatus.OK
    assert result.steps[1].verdict.status == VerdictStatus.BLOCKED


def test_registry_validates_arguments():
    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=None)
    bad = asyncio.run(default_registry.execute(ToolCall(name="click", args={"ref": "abc"}), ctx))
    assert not bad.ok and "Invalid arguments" in bad.error
    too_small = asyncio.run(default_registry.execute(ToolCall(name="click", args={"ref": 0}), ctx))
    assert not too_small.ok and "Invalid arguments" in too_small.error
    bad_url = asyncio.run(default_registry.execute(ToolCall(name="goto", args={"url": "not-a-url"}), ctx))
    assert not bad_url.ok and "Invalid arguments" in bad_url.error
    assert {"goto", "click", "done", "search", "search_page", "fill_credential", "load_playbook",
            "save_session", "upload_file", "click_download", "wait_for_load", "request_human_help"} <= set(default_registry.names())


def test_playbooks_from_firecrawl_pattern():
    from agent.playbook_store import PlaybookStore

    store = PlaybookStore()
    assert store.get("hard-pages") and store.get("e-commerce")
    assert store.for_url("https://finance.yahoo.com/quote/AAPL") is not None
    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=None)
    listed = asyncio.run(default_registry.execute(ToolCall(name="list_playbooks", args={}), ctx))
    assert listed.ok and "hard-pages" in listed.output
    site = asyncio.run(default_registry.execute(
        ToolCall(name="lookup_site_playbook", args={"url": "https://finance.yahoo.com/quote/AAPL"}), ctx
    ))
    assert site.ok and "analysts" in site.output.lower()


def test_session_and_files_from_steel_pattern(tmp_path):
    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=None)
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    saved = asyncio.run(default_registry.execute(
        ToolCall(name="save_session", args={"path": str(tmp_path / "out.json")}), ctx
    ))
    assert saved.ok and "omitted" in saved.output
    loaded = asyncio.run(default_registry.execute(
        ToolCall(name="load_session", args={"path": str(state)}), ctx
    ))
    assert loaded.ok
    blocked = asyncio.run(default_registry.execute(
        ToolCall(name="upload_file", args={"ref": 1, "path": ".env"}), ctx
    ))
    assert not blocked.ok
    wait = asyncio.run(default_registry.execute(
        ToolCall(name="wait_for_load", args={"state": "networkidle"}), ctx
    ))
    assert wait.ok and "wait_load networkidle" in ctx.browser.actions


def test_human_in_loop_uses_ask_user_when_no_browser_dialog():
    async def human(question: str) -> str:
        assert "phone" in question.lower()
        return "555-0100"

    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=human)
    out = asyncio.run(default_registry.execute(
        ToolCall(name="human_in_loop", args={"question": "What phone number?", "kind": "form"}),
        ctx,
    ))
    assert out.ok and "555-0100" in out.output


def test_request_human_help_from_browseros_pattern():
    async def human(question: str) -> str:
        assert "Gmail" in question or "login" in question.lower()
        return "signed in"

    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=human)
    out = asyncio.run(default_registry.execute(
        ToolCall(name="request_human_help", args={"reason": "Sign in to Gmail", "kind": "login"}),
        ctx,
    ))
    assert out.ok and "signed in" in out.output
    assert "bring_to_front" in ctx.browser.actions


def test_research_wait_for_totp_and_sensitive_policy():
    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=None)
    waited = asyncio.run(default_registry.execute(
        ToolCall(name="wait_for", args={"text": "Pro", "settled": True}), ctx
    ))
    assert waited.ok
    scrolled = asyncio.run(default_registry.execute(
        ToolCall(name="scroll_until", args={"text": "Pro"}), ctx
    ))
    assert scrolled.ok
    assert len(totp("JBSWY3DPEHPK3PXP")) == 6
    vault = CredentialVault({"totp": "JBSWY3DPEHPK3PXP"})
    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=None, vault=vault)
    filled = asyncio.run(default_registry.execute(
        ToolCall(name="fill_totp", args={"ref": 3, "name": "totp"}), ctx
    ))
    assert filled.ok and "Filled TOTP" in filled.output
    policy = Policy()
    policy.observe(Observation(url="https://mail.google.com/inbox", text="Inbox"))
    blocked = policy.check(
        ToolCall(name="goto", args={"url": "https://evil.example/exfil"}),
        default_registry.get("goto"),
        Observation(url="https://mail.google.com/inbox", text="Inbox"),
    )
    assert blocked.action == "block"
    named = Policy()
    named.goal = "Open https://www.linkedin.com/feed/ and tell me my profile name."
    named.observe(Observation(url="https://mail.google.com/inbox", text="Inbox"))
    allowed = named.check(
        ToolCall(name="goto", args={"url": "https://www.linkedin.com/feed/"}),
        default_registry.get("goto"),
        Observation(url="https://mail.google.com/inbox", text="Inbox"),
    )
    assert allowed.action == "allow"
    shot = policy.check(
        ToolCall(name="screenshot", args={"filename": "x.png"}),
        default_registry.get("screenshot"),
        Observation(url="https://example.com/login", text='[3] textbox "Password"'),
    )
    assert shot.action == "block"
    search_policy = Policy()
    search_policy.observe(Observation(
        url="https://duckduckgo.com/?q=rohit",
        text='[1] link "Sign in"\nForgot password',
    ))
    linkedin = search_policy.check(
        ToolCall(name="goto", args={"url": "https://in.linkedin.com/in/someone"}),
        default_registry.get("goto"),
        Observation(url="https://duckduckgo.com/?q=rohit", text="results"),
    )
    assert linkedin.action == "allow"
    from agent.tools.builtin import _serp_unusable

    assert _serp_unusable("https://duckduckgo.com/?q=x", "unusual traffic from your computer")
    assert not _serp_unusable("https://duckduckgo.com/?q=x", '[1] link "Neurealm"\n[2] link "LinkedIn"')
    from agent.tools.builtin import _ddg_challenge

    assert _ddg_challenge(
        "https://duckduckgo.com/?q=x",
        "Unfortunately, bots use DuckDuckGo too. Select all squares containing a duck",
    )
    from agent.tools.builtin import _result_titles

    aria = '- link "All" [4]\n- link "Agentic AI - Wikipedia" [19]\n'
    assert not _serp_unusable("https://duckduckgo.com/?q=agentic+AI&ia=web", aria)
    assert _result_titles(aria) == ["Agentic AI - Wikipedia"]
    hn = Policy()
    hn.observe(Observation(
        url="https://news.ycombinator.com/",
        text='[1] link "OpenSSH-server password authentication bypass"',
    ))
    bbc = hn.check(
        ToolCall(name="goto", args={"url": "https://www.bbc.com/news"}),
        default_registry.get("goto"),
        Observation(url="https://news.ycombinator.com/", text="HN"),
    )
    assert bbc.action == "allow"
    submit = Policy().check(
        ToolCall(name="click_text", args={"text": "Submit"}),
        default_registry.get("click_text"),
        Observation(url="https://www.linkedin.com/jobs/view/1", text="Easy Apply"),
    )
    assert submit.action == "confirm"
    card = Policy().check(
        ToolCall(name="fill_field", args={"label": "Card number", "value": "4111111111111111"}),
        default_registry.get("fill_field"),
        Observation(url="https://shop.example/pay", text="Checkout"),
    )
    assert card.action == "block"
    supplied = Policy()
    supplied.goal = "login with username tomsmith and password SuperSecretPassword!"
    pw_ok = supplied.check(
        ToolCall(name="fill_field", args={"label": "Password", "value": "SuperSecretPassword!"}),
        default_registry.get("fill_field"),
        Observation(url="https://the-internet.herokuapp.com/login", text='[3] textbox "Password"'),
    )
    assert pw_ok.action == "allow"
    pw_block = Policy().check(
        ToolCall(name="fill_field", args={"label": "Password", "value": "SuperSecretPassword!"}),
        default_registry.get("fill_field"),
        Observation(url="https://the-internet.herokuapp.com/login", text='[3] textbox "Password"'),
    )
    assert pw_block.action == "block"


def test_search_waits_for_duckduckgo_results_and_skips_google():
    class LateDuckDuckGo:
        def __init__(self):
            self.urls = []
            self.sees = 0

        async def goto(self, url):
            self.urls.append(url)

        async def wait(self, seconds):
            return None

        async def observe(self):
            self.sees += 1
            if self.sees < 3:
                return Observation(url=self.urls[-1], title="DuckDuckGo", text="DuckDuckGo Open menu Search")
            return Observation(
                url=self.urls[-1],
                title="DuckDuckGo",
                text='[4] link "All"\n[9] link "Largest organisms - Wikipedia"\n[10] link "Blue whale - Wikipedia"',
            )

    browser = LateDuckDuckGo()
    ctx = ToolContext(browser=browser, memory=Memory(), ask_user=None)
    result = asyncio.run(default_registry.execute(
        ToolCall(name="search", args={"query": "largest living species"}),
        ctx,
    ))
    assert result.ok
    assert "Largest organisms - Wikipedia" in result.output
    assert browser.urls == ["https://duckduckgo.com/?q=largest+living+species&ia=web"]
    assert not any("google.com" in url for url in browser.urls)

    class AriaDuckDuckGo(LateDuckDuckGo):
        async def observe(self):
            return Observation(
                url=self.urls[-1],
                title="agentic AI at DuckDuckGo",
                text='- link "All" [4]\n- link "Agentic AI - Wikipedia" [19]\n',
            )

    aria = AriaDuckDuckGo()
    aria_ctx = ToolContext(browser=aria, memory=Memory(), ask_user=None)
    aria_result = asyncio.run(default_registry.execute(
        ToolCall(name="search", args={"query": "agentic AI"}),
        aria_ctx,
    ))
    assert aria_result.ok and "Agentic AI - Wikipedia" in aria_result.output
    assert aria.urls == ["https://duckduckgo.com/?q=agentic+AI&ia=web"]


def test_search_page_and_fill_credential_from_browser_use():
    ctx = ToolContext(
        browser=MockBrowser(), memory=Memory(), ask_user=None,
        vault=CredentialVault({"password": "s3cret"}),
    )
    found = asyncio.run(default_registry.execute(ToolCall(name="search_page", args={"pattern": "Pricing"}), ctx))
    assert found.ok and "Pricing" in found.output
    filled = asyncio.run(default_registry.execute(
        ToolCall(name="fill_credential", args={"ref": 3, "name": "password"}), ctx
    ))
    assert filled.ok and "s3cret" not in filled.output
    assert "type 3" in ctx.browser.actions


def test_short_and_long_term_memory(tmp_path):
    from agent.memory import Conversation, ConversationStore, LongTermMemory, Memory
    from agent.types import Step, ToolCall, ToolResult, Verdict, VerdictStatus

    ltm = LongTermMemory(path=None)
    mem = Memory(long_term=ltm)
    mem.begin_run(
        "Open https://en.wikipedia.org/wiki/LinkedIn, find the History section, and give me 4 dated facts.",
        "https://en.wikipedia.org/wiki/LinkedIn",
    )
    assert any("search_page" in line and "History" in line for line in mem.recalled)
    mem.end_run(
        "LinkedIn History 4 dated facts",
        success=True,
        answer="2002 founded; 2011 IPO; 2016 Microsoft; 2021 visor.",
        url="https://en.wikipedia.org/wiki/LinkedIn",
    )
    mem.begin_run("give me more LinkedIn History dates", "https://en.wikipedia.org/wiki/LinkedIn")
    blob = " ".join(mem.prompt_notes())
    assert "2002" in blob
    other = Memory(long_term=ltm, conversation=Conversation())
    other.begin_run("give me more LinkedIn History dates", "https://en.wikipedia.org/wiki/LinkedIn")
    other_blob = " ".join(other.prompt_notes())
    assert "2002 founded" not in other_blob
    failed = Memory(long_term=LongTermMemory(path=None))
    failed.begin_run("History facts", "https://en.wikipedia.org/wiki/LinkedIn")
    failed.add_step(Step(
        index=1,
        subgoal="open history",
        call=ToolCall(name="click_text", args={"text": "History"}),
        result=ToolResult(ok=False, error="covered"),
        verdict=Verdict(status=VerdictStatus.ERROR, reason="timeout"),
    ))
    failed.end_run("History", success=False, answer="stuck", url="https://en.wikipedia.org/wiki/LinkedIn")
    saved_trace = failed.conversation.messages[-1].get("trace") or []
    assert saved_trace and saved_trace[0]["name"] == "click_text"
    assert saved_trace[0]["status"] == "error"
    again = Memory(long_term=failed.long_term, conversation=Conversation())
    again.begin_run("LinkedIn History", "https://en.wikipedia.org/wiki/LinkedIn")
    assert any("TOC" in line or "search_page" in line for line in again.recalled)

    store = ConversationStore(folder=tmp_path / "conversations")
    conv = Conversation(conv_id="conv-test-1")
    mem_s = Memory(long_term=LongTermMemory(path=None), conversation=conv, store=store)
    mem_s.end_run("hello", success=True, answer="hi there", url="https://example.com")
    loaded = store.load("conv-test-1")
    assert loaded is not None and any(m.get("content") == "hello" for m in loaded.messages)


def test_list_notes_is_independent_of_the_browser():
    ctx = ToolContext(browser=MockBrowser(), memory=Memory(), ask_user=None)
    asyncio.run(default_registry.execute(ToolCall(name="note", args={"text": "Pro is $20"}), ctx))
    out = asyncio.run(default_registry.execute(ToolCall(name="list_notes", args={}), ctx))
    assert out.ok and "Pro is $20" in out.output


def test_executor_receives_every_registered_tool_and_untrusted_markers():
    llm = ScriptedLLM([plan("x"), act("done", answer="d")])
    run(AgentCore(MockBrowser(), llm, AgentConfig()))
    exec_call = llm.calls[1]
    names = {t["name"] for t in exec_call["tools"]}
    assert {"goto", "click", "click_text", "done", "call_extra", "more_tools", "human_in_loop"} <= names
    assert "upload_file" not in names
    assert "[[BEGIN_UNTRUSTED_PAGE nonce=" in exec_call["user"]


def test_rewrite_aria_refs_maps_playwright_mcp_style_refs():
    yaml = (
        '- heading "todos" [level=1]\n'
        '- textbox "What needs to be done?" [ref=e5]\n'
        '- checkbox "Toggle Todo" [ref=e10]\n'
        '- checkbox "Toggle Todo" [ref=e10]\n'
    )
    text, mapping = rewrite_aria_refs(yaml)
    assert "[1]" in text and "[2]" in text
    assert "[ref=e5]" not in text
    assert mapping == {1: "e5", 2: "e10"}
    from agent.snapshot import slim_snapshot
    fat = "\n".join([f'[1] link "x{i}"' for i in range(20)] + ['heading "H"'] * 2)
    slim = slim_snapshot(fat, max_chars=400)
    assert "similar" in slim or len(slim) < len(fat)
    mixed = "\n".join(
        [f'[1] button "Home{i}"' for i in range(20)]
        + ['[2] article "Jane posted about shipping a Python browser agent today"']
        + ["visible_text:", "Jane posted about shipping a Python browser agent today"]
    )
    slim2 = slim_snapshot(mixed, max_chars=500)
    assert "Jane posted" in slim2
    assert slim2.find("visible_text:") < slim2.find("Home0") or "Home0" not in slim2
    wiki = "\n".join(
        [f'heading "Section {i}"' for i in range(40)]
        + [
            '[1] link "Jump to content"',
            "visible_text:",
            "Python is a high-level, general-purpose programming language.",
        ]
    )
    slim_wiki = slim_snapshot(wiki, max_chars=800)
    assert "Python is a high-level" in slim_wiki
    assert slim_wiki.find("visible_text:") < slim_wiki.find("Jump to content") or "Jump to content" not in slim_wiki
    _, refs = parse_aria_snapshot(yaml)
    assert refs[1].role == "textbox" and refs[1].name.startswith("What needs")
    assert "ref" not in strip_ref_args({"ref": 1, "x": 1}, {1: {"role": "textbox", "name": "q"}})


def test_to_openai_tools_converts_anthropic_schema():
    tools = _to_openai_tools([{
        "name": "submit_plan",
        "description": "Submit the plan",
        "input_schema": {"type": "object", "properties": {"subgoals": {"type": "array"}}},
    }])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "submit_plan"
    assert tools[0]["function"]["parameters"]["properties"]["subgoals"]["type"] == "array"


def test_llm_from_env_prefers_grok(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-test")
    llm = llm_from_env()
    assert isinstance(llm, GrokLLM)
    assert llm_provider_from_env() == "grok"
    cfg = AgentConfig.from_env()
    assert cfg.provider == "grok"
    assert cfg.planner_model == "grok-4"
    assert cfg.executor_model == "grok-4-fast"


def test_llm_from_env_falls_back_to_claude(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("GROK_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-test")
    from agent.llm import AnthropicLLM
    llm = llm_from_env()
    assert isinstance(llm, AnthropicLLM)
    cfg = AgentConfig.from_env()
    assert cfg.provider == "anthropic"
    assert cfg.planner_model == "claude-sonnet-5-5"


def test_llm_from_env_requires_a_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("GROK_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="No LLM API key"):
        llm_from_env()


def test_grok_complete_parses_tool_calls():
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="",
                    tool_calls=[SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(name="click", arguments='{"ref": 1}'),
                    )],
                ))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    resp = asyncio.run(GrokLLM(client=client).complete(
        model="grok-4",
        system="sys",
        user="do it",
        tools=[{"name": "click", "description": "Click", "input_schema": {"type": "object"}}],
        force_tool="click",
    ))
    assert resp.tool_calls[0].name == "click"
    assert resp.tool_calls[0].args == {"ref": 1}
    assert resp.input_tokens == 11 and resp.output_tokens == 4
    assert captured["tool_choice"] == {"type": "function", "function": {"name": "click"}}
    assert captured["tools"][0]["function"]["name"] == "click"


def test_chrome_profiles_parse_local_state(tmp_path, monkeypatch):
    import json
    from agent import chrome_profiles as cp

    root = tmp_path / "Chrome"
    (root / "Default").mkdir(parents=True)
    (root / "Profile 1").mkdir()
    (root / "Local State").write_text(json.dumps({
        "profile": {"info_cache": {
            "Default": {"name": "Work", "user_name": "me@x.com"},
            "Profile 1": {"gaia_name": "Personal", "user_name": "you@x.com"},
        }}
    }), encoding="utf-8")
    monkeypatch.setattr(cp, "_chrome_roots", lambda: [("Google Chrome", "chrome", root)])
    found = cp.list_profiles()
    emails = " ".join(p.display for p in found)
    assert "me@x.com" in emails and "you@x.com" in emails
    chosen = cp.choose_chrome_profile(found)
    assert chosen is not None and chosen.directory == "Default" and "me@x.com" in chosen.display
    assert cp.normalize_cdp_argv(["--cdphttp://127.0.0.1:9222", "goal"]) == [
        "--cdp", "http://127.0.0.1:9222", "goal",
    ]
    assert cp.normalize_cdp_argv(['--cdpOpen LinkedIn']) == ["--cdp", "Open LinkedIn"]
    default = "Find the Pro plan price on the pricing page"
    assert cp.recover_cdp_goal("Open LinkedIn and summarize my feed", default, default) == (
        "http://127.0.0.1:9222",
        "Open LinkedIn and summarize my feed",
    )
    assert cp.recover_cdp_goal("http://127.0.0.1:9222", "Open LinkedIn", default) == (
        "http://127.0.0.1:9222",
        "Open LinkedIn",
    )


def test_agent_chrome_hide_script_names_only_the_playwright_browser():
    from agent.playwright_browser import _chrome_visibility_script

    hidden = _chrome_visibility_script(False)
    shown = _chrome_visibility_script(True)
    focused = _chrome_visibility_script(True, activate=True)
    assert 'process "Google Chrome for Testing"' in hidden
    assert 'process "Chromium"' in hidden
    assert "repeat with" not in hidden
    assert "set frontmost" not in hidden
    assert "set frontmost" not in shown
    assert "set frontmost" in focused
    assert 'process "Google Chrome"' not in hidden.replace('process "Google Chrome for Testing"', "")


def test_park_returns_focus_without_hiding_chrome(monkeypatch):
    from agent import playwright_browser as pb

    calls = []
    monkeypatch.setattr(
        pb, "_frontmost_identity",
        lambda: ("Google Chrome", "com.google.chrome.for.testing"),
    )
    monkeypatch.setattr(
        pb, "_set_agent_chrome_visible",
        lambda visible, activate=False: calls.append(("visible", visible, activate)) or True,
    )
    monkeypatch.setattr(pb, "_activate_app", lambda name: calls.append(("activate", name)))
    assert pb._park_agent_chrome("Cursor") is True
    assert calls == [("activate", "Cursor")]
    calls.clear()
    monkeypatch.setattr(pb, "_frontmost_identity", lambda: ("Google Chrome", "com.google.Chrome"))
    monkeypatch.setattr(pb, "_frontmost_command", lambda: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    assert pb._park_agent_chrome("Cursor", "/tmp/agent-chrome") is True
    assert calls == []
    monkeypatch.setattr(
        pb, "_frontmost_command",
        lambda: "Google Chrome --user-data-dir=/tmp/agent-chrome --remote-debugging-port=9334",
    )
    assert pb._park_agent_chrome("Cursor", "/tmp/agent-chrome") is True
    assert calls == [("activate", "Cursor")]


def test_background_chrome_launch_does_not_activate_the_app():
    from agent.playwright_browser import _background_chrome_command

    cmd = _background_chrome_command(
        "/Library/Caches/ms-playwright/Google Chrome for Testing.app",
        "/tmp/web-agent-chrome",
        9333,
    )
    assert cmd[:4] == ["open", "-g", "-n", "-a"]
    assert "--args" in cmd
    assert "--remote-debugging-port=9333" in cmd
    assert "--user-data-dir=/tmp/web-agent-chrome" in cmd
    assert not any(part.endswith("Google Chrome.app") for part in cmd)
    user = _background_chrome_command(
        "/Applications/Google Chrome.app",
        "/Users/me/Library/Application Support/Google/Chrome",
        9334,
        "Default",
    )
    assert user[4].endswith("Google Chrome.app")
    assert "--profile-directory=Default" in user
    assert "--remote-debugging-port=9334" in user


def test_park_leaves_everyday_chrome_alone(monkeypatch):
    from agent import playwright_browser as pb

    calls = []
    monkeypatch.setattr(
        pb, "_frontmost_identity",
        lambda: ("Google Chrome", "com.google.Chrome"),
    )
    monkeypatch.setattr(
        pb, "_set_agent_chrome_visible",
        lambda visible, activate=False: calls.append(visible) or True,
    )
    monkeypatch.setattr(pb, "_activate_app", lambda name: calls.append(name))
    assert pb._park_agent_chrome("Cursor") is True
    assert calls == []


def test_human_prompt_blocks_background_hide():
    browser = PlaywrightBrowser()
    browser._hitl_open = True
    browser._schedule_rest()
    assert browser._rest_task is None


def test_aria_snapshot_numbers_are_accepted():
    from agent.core import _missing_ref

    page = Observation(
        url="https://www.google.com/",
        title="Google",
        text='- textbox "Search" [12]\n- button "Google Search" [13]\n',
    )
    allowed = _missing_ref(
        ToolCall(name="type_text", args={"ref": 12, "text": "largest living species", "press_enter": True}),
        page,
    )
    assert allowed == ""
    blocked = _missing_ref(ToolCall(name="click", args={"ref": 120}), page)
    assert "[120]" in blocked
    assert "[12] Search" in blocked


def test_observe_does_not_open_a_browser_until_a_page_is_needed():
    browser = PlaywrightBrowser()
    obs = asyncio.run(browser.observe())
    assert browser._pw is None
    assert "No browser is open" in obs.text


def test_closed_browser_page_is_not_reused():
    browser = PlaywrightBrowser()
    assert browser._alive() is False
    browser._page = SimpleNamespace(is_closed=lambda: True, context=None)
    assert browser._alive() is False
    browser._page = SimpleNamespace(
        is_closed=lambda: False,
        context=SimpleNamespace(browser=SimpleNamespace(is_connected=lambda: False)),
    )
    browser._browser = browser._page.context.browser
    assert browser._alive() is False
    browser._browser = SimpleNamespace(is_connected=lambda: True)
    assert browser._alive() is True


def test_steer_reaches_the_next_decision():
    pending = ["Use the Pricing link, not Pay now."]

    def steer():
        return pending.pop(0) if pending else ""

    llm = ScriptedLLM([
        plan("Open the pricing page"),
        act("click", ref=1),
        act("subgoal_complete", summary="On the pricing page"),
        act("done", answer="$20/month"),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig(), steer=steer))
    assert result.success
    assert any("Use the Pricing link" in (call.get("user") or "") for call in llm.calls)


def test_stop_ends_the_run_before_the_next_action():
    llm = ScriptedLLM([
        plan("Open the pricing page"),
        act("done", answer="should not run"),
    ])
    result = run(AgentCore(MockBrowser(), llm, AgentConfig(), steer=lambda: "__stop__"))
    assert not result.success
    assert result.stop_reason == "stopped"
    assert result.answer == "Stopped."
    assert result.steps == []
