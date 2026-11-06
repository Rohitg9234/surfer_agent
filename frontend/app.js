const state = {
  conversations: [],
  activeId: null,
  view: null,
  running: false,
  steps: [],
  phase: "",
  openTraces: new Set(),
  status: "Connecting",
  searchHits: null,
  browser: { open: false, url: "", title: "" },
  freshId: null,
};

const $ = (id) => document.getElementById(id);
const listEl = $("thread-list");
const transcript = $("transcript");
const titleEl = $("thread-title");
const shortEl = $("short-term");
const longEl = $("long-term");
const input = $("input");
const live = $("live");
const modal = $("modal");

function when(iso) {
  if (!iso) return "";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const delta = (Date.now() - then.getTime()) / 1000;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 86400 * 7) return `${Math.floor(delta / 86400)}d ago`;
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function setStatus(text) {
  state.status = text;
  const routine = !text || ["Connecting", "Working", "Answered", "Stopped", "Sending", "Error"].includes(text);
  if (routine) return;
  live.hidden = false;
  live.classList.remove("bad");
  live.textContent = text;
}

function showError(message) {
  live.hidden = false;
  live.classList.add("bad");
  live.textContent = message || "Something went wrong";
  setStatus("Error");
}

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* keep status text */ }
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  if (response.status === 204) return null;
  const type = response.headers.get("content-type") || "";
  if (!type.includes("application/json")) return response;
  return response.json();
}

function renderThreads() {
  const query = $("search").value.trim().toLowerCase();
  const source = state.searchHits || state.conversations;
  const rows = state.searchHits
    ? source
    : source.filter((row) => !query || `${row.title} ${row.preview}`.toLowerCase().includes(query));
  listEl.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "quiet";
    empty.style.padding = "8px 10px";
    empty.textContent = query ? "No threads match that search." : "No threads yet.";
    listEl.append(empty);
    return;
  }
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "thread-row";
    item.dataset.id = row.id;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread" + (row.id === state.activeId ? " active" : "");
    const strong = document.createElement("strong");
    strong.textContent = displayTitle(row.title);
    const preview = document.createElement("span");
    preview.textContent = row.preview || "Empty chat";
    const time = document.createElement("small");
    time.textContent = when(row.updated_at);
    button.append(strong, preview, time);
    button.addEventListener("click", () => {
      closeThreadMenus();
      if (isNarrow()) {
        document.querySelector(".app").classList.remove("show-history");
        syncPanels();
      }
      openThread(row.id);
    });
    const more = document.createElement("button");
    more.type = "button";
    more.className = "thread-more";
    more.setAttribute("aria-label", "Chat actions");
    more.setAttribute("aria-haspopup", "menu");
    more.setAttribute("aria-expanded", "false");
    more.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="6" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="18" cy="12" r="1.4" fill="currentColor"/></svg>';
    const menu = document.createElement("div");
    menu.className = "thread-menu";
    menu.hidden = true;
    menu.setAttribute("role", "menu");
    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "Rename";
    rename.setAttribute("role", "menuitem");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.setAttribute("role", "menuitem");
    menu.append(rename, remove);
    more.addEventListener("click", (event) => {
      event.stopPropagation();
      const willOpen = menu.hidden;
      closeThreadMenus();
      if (!willOpen) return;
      menu.hidden = false;
      more.setAttribute("aria-expanded", "true");
      const list = listEl.getBoundingClientRect();
      const anchor = more.getBoundingClientRect();
      menu.classList.toggle("up", anchor.bottom + 88 > list.bottom);
    });
    rename.addEventListener("click", (event) => {
      event.stopPropagation();
      closeThreadMenus();
      startRename(row);
    });
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      closeThreadMenus();
      deleteThread(row.id).catch((error) => setStatus(error.message));
    });
    item.append(button, more, menu);
    listEl.append(item);
  }
}

function closeThreadMenus() {
  for (const menu of listEl.querySelectorAll(".thread-menu")) menu.hidden = true;
  for (const more of listEl.querySelectorAll(".thread-more")) more.setAttribute("aria-expanded", "false");
}

function untitledTitle(title) {
  const text = (title || "").trim();
  return !text || text === "New thread" || text === "New chat";
}

function startRename(row) {
  const item = listEl.querySelector('.thread-row[data-id="' + CSS.escape(row.id) + '"]');
  if (!item || item.querySelector(".thread-rename")) return;
  const field = document.createElement("input");
  field.className = "thread-rename";
  field.type = "text";
  field.maxLength = 72;
  field.value = displayTitle(row.title);
  field.setAttribute("aria-label", "Rename chat");
  item.append(field);
  field.focus();
  field.select();
  let settled = false;
  const finish = (save) => {
    if (settled) return;
    settled = true;
    const next = field.value.trim().replace(/\s+/g, " ").slice(0, 72);
    const shown = displayTitle(row.title);
    if (!save || !next || next === shown || (untitledTitle(next) && untitledTitle(row.title))) {
      renderThreads();
      return;
    }
    saveThreadTitle(row.id, next).catch((error) => {
      setStatus(error.message);
      renderThreads();
    });
  };
  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  });
  field.addEventListener("blur", () => finish(true));
}

async function saveThreadTitle(id, next) {
  const saved = await api("/api/conversations/" + encodeURIComponent(id), {
    method: "PATCH",
    body: JSON.stringify({ title: next }),
  });
  const listed = state.conversations.find((item) => item.id === saved.id);
  if (listed) listed.title = saved.title;
  if (state.searchHits) {
    const hit = state.searchHits.find((item) => item.id === saved.id);
    if (hit) hit.title = saved.title;
  }
  if (state.view && state.view.id === id) {
    state.view.title = saved.title;
    if (document.activeElement !== titleEl) titleEl.value = displayTitle(saved.title);
  }
  renderThreads();
}

function linkify(node, text) {
  const pattern = /https?:\/\/[^\s)]+/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index || 0;
    if (index > last) node.append(document.createTextNode(text.slice(last, index)));
    const anchor = document.createElement("a");
    anchor.href = match[0];
    anchor.target = "_blank";
    anchor.rel = "noreferrer";
    anchor.textContent = match[0];
    node.append(anchor);
    last = index + match[0].length;
  }
  if (last < text.length) node.append(document.createTextNode(text.slice(last)));
}

function statusClass(status) {
  if (status === "ok") return "ok";
  if (status === "loop" || status === "no_effect") return "warn";
  return "bad";
}

function formatArgs(args) {
  if (!args || typeof args !== "object") return [];
  return Object.entries(args)
    .filter(([, value]) => value !== "" && value != null)
    .map(([key, value]) => {
      let text = typeof value === "string" ? value : JSON.stringify(value);
      if (text.length > 180) text = text.slice(0, 177) + "…";
      return { key, value: text };
    });
}

function traceMeta(steps, live) {
  const real = steps.filter((step) => step.name !== "steer");
  if (live) {
    const last = real[real.length - 1];
    return last ? `#${last.index} ${last.name}` : "";
  }
  const flagged = real.filter((step) => step.status && step.status !== "ok").length;
  if (!real.length) return "";
  return flagged ? `${real.length} steps · ${flagged} flagged` : `${real.length} steps`;
}

function browserLabel() {
  if (!state.browser.open) return "Opening the browser";
  const url = state.browser.url || "";
  let host = "";
  try {
    if (/^https?:\/\//i.test(url)) host = new URL(url).host.replace(/^www\./, "");
  } catch (_) { /* keep the raw label */ }
  const title = (state.browser.title || "").trim();
  if (title && host) return `${title} · ${host}`;
  if (host) return `Open browser · ${host}`;
  return "Open browser";
}

function renderBrowserRow() {
  const row = document.createElement("button");
  row.type = "button";
  row.id = "trace-browser";
  row.className = "trace-browser" + (state.browser.open ? " ready" : "");
  row.disabled = !state.browser.open;
  const label = document.createElement("span");
  label.textContent = browserLabel();
  row.append(label);
  if (state.browser.open) {
    row.title = state.browser.url || "Open the browser";
    row.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      focusBrowser();
    });
  }
  return row;
}

function updateBrowserRow() {
  const current = document.getElementById("trace-browser");
  if (!current) return;
  current.replaceWith(renderBrowserRow());
}

function renderNote(text, latest) {
  const item = document.createElement("li");
  item.className = "trace-step trace-note" + (latest ? " latest" : "");
  const index = document.createElement("span");
  index.className = "trace-index note";
  index.textContent = "you";
  const body = document.createElement("div");
  body.className = "trace-body";
  const head = document.createElement("div");
  head.className = "trace-head";
  const name = document.createElement("code");
  name.textContent = "your correction";
  head.append(name);
  const detail = document.createElement("p");
  detail.className = "trace-detail";
  detail.textContent = text;
  body.append(head, detail);
  item.append(index, body);
  return item;
}

function renderStep(step, latest) {
  if (step.name === "steer") return renderNote(step.detail || "", latest);
  const item = document.createElement("li");
  item.className = "trace-step" + (latest ? " latest" : "");
  const index = document.createElement("span");
  index.className = "trace-index " + statusClass(step.status);
  index.textContent = String(step.index ?? "");
  const body = document.createElement("div");
  body.className = "trace-body";
  const head = document.createElement("div");
  head.className = "trace-head";
  const status = document.createElement("span");
  status.className = "trace-status " + statusClass(step.status);
  status.textContent = step.status || "step";
  const name = document.createElement("code");
  name.textContent = step.name || "step";
  head.append(status, name);
  body.append(head);
  const args = formatArgs(step.args);
  if (args.length) {
    const list = document.createElement("dl");
    list.className = "trace-args";
    for (const arg of args) {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = arg.key;
      const value = document.createElement("dd");
      value.textContent = arg.value;
      row.append(term, value);
      list.append(row);
    }
    body.append(list);
  }
  if (step.detail) {
    const detail = document.createElement("p");
    detail.className = "trace-detail";
    detail.textContent = step.detail;
    body.append(detail);
  }
  item.append(index, body);
  return item;
}

function chevron() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("class", "chev");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M9 6.5 14.5 12 9 17.5");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "1.8");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  return svg;
}

function buildTrace(steps, { live, key, browser }) {
  const details = document.createElement("details");
  details.className = "trace" + (live ? " live" : "");
  if (live) details.id = "live-trace";
  if (state.openTraces.has(key)) details.open = true;
  const summary = document.createElement("summary");
  const title = document.createElement("span");
  title.className = "trace-title";
  title.textContent = live ? (state.phase || "Working") : "Steps";
  const meta = document.createElement("span");
  meta.className = "trace-meta";
  meta.textContent = traceMeta(steps, live);
  summary.append(chevron(), title);
  if (live) {
    const dot = document.createElement("span");
    dot.className = "pulse";
    dot.setAttribute("aria-hidden", "true");
    summary.append(dot);
  }
  summary.append(meta);
  const list = document.createElement("ol");
  list.className = "trace-list";
  if (live) list.id = "live-steps";
  if (live && !steps.length) {
    const wait = document.createElement("li");
    wait.className = "trace-wait";
    wait.id = "trace-wait";
    wait.textContent = state.phase || "Starting";
    list.append(wait);
  } else {
    steps.forEach((step, index) => {
      list.append(renderStep(step, live && index === steps.length - 1));
    });
  }
  details.append(summary);
  if (state.browser.open && (live || browser)) details.append(renderBrowserRow());
  details.append(list);
  details.addEventListener("toggle", () => {
    if (details.open) state.openTraces.add(key);
    else state.openTraces.delete(key);
  });
  return details;
}

function updateLiveSummary() {
  const trace = document.getElementById("live-trace");
  if (!trace) return;
  const title = trace.querySelector(".trace-title");
  const meta = trace.querySelector(".trace-meta");
  if (title) title.textContent = state.phase || "Working";
  if (meta) meta.textContent = traceMeta(state.steps, true);
  const wait = document.getElementById("trace-wait");
  if (wait) wait.textContent = state.phase || "Starting";
}

function stickToBottom(force) {
  const gap = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight;
  if (force || gap < 140) transcript.scrollTop = transcript.scrollHeight;
  const list = document.getElementById("live-steps");
  if (list) list.scrollTop = list.scrollHeight;
}

function renderMessage(message, steps, options = {}) {
  const wrap = document.createElement("article");
  wrap.className = `bubble ${message.role === "user" ? "user" : "assistant"}`;
  const who = document.createElement("p");
  who.className = "who";
  who.textContent = message.role === "user" ? "You" : "Agent";
  wrap.append(who);
  if ((steps && (steps.length || options.live)) || options.browser) wrap.append(buildTrace(steps || [], options));
  if (message.content) {
    const body = document.createElement("p");
    body.className = "body";
    linkify(body, message.content);
    wrap.append(body);
  }
  return wrap;
}

function displayTitle(title) {
  const text = (title || "").trim();
  if (!text || text === "New thread" || text === "New chat") return "New chat";
  return text;
}

function isFreshChat() {
  return !!(state.view && state.freshId && state.view.id === state.freshId && !(state.view.messages || []).length);
}

function renderTranscript() {
  transcript.replaceChildren();
  const messages = state.view ? state.view.messages : [];
  if (!messages.length && !state.running) {
    const empty = document.createElement("div");
    empty.className = "empty";
    const heading = document.createElement("h2");
    const note = document.createElement("p");
    if (!state.activeId) {
      heading.textContent = "Start a chat.";
      note.textContent = "Pick a chat from the history, or start a new one.";
    } else if (isFreshChat()) {
      heading.textContent = "New chat.";
      note.textContent = "Send a task. The agent opens the web when the task needs a page, and the answer stays in this chat.";
    } else {
      heading.textContent = "Continue where you left off.";
      note.textContent = "This chat is empty. Send a task and the agent will keep the answer here.";
    }
    empty.append(heading, note);
    transcript.append(empty);
    return;
  }
  messages.forEach((message, index) => {
    const trace = Array.isArray(message.trace) && message.trace.length ? message.trace : null;
    const lastAssistant = !state.running && message.role === "assistant" && index === messages.length - 1;
    transcript.append(renderMessage(message, trace, {
      live: false,
      key: message.at || String(index),
      browser: lastAssistant && state.browser.open,
    }));
  });
  if (state.running) {
    transcript.append(renderMessage(
      { role: "assistant", content: "" },
      state.steps,
      { live: true, key: "live" },
    ));
  }
  stickToBottom(true);
}

function renderMemory() {
  shortEl.replaceChildren();
  longEl.replaceChildren();
  const view = state.view;
  const short = view ? view.short_term || [] : [];
  const long = view ? view.long_term || [] : [];
  if (!short.length) {
    const li = document.createElement("li");
    li.className = "quiet";
    li.textContent = "Nothing in the working window yet.";
    shortEl.append(li);
  }
  for (const item of short) {
    const li = document.createElement("li");
    const who = document.createElement("span");
    who.className = "pill";
    who.textContent = item.role === "user" ? "you" : "agent";
    const text = document.createElement("p");
    text.textContent = (item.content || "").slice(0, 280);
    li.append(who, text);
    shortEl.append(li);
  }
  if (!long.length) {
    const li = document.createElement("li");
    li.className = "quiet";
    li.textContent = "No long-term hits for this thread yet. They appear after the agent finishes a turn.";
    longEl.append(li);
  }
  for (const item of long) {
    const li = document.createElement("li");
    const kind = document.createElement("span");
    kind.className = "pill ok";
    kind.textContent = item.kind || "memory";
    const text = document.createElement("p");
    text.textContent = item.text || "";
    const meter = document.createElement("div");
    meter.className = "meter";
    const bar = document.createElement("span");
    const score = Math.max(0, Math.min(1, Number(item.score) || 0));
    bar.style.width = `${Math.round(score * 100)}%`;
    meter.append(bar);
    li.append(kind, text, meter);
    longEl.append(li);
  }
}

function paint() {
  const view = state.view;
  if (document.activeElement !== titleEl) {
    titleEl.value = view ? displayTitle(view.title) : "Choose a chat";
  }
  titleEl.disabled = !view || state.running;
  input.disabled = !view;
  input.placeholder = state.running
    ? "It's going the wrong way — tell it what to do instead…"
    : "Ask the agent to open a page, extract something, or pick up this thread…";
  $("send").textContent = state.running ? "Steer" : "Send";
  $("send").disabled = !view || (state.running && !input.value.trim());
  $("stop").hidden = !state.running;
  $("hint").textContent = state.running
    ? "Enter sends a correction the agent follows on its next step. Stop halts the run."
    : "Enter to send. Shift+Enter for a new line. While it is working, steer it or press Stop.";
  renderThreads();
  renderTranscript();
  renderMemory();
}

async function refreshList() {
  const payload = await api("/api/conversations");
  state.conversations = payload.conversations || [];
  renderThreads();
}

async function openThread(id, { push = true, fresh = false } = {}) {
  state.freshId = fresh ? id : null;
  state.activeId = id;
  state.steps = [];
  state.searchHits = null;
  $("search").value = "";
  const view = await api("/api/conversations/" + encodeURIComponent(id));
  state.view = view;
  if (push) {
    const url = new URL(location.href);
    url.searchParams.set("c", id);
    history.replaceState(null, "", url);
  }
  paint();
}

async function createThread() {
  const view = await api("/api/conversations", { method: "POST", body: "{}" });
  await refreshList();
  await openThread(view.id, { fresh: true });
  input.focus();
}

async function deleteThread(id) {
  const target = id || state.activeId;
  if (!target) return;
  if (!confirm("Delete this chat from history?")) return;
  await api("/api/conversations/" + encodeURIComponent(target), { method: "DELETE" });
  if (state.searchHits) state.searchHits = state.searchHits.filter((item) => item.id !== target);
  if (state.activeId !== target) {
    await refreshList();
    return;
  }
  state.view = null;
  state.activeId = null;
  state.freshId = null;
  const url = new URL(location.href);
  url.searchParams.delete("c");
  history.replaceState(null, "", url);
  await refreshList();
  if (state.conversations[0]) await openThread(state.conversations[0].id);
  else paint();
}

function parseEvent(raw) {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  try {
    return { event, data: data ? JSON.parse(data) : {} };
  } catch (_) {
    return { event, data: { message: data } };
  }
}

async function readStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split = buffer.indexOf("\n\n");
    while (split >= 0) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      if (raw.trim()) await handleEvent(parseEvent(raw));
      split = buffer.indexOf("\n\n");
    }
  }
}

async function handleEvent(message) {
  const data = message.data || {};
  if (message.event === "status") {
    state.phase = data.text || "Working";
    setStatus(state.phase);
    updateLiveSummary();
  } else if (message.event === "browser") {
    state.browser = {
      open: !!data.open,
      url: data.url || "",
      title: data.title || state.browser.title || "",
    };
    updateBrowserRow();
  } else if (message.event === "steer") {
    const note = { index: "you", name: "steer", status: "ok", detail: data.text || "", args: {} };
    state.steps.push(note);
    const list = document.getElementById("live-steps");
    if (!list) {
      renderTranscript();
    } else {
      list.querySelectorAll(".latest").forEach((node) => node.classList.remove("latest"));
      list.append(renderStep(note, true));
      updateLiveSummary();
      stickToBottom();
    }
  } else if (message.event === "step") {
    if (data.url) {
      state.browser = { open: true, url: data.url, title: state.browser.title || "" };
      updateBrowserRow();
    }
    state.steps.push(data);
    const list = document.getElementById("live-steps");
    if (!list) {
      renderTranscript();
    } else {
      const wait = document.getElementById("trace-wait");
      if (wait) wait.remove();
      list.querySelectorAll(".latest").forEach((node) => node.classList.remove("latest"));
      list.append(renderStep(data, true));
      updateLiveSummary();
      stickToBottom();
    }
  } else if (message.event === "confirm") {
    const allowed = await askModal({
      kicker: "Needs a yes",
      title: data.name || "Confirm",
      body: data.reason || "Allow this action?",
      extra: data.args && Object.keys(data.args).length ? JSON.stringify(data.args, null, 2) : "",
      allow: true,
    });
    try {
      await api("/api/decide", {
        method: "POST",
        body: JSON.stringify({ token: data.token, value: allowed ? "yes" : "no" }),
      });
    } catch (_) { /* the run may already have been stopped */ }
  } else if (message.event === "ask") {
    const answer = await askModal({
      kicker: "The agent asks",
      title: "A question",
      body: data.question || "",
      allow: false,
    });
    try {
      await api("/api/decide", {
        method: "POST",
        body: JSON.stringify({ token: data.token, value: answer }),
      });
    } catch (_) { /* the run may already have been stopped */ }
  } else if (message.event === "done") {
    if (modal.open) modal.close();
    state.view = data.conversation || state.view;
    state.steps = [];
    state.phase = "";
    state.running = false;
    state.openTraces.delete("live");
    live.hidden = true;
    setStatus(data.success ? "Answered" : "Stopped");
    await refreshList();
    paint();
  } else if (message.event === "error") {
    showError(data.message || "Something went wrong");
  }
}

function askModal({ kicker, title, body, extra = "", allow }) {
  return new Promise((resolve) => {
    $("modal-kicker").textContent = kicker;
    $("modal-title").textContent = title;
    $("modal-body").textContent = extra ? `${body}\n\n${extra}` : body;
    const field = $("modal-input");
    const yes = $("modal-yes");
    const no = $("modal-no");
    const form = $("modal-form");
    field.hidden = allow;
    field.value = "";
    yes.textContent = allow ? "Allow" : "Send";
    no.hidden = !allow;
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      form.removeEventListener("submit", onSubmit);
      no.removeEventListener("click", onNo);
      modal.removeEventListener("cancel", onCancel);
      if (modal.open) modal.close();
      resolve(value);
    };
    const onNo = () => finish(allow ? false : "");
    const onCancel = (event) => {
      event.preventDefault();
      finish(allow ? false : "");
    };
    const onSubmit = (event) => {
      event.preventDefault();
      finish(allow ? true : field.value);
    };
    no.addEventListener("click", onNo);
    modal.addEventListener("cancel", onCancel);
    form.addEventListener("submit", onSubmit);
    modal.showModal();
    if (!allow) field.focus();
  });
}

async function send(text) {
  if (!state.activeId || state.running) return;
  state.running = true;
  state.steps = [];
  state.phase = "Starting";
  state.openTraces.add("live");
  state.view.messages = state.view.messages.concat([{ role: "user", content: text, at: new Date().toISOString() }]);
  live.hidden = true;
  live.classList.remove("bad");
  setStatus("Working");
  paint();
  try {
    const response = await fetch("/api/conversations/" + encodeURIComponent(state.activeId) + "/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok || !response.body) {
      let detail = "The agent could not start.";
      try { detail = (await response.json()).detail || detail; } catch (_) {}
      throw new Error(typeof detail === "string" ? detail : "The agent could not start.");
    }
    await readStream(response);
  } catch (error) {
    showError(error.message);
  } finally {
    state.running = false;
    paint();
  }
}

let searchTimer = 0;
async function onSearch() {
  const query = $("search").value.trim();
  window.clearTimeout(searchTimer);
  if (query.length < 2) {
    state.searchHits = null;
    renderThreads();
    return;
  }
  searchTimer = window.setTimeout(async () => {
    try {
      const payload = await api("/api/search?q=" + encodeURIComponent(query));
      const hits = payload.hits || [];
      state.searchHits = hits.length ? hits : null;
      renderThreads();
    } catch (_) {
      state.searchHits = null;
      renderThreads();
    }
  }, 220);
}

function isNarrow() {
  return window.matchMedia("(max-width: 1100px)").matches;
}

function syncPanels() {
  const app = document.querySelector(".app");
  const history = app.classList.contains("show-history");
  const memory = app.classList.contains("show-memory");
  $("show-history").hidden = history;
  $("show-memory").setAttribute("aria-pressed", String(memory));
  $("show-memory").textContent = memory ? "Hide memory" : "Memory";
  $("scrim").hidden = !(isNarrow() && (history || memory));
}

async function saveTitle() {
  if (!state.view || state.running) return;
  const id = state.view.id;
  const next = titleEl.value.trim().replace(/\s+/g, " ").slice(0, 72);
  const rawTitle = (state.view.title || "").trim();
  const stored = rawTitle && rawTitle !== "New thread" && rawTitle !== "New chat" ? rawTitle : "";
  const shown = displayTitle(stored);
  if (!next || next === shown || ((next === "New chat" || next === "New thread") && !stored)) {
    if (state.view && state.view.id === id && document.activeElement !== titleEl) titleEl.value = shown;
    return;
  }
  try {
    const saved = await api("/api/conversations/" + encodeURIComponent(id), {
      method: "PATCH",
      body: JSON.stringify({ title: next }),
    });
    const row = state.conversations.find((item) => item.id === saved.id);
    if (row) row.title = saved.title;
    if (state.view && state.view.id === id) {
      state.view = saved;
      if (document.activeElement !== titleEl) titleEl.value = saved.title;
    }
    renderThreads();
  } catch (error) {
    if (state.view && state.view.id === id) titleEl.value = shown;
    setStatus(error.message);
  }
}

function togglePanel(name) {
  document.querySelector(".app").classList.toggle(name === "history" ? "show-history" : "show-memory");
  syncPanels();
}

$("new-thread").addEventListener("click", () => createThread().catch((error) => setStatus(error.message)));
document.addEventListener("click", (event) => {
  if (event.target.closest(".thread-more, .thread-menu, .thread-rename")) return;
  closeThreadMenus();
});
$("search").addEventListener("input", onSearch);
$("hide-history").addEventListener("click", () => {
  document.querySelector(".app").classList.remove("show-history");
  syncPanels();
});
$("show-history").addEventListener("click", () => togglePanel("history"));
titleEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    titleEl.blur();
  } else if (event.key === "Escape") {
    titleEl.value = state.view ? displayTitle(state.view.title) : "Choose a chat";
    titleEl.blur();
  }
});
titleEl.addEventListener("blur", () => { saveTitle(); });
$("show-memory").addEventListener("click", () => togglePanel("memory"));
$("scrim").addEventListener("click", () => {
  if (!isNarrow()) return;
  const app = document.querySelector(".app");
  app.classList.remove("show-history", "show-memory");
  syncPanels();
});
syncPanels();
$("stop").addEventListener("click", () => { stopRun(); });
$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  if (state.running) {
    steer(text);
    paint();
    return;
  }
  send(text);
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
  if (state.running) $("send").disabled = !input.value.trim();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

async function steer(text) {
  try {
    await api("/api/steer", { method: "POST", body: JSON.stringify({ text }) });
  } catch (error) {
    setStatus(error.message);
  }
}

async function stopRun() {
  if (modal.open) modal.close();
  try {
    await api("/api/stop", { method: "POST", body: "{}" });
  } catch (error) {
    setStatus(error.message);
  }
}

async function focusBrowser() {
  if (!state.browser.open) return;
  try {
    const info = await api("/api/browser/focus", { method: "POST", body: "{}" });
    state.browser = {
      open: !!info.open,
      url: info.url || state.browser.url,
      title: info.title || state.browser.title || "",
    };
    updateBrowserRow();
  } catch (error) {
    setStatus(error.message);
    return;
  }
  const url = state.browser.url || "";
  if (/^https?:\/\//i.test(url)) {
    const tab = window.open(url, "_blank", "noopener,noreferrer");
    if (!tab) setStatus("Browser brought forward. Allow pop-ups to also open the page.");
  }
}

async function boot() {
  try {
    const status = await api("/api/status");
    if (status.page && status.page.open) state.browser = {
      open: true,
      url: status.page.url || "",
      title: status.page.title || "",
    };
    await refreshList();
    const requested = new URL(location.href).searchParams.get("c");
    const pick = state.conversations.find((row) => row.id === requested)
      || state.conversations.find((row) => row.message_count > 0)
      || state.conversations[0];
    if (pick) await openThread(pick.id);
    else paint();
  } catch (error) {
    setStatus(error.message || "Cannot reach the server");
    paint();
  }
}

boot();
