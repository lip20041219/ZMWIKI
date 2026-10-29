"use strict";

const state = {
  pages: [],
  current: null,
  editorMode: null, // null | "create" | "edit"
  dirty: false,
  readOnly: false,
  writable: false,
  workspaceConfigured: true,
  agentConfigured: false,
  importConfigured: false,
  model: null, // /api/config payload; never contains the API key
  setupAutoOpened: false,
  setupBusy: false,
  busy: false,
  asking: false,
  searchSeq: 0,
};

const els = {
  status: document.getElementById("status"),
  accessBadge: document.getElementById("access-badge"),
  search: document.getElementById("search"),
  listTitle: document.getElementById("list-title"),
  listCount: document.getElementById("list-count"),
  pageList: document.getElementById("page-list"),
  importFile: document.getElementById("import-file"),
  importSubmit: document.getElementById("import-submit"),
  importHint: document.getElementById("import-hint"),
  importStatus: document.getElementById("import-status"),
  empty: document.getElementById("empty"),
  reader: document.getElementById("reader"),
  pageTitle: document.getElementById("page-title"),
  pagePath: document.getElementById("page-path"),
  pageDescription: document.getElementById("page-description"),
  pageContent: document.getElementById("page-content"),
  pageTags: document.getElementById("page-tags"),
  details: document.getElementById("details"),
  pageDetails: document.getElementById("page-details"),
  pageMeta: document.getElementById("page-meta"),
  outbound: document.getElementById("outbound"),
  inbound: document.getElementById("inbound"),
  sources: document.getElementById("sources"),
  editor: document.getElementById("editor"),
  editorTitle: document.getElementById("editor-title"),
  editorHint: document.getElementById("editor-hint"),
  editorSubmit: document.getElementById("editor-submit"),
  editorCancel: document.getElementById("editor-cancel"),
  editId: document.getElementById("edit-id"),
  editPath: document.getElementById("edit-path"),
  editTitle: document.getElementById("edit-title"),
  editAliases: document.getElementById("edit-aliases"),
  editTags: document.getElementById("edit-tags"),
  editDescription: document.getElementById("edit-description"),
  editLinks: document.getElementById("edit-links"),
  editContent: document.getElementById("edit-content"),
  newPage: document.getElementById("new-page"),
  save: document.getElementById("save-workspace"),
  editPage: document.getElementById("edit-page"),
  deletePage: document.getElementById("delete-page"),
  askForm: document.getElementById("ask-form"),
  askQuestion: document.getElementById("ask-question"),
  askSubmit: document.getElementById("ask-submit"),
  askHint: document.getElementById("ask-hint"),
  askStatus: document.getElementById("ask-status"),
  askResult: document.getElementById("ask-result"),
  askAnswer: document.getElementById("ask-answer"),
  askSteps: document.getElementById("ask-steps"),
  askEvidence: document.getElementById("ask-evidence"),
  setupOpen: document.getElementById("setup-open"),
  setup: document.getElementById("setup"),
  setupForm: document.getElementById("setup-form"),
  setupBaseUrl: document.getElementById("setup-base-url"),
  setupModel: document.getElementById("setup-model"),
  setupApiKey: document.getElementById("setup-api-key"),
  setupKeyHint: document.getElementById("setup-key-hint"),
  setupClearWrap: document.getElementById("setup-clear-wrap"),
  setupClearKey: document.getElementById("setup-clear-key"),
  setupStatus: document.getElementById("setup-status"),
  setupServerStatus: document.getElementById("setup-server-status"),
  setupTest: document.getElementById("setup-test"),
  setupSave: document.getElementById("setup-save"),
  setupClose: document.getElementById("setup-close"),
};

let searchTimer = null;

function setStatus(message, tone = "") {
  els.status.textContent = message;
  els.status.className = tone ? `status ${tone}` : "status";
}

function canWrite() {
  return state.writable && !state.readOnly && !state.busy;
}

function updateSaveButton() {
  els.save.textContent = state.dirty ? "Save •" : "Save";
  els.save.classList.toggle("dirty", state.dirty);
}

function setAccessBadge() {
  if (state.readOnly) {
    els.accessBadge.hidden = false;
    els.accessBadge.textContent = "Read-only";
    els.accessBadge.className = "badge readonly";
  } else if (!state.workspaceConfigured) {
    els.accessBadge.hidden = false;
    els.accessBadge.textContent = "No workspace";
    els.accessBadge.className = "badge readonly";
  } else {
    els.accessBadge.hidden = true;
  }
}

function applyCapabilities() {
  const writable = state.writable && !state.readOnly;
  const locked = state.busy || !writable;
  els.newPage.disabled = locked || state.editorMode !== null;
  els.save.disabled = locked || state.editorMode !== null;
  els.editPage.disabled =
    locked || !state.current || state.editorMode !== null;
  els.deletePage.disabled =
    locked || !state.current || state.editorMode !== null;
  els.editorSubmit.disabled = locked;
  els.editorCancel.disabled = state.busy;
  const importReady = writable && state.importConfigured;
  els.importFile.disabled = state.busy || !importReady;
  els.importSubmit.disabled =
    state.busy || !importReady || els.importFile.files.length === 0;
  els.importHint.textContent = state.importConfigured
    ? "Compiles a UTF-8 .txt file into a wiki page and saves the workspace."
    : "Open Model setup to configure the server.";
  setAccessBadge();
  updateSaveButton();
  applyAskState();
  renderServerStatus();
}

function applyAskState() {
  const disabled = state.asking || !state.agentConfigured;
  els.askSubmit.disabled = disabled;
  els.askQuestion.disabled = state.asking;
  els.askSubmit.textContent = state.asking ? "Asking…" : "Ask";
  els.askHint.textContent = state.agentConfigured
    ? "Multi-step search, read and link traversal."
    : "Open Model setup to configure the server.";
}

function setAskStatus(message, tone = "") {
  els.askStatus.textContent = message;
  els.askStatus.className = tone ? `ask-status ${tone}` : "ask-status";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
  });
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }
  if (!response.ok) {
    const message =
      payload && typeof payload.message === "string"
        ? payload.message
        : `Request failed (${response.status})`;
    const error = new Error(message);
    error.status = response.status;
    error.code = payload ? payload.error : null;
    throw error;
  }
  return payload;
}

function jsonRequest(path, method, payload) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function splitList(value) {
  return Array.from(
    new Set(
      value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
    )
  );
}

function markDirty() {
  state.dirty = true;
  updateSaveButton();
}

function markSaved() {
  state.dirty = false;
  updateSaveButton();
}

async function loadCapabilities() {
  try {
    const data = await api("/api/capabilities");
    state.readOnly = Boolean(data.read_only);
    state.writable = Boolean(data.writable);
    state.workspaceConfigured = Boolean(data.workspace_configured);
    state.agentConfigured = Boolean(data.agent_configured);
    state.importConfigured = Boolean(data.import_configured);
    applyCapabilities();
  } catch (error) {
    // Keep the page usable against older servers; writes will report a clear
    // server error if the server does not support them.
    state.readOnly = false;
    state.writable = true;
    state.workspaceConfigured = true;
    state.agentConfigured = false;
    state.importConfigured = false;
    applyCapabilities();
    setStatus(`Capabilities unavailable: ${error.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Model setup
// ---------------------------------------------------------------------------

function renderServerStatus() {
  if (!els.setupServerStatus) {
    return;
  }
  const model = state.model;
  const workspace = state.readOnly
    ? "read-only"
    : state.workspaceConfigured
      ? "writable"
      : "not configured";
  const modelText =
    model && model.configured
      ? `configured (${model.source})`
      : "not configured";
  els.setupServerStatus.textContent =
    `Server: online · Workspace: ${workspace} · Model: ${modelText}`;
}

function updateSetupButton() {
  const configured = Boolean(state.model && state.model.configured);
  els.setupOpen.textContent = configured ? "Model" : "Set up model";
  els.setupOpen.classList.toggle("attention", !configured);
}

function setSetupStatus(message, tone = "") {
  els.setupStatus.textContent = message;
  els.setupStatus.className = tone ? `setup-status ${tone}` : "setup-status";
}

function fillSetupForm() {
  const model = state.model || {};
  els.setupBaseUrl.value = model.base_url || "";
  els.setupModel.value = model.model || "";
  els.setupApiKey.value = "";
  els.setupApiKey.placeholder = model.has_api_key
    ? "Leave blank to keep the stored key"
    : "Optional for local endpoints";
  els.setupClearKey.checked = false;
  els.setupClearWrap.hidden = !model.stored_api_key;
  els.setupKeyHint.textContent = model.key_from_environment
    ? "A key is set by the server environment (ZMWIKI_API_KEY)."
    : model.stored_api_key
      ? "A key is stored on the server. It is never sent to this page."
      : "The key goes only to the local server and is stored there, not in the browser.";
  renderServerStatus();
}

function openSetup() {
  fillSetupForm();
  setSetupStatus("", "");
  if (!els.setup.open) {
    els.setup.showModal();
  }
}

function closeSetup() {
  if (els.setup.open) {
    els.setup.close();
  }
}

function setSetupBusy(busy) {
  state.setupBusy = busy;
  els.setupTest.disabled = busy;
  els.setupSave.disabled = busy;
}

function setupPayload() {
  const payload = {
    base_url: els.setupBaseUrl.value.trim(),
    model: els.setupModel.value.trim(),
  };
  const key = els.setupApiKey.value.trim();
  if (key) {
    payload.api_key = key;
  } else if (!els.setupClearWrap.hidden && els.setupClearKey.checked) {
    payload.clear_api_key = true;
  }
  return payload;
}

async function loadModelConfig() {
  try {
    state.model = await api("/api/config");
  } catch (error) {
    state.model = { configured: false, source: "unknown" };
  }
  updateSetupButton();
  renderServerStatus();
}

async function testSetupConnection() {
  if (state.setupBusy) {
    return;
  }
  const payload = setupPayload();
  if (!payload.base_url || !payload.model) {
    setSetupStatus("Enter a Base URL and a model name first.", "error");
    return;
  }
  setSetupBusy(true);
  setSetupStatus("Testing connection…", "");
  try {
    const result = await jsonRequest("/api/config/test", "POST", payload);
    setSetupStatus(
      result.message || (result.ok ? "Connected." : "Connection failed."),
      result.ok ? "ok" : "error"
    );
  } catch (error) {
    setSetupStatus(`Connection test failed: ${error.message}`, "error");
  } finally {
    setSetupBusy(false);
  }
}

async function saveSetup(event) {
  event.preventDefault();
  if (state.setupBusy) {
    return;
  }
  const payload = setupPayload();
  if (!payload.base_url || !payload.model) {
    setSetupStatus("Enter a Base URL and a model name first.", "error");
    return;
  }
  setSetupBusy(true);
  setSetupStatus("Saving model settings…", "");
  try {
    state.model = await jsonRequest("/api/config", "POST", payload);
    els.setupApiKey.value = "";
    await loadCapabilities();
    updateSetupButton();
    closeSetup();
    setStatus("Model settings saved.", "ok");
  } catch (error) {
    setSetupStatus(withHint(`Save failed: ${error.message}`, error.code), "error");
  } finally {
    setSetupBusy(false);
  }
}

function maybeOpenSetup() {
  if (state.setupAutoOpened) {
    return;
  }
  if (state.model && state.model.configured) {
    return;
  }
  state.setupAutoOpened = true;
  openSetup();
}

// ---------------------------------------------------------------------------
// Actionable error hints
// ---------------------------------------------------------------------------

const ERROR_HINTS = {
  model_not_configured: "open Model setup to configure the server",
  compile_failed: "open Model setup and run Test connection",
  config_save_failed: "check that the settings file is writable",
  save_failed: "check disk space and write permissions for the workspace file",
  invalid_encoding: "save the file as UTF-8 plain text",
  request_too_large: "the upload limit is 2 MiB",
  page_exists: "the page already exists; rename the file or delete that page",
  path_conflict: "another page already uses that path",
  invalid_filename: "use a plain .txt base name",
  invalid_config: "check the Base URL, model name and API key fields",
  read_only: "this workspace is read-only; use a writable workspace or start with --base-url/--model",
};

function withHint(message, code) {
  const hint = code ? ERROR_HINTS[code] : "";
  return hint ? `${message} — ${hint}` : message;
}

function renderTags(tags) {
  els.pageTags.replaceChildren();
  for (const tag of tags) {
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = tag;
    els.pageTags.append(span);
  }
}

function renderMeta(page) {
  els.pageMeta.replaceChildren();
  const rows = [
    ["ID", page.id],
    ["Path", page.path],
    ["Aliases", (page.aliases || []).join(", ") || "—"],
    ["Tags", (page.tags || []).join(", ") || "—"],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    els.pageMeta.append(dt, dd);
  }
}

function renderLinks(container, pageIds, emptyText) {
  container.replaceChildren();
  if (!pageIds.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = emptyText;
    container.append(empty);
    return;
  }
  for (const pageId of pageIds) {
    const target = state.pages.find((page) => page.id === pageId);
    if (!target) {
      const missing = document.createElement("div");
      missing.className = "link-item missing";
      missing.textContent = `${pageId} (missing)`;
      container.append(missing);
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "link-item";
    button.dataset.pageId = pageId;
    button.textContent = target.title;
    const id = document.createElement("span");
    id.className = "link-id";
    id.textContent = pageId;
    button.append(id);
    container.append(button);
  }
}

function renderSources(sources) {
  els.sources.replaceChildren();
  if (!sources.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = "No source documents.";
    els.sources.append(empty);
    return;
  }
  for (const source of sources) {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source;
    els.sources.append(chip);
  }
}

const ASK_STATUS = {
  answered: { label: "Answered", tone: "ok" },
  insufficient_evidence: { label: "Insufficient evidence", tone: "warn" },
  budget_exhausted: { label: "Step budget exhausted", tone: "error" },
  model_error: { label: "Model error", tone: "error" },
  protocol_error: { label: "Protocol error", tone: "error" },
};

function formatStepParams(step) {
  const params = step.params || {};
  if (step.action === "search" && params.query) {
    return `query: ${params.query}`;
  }
  if (step.action === "read" && Array.isArray(params.paths)) {
    return `paths: ${params.paths.join(", ")}`;
  }
  if (step.action === "follow_links" && params.from_page) {
    return `from: ${params.from_page}`;
  }
  if (step.action === "answer" && Array.isArray(params.sources)) {
    return `sources: ${params.sources.length}`;
  }
  return "";
}

function renderAskSteps(steps) {
  els.askSteps.replaceChildren();
  if (!steps.length) {
    const empty = document.createElement("li");
    empty.className = "step-empty";
    empty.textContent = "No steps recorded.";
    els.askSteps.append(empty);
    return;
  }
  for (const step of steps) {
    const item = document.createElement("li");
    item.className = "step-item";

    const head = document.createElement("div");
    head.className = "step-head";
    const number = document.createElement("span");
    number.className = "step-num";
    number.textContent = String((step.step || 0) + 1).padStart(2, "0");
    const action = document.createElement("span");
    action.className = "step-action";
    action.textContent = step.action || "unknown";
    head.append(number, action);
    item.append(head);

    const summary = document.createElement("p");
    summary.className = "step-summary";
    summary.textContent = step.result_summary || "";
    const extra = formatStepParams(step);
    if (extra) {
      summary.textContent += ` — ${extra}`;
    }
    item.append(summary);

    if (step.error) {
      const error = document.createElement("p");
      error.className = "step-error";
      error.textContent = step.error;
      item.append(error);
    }
    if (Array.isArray(step.search_hit_page_ids) && step.search_hit_page_ids.length) {
      const refs = document.createElement("p");
      refs.className = "step-refs";
      refs.textContent = `search hits: ${step.search_hit_page_ids.join(", ")}`;
      item.append(refs);
    }
    if (Array.isArray(step.read_page_ids) && step.read_page_ids.length) {
      const refs = document.createElement("p");
      refs.className = "step-refs";
      refs.textContent = `read: ${step.read_page_ids.join(", ")}`;
      item.append(refs);
    }
    els.askSteps.append(item);
  }
}

function renderAskEvidence(evidence) {
  els.askEvidence.replaceChildren();
  if (!evidence.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = "No cited pages.";
    els.askEvidence.append(empty);
    return;
  }
  for (const item of evidence) {
    const row = document.createElement(item.id ? "button" : "div");
    row.className = item.id ? "link-item" : "link-item missing";
    if (item.id) {
      row.type = "button";
      row.dataset.pageId = item.id;
    }
    row.textContent = item.title || item.path || "Unknown page";
    const meta = document.createElement("span");
    meta.className = "link-id";
    const parts = [item.path || ""];
    if (Array.isArray(item.sources) && item.sources.length) {
      parts.push(`sources: ${item.sources.join(", ")}`);
    }
    meta.textContent = parts.filter(Boolean).join(" · ");
    row.append(meta);
    els.askEvidence.append(row);
  }
}

function renderAskResult(data) {
  els.askResult.hidden = false;
  els.askAnswer.textContent =
    data.answer || data.error || "No answer produced.";
  renderAskSteps(Array.isArray(data.steps) ? data.steps : []);
  renderAskEvidence(Array.isArray(data.evidence) ? data.evidence : []);
  const info = ASK_STATUS[data.status] || {
    label: data.status || "Unknown",
    tone: "error",
  };
  const stepCount = Array.isArray(data.steps) ? data.steps.length : 0;
  const pagesRead = Number.isInteger(data.pages_read) ? data.pages_read : 0;
  setAskStatus(
    `${info.label} · ${stepCount} steps · ${pagesRead} pages read`,
    info.tone
  );
}

async function submitAsk(event) {
  event.preventDefault();
  if (state.asking) {
    return;
  }
  if (!state.agentConfigured) {
    setAskStatus("Agent Q&A is not configured on the server.", "error");
    return;
  }
  const question = els.askQuestion.value.trim();
  if (!question) {
    setAskStatus("Enter a question first.", "error");
    return;
  }

  state.asking = true;
  applyCapabilities();
  els.askResult.hidden = true;
  els.askAnswer.textContent = "";
  els.askSteps.replaceChildren();
  els.askEvidence.replaceChildren();
  setAskStatus("Running agent…", "");

  try {
    const data = await jsonRequest("/api/ask", "POST", { question });
    renderAskResult(data);
  } catch (error) {
    setAskStatus(withHint(`Ask failed: ${error.message}`, error.code), "error");
  } finally {
    state.asking = false;
    applyCapabilities();
  }
}

function showEmpty() {
  els.reader.hidden = true;
  els.editor.hidden = true;
  els.details.hidden = false;
  els.pageDetails.hidden = true;
  els.empty.hidden = false;
}

function updateActiveListItem(pageId) {
  for (const item of els.pageList.querySelectorAll("[data-page-id]")) {
    const active = item.dataset.pageId === pageId;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", active ? "true" : "false");
  }
}

function renderPage(page) {
  state.current = page;
  state.editorMode = null;
  els.empty.hidden = true;
  els.editor.hidden = true;
  els.reader.hidden = false;
  els.details.hidden = false;
  els.pageDetails.hidden = false;
  updateActiveListItem(page.id);

  els.pageTitle.textContent = page.title || "";
  els.pagePath.textContent = page.path || "";
  els.pageDescription.textContent = page.description || "";
  els.pageDescription.hidden = !page.description;
  els.pageContent.textContent = page.content || "";
  renderTags(page.tags || []);
  renderMeta(page);
  renderLinks(els.outbound, page.links_to || [], "No outgoing links.");
  renderLinks(els.inbound, page.links_from || [], "No backlinks.");
  renderSources(page.sources || []);
  applyCapabilities();
}

function renderPageList(pages, label) {
  els.pageList.replaceChildren();
  els.listTitle.textContent = label;
  els.listCount.textContent = String(pages.length);
  if (!pages.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = "No pages found.";
    els.pageList.append(empty);
    return;
  }
  const activeId = state.current ? state.current.id : null;
  for (const page of pages) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "page-item";
    button.dataset.pageId = page.id;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", page.id === activeId ? "true" : "false");
    if (page.id === activeId) {
      button.classList.add("active");
    }

    const title = document.createElement("span");
    title.className = "page-item-title";
    title.textContent = page.title || page.id;

    const path = document.createElement("span");
    path.className = "page-item-path";
    path.textContent = page.path || "";

    button.append(title, path);
    if (page.tags && page.tags.length) {
      const tags = document.createElement("span");
      tags.className = "page-item-tags";
      tags.textContent = page.tags.join(" · ");
      button.append(tags);
    }
    els.pageList.append(button);
  }
}

async function loadPage(pageId, { silent = false } = {}) {
  try {
    const page = await api(`/api/page?id=${encodeURIComponent(pageId)}`);
    renderPage(page);
    if (!silent) {
      setStatus(`Opened ${page.title}`, "");
    }
  } catch (error) {
    setStatus(`Cannot open page: ${error.message}`, "error");
  }
}

async function loadPages({ selectId = null, silent = false } = {}) {
  try {
    const data = await api("/api/pages");
    state.pages = Array.isArray(data.pages) ? data.pages : [];
    renderPageList(state.pages, "Pages");
    const wanted =
      selectId ||
      (state.current &&
      state.pages.some((page) => page.id === state.current.id)
        ? state.current.id
        : null);
    const target = wanted || (state.pages[0] ? state.pages[0].id : null);
    if (target) {
      await loadPage(target, { silent: true });
      if (!silent) {
        const current = state.current;
        if (current) {
          setStatus(`Opened ${current.title}`, "");
        }
      }
    } else {
      state.current = null;
      showEmpty();
      if (!silent) {
        setStatus(
          state.readOnly
            ? "Read-only workspace is empty."
            : "Workspace is empty. Create a page to begin.",
          state.readOnly ? "warn" : ""
        );
      }
    }
    applyCapabilities();
  } catch (error) {
    setStatus(`Cannot load pages: ${error.message}`, "error");
  }
}

function fillEditor(page) {
  els.editId.value = page ? page.id : "";
  els.editPath.value = page ? page.path : "";
  els.editTitle.value = page ? page.title : "";
  els.editAliases.value = page ? (page.aliases || []).join(", ") : "";
  els.editTags.value = page ? (page.tags || []).join(", ") : "";
  els.editDescription.value = page && page.description ? page.description : "";
  els.editLinks.value = page ? (page.links_to || []).join(", ") : "";
  els.editContent.value = page ? page.content : "";
}

function openCreate() {
  if (!canWrite() || state.editorMode !== null) {
    return;
  }
  state.editorMode = "create";
  els.empty.hidden = true;
  els.reader.hidden = true;
  els.details.hidden = false;
  els.pageDetails.hidden = true;
  els.editor.hidden = false;
  els.editorTitle.textContent = "New page";
  els.editorHint.textContent =
    "ID and path must be unique. Links are page IDs that already exist.";
  els.editId.disabled = false;
  fillEditor(null);
  setStatus("New page — fill the fields and apply.", "");
  applyCapabilities();
  els.editId.focus();
}

function openEdit() {
  if (!canWrite() || !state.current || state.editorMode !== null) {
    return;
  }
  const page = state.current;
  state.editorMode = "edit";
  els.empty.hidden = true;
  els.reader.hidden = true;
  els.editor.hidden = false;
  els.details.hidden = false;
  els.pageDetails.hidden = false;
  els.editorTitle.textContent = `Edit ${page.title}`;
  els.editorHint.textContent =
    "ID is fixed while editing. Links are page IDs that already exist.";
  els.editId.disabled = true;
  fillEditor(page);
  setStatus(`Editing ${page.id}`, "");
  applyCapabilities();
  els.editTitle.focus();
}

function closeEditor() {
  if (state.busy || state.editorMode === null) {
    return;
  }
  state.editorMode = null;
  els.editor.hidden = true;
  els.editId.disabled = false;
  if (state.current) {
    renderPage(state.current);
  } else {
    showEmpty();
  }
  setStatus("Edit cancelled.", "");
  applyCapabilities();
}

function collectEditorPayload() {
  const payload = {
    id: els.editId.value.trim(),
    path: els.editPath.value.trim(),
    title: els.editTitle.value.trim(),
    aliases: splitList(els.editAliases.value),
    tags: splitList(els.editTags.value),
    description: els.editDescription.value.trim() || null,
    links_to: splitList(els.editLinks.value),
    content: els.editContent.value,
  };
  if (!payload.id || !payload.path || !payload.title || !payload.content.trim()) {
    setStatus("ID, path, title and content are required.", "error");
    return null;
  }
  return payload;
}

async function submitEditor(event) {
  event.preventDefault();
  if (!canWrite() || state.editorMode === null) {
    return;
  }
  const payload = collectEditorPayload();
  if (!payload) {
    return;
  }

  const mode = state.editorMode;
  const targetId = mode === "edit" && state.current
    ? state.current.id
    : payload.id;
  state.busy = true;
  applyCapabilities();
  setStatus(mode === "edit" ? "Updating page…" : "Creating page…", "");

  try {
    if (mode === "edit") {
      await jsonRequest(
        `/api/page?id=${encodeURIComponent(targetId)}`,
        "PUT",
        payload
      );
    } else {
      await jsonRequest("/api/pages", "POST", payload);
    }
    state.editorMode = null;
    els.editor.hidden = true;
    els.editId.disabled = false;
    markDirty();
    els.search.value = "";
    await loadPages({ selectId: targetId, silent: true });
    setStatus(
      `${mode === "edit" ? "Updated" : "Created"} ${targetId}. Unsaved changes.`,
      "warn"
    );
  } catch (error) {
    setStatus(withHint(`Apply failed: ${error.message}`, error.code), "error");
  } finally {
    state.busy = false;
    applyCapabilities();
  }
}

async function saveWorkspace() {
  if (!canWrite()) {
    return;
  }
  state.busy = true;
  applyCapabilities();
  setStatus("Saving workspace…", "");
  try {
    await api("/api/workspace/save", { method: "POST" });
    markSaved();
    setStatus("Workspace saved (previous version kept as .bak).", "ok");
  } catch (error) {
    setStatus(withHint(`Save failed: ${error.message}`, error.code), "error");
  } finally {
    state.busy = false;
    applyCapabilities();
  }
}

function setImportStatus(message, tone = "") {
  els.importStatus.textContent = message;
  els.importStatus.className = tone ? `import-status ${tone}` : "import-status";
}

async function submitImport() {
  if (!canWrite() || !state.importConfigured) {
    return;
  }
  const file = els.importFile.files[0];
  if (!file) {
    setImportStatus("Choose a .txt file first.", "error");
    return;
  }
  if (!/\.txt$/i.test(file.name)) {
    setImportStatus("Only UTF-8 .txt files are supported.", "error");
    return;
  }
  if (state.editorMode !== null) {
    setImportStatus("Finish or cancel the page editor first.", "error");
    return;
  }

  state.busy = true;
  applyCapabilities();
  setImportStatus(`Importing ${file.name}…`, "");
  try {
    const data = await api(
      `/api/import?filename=${encodeURIComponent(file.name)}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "text/plain;charset=utf-8",
          "X-ZMWiki-Import": "1",
        },
        body: file,
      }
    );
    const pages = [
      ...(Array.isArray(data.imported_pages) ? data.imported_pages : []),
      ...(Array.isArray(data.updated_pages) ? data.updated_pages : []),
    ];
    const target = pages.length ? pages[0].id : null;
    const links = data.links || {};
    els.importFile.value = "";
    els.search.value = "";
    await loadPages({ selectId: target, silent: true });
    // Import saves the workspace on the server, so nothing is left dirty.
    markSaved();
    const verb = data.status === "updated" ? "Updated" : "Imported";
    setImportStatus(
      `${verb} ${file.name} → ${target || "page"} · ` +
        `${links.resolved || 0} links resolved · workspace saved`,
      "ok"
    );
  } catch (error) {
    setImportStatus(
      withHint(`Import failed: ${error.message}`, error.code),
      "error"
    );
  } finally {
    state.busy = false;
    applyCapabilities();
  }
}

async function deleteCurrentPage() {
  if (!canWrite() || !state.current) {
    return;
  }
  const page = state.current;
  const confirmed = window.confirm(
    `Delete "${page.title}" (${page.id})?\n\n` +
      "Links from other pages to this page will be removed."
  );
  if (!confirmed) {
    return;
  }

  state.busy = true;
  applyCapabilities();
  setStatus(`Deleting ${page.id}…`, "");
  try {
    await api(`/api/page?id=${encodeURIComponent(page.id)}`, {
      method: "DELETE",
    });
    markDirty();
    state.current = null;
    state.editorMode = null;
    els.search.value = "";
    await loadPages({ silent: true });
    setStatus(`Deleted ${page.id}. Unsaved changes.`, "warn");
  } catch (error) {
    setStatus(`Delete failed: ${error.message}`, "error");
  } finally {
    state.busy = false;
    applyCapabilities();
  }
}

async function runSearch() {
  const query = els.search.value.trim();
  const sequence = ++state.searchSeq;
  if (!query) {
    renderPageList(state.pages, "Pages");
    return;
  }
  try {
    const data = await api(
      `/api/search?q=${encodeURIComponent(query)}&top_k=20`
    );
    if (sequence !== state.searchSeq) {
      return;
    }
    const results = (data.results || []).map((row) => ({
      ...row,
      id: row.page_id,
    }));
    renderPageList(results, `Results for “${query}”`);
  } catch (error) {
    if (sequence === state.searchSeq) {
      setStatus(`Search failed: ${error.message}`, "error");
    }
  }
}

function wireEvents() {
  els.newPage.addEventListener("click", openCreate);
  els.save.addEventListener("click", saveWorkspace);
  els.setupOpen.addEventListener("click", openSetup);
  els.setupClose.addEventListener("click", closeSetup);
  els.setupForm.addEventListener("submit", saveSetup);
  els.setupTest.addEventListener("click", testSetupConnection);
  els.importSubmit.addEventListener("click", submitImport);
  els.importFile.addEventListener("change", applyCapabilities);
  els.editPage.addEventListener("click", openEdit);
  els.deletePage.addEventListener("click", deleteCurrentPage);
  els.editor.addEventListener("submit", submitEditor);
  els.editorCancel.addEventListener("click", closeEditor);
  els.askForm.addEventListener("submit", submitAsk);

  els.pageList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-page-id]");
    if (!button) {
      return;
    }
    if (state.editorMode !== null) {
      if (!window.confirm("Discard the current editor changes?")) {
        return;
      }
      closeEditor();
    }
    loadPage(button.dataset.pageId);
  });

  for (const container of [els.outbound, els.inbound, els.askEvidence]) {
    container.addEventListener("click", (event) => {
      const button = event.target.closest("[data-page-id]");
      if (!button) {
        return;
      }
      if (state.editorMode !== null) {
        if (!window.confirm("Discard the current editor changes?")) {
          return;
        }
        closeEditor();
      }
      loadPage(button.dataset.pageId);
    });
  }

  els.search.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(runSearch, 180);
  });

  window.addEventListener("beforeunload", (event) => {
    if (state.dirty || state.editorMode !== null) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
}

async function init() {
  wireEvents();
  await loadCapabilities();
  await loadPages();
  await loadModelConfig();
  maybeOpenSetup();
}

init();
