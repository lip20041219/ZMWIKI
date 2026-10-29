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
  view: "overview",
  lastSaved: null,
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
  askRead: document.getElementById("ask-read"),
  libraryGrid: document.getElementById("library-grid"),
  searchResults: document.getElementById("search-results"),
  searchSummary: document.getElementById("search-summary"),
  linksEmpty: document.getElementById("links-empty"),
  overviewPages: document.getElementById("overview-pages"),
  overviewModel: document.getElementById("overview-model"),
  overviewSave: document.getElementById("overview-save"),
  inspectorModel: document.getElementById("inspector-model"),
  inspectorModelHint: document.getElementById("inspector-model-hint"),
  inspectorPages: document.getElementById("inspector-pages"),
  inspectorSave: document.getElementById("inspector-save"),
  inspectorSaveBtn: document.getElementById("inspector-save-btn"),
  workspaceSave: document.getElementById("workspace-save"),
  workspaceState: document.getElementById("workspace-state"),
  workspacePages: document.getElementById("workspace-pages"),
  workspaceSaved: document.getElementById("workspace-saved"),
  quickModel: document.getElementById("quick-model"),
  quickImport: document.getElementById("quick-import"),
  quickAsk: document.getElementById("quick-ask"),
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
  els.save.textContent = state.dirty ? "保存 •" : "保存";
  els.save.classList.toggle("dirty", state.dirty);
  for (const button of [els.inspectorSaveBtn, els.workspaceSave]) {
    if (button) {
      button.disabled = els.save.disabled;
      button.classList.toggle("dirty", state.dirty);
    }
  }
  updateStatusPanels();
}

function setAccessBadge() {
  if (state.readOnly) {
    els.accessBadge.hidden = false;
    els.accessBadge.textContent = "只读";
    els.accessBadge.className = "badge readonly";
  } else if (!state.workspaceConfigured) {
    els.accessBadge.hidden = false;
    els.accessBadge.textContent = "未配置工作区";
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
    ? "选择 UTF-8 .txt 文件后点击开始导入。"
    : "请先在「模型设置」中配置模型。";
  setAccessBadge();
  updateSaveButton();
  applyAskState();
  renderServerStatus();
  updateStatusPanels();
}

function applyAskState() {
  const disabled = state.asking || !state.agentConfigured;
  els.askSubmit.disabled = disabled;
  els.askQuestion.disabled = state.asking;
  els.askSubmit.textContent = state.asking ? "提问中…" : "提问";
  els.askHint.textContent = state.agentConfigured
    ? "多步检索、阅读与链接跟踪。"
    : "请先在「模型设置」中配置模型。";
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
        : `请求失败（HTTP ${response.status}）`;
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
  state.lastSaved = nowText();
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
    setStatus(`无法获取服务能力：${friendlyError(error)}`, "error");
  }
}

function nowText() {
  return new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

const MODEL_SOURCE = {
  cli: "命令行参数",
  settings: "已保存的设置",
  ui: "界面配置",
  none: "未配置",
  unknown: "未知",
};

function updateStatusPanels() {
  const count = state.pages.length;
  const configured = Boolean(state.model && state.model.configured);
  const source = state.model ? MODEL_SOURCE[state.model.source] || state.model.source : "未知";
  const savedText = state.dirty
    ? "有未保存的修改"
    : state.lastSaved
      ? `已保存 ${state.lastSaved}`
      : "尚未保存";
  const workspaceText = state.readOnly
    ? "只读"
    : state.workspaceConfigured
      ? "可写"
      : "未配置";

  if (els.overviewPages) els.overviewPages.textContent = String(count);
  if (els.overviewModel) els.overviewModel.textContent = configured ? "已配置" : "未配置";
  if (els.overviewSave) {
    els.overviewSave.textContent = state.lastSaved ? state.lastSaved : "尚未保存";
  }
  if (els.inspectorModel) {
    els.inspectorModel.textContent = configured ? `已配置（${source}）` : "未配置";
  }
  if (els.inspectorModelHint) {
    els.inspectorModelHint.textContent = configured
      ? "可以导入文档并提问。"
      : "配置模型后才能导入文档和提问。";
  }
  if (els.inspectorPages) els.inspectorPages.textContent = String(count);
  if (els.inspectorSave) els.inspectorSave.textContent = savedText;
  if (els.workspaceState) els.workspaceState.textContent = workspaceText;
  if (els.workspacePages) els.workspacePages.textContent = String(count);
  if (els.workspaceSaved) els.workspaceSaved.textContent = savedText;
  if (els.quickModel) els.quickModel.classList.toggle("done", configured);
  if (els.quickImport) els.quickImport.classList.toggle("done", count > 0);
  if (els.quickAsk) {
    els.quickAsk.classList.toggle("done", count > 0 && configured);
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
    ? "只读"
    : state.workspaceConfigured
      ? "可写"
      : "未配置";
  const source = model ? MODEL_SOURCE[model.source] || model.source : "未知";
  const modelText =
    model && model.configured
      ? `已配置（${source}）`
      : "未配置";
  els.setupServerStatus.textContent =
    `服务端在线 · 工作区：${workspace} · 模型：${modelText}`;
}

function updateSetupButton() {
  const configured = Boolean(state.model && state.model.configured);
  els.setupOpen.textContent = configured ? "模型设置" : "配置模型";
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
    ? "留空表示继续使用已保存的 Key"
    : "本地服务可留空";
  els.setupClearKey.checked = false;
  els.setupClearWrap.hidden = !model.stored_api_key;
  els.setupKeyHint.textContent = model.key_from_environment
    ? "服务端环境变量（ZMWIKI_API_KEY）中已提供 Key。"
    : model.stored_api_key
      ? "服务端已保存 Key，不会回传到本页面。"
      : "Key 只发送给本机服务端并保存在本机，不写入浏览器。";
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
  updateStatusPanels();
}

async function testSetupConnection() {
  if (state.setupBusy) {
    return;
  }
  const payload = setupPayload();
  if (!payload.base_url || !payload.model) {
    setSetupStatus("请先填写 API 地址和模型名称。", "error");
    return;
  }
  setSetupBusy(true);
  setSetupStatus("正在测试连接…", "");
  try {
    const result = await jsonRequest("/api/config/test", "POST", payload);
    setSetupStatus(
      TEST_MESSAGES[result.code] || "连接测试失败，请检查配置后重试。",
      result.ok ? "ok" : "error"
    );
  } catch (error) {
    setSetupStatus(`测试连接失败：${error.message}`, "error");
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
    setSetupStatus("请先填写 API 地址和模型名称。", "error");
    return;
  }
  setSetupBusy(true);
  setSetupStatus("正在保存模型配置…", "");
  try {
    state.model = await jsonRequest("/api/config", "POST", payload);
    els.setupApiKey.value = "";
    await loadCapabilities();
    updateSetupButton();
    closeSetup();
    if (state.pages.length === 0) {
      switchView("import");
      setStatus("模型已保存。下一步：导入一份 TXT 文档。", "ok");
    } else {
      setStatus("模型设置已保存。", "ok");
    }
  } catch (error) {
    setSetupStatus(`保存失败：${friendlyError(error)}`, "error");
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
  model_not_configured: "请先在「模型设置」中配置模型",
  compile_failed: "请打开「模型设置」并测试连接",
  config_save_failed: "请检查设置文件是否可写",
  save_failed: "请检查磁盘空间和工作区文件权限",
  invalid_encoding: "请将文件保存为 UTF-8 纯文本",
  request_too_large: "单个文件不能超过 2 MiB",
  page_exists: "请重命名文件，或先删除冲突页面",
  path_conflict: "请换一个路径，或删除占用该路径的页面",
  invalid_filename: "请使用纯 .txt 文件名",
  invalid_config: "请检查 API 地址、模型名称和 API Key",
  read_only: "请使用可写工作区，或用命令行参数启动服务",
};

const SERVER_MESSAGES = {
  not_found: "没有找到对应页面。",
  invalid_page: "页面数据不完整或格式不正确。",
  invalid_links: "链接目标不存在，或页面链接到了自己。",
  page_exists: "页面已存在，未覆盖原有内容。",
  path_conflict: "该路径已被其他页面占用。",
  read_only: "当前工作区是只读的。",
  workspace_not_configured: "服务端没有配置工作区文件。",
  save_failed: "工作区写入失败。",
  model_not_configured: "尚未配置模型。",
  compile_failed: "文档编译失败。",
  invalid_filename: "文件名不合法。",
  unsupported_media_type: "请求格式不正确。",
  invalid_encoding: "文件不是有效的 UTF-8 编码。",
  empty_document: "文件内容为空。",
  request_too_large: "文件超过大小限制。",
  invalid_config: "模型配置不完整或格式不正确。",
  config_save_failed: "模型配置写入失败。",
  invalid_question: "问题不能为空，且不超过 4000 字。",
  invalid_query: "搜索关键词不能为空，且不超过 200 字。",
  invalid_top_k: "搜索结果数量参数不合法。",
  invalid_id: "页面 ID 不合法。",
  invalid_json: "请求格式不正确。",
  invalid_body: "请求内容不完整。",
  agent_failed: "Agent 执行失败。",
  internal_error: "服务端内部错误。",
  forbidden_request: "请求被拒绝。",
};

function friendlyError(error) {
  const base = SERVER_MESSAGES[error.code] || "请求失败，请重试。";
  const hint = error.code ? ERROR_HINTS[error.code] : "";
  return hint ? `${base}${hint}。` : base;
}

const TEST_MESSAGES = {
  ok: "连接成功，接口返回了可用的聊天响应。",
  auth_failed: "鉴权失败：请检查 API Key 是否正确、是否有权限使用该模型。",
  endpoint_not_found: "找不到接口：请检查 API 地址，通常以 /v1 结尾。",
  bad_request: "接口拒绝了请求：请检查模型名称是否准确。",
  rate_limited: "请求被限流：请稍后重试，或检查账户额度。",
  service_error: "模型服务返回错误：请稍后重试。",
  connection_failed: "无法连接模型服务：请检查 API 地址、网络和代理设置。",
  invalid_response: "接口响应不是 OpenAI 兼容的聊天格式：请检查 API 地址。",
  invalid_config: "配置不完整：请检查 API 地址、模型名称和 API Key。",
  failed: "连接测试失败：请检查配置后重试。",
};

const ASK_ERROR_TEXT = {
  insufficient_evidence:
    "Wiki 中没有足够的已读证据来回答。请导入相关文档，或换一个更具体的问题。",
  budget_exhausted: "Agent 用完了步骤预算仍未回答。请缩小问题范围。",
  model_error: "模型调用失败。请打开「模型设置」并测试连接。",
  protocol_error: "模型返回了无法解析的动作。请尝试更稳定的模型。",
};

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
    ["页面 ID", page.id],
    ["路径", page.path],
    ["别名", (page.aliases || []).join("、") || "—"],
    ["标签", (page.tags || []).join("、") || "—"],
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
      missing.textContent = `${pageId}（缺失）`;
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
    empty.textContent = "暂无来源文档。";
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
  answered: { label: "已回答", tone: "ok" },
  insufficient_evidence: { label: "证据不足", tone: "warn" },
  budget_exhausted: { label: "步骤预算耗尽", tone: "error" },
  model_error: { label: "模型调用失败", tone: "error" },
  protocol_error: { label: "模型输出格式错误", tone: "error" },
};

function formatStepParams(step) {
  const params = step.params || {};
  if (step.action === "search" && params.query) {
    return `查询：${params.query}`;
  }
  if (step.action === "read" && Array.isArray(params.paths)) {
    return `页面：${params.paths.join("、")}`;
  }
  if (step.action === "follow_links" && params.from_page) {
    return `起点：${params.from_page}`;
  }
  if (step.action === "answer" && Array.isArray(params.sources)) {
    return `引用：${params.sources.length} 个来源`;
  }
  return "";
}

const STEP_ACTION_LABELS = {
  search: "搜索",
  read: "阅读",
  follow_links: "跟踪链接",
  answer: "回答",
  model_error: "模型失败",
  protocol_error: "协议错误",
  duplicate: "重复动作",
};

function stepSummary(step) {
  const hits = Array.isArray(step.search_hit_page_ids)
    ? step.search_hit_page_ids.length
    : 0;
  const reads = Array.isArray(step.read_page_ids)
    ? step.read_page_ids.length
    : 0;
  if (step.action === "search") return `搜索返回 ${hits} 个候选页面。`;
  if (step.action === "read") return `读取了 ${reads} 个页面。`;
  if (step.action === "follow_links") return `沿链接读取了 ${reads} 个页面。`;
  if (step.action === "answer") return "根据已读页面生成回答。";
  if (step.action === "model_error") return "模型调用失败。";
  if (step.action === "protocol_error") return "模型输出无法解析。";
  if (step.action === "duplicate") return "重复动作已跳过。";
  return "已执行一个步骤。";
}

function renderAskSteps(steps) {
  els.askSteps.replaceChildren();
  if (!steps.length) {
    const empty = document.createElement("li");
    empty.className = "step-empty";
    empty.textContent = "没有记录到步骤。";
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
    action.textContent = STEP_ACTION_LABELS[step.action] || step.action || "未知动作";
    head.append(number, action);
    item.append(head);

    const summary = document.createElement("p");
    summary.className = "step-summary";
    summary.textContent = stepSummary(step);
    const extra = formatStepParams(step);
    if (extra) {
      summary.textContent += ` · ${extra}`;
    }
    item.append(summary);

    if (step.error) {
      const error = document.createElement("p");
      error.className = "step-error";
      error.textContent = `错误：${step.error}`;
      item.append(error);
    }
    if (Array.isArray(step.search_hit_page_ids) && step.search_hit_page_ids.length) {
      const refs = document.createElement("p");
      refs.className = "step-refs";
      refs.textContent = `搜索命中：${step.search_hit_page_ids.join("、")}`;
      item.append(refs);
    }
    if (Array.isArray(step.read_page_ids) && step.read_page_ids.length) {
      const refs = document.createElement("p");
      refs.className = "step-refs";
      refs.textContent = `已读：${step.read_page_ids.join("、")}`;
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
    empty.textContent = "没有引用页面。";
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
    row.textContent = item.title || item.path || "未知页面";
    const meta = document.createElement("span");
    meta.className = "link-id";
    const parts = [item.path || ""];
    if (Array.isArray(item.sources) && item.sources.length) {
      parts.push(`来源：${item.sources.join("、")}`);
    }
    meta.textContent = parts.filter(Boolean).join(" · ");
    row.append(meta);
    els.askEvidence.append(row);
  }
}

function renderAskResult(data) {
  els.askResult.hidden = false;
  els.askAnswer.textContent =
    data.answer || ASK_ERROR_TEXT[data.status] || "没有生成回答。";
  renderAskSteps(Array.isArray(data.steps) ? data.steps : []);
  renderAskEvidence(Array.isArray(data.evidence) ? data.evidence : []);
  const readIds = Array.isArray(data.read_page_ids) ? data.read_page_ids : [];
  els.askRead.textContent = readIds.length ? readIds.join("、") : "—";
  const info = ASK_STATUS[data.status] || {
    label: data.status || "未知状态",
    tone: "error",
  };
  const stepCount = Array.isArray(data.steps) ? data.steps.length : 0;
  const pagesRead = Number.isInteger(data.pages_read) ? data.pages_read : 0;
  setAskStatus(
    `${info.label} · ${stepCount} 步 · 已读 ${pagesRead} 页`,
    info.tone
  );
}

async function submitAsk(event) {
  event.preventDefault();
  if (state.asking) {
    return;
  }
  if (!state.agentConfigured) {
    setAskStatus("还没有配置模型。请先在「模型设置」中完成配置。", "error");
    return;
  }
  const question = els.askQuestion.value.trim();
  if (!question) {
    setAskStatus("请先输入一个问题。", "error");
    return;
  }

  state.asking = true;
  applyCapabilities();
  els.askResult.hidden = true;
  els.askAnswer.textContent = "";
  els.askSteps.replaceChildren();
  els.askEvidence.replaceChildren();
  els.askRead.textContent = "—";
  setAskStatus("Agent 正在检索并阅读页面…", "");

  try {
    const data = await jsonRequest("/api/ask", "POST", { question });
    renderAskResult(data);
  } catch (error) {
    setAskStatus(`提问失败：${friendlyError(error)}`, "error");
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
  els.linksEmpty.hidden = false;
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
  els.linksEmpty.hidden = true;
  updateActiveListItem(page.id);

  els.pageTitle.textContent = page.title || "";
  els.pagePath.textContent = page.path || "";
  els.pageDescription.textContent = page.description || "";
  els.pageDescription.hidden = !page.description;
  els.pageContent.textContent = page.content || "";
  renderTags(page.tags || []);
  renderMeta(page);
  renderLinks(els.outbound, page.links_to || [], "暂无出站链接。");
  renderLinks(els.inbound, page.links_from || [], "暂无反向链接。");
  renderSources(page.sources || []);
  updateStatusPanels();
  applyCapabilities();
}

function renderPageList(pages) {
  els.pageList.replaceChildren();
  els.listTitle.textContent = "全部页面";
  els.listCount.textContent = String(pages.length);
  if (!pages.length) {
    const empty = document.createElement("p");
    empty.className = "list-empty";
    empty.textContent = "知识库里还没有页面。导入一份 TXT 文档开始吧。";
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
      setStatus(`已打开「${page.title}」`, "");
    }
  } catch (error) {
    setStatus(`无法打开页面：${friendlyError(error)}`, "error");
  }
}

async function loadPages({ selectId = null, silent = false } = {}) {
  try {
    const data = await api("/api/pages");
    state.pages = Array.isArray(data.pages) ? data.pages : [];
    renderPageList(state.pages);
    renderLibrary();
    updateStatusPanels();
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
          setStatus(`已打开「${current.title}」`, "");
        }
      }
    } else {
      state.current = null;
      showEmpty();
      if (!silent) {
        setStatus(
          state.readOnly
            ? "只读工作区里还没有页面。"
            : "工作区还是空的，导入文档或新建页面开始吧。",
          state.readOnly ? "warn" : ""
        );
      }
    }
    updateStatusPanels();
    applyCapabilities();
  } catch (error) {
    setStatus(`无法加载页面列表：${friendlyError(error)}`, "error");
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
  switchView("pages");
  state.editorMode = "create";
  els.empty.hidden = true;
  els.reader.hidden = true;
  els.details.hidden = false;
  els.pageDetails.hidden = true;
  els.linksEmpty.hidden = false;
  els.editor.hidden = false;
  els.editorTitle.textContent = "新建页面";
  els.editorHint.textContent = "页面 ID 和路径必须唯一；出站链接填写已存在的页面 ID。";
  els.editId.disabled = false;
  fillEditor(null);
  setStatus("新建页面 — 填写字段后点击「应用修改」。", "");
  applyCapabilities();
  els.editId.focus();
}

function openEdit() {
  if (!canWrite() || !state.current || state.editorMode !== null) {
    return;
  }
  const page = state.current;
  switchView("pages");
  state.editorMode = "edit";
  els.empty.hidden = true;
  els.reader.hidden = true;
  els.editor.hidden = false;
  els.details.hidden = false;
  els.pageDetails.hidden = false;
  els.linksEmpty.hidden = true;
  els.editorTitle.textContent = `编辑「${page.title}」`;
  els.editorHint.textContent = "编辑时页面 ID 固定；出站链接填写已存在的页面 ID。";
  els.editId.disabled = true;
  fillEditor(page);
  setStatus(`正在编辑 ${page.id}`, "");
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
  setStatus("已取消编辑。", "");
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
    setStatus("页面 ID、路径、标题和正文为必填项。", "error");
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
  setStatus(mode === "edit" ? "正在更新页面…" : "正在创建页面…", "");

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
    switchView("pages");
    await loadPages({ selectId: targetId, silent: true });
    setStatus(
      `${mode === "edit" ? "已更新" : "已创建"} ${targetId}，有未保存的修改。`,
      "warn"
    );
  } catch (error) {
    setStatus(`应用修改失败：${friendlyError(error)}`, "error");
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
  setStatus("正在保存工作区…", "");
  try {
    await api("/api/workspace/save", { method: "POST" });
    markSaved();
    setStatus("工作区已保存（上一版保留为 .bak）。", "ok");
  } catch (error) {
    setStatus(`保存失败：${friendlyError(error)}`, "error");
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
    setImportStatus("请先选择一个 .txt 文件。", "error");
    return;
  }
  if (!/\.txt$/i.test(file.name)) {
    setImportStatus("只支持 UTF-8 编码的 .txt 文件。", "error");
    return;
  }
  if (state.editorMode !== null) {
    setImportStatus("请先完成或取消当前的页面编辑。", "error");
    return;
  }

  state.busy = true;
  applyCapabilities();
  setImportStatus(`正在导入 ${file.name}…`, "");
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
    switchView("pages");
    await loadPages({ selectId: target, silent: true });
    // Import saves the workspace on the server, so nothing is left dirty.
    markSaved();
    const verb = data.status === "updated" ? "已更新" : "已导入";
    setImportStatus(
      `${verb} ${file.name} → 页面 ${target || "—"} · ` +
        `解析 ${links.resolved || 0} 条链接 · 工作区已保存`,
      "ok"
    );
  } catch (error) {
    setImportStatus(`导入失败：${friendlyError(error)}`, "error");
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
    `确定删除「${page.title}」（${page.id}）吗？\n\n` +
      "其他页面指向它的链接也会被移除。"
  );
  if (!confirmed) {
    return;
  }

  state.busy = true;
  applyCapabilities();
  setStatus(`正在删除 ${page.id}…`, "");
  try {
    await api(`/api/page?id=${encodeURIComponent(page.id)}`, {
      method: "DELETE",
    });
    markDirty();
    state.current = null;
    state.editorMode = null;
    await loadPages({ silent: true });
    setStatus(`已删除 ${page.id}，有未保存的修改。`, "warn");
  } catch (error) {
    setStatus(`删除失败：${friendlyError(error)}`, "error");
  } finally {
    state.busy = false;
    applyCapabilities();
  }
}

async function runSearch() {
  const query = els.search.value.trim();
  const sequence = ++state.searchSeq;
  if (!query) {
    els.searchResults.replaceChildren();
    els.searchSummary.textContent = "输入关键词开始搜索。";
    return;
  }
  els.searchSummary.textContent = "正在搜索…";
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
    els.searchSummary.textContent = results.length
      ? `“${query}” 找到 ${results.length} 个页面，点击卡片打开。`
      : `没有找到与“${query}”匹配的页面。换一个关键词，或先导入相关文档。`;
    renderPageCards(els.searchResults, results);
  } catch (error) {
    if (sequence === state.searchSeq) {
      els.searchSummary.textContent = `搜索失败：${friendlyError(error)}`;
      els.searchResults.replaceChildren();
    }
  }
}

function renderPageCards(container, pages, emptyText = "") {
  container.replaceChildren();
  if (!pages.length) {
    if (emptyText) {
      const empty = document.createElement("p");
      empty.className = "list-empty";
      empty.textContent = emptyText;
      container.append(empty);
    }
    return;
  }
  for (const page of pages) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "page-card";
    card.dataset.pageId = page.id;

    const title = document.createElement("span");
    title.className = "page-card-title";
    title.textContent = page.title || page.id;

    const path = document.createElement("span");
    path.className = "page-card-path";
    path.textContent = page.path || "";

    card.append(title, path);
    if (page.tags && page.tags.length) {
      const tags = document.createElement("span");
      tags.className = "page-card-tags";
      tags.textContent = page.tags.join(" · ");
      card.append(tags);
    }
    const id = document.createElement("span");
    id.className = "page-card-id";
    id.textContent = page.id;
    card.append(id);
    container.append(card);
  }
}

function renderLibrary() {
  renderPageCards(
    els.libraryGrid,
    state.pages,
    "知识库还是空的。导入一份 TXT 文档开始吧。"
  );
}

function switchView(name) {
  if (name !== "pages" && state.editorMode !== null) {
    if (!window.confirm("放弃当前未应用的编辑？")) {
      return;
    }
    closeEditor();
  }
  state.view = name;
  for (const panel of document.querySelectorAll("[data-view-panel]")) {
    panel.hidden = panel.dataset.viewPanel !== name;
  }
  for (const button of document.querySelectorAll(".nav-item[data-view]")) {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    if (active) {
      button.setAttribute("aria-current", "page");
    } else {
      button.removeAttribute("aria-current");
    }
  }
  if (name === "search" && els.search) {
    window.setTimeout(() => els.search.focus(), 0);
  }
}

function openPageById(pageId) {
  if (state.editorMode !== null) {
    if (!window.confirm("放弃当前未应用的编辑？")) {
      return;
    }
    closeEditor();
  }
  switchView("pages");
  loadPage(pageId);
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
  if (els.inspectorSaveBtn) {
    els.inspectorSaveBtn.addEventListener("click", saveWorkspace);
  }
  if (els.workspaceSave) {
    els.workspaceSave.addEventListener("click", saveWorkspace);
  }

  for (const button of document.querySelectorAll("[data-view]")) {
    button.addEventListener("click", () => switchView(button.dataset.view));
  }
  for (const button of document.querySelectorAll('[data-action="open-setup"]')) {
    button.addEventListener("click", openSetup);
  }
  for (const button of document.querySelectorAll('[data-action="create"]')) {
    button.addEventListener("click", openCreate);
  }
  for (const chip of document.querySelectorAll(".chip[data-question]")) {
    chip.addEventListener("click", () => {
      els.askQuestion.value = chip.dataset.question;
      els.askQuestion.focus();
    });
  }

  els.pageList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-page-id]");
    if (!button) {
      return;
    }
    openPageById(button.dataset.pageId);
  });

  for (const container of [els.libraryGrid, els.searchResults]) {
    container.addEventListener("click", (event) => {
      const card = event.target.closest("[data-page-id]");
      if (!card) {
        return;
      }
      openPageById(card.dataset.pageId);
    });
  }

  for (const container of [els.outbound, els.inbound, els.askEvidence]) {
    container.addEventListener("click", (event) => {
      const button = event.target.closest("[data-page-id]");
      if (!button) {
        return;
      }
      openPageById(button.dataset.pageId);
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
  switchView("overview");
  await loadCapabilities();
  await loadPages();
  await loadModelConfig();
  maybeOpenSetup();
}

init();
