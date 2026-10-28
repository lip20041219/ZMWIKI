"""Retro terminal UI for ZMWiki.

The TUI talks directly to the in-memory WikiIndex. Local document imports
compile and persist a page (re-importing the same source updates it); other
page edits are written to the workspace file explicitly with ``s``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from zmwiki.file_ingestion import PlainTextFileIngestor
from zmwiki.web.workspace import (
    SourceRegistry,
    WorkspaceError,
    load_workspace_state,
    save_workspace,
    source_key,
)
from zmwiki.wiki.wikipedia import (
    WikipediaArticle,
    WikipediaConnector,
    WikipediaError,
    WikipediaLink,
    WikipediaSearchResult,
)
from zmwiki.wiki.wikipedia_import import WikipediaImportError, import_wikipedia_article
from zmwiki.wiki.wiki_agent import AgentResult, AgentStatus, WikiAgent
from zmwiki.wiki.wiki_compiler import CompiledPage, WikiCompiler, WikiCompilerError
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_link_resolver import resolve_candidate_links
from zmwiki.wiki.wiki_retriever import WikiSearchResult, wiki_search
from zmwiki.wiki.wiki_schema import WikiPage

__all__ = ["ZMWikiApp", "main"]

# 仓库根目录下的示例工作区，按包位置定位，避免依赖当前工作目录。
# ponytail: 假定 editable 安装；若改为 wheel 安装，默认路径不存在会报错。
_DEFAULT_WORKSPACE = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "corpus"
    / "wiki_workspace.json"
)

BANNER = r"""
███████╗███╗   ███╗██╗    ██╗██╗██╗  ██╗██╗
╚══███╔╝████╗ ████║██║    ██║██║██║ ██╔╝██║
  ███╔╝ ██╔████╔██║██║ █╗ ██║██║█████╔╝ ██║
 ███╔╝  ██║╚██╔╝██║██║███╗██║██║██╔═██╗ ██║
███████╗██║ ╚═╝ ██║╚███╔███╔╝██║██║  ██╗██║
╚══════╝╚═╝     ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═╝╚═╝
""".strip("\n")

FOOTER_KEYS = (
    "↑↓ 移动 · ENTER 打开 · 1-9 跳转 · / 搜索 · N 新建 · E 编辑 · "
    "D 删除 · W 维基 · L 本地 · A 提问 · I 导入 · S 保存 · ESC 返回 · Q 退出"
)

CSS = """
Screen {
    background: #000000;
    color: #c9ffda;
}

#app-grid {
    height: 100%;
    padding: 0 1;
}

#topbar {
    height: 8;
    border: heavy #00ff88;
    background: #020905;
    padding: 0 2;
    layout: horizontal;
}

#brand {
    width: 1fr;
    color: #39ff88;
    text-style: bold;
}

#topmeta {
    width: auto;
    height: auto;
    color: #4fbf7f;
    text-align: right;
    padding-top: 1;
}

#main {
    height: 1fr;
    margin-top: 1;
}

#left {
    width: 26%;
    border: round #0a7a45;
    padding: 0 1;
}

#center {
    width: 1fr;
    border: round #0a7a45;
    margin: 0 1;
    padding: 0 2;
}

#right {
    width: 28%;
    border: round #0a7a45;
    padding: 0 1;
}

#search-label, #list-label, #wiki-label, #ask-label, #doc-label, #meta-label, #outbound-label, #inbound-label {
    color: #3fbf77;
    text-style: bold;
    height: auto;
}

#search-label, #list-label, #wiki-label, #ask-label, #doc-label {
    margin-top: 1;
}

#meta-label, #outbound-label, #inbound-label {
    margin-top: 1;
}

Input {
    height: 3;
    background: #001008;
    border: tall #0a7a45;
    color: #d8ffe4;
}

Input:focus {
    border: tall #00ff88;
}

#pages, #outbound, #inbound {
    height: 1fr;
    background: #000000;
    border: none;
}

OptionList > .option-list--option {
    color: #a8e6bf;
    padding: 0 1;
}

OptionList > .option-list--option-highlighted {
    background: #00ff88;
    color: #001008;
    text-style: bold;
}

OptionList > .option-list--option-disabled {
    color: #3f6b53;
}

#page-title {
    height: auto;
    color: #00ff88;
    text-style: bold;
}

#page-path {
    height: auto;
    color: #4f9f70;
    margin-bottom: 1;
}

#reader-scroll {
    height: 1fr;
}

#reader {
    width: 100%;
    color: #d8ffe4;
}

#page-meta {
    height: auto;
    color: #8fdcae;
}

#statusbar {
    height: 3;
    border-top: solid #0a7a45;
    color: #4f9f70;
    padding: 0 1;
}
"""


_AGENT_STATUS_LABELS = {
    AgentStatus.ANSWERED: "已回答",
    AgentStatus.INSUFFICIENT_EVIDENCE: "证据不足",
    AgentStatus.BUDGET_EXHAUSTED: "步骤预算耗尽",
    AgentStatus.MODEL_ERROR: "模型错误",
    AgentStatus.PROTOCOL_ERROR: "协议错误",
}


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_WORKSPACE_ERROR_HINTS = {
    "workspace target directory does not exist":
        "目标目录不存在，请检查工作区路径",
    "workspace file cannot be written":
        "无法写入文件，请检查权限或磁盘空间",
}


def _workspace_error_text(exc: WorkspaceError) -> str:
    """Short, actionable Chinese text for a WorkspaceError."""
    message = str(exc)
    hint = _WORKSPACE_ERROR_HINTS.get(message)
    if hint is not None:
        return hint
    if message.startswith("workspace serialization failed"):
        return "工作区序列化失败，页面数据无法写入"
    return message


def _exception_text(exc: Exception, limit: int = 100) -> str:
    """Type + message without a traceback, for user-facing status lines."""
    if isinstance(exc, WorkspaceError):
        return _workspace_error_text(exc)
    return f"{type(exc).__name__}: {_clip(exc, limit)}"


def _answer_body(
    result: AgentResult, label: str, sources: list[WikiPage]
) -> str:
    if result.status == AgentStatus.ANSWERED:
        body = result.answer or "（空回答）"
    else:
        body = f"[{label}]\n\n{result.error or '未生成回答'}"
        if result.answer:
            body += f"\n\n模型备注（未经验证）：\n{result.answer}"
    if sources:
        body += "\n\n── 来源 ──\n" + "\n".join(
            f"- {page.title} (id={page.id})" for page in sources
        )
    else:
        body += "\n\n── 来源 ──\n 无经核实的来源"
    return body


class PageEditorScreen(ModalScreen[WikiPage | None]):
    """Modal editor for creating or updating one local WikiPage."""

    CSS = """
    PageEditorScreen {
        align: center middle;
    }
    #editor-dialog {
        width: 86%;
        height: 88%;
        border: heavy #00ff88;
        background: #001008;
        padding: 1 2;
    }
    #editor-title {
        color: #00ff88;
        text-style: bold;
        height: auto;
        margin-bottom: 1;
    }
    #editor-dialog Input {
        margin-bottom: 1;
    }
    #editor-content {
        height: 1fr;
        border: tall #0a7a45;
        background: #000000;
    }
    #editor-content:focus {
        border: tall #00ff88;
    }
    #editor-error {
        color: #ff6b6b;
        height: auto;
        margin-top: 1;
    }
    #editor-help {
        color: #4f9f70;
        height: auto;
    }
    """

    BINDINGS = [
        ("ctrl+s", "save", "保存"),
        ("escape", "cancel", "取消"),
    ]

    def __init__(
        self,
        page: WikiPage | None,
        validate: Callable[[WikiPage], str | None],
    ) -> None:
        super().__init__()
        self._original = page
        self._validate = validate

    def compose(self) -> ComposeResult:
        editing = self._original is not None
        with Vertical(id="editor-dialog"):
            yield Static(
                f"编辑页面 // {self._original.id}" if editing else "新建页面",
                id="editor-title",
            )
            yield Input(
                value=self._original.id if editing else "",
                placeholder="页面 ID（唯一，不能包含换行）",
                id="edit-id",
                disabled=editing,
            )
            yield Input(
                value=self._original.path if editing else "",
                placeholder="路径，例如 people/alice",
                id="edit-path",
            )
            yield Input(
                value=self._original.title if editing else "",
                placeholder="标题",
                id="edit-title",
            )
            yield Input(
                value=", ".join(self._original.tags) if editing else "",
                placeholder="标签（逗号分隔，可选）",
                id="edit-tags",
            )
            yield Input(
                value=", ".join(self._original.links_to) if editing else "",
                placeholder="出链 page ID（逗号分隔，可选）",
                id="edit-links",
            )
            yield TextArea(
                self._original.content if editing else "",
                id="edit-content",
                placeholder="页面正文…",
            )
            yield Static("", id="editor-error")
            yield Static("Ctrl+S 保存 · ESC 取消", id="editor-help")

    def on_mount(self) -> None:
        target = "#edit-title" if self._original is not None else "#edit-id"
        self.query_one(target, Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_save(self) -> None:
        page_id = self.query_one("#edit-id", Input).value.strip()
        path = self.query_one("#edit-path", Input).value.strip()
        title = self.query_one("#edit-title", Input).value.strip()
        content = self.query_one("#edit-content", TextArea).text.strip()
        for value, label in (
            (page_id, "页面 ID"),
            (path, "路径"),
            (title, "标题"),
            (content, "正文"),
        ):
            if not value:
                self.query_one("#editor-error", Static).update(
                    f"{label} 不能为空"
                )
                return
        raw_tags = self.query_one("#edit-tags", Input).value.replace("，", ",")
        tags = tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip())
        raw_links = self.query_one("#edit-links", Input).value.replace("，", ",")
        links_to = tuple(dict.fromkeys(
            link.strip() for link in raw_links.split(",") if link.strip()
        ))
        try:
            if self._original is None:
                page = WikiPage(
                    id=page_id,
                    path=path,
                    title=title,
                    content=content,
                    tags=tags,
                    links_to=links_to,
                )
            else:
                page = replace(
                    self._original,
                    path=path,
                    title=title,
                    content=content,
                    tags=tags,
                    links_to=links_to,
                )
        except (TypeError, ValueError) as exc:
            self.query_one("#editor-error", Static).update(f"无法保存：{exc}")
            return
        except Exception as exc:  # keep tracebacks out of the editor
            self.query_one("#editor-error", Static).update(
                f"无法保存：{_exception_text(exc, 80)}"
            )
            return
        error = self._validate(page)
        if error is not None:
            self.query_one("#editor-error", Static).update(error)
            return
        self.dismiss(page)


class ZMWikiApp(App[None]):
    """ZMWiki terminal browser with local search, page editing and import."""

    TITLE = "ZMWiki"
    SUB_TITLE = "本地知识库终端"
    CSS = CSS

    # Native bindings only fire when the focused widget does not consume the
    # key, so typing letters into Input/TextArea never triggers these actions.
    BINDINGS = [
        Binding("n,N", "new_page", "新建"),
        Binding("e,E", "edit_page", "编辑"),
        Binding("d,D", "delete_page", "删除"),
        Binding("ctrl+z", "undo_edit", "撤销"),
        Binding("a,A", "ask", "提问"),
        Binding("w,W", "wikipedia_search", "维基"),
        Binding("i,I", "import_wikipedia", "导入"),
        Binding("l,L", "import_local_document", "本地文档"),
        Binding("u,U", "sync_sources", "源同步"),
        Binding("s,S", "save_workspace", "保存"),
        Binding("ctrl+s", "save_workspace", "保存", show=False),
    ]

    def __init__(
        self,
        index: WikiIndex,
        *,
        workspace_path: str | Path | None = None,
        sources: SourceRegistry | None = None,
        wikipedia: WikipediaConnector | None = None,
        agent_factory: Callable[[WikiIndex], WikiAgent] | None = None,
        compiler_factory: Callable[[], WikiCompiler] | None = None,
    ) -> None:
        super().__init__()
        self.index = index
        self.workspace_path = (
            Path(workspace_path) if workspace_path is not None else None
        )
        # The bundled demo workspace is read-only: never write user edits or
        # imports back into the repository's fixture.
        self._is_demo_workspace = (
            self.workspace_path is not None
            and source_key(self.workspace_path) == source_key(_DEFAULT_WORKSPACE)
        )
        if sources is None and self.workspace_path is not None:
            # Callers that only pass a WikiIndex (e.g. load_workspace) must
            # not wipe the on-disk source registry when the app saves.
            try:
                sources = load_workspace_state(self.workspace_path)[1]
            except WorkspaceError:
                sources = {}
        self.sources: SourceRegistry = dict(sources or {})
        self.wikipedia = wikipedia if wikipedia is not None else WikipediaConnector()
        self.agent_factory = agent_factory
        self.compiler_factory = compiler_factory
        self.last_agent_result: AgentResult | None = None
        self.last_agent_question: str | None = None
        self._agent_panels_active = False
        self._agent_source_navigation = False
        self.pages: list[WikiPage] = []
        self.current_page: WikiPage | None = None
        self.mode = "pages"  # "pages" | "search" | "wikipedia" | "ask"
        self.search_results: dict[str, WikiSearchResult] = {}
        self.wikipedia_results: dict[str, WikipediaSearchResult] = {}
        self.preview_article: WikipediaArticle | None = None
        self.preview_links: list[WikipediaLink] | None = None
        self.preview_link_error: str | None = None
        self._overwrite_armed = False
        self._delete_armed = False
        # (page before the edit or None for a create, page stored after it)
        self._undo_edit: tuple[WikiPage | None, WikiPage] | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="app-grid"):
            with Horizontal(id="topbar"):
                yield Static(BANNER, id="brand", markup=False)
                yield Static("", id="topmeta", markup=False)
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield Static("本地搜索", id="search-label")
                    yield Input(placeholder="搜索页面…", id="search")
                    yield Static("页面列表", id="list-label")
                    yield OptionList(id="pages")
                    yield Static("维基百科", id="wiki-label")
                    yield Input(placeholder="搜索维基百科…", id="wiki-search")
                    yield Static("提问", id="ask-label")
                    yield Input(placeholder="向本地知识库提问…", id="ask")
                    yield Static("本地文档", id="doc-label")
                    yield Input(placeholder=".txt 文件路径，回车编译导入", id="doc-path")
                with Vertical(id="center"):
                    yield Static("", id="page-title")
                    yield Static("", id="page-path")
                    with VerticalScroll(id="reader-scroll"):
                        yield Static("", id="reader", markup=False)
                with Vertical(id="right"):
                    yield Static("页面信息", id="meta-label")
                    yield Static("", id="page-meta", markup=False)
                    yield Static("出链", id="outbound-label")
                    yield OptionList(id="outbound")
                    yield Static("入链", id="inbound-label")
                    yield OptionList(id="inbound")
            yield Static("", id="statusbar")

    async def on_mount(self) -> None:
        self.pages = self.index.list_pages()
        self.query_one("#topmeta", Static).update(
            "本地知识库\n"
            "本地会话\n"
            f"已索引 {len(self.pages)} 页"
        )
        if not self.pages:
            self._show_empty_workspace()
            self.query_one("#pages", OptionList).focus()
            return
        self._populate_pages(self.pages[0].id)
        self._open_page(self.pages[0].id)
        self.query_one("#pages", OptionList).focus()

    # ------------------------------------------------------------------
    # Page list, search, previews
    # ------------------------------------------------------------------

    def _populate_pages(self, highlight_id: str | None = None) -> None:
        self.mode = "pages"
        self._agent_source_navigation = False
        self._agent_panels_active = False
        self.search_results = {}
        self.wikipedia_results = {}
        self.preview_article = None
        self.preview_links = None
        self.preview_link_error = None
        self._update_topmeta()
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        self.query_one("#list-label", Static).update(f"页面 ({len(self.pages)})")
        if not self.pages:
            list_widget.add_option(Option("  空工作区", disabled=True))
            return
        for position, page in enumerate(self.pages, start=1):
            list_widget.add_option(
                Option(f"{position:02d}  {page.title}", id=page.id)
            )
        highlighted = 0
        if highlight_id is not None:
            ids = [page.id for page in self.pages]
            if highlight_id in ids:
                highlighted = ids.index(highlight_id)
        list_widget.highlighted = highlighted

    def _run_search(self, raw_query: str) -> None:
        self.preview_article = None
        self._agent_source_navigation = False
        self._set_panel_labels("出链", "入链")
        query = raw_query.strip()
        if not query:
            self._populate_pages(self.current_page.id if self.current_page else None)
            if self.current_page is not None:
                self._preview_page(self.current_page.id)
                self._set_status(f"页面列表 // {self.current_page.id}")
            return

        results = wiki_search(self.index, query, top_k=20)
        self.mode = "search"
        self.search_results = {result.page_id: result for result in results}
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        self.query_one("#list-label", Static).update(
            f"搜索结果 ({len(results)})"
        )

        if not results:
            list_widget.add_option(Option("  无匹配结果", disabled=True))
            self.query_one("#meta-label", Static).update("搜索")
            self.query_one("#page-meta", Static).update(
                f"查询     {query}\n结果     0\n\n没有页面匹配此查询。"
            )
            self._fill_links(self.query_one("#outbound", OptionList), [], "—")
            self._fill_links(self.query_one("#inbound", OptionList), [], "—")
            self._set_status(f"无匹配 // {query!r}")
            return

        for position, result in enumerate(results, start=1):
            list_widget.add_option(
                Option(
                    f"{position:02d}  {result.title}  [{result.score:.2f}]",
                    id=result.page_id,
                )
            )
        list_widget.highlighted = 0
        self._preview_search(results[0].page_id)
        self._set_status(f"搜索 // {len(results)} 条结果 // {query!r}")
        list_widget.focus()

    def _preview_page(self, page_id: str) -> None:
        page = self.index.get(page_id)
        if page is None:
            return
        self.query_one("#meta-label", Static).update("页面信息")
        self._render_page_side(page)

    def _preview_search(self, page_id: str) -> None:
        result = self.search_results.get(page_id)
        if result is None:
            return
        self._set_panel_labels("出链", "入链")
        self.query_one("#meta-label", Static).update("搜索命中")
        description = getattr(result, "description", None) or "—"
        self.query_one("#page-meta", Static).update(
            f"编号     {result.page_id}\n"
            f"路径     {result.path}\n"
            f"得分     {result.score:.4f}\n"
            f"标签     {', '.join(result.tags) or '—'}\n"
            f"描述     {description}"
        )
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")

    # ------------------------------------------------------------------
    # Wikipedia search, preview, import, save
    # ------------------------------------------------------------------

    def _start_wikipedia_search(self, raw_query: str) -> None:
        self._agent_source_navigation = False
        self._agent_panels_active = False
        query = raw_query.strip()
        if not query:
            self._set_status("维基百科 // 查询为空")
            return
        self._set_status(f"正在搜索维基百科 // {query!r}")
        self._search_wikipedia_worker(query)

    @work(thread=True)
    def _search_wikipedia_worker(self, query: str) -> None:
        try:
            results = self.wikipedia.search(query, limit=8)
        except (WikipediaError, ValueError) as exc:
            self.call_from_thread(self._show_wikipedia_search_error, str(exc))
            return
        self.call_from_thread(self._show_wikipedia_results, query, results)

    def _show_wikipedia_results(
        self, query: str, results: list[WikipediaSearchResult]
    ) -> None:
        self.mode = "wikipedia"
        self.preview_article = None
        self.wikipedia_results = {str(result.page_id): result for result in results}
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        self.query_one("#list-label", Static).update(
            f"维基百科结果 ({len(results)})"
        )
        if not results:
            list_widget.add_option(Option("  无维基百科结果", disabled=True))
            self._set_panel_labels("出链", "入链")
            self._fill_links(self.query_one("#outbound", OptionList), [], "—")
            self._fill_links(self.query_one("#inbound", OptionList), [], "—")
            self.query_one("#meta-label", Static).update("维基百科")
            self.query_one("#page-meta", Static).update(
                f"查询     {query}\n结果     0\n\n没有文章匹配此查询。"
            )
            self._set_status(f"维基百科 // 无结果 // {query!r}")
            return
        for position, result in enumerate(results, start=1):
            list_widget.add_option(
                Option(f"{position:02d}  {result.title}", id=str(result.page_id))
            )
        list_widget.highlighted = 0
        self._preview_wikipedia_hit(str(results[0].page_id))
        self._set_status(
            f"维基百科 // {len(results)} 条结果 // ENTER 阅读"
        )
        list_widget.focus()

    def _preview_wikipedia_hit(self, option_id: str) -> None:
        result = self.wikipedia_results.get(option_id)
        if result is None:
            return
        self._set_panel_labels("出链", "入链")
        self.query_one("#meta-label", Static).update("维基百科结果")
        self.query_one("#page-meta", Static).update(
            f"标题     {result.title}\n"
            f"页面编号 {result.page_id}\n"
            f"来源     {result.url}\n\n"
            f"{result.summary or '—'}\n\n"
            "按 ENTER 阅读"
        )
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")
        self._set_status(f"维基百科结果 // {result.title} // 按 ENTER 阅读")

    def _read_wikipedia_article(self, option_id: str) -> None:
        result = self.wikipedia_results.get(option_id)
        if result is None:
            return
        self._set_status(f"正在读取维基百科 // {result.title}")
        self._read_wikipedia_worker(result.page_id)

    @work(thread=True)
    def _read_wikipedia_worker(self, page_id: int) -> None:
        try:
            article = self.wikipedia.read_article(page_id=page_id)
        except (WikipediaError, ValueError) as exc:
            self.call_from_thread(self._show_wikipedia_read_error, str(exc))
            return
        links: list[WikipediaLink] | None = None
        link_error: str | None = None
        try:
            links = self.wikipedia.read_links(page_id)
        except (WikipediaError, ValueError) as exc:
            link_error = str(exc)
        self.call_from_thread(self._show_article_preview, article, links, link_error)

    def _show_article_preview(
        self,
        article: WikipediaArticle,
        links: list[WikipediaLink] | None,
        link_error: str | None,
    ) -> None:
        self.preview_article = article
        self.preview_links = links
        self.preview_link_error = link_error
        self._set_panel_labels("出链", "入链")
        self.query_one("#page-title", Static).update(article.title)
        self.query_one("#page-path", Static).update(article.url)
        self.query_one("#reader", Static).update(
            article.text.rstrip()
            + "\n\n── 预览结束 ──\n\n按 I 导入 · ESC 取消"
        )
        self.query_one("#reader-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#meta-label", Static).update("维基百科条目")
        if link_error is not None:
            link_line = f"链接抓取失败  {link_error}"
        else:
            link_line = f"内部链接  {len(links)} 个目标"
        self.query_one("#page-meta", Static).update(
            f"页面编号 {article.page_id if article.page_id is not None else '—'}\n"
            f"修订版本 {article.revision_id if article.revision_id is not None else '—'}\n"
            f"来源     {article.url}\n"
            f"{link_line}\n\n"
            "尚未导入"
        )
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")
        if link_error is not None:
            self._set_status(
                "维基百科预览 // 链接抓取失败 // 按 I 导入 · ESC 取消"
            )
        else:
            self._set_status(
                f"维基百科预览 // {len(links)} 条链接 // 按 I 导入 · ESC 取消"
            )

    def _import_preview(self) -> None:
        article = self.preview_article
        if article is None:
            self._set_status("无可导入内容 // 请先阅读一篇文章")
            return
        link_error = self.preview_link_error
        try:
            result = import_wikipedia_article(
                self.index, article, links=self.preview_links
            )
        except (WikipediaImportError, TypeError, ValueError) as exc:
            self._set_status(f"导入失败 // {exc}")
            return

        self.pages = self.index.list_pages()
        self._populate_pages(result.page.id)
        self._open_page(result.page.id)
        if result.created:
            message = f"已导入 // {result.page.id} // 解析 {result.links_resolved} 条链接"
        else:
            message = f"页面已存在 // 未覆盖 // {result.page.id}"
        if link_error is not None:
            message += " // 链接抓取失败"
        self._set_status(message)

    # ------------------------------------------------------------------
    # Local document compilation
    # ------------------------------------------------------------------

    def _start_document_import(self, raw_path: str) -> None:
        path_text = raw_path.strip().strip('"').strip("'")
        if not path_text:
            self._set_status("本地导入 // 路径不能为空")
            return
        if self.compiler_factory is None:
            self._set_status(
                "本地导入不可用 // 请以 --base-url 和 --model 启动"
            )
            return
        path = Path(path_text).expanduser()
        if not path.exists():
            self._set_status(f"本地导入失败 // 文件不存在：{path}")
            return
        if not path.is_file():
            self._set_status(f"本地导入失败 // 不是文件：{path}")
            return
        if path.suffix.lower() != ".txt":
            self._set_status("本地导入失败 // 仅支持 UTF-8 .txt 文件")
            return
        try:
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            self._set_status(
                f"本地导入失败 // {path.name} // 无法读取文件："
                f"{_clip(str(exc), 80)}"
            )
            return
        document_id, previous_key = self._resolve_import_source(path, fingerprint)
        self._set_status(f"正在编译本地文档 // {path.name}")
        self._import_document_worker(
            path, document_id, fingerprint, previous_key
        )

    def _resolve_import_source(
        self, path: Path, fingerprint: str
    ) -> tuple[str, str | None]:
        """Return (document_id, previous registry key) for this import.

        A registered path keeps its document id, so a re-import always maps
        to the same page. A file whose content uniquely matches a registered
        page whose old path no longer exists is treated as that file renamed.
        Anything else is a new source and gets the stem-based id.
        """
        key = source_key(path)
        entry = self.sources.get(key)
        if entry is not None:
            return entry["document_id"], None
        renamed = [
            (k, e)
            for k, e in self.sources.items()
            if e["fingerprint"] == fingerprint
            and self.index.get(e["page_id"]) is not None
            and not Path(k).exists()
        ]
        if len(renamed) == 1:
            return renamed[0][1]["document_id"], renamed[0][0]
        return f"local-{path.stem}", None

    @work(thread=True)
    def _import_document_worker(
        self,
        path: Path,
        document_id: str,
        fingerprint: str,
        previous_key: str | None,
    ) -> None:
        factory = self.compiler_factory
        if factory is None:
            self.call_from_thread(
                self._show_document_import_error, path,
                "compiler is not configured",
            )
            return
        try:
            document = PlainTextFileIngestor().ingest(
                path, document_id=document_id
            )
            compiled = factory().compile(document)
        except (WikiCompilerError, OSError, ValueError, TypeError) as exc:
            self.call_from_thread(
                self._show_document_import_error, path, str(exc)
            )
            return
        except Exception as exc:  # keep the TUI alive on unexpected failures
            self.call_from_thread(
                self._show_document_import_error, path,
                _exception_text(exc),
            )
            return
        self.call_from_thread(
            self._apply_compiled_document,
            path, compiled, fingerprint, previous_key,
        )

    def _show_document_import_error(self, path: Path, message: str) -> None:
        self._set_status(
            f"本地导入失败 // {path.name} // {_clip(message, 120)}"
        )

    def _restore_index_snapshot(self, before: dict[str, WikiPage]) -> None:
        for page_id in [page.id for page in self.index.list_pages()]:
            if page_id not in before:
                self.index.delete(page_id)
        for page in before.values():
            self.index.upsert(page)

    @staticmethod
    def _same_import_source(existing: WikiPage, page: WikiPage) -> bool:
        """True when ``existing`` was compiled from the same source document."""
        source_id = page.metadata.get("source_document_id")
        if not isinstance(source_id, str) or not source_id:
            return False
        return (
            existing.metadata.get("source_document_id") == source_id
            or source_id in existing.sources
        )

    def _apply_compiled_document(
        self,
        path: Path,
        compiled: CompiledPage,
        fingerprint: str,
        previous_key: str | None,
    ) -> None:
        page = compiled.page
        key = source_key(path)
        existing = self.index.get(page.id)
        # A page registered to another live source path must not be taken
        # over (e.g. two files sharing a stem). A rename's old path is
        # allowed because ``previous_key`` names it explicitly.
        owned_elsewhere = any(
            entry["page_id"] == page.id and k != previous_key
            for k, entry in self.sources.items()
            if k != key
        )
        updating = existing is not None and not owned_elsewhere and (
            key in self.sources
            or previous_key is not None
            or self._same_import_source(existing, page)
        )
        if existing is not None and not updating:
            detail = "由其他源文件管理，未覆盖" if owned_elsewhere else "未覆盖"
            self._show_document_import_error(
                path, f"页面已存在：{page.id}（{detail}）"
            )
            return
        occupant = self.index.get_by_path(page.path)
        if occupant is not None and occupant.id != page.id:
            self._show_document_import_error(
                path,
                f"路径已被页面 {occupant.id} 使用：{page.path}",
            )
            return

        before = {p.id: p for p in self.index.list_pages()}
        before_sources = dict(self.sources)
        saved = False
        try:
            # Replace the old version instead of appending a duplicate id.
            pages = [p for p in self.index.list_pages() if p.id != page.id]
            if updating:
                # Links the user added by hand are not in "generated_links";
                # carry them over instead of silently dropping them.
                generated = existing.metadata.get("generated_links")
                old_generated = (
                    {t for t in generated if isinstance(t, str)}
                    if isinstance(generated, list)
                    else set()
                )
                manual_links = tuple(
                    t for t in existing.links_to if t not in old_generated
                )
                if manual_links:
                    page = replace(page, links_to=manual_links)
                # Drop backlinks contributed by the old version. Otherwise the
                # resolver's existing-link union resurrects links that the
                # edited document no longer declares.
                pages = [
                    replace(
                        p,
                        links_from=tuple(
                            s for s in p.links_from if s != page.id
                        ),
                    ) if page.id in p.links_from else p
                    for p in pages
                ]
            pages.append(page)
            resolved_pages, report = resolve_candidate_links(pages)
            # Record which outgoing links came from candidate resolution so
            # a later re-import can tell them from hand-added ones.
            resolved_for_page = sorted(
                {
                    link.target_page_id
                    for link in report.resolved_links
                    if link.source_page_id == page.id
                }
            )
            for resolved in resolved_pages:
                if resolved.id == page.id:
                    metadata = dict(resolved.metadata)
                    metadata["generated_links"] = resolved_for_page
                    resolved = replace(resolved, metadata=metadata)
                self.index.upsert(resolved)
            document_id = page.metadata.get("source_document_id")
            if isinstance(document_id, str) and document_id:
                for stale_key in [
                    k
                    for k, entry in self.sources.items()
                    if entry["page_id"] == page.id and k != key
                ]:
                    del self.sources[stale_key]
                self.sources[key] = {
                    "page_id": page.id,
                    "document_id": document_id,
                    "fingerprint": fingerprint,
                }
            if self.workspace_path is not None and not self._is_demo_workspace:
                save_workspace(
                    self.index, self.workspace_path,
                    overwrite=True, sources=self.sources,
                )
                saved = True
        except (WorkspaceError, OSError, TypeError, ValueError) as exc:
            self._restore_index_snapshot(before)
            self.sources = before_sources
            self._show_document_import_error(
                path, f"{_exception_text(exc)} // 已回滚"
            )
            return
        except Exception as exc:  # never leave a half-applied import
            self._restore_index_snapshot(before)
            self.sources = before_sources
            self._show_document_import_error(
                path, f"{_exception_text(exc)} // 已回滚"
            )
            return

        self.pages = self.index.list_pages()
        self._populate_pages(page.id)
        self._open_page(page.id)
        self.query_one("#doc-path", Input).value = ""
        link_text = (
            f"链接 {report.resolved_count} 解析 / "
            f"{report.ambiguous_count} 歧义 / {report.unresolved_count} 未解析"
        )
        saved_text = self._saved_status(saved)
        action = "已更新" if updating else "已导入"
        self._set_status(
            f"{action} // {page.id} // {link_text} // {saved_text}"
        )
        self.query_one("#doc-path", Input).focus()

    # ------------------------------------------------------------------
    # Local wiki agent Q&A
    # ------------------------------------------------------------------

    def _start_ask(self, raw_question: str) -> None:
        self._agent_source_navigation = False
        self._agent_panels_active = False
        question = raw_question.strip()
        if not question:
            self._set_status("提问 // 问题不能为空")
            return
        if self.agent_factory is None:
            self._set_status(
                "问答不可用 // 请以 --base-url 和 --model 启动"
            )
            return
        self._set_status(f"Agent 运行中 // {question!r}")
        self._ask_agent_worker(question)

    @work(thread=True)
    def _ask_agent_worker(self, question: str) -> None:
        try:
            agent = self.agent_factory(self.index)
            result = agent.answer(question)
        except Exception as exc:  # config/transport failures must not kill the UI
            self.call_from_thread(self._show_ask_error, str(exc))
            return
        self.call_from_thread(self._show_agent_result, question, result)

    def _show_agent_result(self, question: str, result: AgentResult) -> None:
        self.last_agent_result = result
        self.last_agent_question = question
        self._agent_source_navigation = False
        self.preview_article = None
        self.preview_links = None
        self.mode = "ask"

        label = _AGENT_STATUS_LABELS.get(result.status, result.status.upper())
        sources = self._source_pages(result.sources, result.read_page_ids)
        self._set_panel_labels("答案来源", "已读页面")
        valid_sources, missing_sources = self._fill_agent_sources(
            result.sources, result.read_page_ids
        )
        missing_reads = self._fill_agent_read_pages(result.read_page_ids)
        self.query_one("#page-title", Static).update(f"提问 // {question}")
        self.query_one("#page-path", Static).update(
            f"状态 {label}  //  步骤 {len(result.steps)}  //  "
            f"已读页面 {result.pages_read}"
        )
        self.query_one("#reader", Static).update(
            _answer_body(result, label, sources)
        )
        self.query_one("#reader-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#meta-label", Static).update("Agent 运行")
        self.query_one("#page-meta", Static).update(
            self._agent_diagnostics(result, label, sources)
        )
        status = (
            f"问答 // {label} // {len(result.steps)} 步 // "
            f"阅读 {result.pages_read} 页"
        )
        if missing_sources:
            status += f" // {missing_sources} 个来源缺失"
        if missing_reads:
            status += f" // {missing_reads} 个已读页面缺失"
        self._set_status(status)
        if valid_sources and not isinstance(self.screen, PageEditorScreen):
            self.query_one("#outbound", OptionList).focus()

    def _show_ask_error(self, message: str) -> None:
        self.mode = "ask"
        self.last_agent_result = None
        self.last_agent_question = None
        self._agent_source_navigation = False
        self._set_panel_labels("出链", "入链")
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")
        self.query_one("#page-title", Static).update("提问 // 失败")
        self.query_one("#page-path", Static).update("状态 失败")
        self.query_one("#reader", Static).update(f"[提问失败]\n\n{message}")
        self.query_one("#reader-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#meta-label", Static).update("Agent 错误")
        self.query_one("#page-meta", Static).update(f"状态     失败\n\n{message}")
        self._set_status(f"问答失败 // {message}")

    def _source_pages(
        self, paths: tuple[str, ...], read_page_ids: tuple[str, ...] = ()
    ) -> list[WikiPage]:
        read_ids = set(read_page_ids)
        pages: list[WikiPage] = []
        seen: set[str] = set()
        for path in paths:
            page = self.index.get_by_path(path)
            if (
                page is not None
                and (not read_ids or page.id in read_ids)
                and page.id not in seen
            ):
                seen.add(page.id)
                pages.append(page)
        return pages

    def _agent_diagnostics(
        self, result: AgentResult, label: str, sources: list[WikiPage]
    ) -> str:
        lines = [
            f"状态     {label}",
            f"步骤     {len(result.steps)}",
            f"阅读     {result.pages_read} 页",
            f"停止     {_clip(result.error or '已生成回答', 48)}",
            "",
            "动作",
        ]
        for step in result.steps[:10]:
            lines.append(
                f"{step.step + 1:02d} {step.action:<13} "
                f"{_clip(step.result_summary, 32)}"
            )
        if len(result.steps) > 10:
            lines.append(f"   ... 另有 {len(result.steps) - 10} 条")
        if result.read_page_ids:
            lines.extend(["", "已读页面"])
            for page_id in result.read_page_ids:
                page = self.index.get(page_id)
                lines.append(
                    f"- {_clip(page.title, 20)} ({page.id})"
                    if page is not None else f"- {page_id}"
                )
        if sources:
            lines.extend(["", "来源"])
            for page in sources:
                lines.append(f"- {_clip(page.title, 20)} ({page.id})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Local page create / edit / delete
    # ------------------------------------------------------------------

    def _open_page_editor(self, page: WikiPage | None) -> None:
        if self.preview_article is not None:
            self._set_status("请先按 ESC 取消维基百科预览")
            return

        def validate(candidate: WikiPage) -> str | None:
            if page is None and self.index.get(candidate.id) is not None:
                return f"页面 ID 已存在：{candidate.id}"
            occupant = self.index.get_by_path(candidate.path)
            if occupant is not None and occupant.id != candidate.id:
                return f"路径已被页面 {occupant.id} 使用：{candidate.path}"
            if candidate.id in candidate.links_to:
                return "页面不能链接到自身"
            missing = [
                target for target in candidate.links_to
                if self.index.get(target) is None
            ]
            if missing:
                return f"出链目标不存在：{', '.join(missing)}"
            return None

        self.push_screen(
            PageEditorScreen(page, validate),
            callback=lambda result: self._apply_page_edit(page, result),
        )

    def _apply_page_edit(
        self, original: WikiPage | None, page: WikiPage | None
    ) -> None:
        self._overwrite_armed = False
        self._delete_armed = False
        if page is None:
            self._set_status("已取消页面编辑")
            return
        before = {p.id: p for p in self.index.list_pages()}
        try:
            self.index.upsert(page)
            self._sync_links_from()
        except (TypeError, ValueError) as exc:
            self._restore_index_snapshot(before)
            self._set_status(
                f"保存页面失败 // {_clip(str(exc), 100)} // 已回滚"
            )
            return
        except Exception as exc:  # never leave a half-applied edit
            self._restore_index_snapshot(before)
            self._set_status(
                f"保存页面失败 // {_exception_text(exc)} // 已回滚"
            )
            return
        stored = self.index.get(page.id)
        self._undo_edit = (
            (original, stored) if stored is not None else None
        )
        self.pages = self.index.list_pages()
        self.current_page = page
        self._populate_pages(page.id)
        self._open_page(page.id)
        action = "已创建" if original is None else "已更新"
        self._set_status(f"{action} // {page.id} // 按 S 保存")

    def _sync_links_from(self) -> None:
        """Recompute every page's backlinks from the authoritative links_to."""
        pages = self.index.list_pages()
        incoming: dict[str, list[str]] = {page.id: [] for page in pages}
        for page in pages:
            for target in page.links_to:
                if target in incoming:
                    incoming[target].append(page.id)
        for page in pages:
            desired = tuple(sorted(incoming[page.id]))
            if page.links_from != desired:
                self.index.upsert(replace(page, links_from=desired))

    def _undo_last_edit(self) -> None:
        """Undo the most recent page-editor save (single level).

        Scope: only the last create/edit made through the page editor. It
        never undoes imports, deletes or source sync. If the page changed
        since that save, undo refuses rather than overwriting newer content.
        """
        if isinstance(self.screen, PageEditorScreen):
            return
        if self.preview_article is not None:
            self._set_status("请先按 ESC 取消维基百科预览")
            return
        if self._undo_edit is None:
            self._set_status("无法撤销 // 没有可撤销的编辑")
            return
        original, edited = self._undo_edit
        current = self.index.get(edited.id)
        if current is None:
            self._undo_edit = None
            self._set_status(
                f"无法撤销 // 页面已不存在：{edited.id} // 撤销记录已清除"
            )
            return
        # links_from is derived and recomputed after restoring, so changes
        # there (another page linked here) must not block the undo.
        if replace(current, links_from=()) != replace(edited, links_from=()):
            self._undo_edit = None
            self._set_status(
                f"无法撤销 // 页面在编辑后被修改：{edited.id} // 未覆盖当前内容"
            )
            return
        if original is not None:
            occupant = self.index.get_by_path(original.path)
            if occupant is not None and occupant.id != original.id:
                self._undo_edit = None
                self._set_status(
                    f"无法撤销 // 原路径 {original.path} 已被页面 "
                    f"{occupant.id} 使用 // 请先移动该页面"
                )
                return
            missing = [
                target for target in original.links_to
                if target != original.id and self.index.get(target) is None
            ]
            if missing:
                self._undo_edit = None
                self._set_status(
                    f"无法撤销 // 链接目标已不存在：{', '.join(missing)} "
                    "// 请先恢复目标页面"
                )
                return
        before = {p.id: p for p in self.index.list_pages()}
        try:
            if original is None:
                self._remove_page_from_index(current)
            else:
                self.index.upsert(original)
                self._sync_links_from()
        except (TypeError, ValueError) as exc:
            self._restore_index_snapshot(before)
            self._set_status(f"撤销失败 // {_clip(str(exc), 100)} // 已回滚")
            return
        except Exception as exc:  # never leave a half-applied undo
            self._restore_index_snapshot(before)
            self._set_status(
                f"撤销失败 // {_exception_text(exc)} // 已回滚"
            )
            return
        self._undo_edit = None
        self.pages = self.index.list_pages()
        if original is None:
            self.current_page = None
            if self.pages:
                self.current_page = self.pages[0]
            self._restore_pages()
            self._set_status(f"已撤销新建 // {edited.id} // 按 S 保存")
            return
        self.current_page = self.index.get(original.id)
        self._populate_pages(original.id)
        self._open_page(original.id)
        self._set_status(f"已撤销编辑 // {original.id} // 按 S 保存")

    def _edit_current_page(self) -> None:
        if self.preview_article is not None:
            self._set_status("请先按 ESC 取消维基百科预览")
            return
        if self.current_page is None:
            self._set_status("无法编辑 // 没有打开的页面")
            return
        self._open_page_editor(self.current_page)

    def _request_delete_current_page(self) -> None:
        if self.preview_article is not None:
            self._set_status("请先按 ESC 取消维基百科预览")
            return
        if self.current_page is None:
            self._set_status("无法删除 // 没有打开的页面")
            return
        if not self._delete_armed:
            self._delete_armed = True
            self._set_status(
                f"再次按 D 删除 // {self.current_page.id} · ESC 取消"
            )
            return
        self._delete_armed = False
        self._delete_current_page()

    def _remove_page_from_index(self, page: WikiPage) -> None:
        """Delete a page and strip it from every other page's links.

        Keeps the link graph consistent without changing WikiIndex.delete()'s
        documented low-level behavior.
        """
        for other in self.index.list_pages():
            if other.id == page.id:
                continue
            links_to = tuple(t for t in other.links_to if t != page.id)
            links_from = tuple(s for s in other.links_from if s != page.id)
            if links_to != other.links_to or links_from != other.links_from:
                self.index.upsert(
                    replace(other, links_to=links_to, links_from=links_from)
                )
        self.index.delete(page.id)

    def _delete_current_page(self) -> None:
        page = self.current_page
        if page is None:
            return
        self._remove_page_from_index(page)
        self.sources = {
            key: entry
            for key, entry in self.sources.items()
            if entry["page_id"] != page.id
        }
        self.pages = self.index.list_pages()
        self.current_page = None
        self.mode = "pages"
        if self.pages:
            self._populate_pages(self.pages[0].id)
            self._open_page(self.pages[0].id)
            self._set_status(f"已删除 // {page.id} // 按 S 保存")
        else:
            self._populate_pages()
            self._show_empty_workspace()
            self._set_status(
                f"已删除 // {page.id} // 按 N 新建页面 // 按 S 保存"
            )

    @staticmethod
    def _page_is_source_managed(
        page: WikiPage, entry: dict[str, str]
    ) -> bool:
        """True when the registry can confirm the page came from this source."""
        document_id = page.metadata.get("source_document_id")
        return (
            isinstance(document_id, str)
            and document_id == entry["document_id"]
            and document_id in page.sources
        )

    def action_sync_sources(self) -> None:
        self._reset_arming()
        self._sync_source_lifecycle()

    def _sync_source_lifecycle(self) -> None:
        """Explicit cleanup: drop registrations and pages whose file is gone.

        A page is only deleted when the registry entry and the page's own
        provenance agree; manually created or otherwise-owned pages are kept.
        """
        if self.preview_article is not None:
            self._set_status("请先按 ESC 取消维基百科预览")
            return
        missing = [key for key in self.sources if not Path(key).exists()]
        if not missing:
            if not self.sources:
                self._set_status("源同步 // 没有已登记的源文件")
            else:
                self._set_status(
                    f"源同步 // {len(self.sources)} 个登记源均存在"
                )
            return
        before = {p.id: p for p in self.index.list_pages()}
        before_sources = dict(self.sources)
        removed = 0
        protected = 0
        dropped = 0
        saved = False
        try:
            for key in missing:
                entry = self.sources[key]
                page = self.index.get(entry["page_id"])
                if page is None:
                    del self.sources[key]
                    dropped += 1
                    continue
                if not self._page_is_source_managed(page, entry):
                    protected += 1
                    continue
                self._remove_page_from_index(page)
                del self.sources[key]
                removed += 1
            if self.workspace_path is not None and not self._is_demo_workspace:
                save_workspace(
                    self.index, self.workspace_path,
                    overwrite=True, sources=self.sources,
                )
                saved = True
        except (WorkspaceError, OSError, TypeError, ValueError) as exc:
            self._restore_index_snapshot(before)
            self.sources = before_sources
            self._set_status(
                f"源同步失败 // {_exception_text(exc)} // 已回滚"
            )
            return
        except Exception as exc:  # never leave a half-applied cleanup
            self._restore_index_snapshot(before)
            self.sources = before_sources
            self._set_status(
                f"源同步失败 // {_exception_text(exc)} // 已回滚"
            )
            return
        self.pages = self.index.list_pages()
        if (
            self.current_page is not None
            and self.index.get(self.current_page.id) is None
        ):
            self.current_page = None
        if self.current_page is None and self.pages:
            self.current_page = self.pages[0]
        self._restore_pages()
        saved_text = self._saved_status(saved)
        protected_text = (
            f"保护 {protected} 页（来源不匹配）"
            if protected
            else "保护 0 页"
        )
        self._set_status(
            f"源同步 // 清理 {removed} 页 // {protected_text} "
            f"// 移除 {dropped} 条失效登记 // {saved_text}"
        )

    def _saved_status(self, saved: bool) -> str:
        if saved:
            return "已保存"
        if self._is_demo_workspace:
            return "未持久化（演示工作区只读）"
        return "未持久化（未指定工作区）"

    def _save_workspace(self) -> None:
        if self.workspace_path is None:
            self._set_status("无法保存 // 未指定工作区路径")
            return
        if self._is_demo_workspace:
            self._set_status(
                "无法保存 // 演示工作区只读，请用 --workspace 新建自己的工作区"
            )
            return
        if self.workspace_path.exists() and not self._overwrite_armed:
            self._overwrite_armed = True
            self._set_status(
                "工作区已存在 // 再次按 S 覆盖 · ESC 取消"
            )
            return
        try:
            save_workspace(
                self.index,
                self.workspace_path,
                overwrite=self._overwrite_armed,
                sources=self.sources,
            )
        except WorkspaceError as exc:
            self._set_status(f"保存失败 // {_workspace_error_text(exc)}")
            return
        except Exception as exc:  # keep tracebacks out of the UI
            self._set_status(f"保存失败 // {_exception_text(exc)}")
            return
        self._overwrite_armed = False
        self._set_status(f"已保存 // {self.workspace_path}")

    def _cancel_wikipedia(self) -> None:
        self.preview_article = None
        self.preview_links = None
        self.preview_link_error = None
        self.wikipedia_results = {}
        self._overwrite_armed = False
        self._restore_pages()

    def _restore_pages(self) -> None:
        self.pages = self.index.list_pages()
        if self.current_page is not None:
            self._populate_pages(self.current_page.id)
            self._open_page(self.current_page.id)
            self._set_status(f"页面列表 // {self.current_page.id}")
        else:
            self.query_one("#pages", OptionList).clear_options()
            self._show_empty_workspace()
        self.query_one("#pages", OptionList).focus()

    def _show_wikipedia_search_error(self, message: str) -> None:
        self.mode = "wikipedia"
        self.preview_article = None
        self.wikipedia_results = {}
        self._set_panel_labels("出链", "入链")
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        list_widget.add_option(Option("  维基百科搜索失败", disabled=True))
        self.query_one("#list-label", Static).update("维基百科结果 (0)")
        self.query_one("#meta-label", Static).update("维基百科错误")
        self.query_one("#page-meta", Static).update(f"搜索失败\n\n{message}")
        self._set_status(f"维基百科搜索失败 // {message}")

    def _show_wikipedia_read_error(self, message: str) -> None:
        self.query_one("#meta-label", Static).update("维基百科错误")
        self.query_one("#page-meta", Static).update(f"读取失败\n\n{message}")
        self._set_status(f"维基百科读取失败 // {message}")

    def _open_page(self, page_id: str) -> None:
        page = self.index.get(page_id)
        if page is None:
            self._set_status(f"未找到页面 // {page_id}")
            return
        self.current_page = page
        self.query_one("#page-title", Static).update(page.title)
        self.query_one("#page-path", Static).update(
            f"{page.path}  //  id={page.id}"
        )
        self.query_one("#reader", Static).update(
            page.content.rstrip() + "\n\n── 页面结束 ──"
        )
        self.query_one("#reader-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#meta-label", Static).update("页面信息")
        self._render_page_side(page)
        self._set_status(f"打开 // {page.id}")
        if self.mode == "pages":
            self._highlight_in_pages(page.id)

    def _highlight_in_pages(self, page_id: str) -> None:
        list_widget = self.query_one("#pages", OptionList)
        ids = [
            option.id
            for option in (
                list_widget.get_option_at_index(i)
                for i in range(list_widget.option_count)
            )
        ]
        if page_id in ids:
            list_widget.highlighted = ids.index(page_id)

    def _set_panel_labels(self, outbound: str, inbound: str) -> None:
        self.query_one("#outbound-label", Static).update(outbound)
        self.query_one("#inbound-label", Static).update(inbound)
        self._agent_panels_active = outbound == "答案来源"

    def _fill_agent_sources(
        self, paths: tuple[str, ...], read_page_ids: tuple[str, ...] = ()
    ) -> tuple[int, int]:
        """Fill the right panel with clickable answer sources.

        Returns (valid_source_count, missing_source_count). Missing or
        mismatched paths stay visible as disabled entries so the user sees why
        they cannot open.
        """
        widget = self.query_one("#outbound", OptionList)
        widget.clear_options()
        read_ids = set(read_page_ids)
        seen_paths: set[str] = set()
        seen_ids: set[str] = set()
        first_valid: int | None = None
        missing = 0
        for path in paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                page = self.index.get_by_path(path)
            except (TypeError, ValueError):
                page = None
            if page is None or (read_ids and page.id not in read_ids):
                widget.add_option(
                    Option(f"  {path}  (来源缺失)", disabled=True)
                )
                missing += 1
            elif page.id not in seen_ids:
                seen_ids.add(page.id)
                if first_valid is None:
                    first_valid = widget.option_count
                widget.add_option(Option(f"  {page.title}", id=page.id))
        if widget.option_count == 0:
            widget.add_option(Option("  无经核实的来源", disabled=True))
        widget.highlighted = first_valid if first_valid is not None else 0
        return len(seen_ids), missing

    def _fill_agent_read_pages(self, page_ids: tuple[str, ...]) -> int:
        """Fill the right panel with clickable pages the Agent actually read."""
        widget = self.query_one("#inbound", OptionList)
        widget.clear_options()
        seen: set[str] = set()
        first_valid: int | None = None
        missing = 0
        for page_id in page_ids:
            if page_id in seen:
                continue
            seen.add(page_id)
            page = self.index.get(page_id)
            if page is None:
                widget.add_option(
                    Option(f"  {page_id}  (缺失)", disabled=True)
                )
                missing += 1
            else:
                if first_valid is None:
                    first_valid = widget.option_count
                widget.add_option(Option(f"  {page.title}", id=page.id))
        if widget.option_count == 0:
            widget.add_option(Option("  未读取页面", disabled=True))
        widget.highlighted = first_valid if first_valid is not None else 0
        return missing

    def _render_page_side(self, page: WikiPage) -> None:
        self._set_panel_labels("出链", "入链")
        self.query_one("#page-meta", Static).update(
            f"编号     {page.id}\n"
            f"路径     {page.path}\n"
            f"标签     {', '.join(page.tags) or '—'}\n"
            f"来源     {', '.join(page.sources) or '—'}\n"
            f"链接     {len(page.links_to)} 出 / {len(page.links_from)} 入"
        )
        self._fill_links(
            self.query_one("#outbound", OptionList), page.links_to,
            "无出链",
        )
        self._fill_links(
            self.query_one("#inbound", OptionList), page.links_from,
            "无入链",
        )

    def _fill_links(
        self, widget: OptionList, page_ids: tuple[str, ...], empty_text: str,
    ) -> None:
        widget.clear_options()
        if not page_ids:
            widget.add_option(Option(f"  {empty_text}", disabled=True))
            return
        for page_id in page_ids:
            target = self.index.get(page_id)
            if target is None:
                widget.add_option(
                    Option(f"  {page_id}  (缺失)", id=page_id, disabled=True)
                )
            else:
                widget.add_option(Option(f"  {target.title}", id=page_id))
        widget.highlighted = 0

    def _update_topmeta(self) -> None:
        self.query_one("#topmeta", Static).update(
            "本地知识库\n"
            "本地会话\n"
            f"已索引 {len(self.pages)} 页"
        )

    def _show_empty_workspace(self) -> None:
        self.pages = self.index.list_pages()
        self.current_page = None
        self._set_panel_labels("出链", "入链")
        self._update_topmeta()
        self.query_one("#list-label", Static).update("页面 (0)")
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        list_widget.add_option(Option("  空工作区", disabled=True))
        self.query_one("#page-title", Static).update("暂无页面")
        self.query_one("#page-path", Static).update("")
        self.query_one("#reader", Static).update(
            "此工作区还没有任何页面。\n\n"
            "按 N 新建页面，按 L 导入本地 .txt 文档，"
            "按 W 搜索维基百科并按 I 导入，或向 JSON 工作区添加页面后"
            "重启 ZMWiki。"
        )
        self.query_one("#meta-label", Static).update("页面信息")
        self.query_one("#page-meta", Static).update("空工作区")
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")
        self._set_status("空工作区 // 按 N 新建页面")

    def _set_status(self, message: str) -> None:
        self.query_one("#statusbar", Static).update(
            f" {message}   //   {len(self.pages)} 页   //   {FOOTER_KEYS}"
        )

    # ------------------------------------------------------------------
    # Global shortcut actions
    # ------------------------------------------------------------------

    def _reset_arming(self) -> None:
        self._overwrite_armed = False
        self._delete_armed = False

    def action_new_page(self) -> None:
        self._reset_arming()
        self._open_page_editor(None)

    def action_edit_page(self) -> None:
        self._reset_arming()
        self._edit_current_page()

    def action_undo_edit(self) -> None:
        self._reset_arming()
        self._undo_last_edit()

    def action_delete_page(self) -> None:
        self._overwrite_armed = False
        self._request_delete_current_page()

    def action_ask(self) -> None:
        self._reset_arming()
        self.query_one("#ask", Input).focus()

    def action_wikipedia_search(self) -> None:
        self._reset_arming()
        self.query_one("#wiki-search", Input).focus()

    def action_import_wikipedia(self) -> None:
        self._reset_arming()
        self._import_preview()

    def action_import_local_document(self) -> None:
        self._reset_arming()
        self.query_one("#doc-path", Input).focus()

    def action_save_workspace(self) -> None:
        self._delete_armed = False
        self._save_workspace()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            self._run_search(event.value)
        elif event.input.id == "wiki-search":
            self._start_wikipedia_search(event.value)
        elif event.input.id == "ask":
            self._start_ask(event.value)
        elif event.input.id == "doc-path":
            self._start_document_import(event.value)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_id is None:
            return
        if event.option_list.id == "pages":
            if self.mode == "wikipedia":
                self._preview_wikipedia_hit(event.option_id)
            elif self.mode == "search":
                self._preview_search(event.option_id)
            else:
                self._preview_page(event.option_id)

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_id is None:
            return
        if event.option_list.id == "pages" and self.mode == "wikipedia":
            self._read_wikipedia_article(event.option_id)
            return
        if event.option_list.id in ("pages", "outbound", "inbound"):
            if (
                event.option_list.id in ("outbound", "inbound")
                and self._agent_panels_active
                and self.last_agent_result is not None
            ):
                self._agent_source_navigation = True
            self._open_page(event.option_id)

    def on_key(self, event: events.Key) -> None:
        if isinstance(self.screen, PageEditorScreen):
            return
        focused = self.focused

        if event.key == "escape" and self._delete_armed:
            self._delete_armed = False
            self._set_status("已取消删除")
            event.stop()
            return

        if event.key not in ("s", "S", "ctrl+s"):
            self._overwrite_armed = False
        if event.key not in ("d", "D"):
            self._delete_armed = False

        if event.key == "q" and not isinstance(focused, Input):
            self.exit(0)
            event.stop()
            return

        if event.key == "slash" and not isinstance(focused, Input):
            self.query_one("#search", Input).focus()
            event.stop()
            return

        if event.key == "escape" and isinstance(focused, Input):
            focused.value = ""
            if focused.id == "wiki-search":
                self._cancel_wikipedia()
            elif focused.id == "ask":
                self._restore_pages()
            else:
                self._populate_pages(
                    self.current_page.id if self.current_page else None
                )
                if self.current_page is not None:
                    self._preview_page(self.current_page.id)
                    self._set_status(f"页面列表 // {self.current_page.id}")
                self.query_one("#pages", OptionList).focus()
            event.stop()
            return

        if event.key == "escape" and self._agent_source_navigation:
            self._agent_source_navigation = False
            if (
                self.last_agent_question is not None
                and self.last_agent_result is not None
            ):
                self._show_agent_result(
                    self.last_agent_question, self.last_agent_result
                )
            event.stop()
            return

        if event.key == "escape" and self.mode == "ask":
            self._restore_pages()
            event.stop()
            return

        if event.key == "escape" and (
            self.mode == "wikipedia" or self.preview_article is not None
        ):
            self._cancel_wikipedia()
            event.stop()
            return

        if event.key in "123456789" and not isinstance(focused, Input):
            target = self.query_one("#pages", OptionList)
            if isinstance(focused, OptionList) and focused.id in ("outbound", "inbound"):
                target = focused
            position = int(event.key) - 1
            if position < target.option_count:
                option = target.get_option_at_index(position)
                if option.id is not None and not option.disabled:
                    self._open_page(option.id)
                    event.stop()


def _load_or_create_workspace(
    path: str | Path, *, create: bool = False
) -> tuple[WikiIndex, SourceRegistry]:
    """Load a workspace, or create an empty one when ``--create`` is used."""
    target = Path(path)
    if create:
        if target.exists():
            raise WorkspaceError("workspace file already exists")
        save_workspace(WikiIndex(), target)
    return load_workspace_state(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zmwiki",
        description=(
            "浏览 ZMWiki JSON 工作区，导入维基百科文章，"
            "并通过 WikiAgent 向本地知识库提问。"
        ),
    )
    parser.add_argument("--workspace", default=str(_DEFAULT_WORKSPACE),
                        help="JSON 工作区文件（读取，按下保存键时写回）；"
                             "默认使用仓库内 benchmark/corpus/wiki_workspace.json")
    parser.add_argument("--create", action="store_true",
                        help="创建新的空工作区文件（目标文件必须不存在）")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI 兼容 API 基础地址（启用提问 / 本地文档编译）")
    parser.add_argument("--model", default=None,
                        help="提问和文档编译所用的模型标识")
    parser.add_argument("--api-key", default=None,
                        help="提问所用的 API 密钥（也可设置 ZMWIKI_API_KEY）")
    args = parser.parse_args(argv)

    if (args.base_url is None) != (args.model is None):
        print("错误：--base-url 和 --model 必须同时提供",
              file=sys.stderr)
        return 2

    try:
        index, sources = _load_or_create_workspace(
            args.workspace, create=args.create
        )
    except WorkspaceError as exc:
        message = str(exc)
        if not args.create and message == "workspace file not found":
            message += "（新建空工作区请加 --create）"
        print(f"错误：{message}", file=sys.stderr)
        return 2

    agent_factory: Callable[[WikiIndex], WikiAgent] | None = None
    if args.base_url is not None:
        def build_agent(index: WikiIndex) -> WikiAgent:
            return WikiAgent(
                index,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
            )

        agent_factory = build_agent

    compiler_factory: Callable[[], WikiCompiler] | None = None
    if args.base_url is not None:
        def build_compiler() -> WikiCompiler:
            return WikiCompiler(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
            )

        compiler_factory = build_compiler

    ZMWikiApp(
        index,
        workspace_path=args.workspace,
        sources=sources,
        agent_factory=agent_factory,
        compiler_factory=compiler_factory,
    ).run()
    return 0
