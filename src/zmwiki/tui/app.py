"""Read-only retro terminal UI for ZMWiki.

The TUI talks directly to the in-memory WikiIndex. It does not use the HTTP
server, and it never writes to the workspace.
"""

from __future__ import annotations

import argparse
import sys

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from zmwiki.web.workspace import WorkspaceError, load_workspace
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_retriever import WikiSearchResult, wiki_search
from zmwiki.wiki.wiki_schema import WikiPage

__all__ = ["ZMWikiApp", "main"]

BANNER = r"""
███████╗███╗   ███╗██╗    ██╗██╗██╗  ██╗██╗
╚══███╔╝████╗ ████║██║    ██║██║██║ ██╔╝██║
  ███╔╝ ██╔████╔██║██║ █╗ ██║██║█████╔╝ ██║
 ███╔╝  ██║╚██╔╝██║██║███╗██║██║██╔═██╗ ██║
███████╗██║ ╚═╝ ██║╚███╔███╔╝██║██║  ██╗██║
╚══════╝╚═╝     ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═╝╚═╝
""".strip("\n")

FOOTER_KEYS = "↑↓ MOVE · ENTER OPEN · 1-9 JUMP · / SEARCH · TAB PANE · Q QUIT"

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

#search-label, #list-label, #meta-label, #outbound-label, #inbound-label {
    color: #3fbf77;
    text-style: bold;
    height: auto;
}

#search-label, #list-label {
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
    height: 2;
    border-top: solid #0a7a45;
    color: #4f9f70;
    padding: 0 1;
}
"""


class ZMWikiApp(App[None]):
    """Read-only ZMWiki browser with search and link navigation."""

    TITLE = "ZMWiki"
    SUB_TITLE = "local knowledge terminal"
    CSS = CSS

    def __init__(self, index: WikiIndex) -> None:
        super().__init__()
        self.index = index
        self.pages: list[WikiPage] = []
        self.current_page: WikiPage | None = None
        self.mode = "pages"  # "pages" | "search"
        self.search_results: dict[str, WikiSearchResult] = {}

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
                    yield Static("SEARCH", id="search-label")
                    yield Input(placeholder="query pages…", id="search")
                    yield Static("PAGES", id="list-label")
                    yield OptionList(id="pages")
                with Vertical(id="center"):
                    yield Static("", id="page-title")
                    yield Static("", id="page-path")
                    with VerticalScroll(id="reader-scroll"):
                        yield Static("", id="reader", markup=False)
                with Vertical(id="right"):
                    yield Static("RECORD", id="meta-label")
                    yield Static("", id="page-meta", markup=False)
                    yield Static("OUTBOUND LINKS", id="outbound-label")
                    yield OptionList(id="outbound")
                    yield Static("INBOUND LINKS", id="inbound-label")
                    yield OptionList(id="inbound")
            yield Static("", id="statusbar")

    async def on_mount(self) -> None:
        self.pages = self.index.list_pages()
        self.query_one("#topmeta", Static).update(
            "LOCAL KNOWLEDGE BASE\n"
            "READ-ONLY SESSION\n"
            f"{len(self.pages)} PAGES INDEXED"
        )
        if not self.pages:
            self._show_empty_workspace()
            return
        self._populate_pages(self.pages[0].id)
        self._open_page(self.pages[0].id)
        self.query_one("#pages", OptionList).focus()

    # ------------------------------------------------------------------
    # Page list, search, previews
    # ------------------------------------------------------------------

    def _populate_pages(self, highlight_id: str | None = None) -> None:
        self.mode = "pages"
        self.search_results = {}
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        self.query_one("#list-label", Static).update(f"PAGES ({len(self.pages)})")
        if not self.pages:
            list_widget.add_option(Option("  EMPTY WORKSPACE", disabled=True))
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
        query = raw_query.strip()
        if not query:
            self._populate_pages(self.current_page.id if self.current_page else None)
            if self.current_page is not None:
                self._preview_page(self.current_page.id)
                self._set_status(f"PAGES // {self.current_page.id}")
            return

        results = wiki_search(self.index, query, top_k=20)
        self.mode = "search"
        self.search_results = {result.page_id: result for result in results}
        list_widget = self.query_one("#pages", OptionList)
        list_widget.clear_options()
        self.query_one("#list-label", Static).update(
            f"SEARCH RESULTS ({len(results)})"
        )

        if not results:
            list_widget.add_option(Option("  NO MATCHES", disabled=True))
            self.query_one("#meta-label", Static).update("SEARCH")
            self.query_one("#page-meta", Static).update(
                f"QUERY     {query}\nRESULTS   0\n\nNo page matched this query."
            )
            self._fill_links(self.query_one("#outbound", OptionList), [], "—")
            self._fill_links(self.query_one("#inbound", OptionList), [], "—")
            self._set_status(f"NO MATCHES // {query!r}")
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
        self._set_status(f"SEARCH // {len(results)} HIT(S) // {query!r}")
        list_widget.focus()

    def _preview_page(self, page_id: str) -> None:
        page = self.index.get(page_id)
        if page is None:
            return
        self.query_one("#meta-label", Static).update("RECORD")
        self._render_page_side(page)

    def _preview_search(self, page_id: str) -> None:
        result = self.search_results.get(page_id)
        if result is None:
            return
        self.query_one("#meta-label", Static).update("SEARCH HIT")
        description = getattr(result, "description", None) or "—"
        self.query_one("#page-meta", Static).update(
            f"ID       {result.page_id}\n"
            f"PATH     {result.path}\n"
            f"SCORE    {result.score:.4f}\n"
            f"TAGS     {', '.join(result.tags) or '—'}\n"
            f"DESC     {description}"
        )
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")

    def _open_page(self, page_id: str) -> None:
        page = self.index.get(page_id)
        if page is None:
            self._set_status(f"PAGE NOT FOUND // {page_id}")
            return
        self.current_page = page
        self.query_one("#page-title", Static).update(page.title)
        self.query_one("#page-path", Static).update(
            f"{page.path}  //  id={page.id}"
        )
        self.query_one("#reader", Static).update(
            page.content.rstrip() + "\n\n── END OF PAGE ──"
        )
        self.query_one("#reader-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#meta-label", Static).update("RECORD")
        self._render_page_side(page)
        self._set_status(f"OPEN // {page.id}")
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

    def _render_page_side(self, page: WikiPage) -> None:
        self.query_one("#page-meta", Static).update(
            f"ID       {page.id}\n"
            f"PATH     {page.path}\n"
            f"TAGS     {', '.join(page.tags) or '—'}\n"
            f"SOURCES  {', '.join(page.sources) or '—'}\n"
            f"LINKS    {len(page.links_to)} out / {len(page.links_from)} in"
        )
        self._fill_links(
            self.query_one("#outbound", OptionList), page.links_to,
            "NO OUTBOUND LINKS",
        )
        self._fill_links(
            self.query_one("#inbound", OptionList), page.links_from,
            "NO INBOUND LINKS",
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
                    Option(f"  {page_id}  [MISSING]", id=page_id, disabled=True)
                )
            else:
                widget.add_option(Option(f"  {target.title}", id=page_id))
        widget.highlighted = 0

    def _show_empty_workspace(self) -> None:
        self.query_one("#list-label", Static).update("PAGES (0)")
        self.query_one("#pages", OptionList).add_option(
            Option("  EMPTY WORKSPACE", disabled=True)
        )
        self.query_one("#page-title", Static).update("NO PAGES")
        self.query_one("#page-path", Static).update("")
        self.query_one("#reader", Static).update(
            "This workspace contains no wiki pages.\n\n"
            "Add pages to the JSON workspace and restart ZMWiki."
        )
        self.query_one("#meta-label", Static).update("RECORD")
        self.query_one("#page-meta", Static).update("EMPTY WORKSPACE")
        self._fill_links(self.query_one("#outbound", OptionList), [], "—")
        self._fill_links(self.query_one("#inbound", OptionList), [], "—")
        self._set_status("EMPTY WORKSPACE")

    def _set_status(self, message: str) -> None:
        self.query_one("#statusbar", Static).update(
            f" {message}   //   {len(self.pages)} PAGES   //   {FOOTER_KEYS}"
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            self._run_search(event.value)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_id is None:
            return
        if event.option_list.id == "pages":
            if self.mode == "search":
                self._preview_search(event.option_id)
            else:
                self._preview_page(event.option_id)

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_id is None:
            return
        if event.option_list.id in ("pages", "outbound", "inbound"):
            self._open_page(event.option_id)

    def on_key(self, event: events.Key) -> None:
        focused = self.focused

        if event.key == "q" and not isinstance(focused, Input):
            self.exit(0)
            event.stop()
            return

        if event.key == "slash" and not isinstance(focused, Input):
            self.query_one("#search", Input).focus()
            event.stop()
            return

        if event.key == "escape" and isinstance(focused, Input):
            self.query_one("#search", Input).value = ""
            self._populate_pages(self.current_page.id if self.current_page else None)
            if self.current_page is not None:
                self._preview_page(self.current_page.id)
                self._set_status(f"PAGES // {self.current_page.id}")
            self.query_one("#pages", OptionList).focus()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zmwiki.tui",
        description="Browse a read-only ZMWiki JSON workspace in the terminal.",
    )
    parser.add_argument("--workspace", required=True,
                        help="JSON workspace file (read-only)")
    args = parser.parse_args(argv)

    try:
        index = load_workspace(args.workspace)
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ZMWikiApp(index).run()
    return 0
