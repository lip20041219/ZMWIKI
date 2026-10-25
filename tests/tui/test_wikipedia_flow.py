"""Headless TUI tests for the Wikipedia search -> preview -> import -> save flow.

No network: a fake connector stands in for WikipediaConnector. No real
terminal: Textual's run_test pilot drives the app.
"""

import asyncio
import dataclasses
import json

from textual.widgets import Input, OptionList, Static

from zmwiki.tui.app import ZMWikiApp
from zmwiki.web.workspace import load_workspace
from zmwiki.wiki.wikipedia import (
    WikipediaArticle,
    WikipediaLink,
    WikipediaSearchResult,
    WikipediaTransportError,
)
from zmwiki.wiki.wikipedia_import import import_wikipedia_article
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage

ARTICLE_URL = "https://en.wikipedia.org/wiki/Alan_Turing"
ARTICLE_TEXT = "Alan Mathison Turing was an English mathematician and logician."

HIT = WikipediaSearchResult(
    title="Alan Turing",
    page_id=11675,
    url=ARTICLE_URL,
    summary="English mathematician and logician.",
)
ARTICLE = WikipediaArticle(
    title="Alan Turing",
    text=ARTICLE_TEXT,
    url=ARTICLE_URL,
    page_id=11675,
    revision_id=123456,
)


def _article_for(page_id, title):
    return WikipediaArticle(
        title=title,
        text=f"Body of {title}.",
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        page_id=page_id,
        revision_id=page_id * 10,
    )


class FakeWikipedia:
    def __init__(
        self,
        results=(HIT,),
        article=ARTICLE,
        *,
        search_error=None,
        read_error=None,
        links=(),
        link_error=None,
    ):
        self.results = list(results)
        self.article = article
        self.search_error = search_error
        self.read_error = read_error
        self.links = list(links)
        self.link_error = link_error
        self.searches: list[tuple[str, int]] = []
        self.reads: list[int] = []
        self.link_reads: list[int] = []

    def search(self, query, *, limit=5):
        self.searches.append((query, limit))
        if self.search_error is not None:
            raise self.search_error
        return list(self.results)

    def read_article(self, *, page_id=None, url=None):
        self.reads.append(page_id)
        if self.read_error is not None:
            raise self.read_error
        return self.article

    def read_links(self, page_id):
        self.link_reads.append(page_id)
        if self.link_error is not None:
            raise self.link_error
        return list(self.links)


def _workspace(tmp_path):
    path = tmp_path / "workspace.json"
    page = WikiPage(
        id="alice", path="people/alice", title="Alice Smith",
        content="Alice leads engineering.",
    )
    path.write_text(
        json.dumps({"version": 1, "pages": [page.to_dict()]}), encoding="utf-8"
    )
    return path


def _app(tmp_path, fake=None, *, workspace_path=None):
    path = _workspace(tmp_path) if workspace_path is None else workspace_path
    app = ZMWikiApp(
        load_workspace(path) if path.exists() else WikiIndex(),
        workspace_path=path,
        wikipedia=fake if fake is not None else FakeWikipedia(),
    )
    return app


async def _search(pilot, app, query="alan turing"):
    await pilot.press("w")
    await pilot.pause()
    search = app.query_one("#wiki-search", Input)
    assert search.has_focus
    search.value = query
    await pilot.pause()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _preview(pilot, app):
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def _run(scenario):
    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Search and preview
# ---------------------------------------------------------------------------


def test_search_lists_results_with_summary(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)

            assert app.mode == "wikipedia"
            assert app.wikipedia.searches == [("alan turing", 8)]
            pages = app.query_one("#pages", OptionList)
            assert pages.option_count == 1
            assert pages.get_option_at_index(0).id == "11675"
            assert "维基百科结果 (1)" in app.query_one(
                "#list-label", Static
            ).content
            meta = app.query_one("#page-meta", Static).content
            assert "English mathematician and logician." in meta
            assert ARTICLE_URL in meta

    _run(scenario)


def test_selected_result_previews_the_article(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)

            assert app.preview_article == ARTICLE
            assert app.wikipedia.reads == [11675]
            assert app.query_one("#page-title", Static).content == "Alan Turing"
            assert ARTICLE_TEXT in app.query_one("#reader", Static).content
            assert app.query_one("#page-path", Static).content == ARTICLE_URL
            meta = app.query_one("#page-meta", Static).content
            assert "123456" in meta
            assert "尚未导入" in meta
            assert app.index.get("wikipedia-11675") is None
            assert app.index.count() == 1

    _run(scenario)


def test_empty_results_show_clear_feedback(tmp_path):
    async def scenario():
        app = _app(tmp_path, FakeWikipedia(results=()))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app, "no such thing")

            pages = app.query_one("#pages", OptionList)
            assert pages.option_count == 1
            assert pages.get_option_at_index(0).disabled
            assert "无维基百科结果" in str(pages.get_option_at_index(0).prompt)
            assert "无结果" in app.query_one("#statusbar", Static).content

    _run(scenario)


# ---------------------------------------------------------------------------
# Import and duplicate handling
# ---------------------------------------------------------------------------


def test_import_adds_page_and_local_search_finds_it(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)

            await pilot.press("i")
            await pilot.pause()

            assert app.index.count() == 2
            imported = app.index.get("wikipedia-11675")
            assert imported is not None
            assert imported.content == ARTICLE_TEXT
            assert "已导入" in app.query_one("#statusbar", Static).content

            search = app.query_one("#search", Input)
            search.focus()
            search.value = "mathematician"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.mode == "search"
            result_ids = [
                app.query_one("#pages", OptionList).get_option_at_index(i).id
                for i in range(app.query_one("#pages", OptionList).option_count)
            ]
            assert "wikipedia-11675" in result_ids

    _run(scenario)


def test_cancel_preview_does_not_touch_the_index(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)

            await pilot.press("escape")
            await pilot.pause()

            assert app.preview_article is None
            assert app.mode == "pages"
            assert app.index.count() == 1
            assert app.index.get("wikipedia-11675") is None
            assert app.current_page.id == "alice"

    _run(scenario)


def test_duplicate_import_reports_exists_and_keeps_user_edits(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)
            await pilot.press("i")
            await pilot.pause()

            imported = app.index.get("wikipedia-11675")
            edited = dataclasses.replace(imported, content="Hand-edited notes.")
            app.index.upsert(edited)

            await _search(pilot, app)
            await _preview(pilot, app)
            await pilot.press("i")
            await pilot.pause()

            assert app.index.count() == 2
            assert "已存在" in app.query_one("#statusbar", Static).content
            assert app.index.get("wikipedia-11675").content == "Hand-edited notes."

    _run(scenario)


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_search_network_error_is_reported_without_crashing(tmp_path):
    async def scenario():
        app = _app(tmp_path, FakeWikipedia(search_error=WikipediaTransportError("timed out")))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)

            status = app.query_one("#statusbar", Static).content
            assert "搜索失败" in status
            assert "timed out" in status
            assert "搜索失败" in app.query_one("#page-meta", Static).content
            assert app.current_page.id == "alice"

            await pilot.press("escape")
            await pilot.pause()
            assert app.mode == "pages"

    _run(scenario)


def test_article_read_error_is_reported_without_crashing(tmp_path):
    async def scenario():
        app = _app(tmp_path, FakeWikipedia(read_error=WikipediaTransportError("bad gateway")))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)

            assert app.preview_article is None
            assert "读取失败" in app.query_one("#statusbar", Static).content
            assert "bad gateway" in app.query_one("#statusbar", Static).content
            assert app.index.count() == 1

    _run(scenario)


def test_link_fetch_failure_keeps_the_article_importable(tmp_path):
    async def scenario():
        fake = FakeWikipedia(link_error=WikipediaTransportError("links unavailable"))
        app = _app(tmp_path, fake)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)

            assert app.preview_article is not None
            assert "链接抓取失败" in app.query_one("#page-meta", Static).content

            await pilot.press("i")
            await pilot.pause()

            assert app.index.get("wikipedia-11675") is not None
            assert app.index.get("wikipedia-11675").content == ARTICLE_TEXT
            status = app.query_one("#statusbar", Static).content
            assert "已导入" in status
            assert "链接抓取失败" in status

    _run(scenario)


def test_imported_links_are_navigable_with_the_existing_panels():
    async def scenario():
        index = WikiIndex()
        import_wikipedia_article(index, _article_for(2, "Beta"), links=())
        import_wikipedia_article(
            index,
            _article_for(1, "Alpha"),
            links=(WikipediaLink(title="Beta", page_id=2),),
        )
        app = ZMWikiApp(index, wikipedia=FakeWikipedia())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_page.id == "wikipedia-1"

            outbound = app.query_one("#outbound", OptionList)
            assert outbound.option_count == 1
            assert outbound.get_option_at_index(0).id == "wikipedia-2"

            outbound.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.current_page.id == "wikipedia-2"

            inbound = app.query_one("#inbound", OptionList)
            assert inbound.get_option_at_index(0).id == "wikipedia-1"

    _run(scenario)


def test_no_wikipedia_request_on_startup(tmp_path):
    async def scenario():
        fake = FakeWikipedia()
        app = _app(tmp_path, fake)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert fake.searches == []
            assert fake.reads == []

    _run(scenario)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_refuses_overwrite_until_explicitly_confirmed(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        workspace = app.workspace_path
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)
            await pilot.press("i")
            await pilot.pause()

            await pilot.press("s")
            await pilot.pause()
            assert "再次按 S" in app.query_one("#statusbar", Static).content
            assert load_workspace(workspace).get("wikipedia-11675") is None

            await pilot.press("s")
            await pilot.pause()
            assert "已保存" in app.query_one("#statusbar", Static).content

            reloaded = load_workspace(workspace)
            page = reloaded.get("wikipedia-11675")
            assert page is not None
            assert page.title == "Alan Turing"
            assert page.content == ARTICLE_TEXT
            assert page.sources == (ARTICLE_URL,)
            assert page.metadata["wikipedia_page_id"] == 11675
            assert page.metadata["wikipedia_revision_id"] == 123456

    _run(scenario)


def test_save_failure_is_reported_not_swallowed(tmp_path):
    async def scenario():
        app = _app(tmp_path, workspace_path=tmp_path / "missing" / "wiki.json")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)
            await pilot.press("i")
            await pilot.pause()

            await pilot.press("s")
            await pilot.pause()

            assert "保存失败" in app.query_one("#statusbar", Static).content
            assert not (tmp_path / "missing" / "wiki.json").exists()

    _run(scenario)


def test_save_to_new_workspace_path(tmp_path):
    async def scenario():
        new_path = tmp_path / "new-workspace.json"
        app = _app(tmp_path, workspace_path=new_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _search(pilot, app)
            await _preview(pilot, app)
            await pilot.press("i")
            await pilot.pause()

            await pilot.press("s")
            await pilot.pause()

            assert "已保存" in app.query_one("#statusbar", Static).content
            assert load_workspace(new_path).get("wikipedia-11675") is not None

    _run(scenario)
