"""Headless tests for the read-only ZMWiki terminal UI."""

import asyncio
import json

from textual.widgets import Input, OptionList, Static

from zmwiki.tui.app import ZMWikiApp, main
from zmwiki.web.workspace import load_workspace
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


def _pages() -> list[WikiPage]:
    return [
        WikiPage(
            id="alice", path="people/alice", title="Alice Smith",
            content="Alice leads engineering.",
            tags=("person",), sources=("doc-1",),
            links_to=("project/phoenix",),
        ),
        WikiPage(
            id="project/phoenix", path="projects/phoenix",
            title="Project Phoenix",
            content="Phoenix budget is 500 thousand dollars.",
            links_from=("alice",),
        ),
        WikiPage(
            id="bob", path="people/bob", title="Bob Jones",
            content="Bob is the director of engineering.",
        ),
    ]


def _workspace(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({
        "version": 1,
        "pages": [page.to_dict() for page in _pages()],
    }), encoding="utf-8")
    return path


def _app(tmp_path) -> ZMWikiApp:
    return ZMWikiApp(load_workspace(_workspace(tmp_path)))


def test_starts_and_reads_first_page(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_page is not None
            assert app.current_page.id == "alice"
            assert app.query_one("#page-title", Static).content == "Alice Smith"
            assert "Alice leads engineering." in app.query_one("#reader", Static).content
            await pilot.press("q")
        assert app.return_code == 0

    asyncio.run(scenario())


def test_page_list_sorted_and_enter_opens(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            pages = app.query_one("#pages", OptionList)
            ids = [
                pages.get_option_at_index(i).id
                for i in range(pages.option_count)
            ]
            assert ids == ["alice", "bob", "project/phoenix"]

            pages.focus()
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()
            assert app.current_page.id == "bob"

    asyncio.run(scenario())


def test_numeric_navigation_opens_page(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("3")
            await pilot.pause()
            assert app.current_page.id == "project/phoenix"

            await pilot.press("1")
            await pilot.pause()
            assert app.current_page.id == "alice"

    asyncio.run(scenario())


def test_search_and_open_result(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            search = app.query_one("#search", Input)
            search.focus()
            search.value = "phoenix"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            assert app.mode == "search"
            pages = app.query_one("#pages", OptionList)
            assert pages.option_count == 1
            assert pages.get_option_at_index(0).id == "project/phoenix"
            assert "SCORE" in app.query_one("#page-meta", Static).content

            await pilot.press("enter")
            await pilot.pause()
            assert app.current_page.id == "project/phoenix"

    asyncio.run(scenario())


def test_search_without_matches_shows_empty_state(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            search = app.query_one("#search", Input)
            search.focus()
            search.value = "zzzz-not-found"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            pages = app.query_one("#pages", OptionList)
            assert pages.option_count == 1
            option = pages.get_option_at_index(0)
            assert option.disabled
            assert "NO MATCHES" in str(option.prompt)
            assert "NO MATCHES" in app.query_one("#statusbar", Static).content

    asyncio.run(scenario())


def test_links_are_shown_and_navigable(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            outbound = app.query_one("#outbound", OptionList)
            assert outbound.option_count == 1
            assert outbound.get_option_at_index(0).id == "project/phoenix"

            outbound.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.current_page.id == "project/phoenix"

            inbound = app.query_one("#inbound", OptionList)
            assert inbound.option_count == 1
            assert inbound.get_option_at_index(0).id == "alice"

    asyncio.run(scenario())


def test_empty_workspace_state():
    async def scenario():
        app = ZMWikiApp(WikiIndex())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_page is None
            pages = app.query_one("#pages", OptionList)
            assert pages.option_count == 1
            assert pages.get_option_at_index(0).disabled
            assert "EMPTY WORKSPACE" in str(pages.get_option_at_index(0).prompt)
            assert "no wiki pages" in app.query_one("#reader", Static).content.lower()

    asyncio.run(scenario())


def test_invalid_workspace_reports_clear_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    assert main(["--workspace", str(bad)]) == 2

    captured = capsys.readouterr()
    assert "not valid JSON" in captured.err
    assert "Traceback" not in captured.err
