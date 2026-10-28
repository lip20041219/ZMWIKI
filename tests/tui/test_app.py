"""Headless tests for the read-only ZMWiki terminal UI."""

import asyncio
import json
from pathlib import Path

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
            assert "得分" in app.query_one("#page-meta", Static).content

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
            assert "无匹配结果" in str(option.prompt)
            assert "无匹配" in app.query_one("#statusbar", Static).content

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
            assert "空工作区" in str(pages.get_option_at_index(0).prompt)
            assert "没有任何页面" in app.query_one("#reader", Static).content

    asyncio.run(scenario())


def test_ui_text_is_chinese_and_ask_entry_is_visible(tmp_path):
    async def scenario():
        app = _app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#search-label", Static).content == "本地搜索"
            assert app.query_one("#list-label", Static).content.startswith("页面")
            assert app.query_one("#wiki-label", Static).content == "维基百科"
            assert app.query_one("#ask-label", Static).content == "提问"
            assert "搜索页面" in app.query_one("#search", Input).placeholder
            assert "搜索维基百科" in app.query_one(
                "#wiki-search", Input
            ).placeholder
            assert "提问" in app.query_one("#ask", Input).placeholder
            status = app.query_one("#statusbar", Static).content
            assert "A 提问" in status
            assert "S 保存" in status
            assert "Q 退出" in status

            await pilot.press("a")
            await pilot.pause()
            assert app.query_one("#ask", Input).has_focus

    asyncio.run(scenario())


def test_invalid_workspace_reports_clear_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    assert main(["--workspace", str(bad)]) == 2

    captured = capsys.readouterr()
    assert "not valid JSON" in captured.err
    assert "Traceback" not in captured.err


def test_workspace_with_conflicting_paths_reports_clear_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "version": 1,
        "pages": [
            {"id": "a", "path": "same/path", "title": "A", "content": "a"},
            {"id": "b", "path": "same/path", "title": "B", "content": "b"},
        ],
    }), encoding="utf-8")

    assert main(["--workspace", str(bad)]) == 2

    captured = capsys.readouterr()
    assert "invalid page at index 1" in captured.err
    assert "Traceback" not in captured.err


def test_readme_documents_core_tui_shortcuts():
    """Documented shortcuts must stay wired to actual App bindings."""
    readme = Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text(encoding="utf-8")
    for shortcut in ("`L`", "`U`", "`Ctrl+Z`"):
        assert shortcut in text

    declared = {
        part.strip().lower()
        for binding in ZMWikiApp.BINDINGS
        for part in binding.key.split(",")
    }
    assert {"l", "u", "ctrl+z"} <= declared
