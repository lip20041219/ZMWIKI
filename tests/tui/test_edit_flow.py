"""Headless tests for local page create / edit / delete and save/reload."""

import asyncio
import json

import pytest
from textual.widgets import Input, OptionList, Static, TextArea

from zmwiki.llm_generators import HttpResponse
from zmwiki.tui.app import (
    PageEditorScreen,
    ZMWikiApp,
    _load_or_create_workspace,
    main,
)
from zmwiki.web.workspace import WorkspaceError, load_workspace
from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


def _workspace(tmp_path, pages):
    path = tmp_path / "workspace.json"
    path.write_text(
        json.dumps({"version": 1, "pages": [p.to_dict() for p in pages]}),
        encoding="utf-8",
    )
    return path


def _app(tmp_path, pages, *, agent_factory=None):
    path = _workspace(tmp_path, pages)
    return ZMWikiApp(
        load_workspace(path), workspace_path=path, agent_factory=agent_factory
    )


async def _fill_editor(
    pilot,
    app,
    *,
    page_id=None,
    path,
    title,
    content,
    tags="",
    links=None,
):
    assert isinstance(app.screen, PageEditorScreen)
    if page_id is not None:
        app.screen.query_one("#edit-id", Input).value = page_id
    app.screen.query_one("#edit-path", Input).value = path
    app.screen.query_one("#edit-title", Input).value = title
    app.screen.query_one("#edit-tags", Input).value = tags
    if links is not None:
        app.screen.query_one("#edit-links", Input).value = links
    app.screen.query_one("#edit-content", TextArea).text = content
    await pilot.pause()
    await pilot.press("ctrl+s")
    await pilot.pause()


async def _save_overwrite(pilot, app):
    await pilot.press("s")
    await pilot.pause()
    assert "再次按 S" in app.query_one("#statusbar", Static).content
    await pilot.press("s")
    await pilot.pause()
    assert "已保存" in app.query_one("#statusbar", Static).content


def test_create_page_saves_and_reloads(tmp_path):
    async def scenario():
        path = tmp_path / "fresh.json"
        index = _load_or_create_workspace(path, create=True)
        app = ZMWikiApp(index, workspace_path=path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_page is None

            await pilot.press("n")
            await pilot.pause()
            await _fill_editor(
                pilot,
                app,
                page_id="carol",
                path="people/carol",
                title="Carol Chen",
                content="Carol Chen writes documentation.",
                tags="person, writer",
            )

            assert app.current_page is not None
            assert app.current_page.id == "carol"
            assert "已创建" in app.query_one("#statusbar", Static).content
            # Ctrl+S in the editor saves the page, not the workspace.
            assert load_workspace(path).count() == 0
            await _save_overwrite(pilot, app)

        reloaded = load_workspace(path)
        page = reloaded.get("carol")
        assert page is not None
        assert page.path == "people/carol"
        assert page.title == "Carol Chen"
        assert page.content == "Carol Chen writes documentation."
        assert page.tags == ("person", "writer")

    asyncio.run(scenario())


def test_edit_page_preserves_other_fields_and_saves(tmp_path):
    alice = WikiPage(
        id="alice",
        path="people/alice",
        title="Alice",
        content="Old body.",
        aliases=("A.",),
        tags=("person",),
        sources=("doc-1",),
        links_to=("bob",),
    )
    bob = WikiPage(
        id="bob",
        path="people/bob",
        title="Bob",
        content="Bob body.",
        links_from=("alice",),
    )

    async def scenario():
        app = _app(tmp_path, [alice, bob])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("E")
            await pilot.pause()

            assert isinstance(app.screen, PageEditorScreen)
            assert app.screen.query_one("#edit-id", Input).disabled
            assert app.screen.query_one("#edit-id", Input).value == "alice"

            app.screen.query_one("#edit-path", Input).value = "people/alice-smith"
            app.screen.query_one("#edit-title", Input).value = "Alice Smith"
            app.screen.query_one("#edit-content", TextArea).text = "New body."
            app.screen.query_one("#edit-tags", Input).value = "person, lead"
            await pilot.press("ctrl+s")
            await pilot.pause()

            page = app.index.get("alice")
            assert page is not None
            assert page.path == "people/alice-smith"
            assert app.index.get_by_path("people/alice") is None
            assert page.title == "Alice Smith"
            assert page.content == "New body."
            assert page.tags == ("person", "lead")
            assert page.aliases == ("A.",)
            assert page.sources == ("doc-1",)
            assert page.links_to == ("bob",)
            assert app.index.get("bob").links_from == ("alice",)
            assert "已更新" in app.query_one("#statusbar", Static).content
            await _save_overwrite(pilot, app)

        reloaded = load_workspace(app.workspace_path)
        page = reloaded.get("alice")
        assert page is not None
        assert page.path == "people/alice-smith"
        assert page.title == "Alice Smith"
        assert page.content == "New body."
        assert page.tags == ("person", "lead")
        assert page.aliases == ("A.",)
        assert page.sources == ("doc-1",)
        assert page.links_to == ("bob",)

    asyncio.run(scenario())


def test_create_and_edit_page_links_keep_backlinks_consistent(tmp_path):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )
    bob = WikiPage(
        id="bob", path="people/bob", title="Bob", content="Bob body."
    )

    async def scenario():
        app = _app(tmp_path, [alice, bob])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            await _fill_editor(
                pilot, app, page_id="carol", path="people/carol",
                title="Carol", content="Carol body.", links="alice",
            )
            assert app.index.get("alice").links_from == ("carol",)
            assert app.index.check_links().is_clean

            await pilot.press("e")
            await pilot.pause()
            app.screen.query_one("#edit-links", Input).value = "bob"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert app.index.get("alice").links_from == ()
            assert app.index.get("bob").links_from == ("carol",)
            assert app.index.check_links().is_clean
            await _save_overwrite(pilot, app)

        reloaded = load_workspace(app.workspace_path)
        assert reloaded.get("carol").links_to == ("bob",)
        assert reloaded.get("alice").links_from == ()
        assert reloaded.get("bob").links_from == ("carol",)
        assert reloaded.check_links().is_clean

    asyncio.run(scenario())


def test_delete_page_cleans_links_and_saves(tmp_path):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body.",
        links_to=("bob",),
    )
    bob = WikiPage(
        id="bob", path="people/bob", title="Bob", content="Bob body.",
        links_from=("alice",),
    )

    async def scenario():
        app = _app(tmp_path, [alice, bob])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_page.id == "alice"

            await pilot.press("d")
            await pilot.pause()
            assert "再次按 D" in app.query_one("#statusbar", Static).content
            assert app.index.get("alice") is not None

            await pilot.press("d")
            await pilot.pause()
            assert app.index.get("alice") is None
            assert app.index.get("bob").links_from == ()
            assert app.index.check_links().is_clean
            assert "已删除" in app.query_one("#statusbar", Static).content
            await _save_overwrite(pilot, app)

        reloaded = load_workspace(app.workspace_path)
        assert reloaded.get("alice") is None
        assert reloaded.get("bob").links_from == ()

    asyncio.run(scenario())


def test_delete_last_page_returns_to_empty_state_and_saves(tmp_path):
    page = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )

    async def scenario():
        app = _app(tmp_path, [page])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            assert "再次按 D" in app.query_one("#statusbar", Static).content
            await pilot.press("d")
            await pilot.pause()

            assert app.current_page is None
            assert "没有任何页面" in app.query_one("#reader", Static).content
            assert "按 N 新建页面" in app.query_one("#statusbar", Static).content
            await _save_overwrite(pilot, app)

        reloaded = load_workspace(app.workspace_path)
        assert reloaded.count() == 0

    asyncio.run(scenario())


def test_editor_rejects_duplicate_id_and_path(tmp_path):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )
    bob = WikiPage(
        id="bob", path="people/bob", title="Bob", content="Bob body."
    )

    async def scenario():
        app = _app(tmp_path, [alice, bob])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, PageEditorScreen)
            assert "页面 ID 不能为空" in app.screen.query_one(
                "#editor-error", Static
            ).content

            await _fill_editor(
                pilot, app, page_id="alice", path="people/new",
                title="Duplicate", content="Body.",
            )
            assert isinstance(app.screen, PageEditorScreen)
            assert "页面 ID 已存在" in app.screen.query_one(
                "#editor-error", Static
            ).content

            app.screen.query_one("#edit-id", Input).value = "new"
            app.screen.query_one("#edit-path", Input).value = "people/bob"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, PageEditorScreen)
            assert "路径已被页面 bob 使用" in app.screen.query_one(
                "#editor-error", Static
            ).content

            app.screen.query_one("#edit-path", Input).value = "people/new"
            app.screen.query_one("#edit-links", Input).value = "new"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "链接到自身" in app.screen.query_one(
                "#editor-error", Static
            ).content

            app.screen.query_one("#edit-links", Input).value = "missing"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "出链目标不存在" in app.screen.query_one(
                "#editor-error", Static
            ).content

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, PageEditorScreen)
            assert app.index.count() == 2

    asyncio.run(scenario())


def test_uppercase_shortcuts_open_forms_immediately(tmp_path):
    async def scenario():
        app = ZMWikiApp(WikiIndex())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#brand", Static).can_focus is False

            await pilot.press("N")
            await pilot.pause()
            assert isinstance(app.screen, PageEditorScreen)
            assert app.focused.id == "edit-id"

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, PageEditorScreen)

            await pilot.press("E")
            await pilot.pause()
            assert "无法编辑" in app.query_one("#statusbar", Static).content

            await pilot.press("D")
            await pilot.pause()
            assert "无法删除" in app.query_one("#statusbar", Static).content

    asyncio.run(scenario())


def test_letters_in_editor_do_not_trigger_global_shortcuts(tmp_path):
    page = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )

    async def scenario():
        app = _app(tmp_path, [page])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("e")
            await pilot.pause()
            assert isinstance(app.screen, PageEditorScreen)

            title = app.screen.query_one("#edit-title", Input)
            title.value = ""
            title.focus()
            await pilot.press("n", "e", "a", "d", "w", "i", "s")
            await pilot.press("N", "E", "A", "D", "W", "I", "S")
            await pilot.pause()
            assert isinstance(app.screen, PageEditorScreen)
            assert title.value == "neadwisNEADWIS"
            assert app.index.count() == 1

            content = app.screen.query_one("#edit-content", TextArea)
            content.text = ""
            content.focus()
            await pilot.press("n", "e", "a")
            await pilot.pause()
            assert isinstance(app.screen, PageEditorScreen)
            assert "nea" in content.text

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, PageEditorScreen)
            assert app.index.count() == 1

    asyncio.run(scenario())


def test_ctrl_s_saves_workspace_without_moving_focus(tmp_path):
    page = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )

    async def scenario():
        app = _app(tmp_path, [page])
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            pages = app.query_one("#pages", OptionList)
            pages.focus()

            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "再次按 S" in app.query_one("#statusbar", Static).content
            assert app.focused is pages

            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "已保存" in app.query_one("#statusbar", Static).content
            assert app.focused is pages
            assert load_workspace(app.workspace_path).get("alice") is not None

    asyncio.run(scenario())


def test_create_flag_creates_empty_workspace(tmp_path):
    path = tmp_path / "fresh.json"
    index = _load_or_create_workspace(path, create=True)
    assert index.count() == 0
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1, "pages": [],
    }

    with pytest.raises(WorkspaceError):
        _load_or_create_workspace(path, create=True)


def test_create_flag_refuses_existing_file_via_cli(tmp_path, capsys):
    path = _workspace(tmp_path, [])
    assert main(["--workspace", str(path), "--create"]) == 2
    assert "already exists" in capsys.readouterr().err


def test_create_flag_starts_with_an_empty_index(monkeypatch, tmp_path):
    path = tmp_path / "fresh.json"
    seen = {}

    def fake_run(self):
        seen["count"] = self.index.count()
        return 0

    monkeypatch.setattr(ZMWikiApp, "run", fake_run)
    assert main(["--workspace", str(path), "--create"]) == 0
    assert seen["count"] == 0
    assert path.exists()


def test_missing_workspace_suggests_create_flag(tmp_path, capsys):
    path = tmp_path / "missing.json"
    assert main(["--workspace", str(path)]) == 2
    assert "--create" in capsys.readouterr().err
    assert not path.exists()


class _Transport:
    def __init__(self, responses):
        self._responses = list(responses)

    def __call__(self, request):
        action = self._responses.pop(0)
        return HttpResponse(
            status=200,
            body=json.dumps(
                {"choices": [{"message": {"role": "assistant",
                                          "content": json.dumps(action)}}]}
            ).encode("utf-8"),
        )


def _agent_factory(responses):
    def build(index):
        return WikiAgent(
            index,
            base_url="https://api.example.com/v1",
            model="test-model",
            transport=_Transport(responses),
        )

    return build


def test_terminal_full_flow_create_save_reload_search_and_ask(tmp_path):
    async def scenario():
        path = tmp_path / "flow.json"
        responses = [
            {"action": "search", "query": "Carol"},
            {"action": "read", "paths": ["people/carol"]},
            {
                "action": "answer",
                "text": "Carol Chen writes documentation.",
                "sources": ["carol"],
            },
        ]
        app = ZMWikiApp(
            _load_or_create_workspace(path, create=True),
            workspace_path=path,
            agent_factory=_agent_factory(responses),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            await _fill_editor(
                pilot, app, page_id="carol", path="people/carol",
                title="Carol Chen", content="Carol Chen writes documentation.",
                tags="person",
            )
            await _save_overwrite(pilot, app)

        reloaded = load_workspace(path)
        app = ZMWikiApp(
            reloaded,
            workspace_path=path,
            agent_factory=_agent_factory(responses),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.current_page.id == "carol"

            search = app.query_one("#search", Input)
            search.focus()
            search.value = "Carol"
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#pages", OptionList).option_count == 1
            await pilot.press("enter")
            await pilot.pause()
            assert app.current_page.id == "carol"

            await pilot.press("a")
            await pilot.pause()
            ask = app.query_one("#ask", Input)
            ask.value = "Who writes documentation?"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert app.last_agent_result.status == AgentStatus.ANSWERED
            reader = app.query_one("#reader", Static).content
            assert "Carol Chen writes documentation." in reader
            assert "来源" in reader
            assert "Carol Chen" in reader

    asyncio.run(scenario())
