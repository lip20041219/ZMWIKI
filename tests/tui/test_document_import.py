"""TUI local-document import tests.

A fake compiler stands in for the real LLM so the import plumbing,
conflict handling, link resolution, and persistence are deterministic.
"""

import asyncio
import hashlib
import json
from dataclasses import replace

from textual.widgets import Input, OptionList, Static, TextArea

from zmwiki.tui.app import ZMWikiApp, _load_or_create_workspace
from zmwiki.web.workspace import (
    WorkspaceError,
    load_workspace,
    load_workspace_state,
    source_key,
)
from zmwiki.wiki.wiki_compiler import CompiledPage, WikiCompilerError
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_retriever import wiki_search
from zmwiki.wiki.wiki_schema import WikiPage


class FakeCompiler:
    def __init__(
        self,
        *,
        title="Imported Note",
        path="notes/imported",
        content=None,
        aliases=(),
        tags=(),
        description=None,
        candidate_links=(),
        error=None,
    ):
        self.title = title
        self.path = path
        self.content = content
        self.aliases = tuple(aliases)
        self.tags = tuple(tags)
        self.description = description
        self.candidate_links = tuple(candidate_links)
        self.error = error
        self.calls = []

    def compile(self, document):
        self.calls.append(document)
        if self.error is not None:
            raise self.error
        metadata = {"source_document_id": document.id}
        if self.candidate_links:
            metadata["candidate_links"] = list(self.candidate_links)
        page = WikiPage(
            id=f"page-{document.id}",
            path=self.path,
            title=self.title,
            content=self.content if self.content is not None else document.content,
            aliases=self.aliases,
            tags=self.tags,
            description=self.description,
            sources=(document.id,),
            metadata=metadata,
        )
        return CompiledPage(
            page=page,
            candidate_links=self.candidate_links,
            raw_response="{}",
        )


def _workspace(tmp_path, pages=(), sources=None):
    path = tmp_path / "workspace.json"
    payload = {"version": 1, "pages": [p.to_dict() for p in pages]}
    if sources:
        payload["sources"] = sources
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _app(tmp_path, compiler, *, pages=(), workspace_path=None):
    path = _workspace(tmp_path, pages) if workspace_path is None else workspace_path
    if path.exists():
        index, sources = load_workspace_state(path)
    else:
        index, sources = WikiIndex(), {}
    return ZMWikiApp(
        index,
        workspace_path=path,
        sources=sources,
        compiler_factory=(lambda: compiler) if compiler is not None else None,
    )


async def _import(pilot, app, path):
    doc_input = app.query_one("#doc-path", Input)
    doc_input.focus()
    doc_input.value = str(path)
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def _run(scenario):
    asyncio.run(scenario())


async def _save_editor_page(
    pilot, app, *, page_id=None, path, title, content, links=None
):
    assert app.screen.__class__.__name__ == "PageEditorScreen"
    if page_id is not None:
        app.screen.query_one("#edit-id", Input).value = page_id
    app.screen.query_one("#edit-path", Input).value = path
    app.screen.query_one("#edit-title", Input).value = title
    app.screen.query_one("#edit-content", TextArea).text = content
    if links is not None:
        app.screen.query_one("#edit-links", Input).value = links
    await pilot.pause()
    await pilot.press("ctrl+s")
    await pilot.pause()


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_import_local_txt_compiles_searches_and_persists(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text(
        "Alpha knowledge: the lantern is blue.", encoding="utf-8"
    )
    workspace = _workspace(tmp_path)
    compiler = FakeCompiler(
        title="Alpha Note",
        path="notes/alpha",
        aliases=("Alpha",),
        tags=("note",),
        description="A local note.",
    )
    app = _app(tmp_path, compiler, workspace_path=workspace)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("l")
            await pilot.pause()
            assert app.query_one("#doc-path", Input).has_focus

            await _import(pilot, app, source)

            page = app.index.get("page-local-alpha")
            assert page is not None
            assert page.path == "notes/alpha"
            assert page.title == "Alpha Note"
            assert page.content == "Alpha knowledge: the lantern is blue."
            assert page.aliases == ("Alpha",)
            assert page.tags == ("note",)
            assert page.description == "A local note."
            assert page.sources == ("local-alpha",)
            assert page.metadata["source_document_id"] == "local-alpha"
            assert app.current_page.id == "page-local-alpha"
            assert app.query_one("#doc-path", Input).value == ""
            assert "已导入" in app.query_one("#statusbar", Static).content
            assert "已保存" in app.query_one("#statusbar", Static).content

            # Immediately searchable without a restart.
            assert [r.page_id for r in wiki_search(app.index, "lantern")] == [
                "page-local-alpha"
            ]

        reloaded = load_workspace(workspace)
        page = reloaded.get("page-local-alpha")
        assert page is not None
        assert page.title == "Alpha Note"
        assert page.content == "Alpha knowledge: the lantern is blue."
        assert page.aliases == ("Alpha",)

    _run(scenario)


def test_import_reports_missing_unsupported_and_compiler_failures(tmp_path):
    workspace = _workspace(tmp_path)
    compiler = FakeCompiler()
    app = _app(tmp_path, compiler, workspace_path=workspace)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            await _import(pilot, app, tmp_path / "missing.txt")
            assert "文件不存在" in app.query_one("#statusbar", Static).content
            assert compiler.calls == []

            markdown = tmp_path / "note.md"
            markdown.write_text("# not supported", encoding="utf-8")
            await _import(pilot, app, markdown)
            assert "仅支持" in app.query_one("#statusbar", Static).content
            assert compiler.calls == []

            failing = FakeCompiler(error=WikiCompilerError("compile boom"))
            app.compiler_factory = lambda: failing
            source = tmp_path / "broken.txt"
            source.write_text("content", encoding="utf-8")
            await _import(pilot, app, source)
            status = app.query_one("#statusbar", Static).content
            assert "本地导入失败" in status
            assert "compile boom" in status
            assert app.index.count() == 0

        assert load_workspace(workspace).count() == 0

    _run(scenario)


def test_import_without_compiler_configuration_is_rejected(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("content", encoding="utf-8")
    app = _app(tmp_path, None)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            assert "本地导入不可用" in app.query_one("#statusbar", Static).content
            assert app.index.count() == 0

    _run(scenario)


def test_duplicate_import_does_not_overwrite_existing_page(tmp_path):
    existing = WikiPage(
        id="page-local-alpha",
        path="notes/original",
        title="Original",
        content="Original knowledge.",
        links_to=("missing-target",),
    )
    workspace = _workspace(tmp_path, [existing])
    original_bytes = workspace.read_bytes()
    compiler = FakeCompiler(title="Replacement", path="notes/alpha")
    app = _app(tmp_path, compiler, workspace_path=workspace)

    async def scenario():
        source = tmp_path / "alpha.txt"
        source.write_text("replacement", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            status = app.query_one("#statusbar", Static).content
            assert "页面已存在" in status
            assert "未覆盖" in status
            page = app.index.get("page-local-alpha")
            assert page.content == "Original knowledge."
            assert page.links_to == ("missing-target",)
            assert page.links_from == ()

    _run(scenario)
    assert workspace.read_bytes() == original_bytes


def test_path_conflict_does_not_overwrite_existing_page(tmp_path):
    existing = WikiPage(
        id="other",
        path="notes/imported",
        title="Existing",
        content="Existing knowledge.",
        links_to=("missing-target",),
    )
    workspace = _workspace(tmp_path, [existing])
    original_bytes = workspace.read_bytes()
    compiler = FakeCompiler(path="notes/imported")
    app = _app(tmp_path, compiler, workspace_path=workspace)

    async def scenario():
        source = tmp_path / "alpha.txt"
        source.write_text("replacement", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            status = app.query_one("#statusbar", Static).content
            assert "路径已被页面 other 使用" in status
            page = app.index.get("other")
            assert page.content == "Existing knowledge."
            assert page.links_to == ("missing-target",)
            assert page.links_from == ()
            assert app.index.get("page-local-alpha") is None

    _run(scenario)
    assert workspace.read_bytes() == original_bytes


def test_import_resolves_candidate_links_against_existing_pages(tmp_path):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )
    workspace = _workspace(tmp_path, [alice])
    compiler = FakeCompiler(candidate_links=("Alice",))
    app = _app(tmp_path, compiler, workspace_path=workspace)

    async def scenario():
        source = tmp_path / "note.txt"
        source.write_text("A note about Alice.", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)

            page = app.index.get("page-local-note")
            assert page is not None
            assert page.links_to == ("alice",)
            assert "page-local-note" in app.index.get("alice").links_from

        reloaded = load_workspace(workspace)
        page = reloaded.get("page-local-note")
        assert page is not None
        assert page.links_to == ("alice",)
        assert "page-local-note" in reloaded.get("alice").links_from

    _run(scenario)


def test_save_failure_rolls_back_the_in_memory_import(tmp_path):
    missing_dir = tmp_path / "missing" / "workspace.json"
    compiler = FakeCompiler()
    app = _app(tmp_path, compiler, workspace_path=missing_dir)

    async def scenario():
        source = tmp_path / "note.txt"
        source.write_text("content", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            status = app.query_one("#statusbar", Static).content
            assert "本地导入失败" in status
            assert "已回滚" in status
            assert app.index.count() == 0
            assert app.current_page is None

    _run(scenario)


def test_save_failure_rolls_back_links_changed_on_existing_pages(
    tmp_path, monkeypatch
):
    # The new page's title resolves a candidate link already stored on Alice.
    # A failed save must restore Alice's links and leave the disk untouched.
    alice = WikiPage(
        id="alice",
        path="people/alice",
        title="Alice",
        content="Alice body.",
        metadata={"candidate_links": ["Imported Note"]},
    )
    workspace = _workspace(tmp_path, [alice])
    original_bytes = workspace.read_bytes()
    compiler = FakeCompiler(title="Imported Note", path="notes/imported")
    app = _app(tmp_path, compiler, pages=[alice], workspace_path=workspace)

    def fail_save(*args, **kwargs):
        raise WorkspaceError("save boom")

    monkeypatch.setattr("zmwiki.tui.app.save_workspace", fail_save)

    async def scenario():
        source = tmp_path / "note.txt"
        source.write_text("A note about Alice.", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)

            status = app.query_one("#statusbar", Static).content
            assert "本地导入失败" in status
            assert "save boom" in status
            assert "已回滚" in status
            assert app.index.count() == 1
            restored = app.index.get("alice")
            assert restored is not None
            assert restored.links_to == ()
            assert restored.links_from == ()
            assert restored.metadata["candidate_links"] == ["Imported Note"]
            assert app.index.get("page-local-note") is None
            assert app.index.check_links().is_clean
            assert app.current_page.id == "alice"

            # The rollback also rebuilds the search indexes.
            assert [r.page_id for r in wiki_search(app.index, "Imported")] == []
            assert [r.page_id for r in wiki_search(app.index, "Alice")] == [
                "alice"
            ]

    _run(scenario)
    assert workspace.read_bytes() == original_bytes


def test_successful_import_persists_links_resolved_from_existing_pages(tmp_path):
    alice = WikiPage(
        id="alice",
        path="people/alice",
        title="Alice",
        content="Alice body.",
        metadata={"candidate_links": ["Imported Note"]},
    )
    workspace = _workspace(tmp_path, [alice])
    compiler = FakeCompiler(title="Imported Note", path="notes/imported")
    app = _app(tmp_path, compiler, pages=[alice], workspace_path=workspace)

    async def scenario():
        source = tmp_path / "note.txt"
        source.write_text("A note about Alice.", encoding="utf-8")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)

            updated_alice = app.index.get("alice")
            imported = app.index.get("page-local-note")
            assert updated_alice is not None
            assert imported is not None
            assert updated_alice.links_to == ("page-local-note",)
            assert imported.links_from == ("alice",)
            assert app.index.check_links().is_clean

        reloaded = load_workspace(workspace)
        assert reloaded.get("alice").links_to == ("page-local-note",)
        assert reloaded.get("page-local-note").links_from == ("alice",)
        assert reloaded.check_links().is_clean

    _run(scenario)


def test_reimport_same_source_updates_page_and_persists(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha old body with oldtoken.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    app = _app(
        tmp_path,
        FakeCompiler(title="Alpha Note", path="notes/alpha", tags=("first",)),
        workspace_path=workspace,
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            assert app.index.count() == 1

            source.write_text("Alpha new body with newtoken.", encoding="utf-8")
            app.compiler_factory = lambda: FakeCompiler(
                title="Alpha Note v2", path="notes/alpha-v2", tags=("updated",)
            )
            await _import(pilot, app, source)

            page = app.index.get("page-local-alpha")
            assert page is not None
            assert page.title == "Alpha Note v2"
            assert page.path == "notes/alpha-v2"
            assert page.content == "Alpha new body with newtoken."
            assert page.tags == ("updated",)
            assert app.index.count() == 1
            assert app.index.get_by_path("notes/alpha") is None
            assert [r.page_id for r in wiki_search(app.index, "newtoken")] == [
                "page-local-alpha"
            ]
            assert wiki_search(app.index, "oldtoken") == []
            assert "已更新" in app.query_one("#statusbar", Static).content

        reloaded = load_workspace(workspace)
        assert reloaded.count() == 1
        page = reloaded.get("page-local-alpha")
        assert page is not None
        assert page.title == "Alpha Note v2"
        assert page.path == "notes/alpha-v2"
        assert page.content == "Alpha new body with newtoken."
        assert page.tags == ("updated",)
        assert [r.page_id for r in wiki_search(reloaded, "newtoken")] == [
            "page-local-alpha"
        ]
        assert wiki_search(reloaded, "oldtoken") == []

    _run(scenario)


def test_reimport_replaces_stale_links_and_backlinks(tmp_path):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )
    bob = WikiPage(
        id="bob", path="people/bob", title="Bob", content="Bob body."
    )
    workspace = _workspace(tmp_path, [alice, bob])
    source = tmp_path / "note.txt"
    source.write_text("A note.", encoding="utf-8")
    app = _app(
        tmp_path,
        FakeCompiler(candidate_links=("Alice",)),
        workspace_path=workspace,
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            note = app.index.get("page-local-note")
            assert note.links_to == ("alice",)
            assert "page-local-note" in app.index.get("alice").links_from

            app.compiler_factory = lambda: FakeCompiler(
                candidate_links=("Bob",)
            )
            await _import(pilot, app, source)

            note = app.index.get("page-local-note")
            assert note.links_to == ("bob",)
            assert app.index.get("alice").links_from == ()
            assert app.index.get("bob").links_from == ("page-local-note",)
            assert app.index.check_links().is_clean

        reloaded = load_workspace(workspace)
        assert reloaded.get("page-local-note").links_to == ("bob",)
        assert reloaded.get("alice").links_from == ()
        assert reloaded.get("bob").links_from == ("page-local-note",)
        assert reloaded.check_links().is_clean

    _run(scenario)


def test_failed_reimport_restores_previous_page_and_links(
    tmp_path, monkeypatch
):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )
    bob = WikiPage(
        id="bob", path="people/bob", title="Bob", content="Bob body."
    )
    workspace = _workspace(tmp_path, [alice, bob])
    source = tmp_path / "note.txt"
    source.write_text("old source", encoding="utf-8")
    app = _app(
        tmp_path,
        FakeCompiler(content="oldtoken body", candidate_links=("Alice",)),
        workspace_path=workspace,
    )
    saved_bytes = []

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            saved_bytes.append(workspace.read_bytes())
            assert app.index.get("page-local-note").links_to == ("alice",)

            def fail_save(*args, **kwargs):
                raise WorkspaceError("save boom")

            monkeypatch.setattr("zmwiki.tui.app.save_workspace", fail_save)
            app.compiler_factory = lambda: FakeCompiler(
                content="newtoken body", candidate_links=("Bob",)
            )
            await _import(pilot, app, source)

            status = app.query_one("#statusbar", Static).content
            assert "本地导入失败" in status
            assert "save boom" in status
            assert "已回滚" in status
            restored = app.index.get("page-local-note")
            assert restored is not None
            assert restored.content == "oldtoken body"
            assert restored.links_to == ("alice",)
            assert app.index.get("alice").links_from == ("page-local-note",)
            assert app.index.get("bob").links_from == ()
            assert app.index.check_links().is_clean
            assert wiki_search(app.index, "newtoken") == []
            assert [r.page_id for r in wiki_search(app.index, "oldtoken")] == [
                "page-local-note"
            ]

    _run(scenario)
    assert workspace.read_bytes() == saved_bytes[0]


def test_import_registers_source_and_reimport_maps_to_same_page(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha old body.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    compiler = FakeCompiler(title="Alpha Note", path="notes/alpha")
    app = _app(tmp_path, compiler, workspace_path=workspace)
    key = source_key(source)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)

            assert app.sources == {
                key: {
                    "page_id": "page-local-alpha",
                    "document_id": "local-alpha",
                    "fingerprint": _fingerprint(source),
                }
            }

            source.write_text("Alpha new body.", encoding="utf-8")
            await _import(pilot, app, source)

            assert app.index.count() == 1
            page = app.index.get("page-local-alpha")
            assert page.content == "Alpha new body."
            assert "已更新" in app.query_one("#statusbar", Static).content
            assert app.sources[key]["fingerprint"] == _fingerprint(source)

        index, sources = load_workspace_state(workspace)
        assert index.count() == 1
        assert sources[key]["page_id"] == "page-local-alpha"

    _run(scenario)


def test_renamed_source_reuses_registered_page(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Stable body.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    compiler = FakeCompiler(title="Alpha Note", path="notes/alpha")
    app = _app(tmp_path, compiler, workspace_path=workspace)
    old_key = source_key(source)
    renamed = tmp_path / "renamed.txt"

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            assert app.index.count() == 1

            source.rename(renamed)
            await _import(pilot, app, renamed)

            assert app.index.count() == 1
            assert app.index.get("page-local-alpha") is not None
            assert app.index.get("page-local-renamed") is None
            assert "已更新" in app.query_one("#statusbar", Static).content
            assert old_key not in app.sources
            assert source_key(renamed) in app.sources

        index, sources = load_workspace_state(workspace)
        assert index.count() == 1
        assert index.get("page-local-alpha") is not None
        assert set(sources) == {source_key(renamed)}
        assert sources[source_key(renamed)]["document_id"] == "local-alpha"

    _run(scenario)


def test_same_stem_from_another_path_does_not_take_over_page(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Original body.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    compiler = FakeCompiler(title="Alpha Note", path="notes/alpha")
    app = _app(tmp_path, compiler, workspace_path=workspace)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            original_bytes = workspace.read_bytes()

            sub = tmp_path / "sub"
            sub.mkdir()
            copy = sub / "alpha.txt"
            copy.write_bytes(source.read_bytes())
            await _import(pilot, app, copy)

            status = app.query_one("#statusbar", Static).content
            assert "页面已存在" in status
            assert "由其他源文件管理" in status
            assert app.index.count() == 1
            assert app.index.get("page-local-alpha").content == "Original body."
            assert workspace.read_bytes() == original_bytes

    _run(scenario)


def test_source_sync_removes_page_after_source_file_deleted(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha body.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    app = _app(
        tmp_path,
        FakeCompiler(title="Alpha Note", path="notes/alpha"),
        workspace_path=workspace,
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            assert app.index.count() == 1

            source.unlink()
            app.query_one("#pages", OptionList).focus()
            await pilot.press("u")
            await pilot.pause()

            status = app.query_one("#statusbar", Static).content
            assert "源同步" in status
            assert "清理 1 页" in status
            assert "已保存" in status
            assert app.index.count() == 0
            assert app.sources == {}
            assert app.current_page is None

        index, sources = load_workspace_state(workspace)
        assert index.count() == 0
        assert sources == {}

    _run(scenario)


def test_source_sync_keeps_page_not_owned_by_source(tmp_path):
    manual = WikiPage(
        id="manual", path="notes/manual", title="Manual", content="Hand written."
    )
    gone = tmp_path / "gone.txt"
    registry = {
        source_key(gone): {
            "page_id": "manual",
            "document_id": "local-gone",
            "fingerprint": "deadbeef",
        }
    }
    workspace = _workspace(tmp_path, [manual], registry)
    app = _app(tmp_path, None, workspace_path=workspace)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.query_one("#pages", OptionList).focus()
            await pilot.press("u")
            await pilot.pause()

            status = app.query_one("#statusbar", Static).content
            assert "保护 1 页" in status
            assert "来源不匹配" in status
            assert "清理 0 页" in status
            assert app.index.get("manual") is not None
            assert list(app.sources) == [source_key(gone)]

    _run(scenario)


def test_reimport_preserves_manual_links_and_drops_generated_ones(tmp_path):
    alice = WikiPage(
        id="alice", path="people/alice", title="Alice", content="Alice body."
    )
    bob = WikiPage(
        id="bob", path="people/bob", title="Bob", content="Bob body."
    )
    source = tmp_path / "note.txt"
    source.write_text("A note.", encoding="utf-8")
    workspace = _workspace(tmp_path, [alice, bob])
    app = _app(
        tmp_path,
        FakeCompiler(
            title="Imported Note", path="notes/imported",
            candidate_links=("Alice",),
        ),
        workspace_path=workspace,
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            note = app.index.get("page-local-note")
            assert note.links_to == ("alice",)
            assert note.metadata["generated_links"] == ["alice"]

            # The editor keeps metadata, so this mirrors a hand-added link.
            app._apply_page_edit(
                note, replace(note, links_to=("alice", "bob"))
            )

            app.compiler_factory = lambda: FakeCompiler(
                title="Imported Note", path="notes/imported"
            )
            await _import(pilot, app, source)

            note = app.index.get("page-local-note")
            assert note.links_to == ("bob",)
            assert note.metadata["generated_links"] == []
            assert app.index.get("alice").links_from == ()
            assert app.index.get("bob").links_from == ("page-local-note",)
            assert app.index.check_links().is_clean

        reloaded, _ = load_workspace_state(workspace)
        assert reloaded.get("page-local-note").links_to == ("bob",)
        assert reloaded.get("bob").links_from == ("page-local-note",)
        assert reloaded.check_links().is_clean

    _run(scenario)


def test_failed_reimport_restores_registry(tmp_path, monkeypatch):
    source = tmp_path / "alpha.txt"
    source.write_text("old body", encoding="utf-8")
    workspace = _workspace(tmp_path)
    app = _app(
        tmp_path,
        FakeCompiler(title="Alpha Note", path="notes/alpha"),
        workspace_path=workspace,
    )
    key = source_key(source)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            saved_bytes = workspace.read_bytes()
            saved_sources = dict(app.sources)
            assert saved_sources[key]["fingerprint"] == _fingerprint(source)

            source.write_text("new body", encoding="utf-8")

            def fail_save(*args, **kwargs):
                raise WorkspaceError("save boom")

            monkeypatch.setattr("zmwiki.tui.app.save_workspace", fail_save)
            await _import(pilot, app, source)

            status = app.query_one("#statusbar", Static).content
            assert "save boom" in status
            assert "已回滚" in status
            assert app.sources == saved_sources
            assert app.index.get("page-local-alpha").content == "old body"
            assert workspace.read_bytes() == saved_bytes

    _run(scenario)


def test_save_without_explicit_sources_preserves_existing_registry(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha body.", encoding="utf-8")
    page = WikiPage(
        id="page-local-alpha",
        path="notes/alpha",
        title="Alpha Note",
        content="Alpha body.",
        sources=("local-alpha",),
        metadata={"source_document_id": "local-alpha"},
    )
    registry = {
        source_key(source): {
            "page_id": "page-local-alpha",
            "document_id": "local-alpha",
            "fingerprint": _fingerprint(source),
        }
    }
    workspace = _workspace(tmp_path, [page], registry)
    app = ZMWikiApp(load_workspace(workspace), workspace_path=workspace)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.sources == registry

            app.query_one("#pages", OptionList).focus()
            await pilot.press("s")  # arm overwrite
            await pilot.press("s")  # write
            await pilot.pause()
            assert "已保存" in app.query_one("#statusbar", Static).content

        index, sources = load_workspace_state(workspace)
        assert sources == registry
        assert index.get("page-local-alpha") is not None

    _run(scenario)


def test_explicit_sources_registry_records_new_and_updated_entries(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha body.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    app = ZMWikiApp(
        load_workspace(workspace),
        workspace_path=workspace,
        sources={},
        compiler_factory=lambda: FakeCompiler(
            title="Alpha Note", path="notes/alpha"
        ),
    )
    key = source_key(source)

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            assert app.sources[key]["document_id"] == "local-alpha"

            source.write_text("Alpha v2 body.", encoding="utf-8")
            await _import(pilot, app, source)

            assert app.sources[key]["fingerprint"] == _fingerprint(source)
            assert app.index.count() == 1
            assert "已更新" in app.query_one("#statusbar", Static).content

    _run(scenario)
    assert load_workspace_state(workspace)[1] == app.sources


def test_reimport_after_restart_reuses_page_and_document_id(tmp_path):
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha body.", encoding="utf-8")
    workspace = _workspace(tmp_path)

    def compiler_factory():
        return FakeCompiler(title="Alpha Note", path="notes/alpha")

    first = ZMWikiApp(
        load_workspace(workspace),
        workspace_path=workspace,
        sources={},
        compiler_factory=compiler_factory,
    )

    async def scenario():
        async with first.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, first, source)
            assert first.index.get("page-local-alpha") is not None

        index, sources = load_workspace_state(workspace)
        second = ZMWikiApp(
            index,
            workspace_path=workspace,
            sources=sources,
            compiler_factory=compiler_factory,
        )
        async with second.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, second, source)

            assert second.index.count() == 1
            assert second.index.get("page-local-alpha") is not None
            assert "已更新" in second.query_one("#statusbar", Static).content
            assert (
                second.sources[source_key(source)]["document_id"]
                == "local-alpha"
            )

    _run(scenario)


def test_source_sync_save_failure_rolls_back_all_state(
    tmp_path, monkeypatch
):
    alpha = tmp_path / "alpha.txt"
    alpha.write_text("Alpha body with alphatoken.", encoding="utf-8")
    beta = tmp_path / "beta.txt"
    beta.write_text("Beta body with betatoken.", encoding="utf-8")
    workspace = _workspace(tmp_path)
    app = _app(
        tmp_path,
        FakeCompiler(title="Alpha Note", path="notes/alpha"),
        workspace_path=workspace,
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, alpha)
            app.compiler_factory = lambda: FakeCompiler(
                title="Beta Note", path="notes/beta"
            )
            await _import(pilot, app, beta)
            assert app.index.count() == 2
            saved_bytes = workspace.read_bytes()
            saved_sources = dict(app.sources)

            alpha.unlink()
            beta.unlink()

            def fail_save(*args, **kwargs):
                raise WorkspaceError("save boom")

            monkeypatch.setattr("zmwiki.tui.app.save_workspace", fail_save)
            app.query_one("#pages", OptionList).focus()
            await pilot.press("u")
            await pilot.pause()

            status = app.query_one("#statusbar", Static).content
            assert "源同步失败" in status
            assert "save boom" in status
            assert "已回滚" in status
            assert app.index.count() == 2
            assert app.index.get("page-local-alpha") is not None
            assert app.index.get("page-local-beta") is not None
            assert app.sources == saved_sources
            assert [
                r.page_id for r in wiki_search(app.index, "alphatoken")
            ] == ["page-local-alpha"]
            assert [
                r.page_id for r in wiki_search(app.index, "betatoken")
            ] == ["page-local-beta"]
            assert app.index.check_links().is_clean
            assert workspace.read_bytes() == saved_bytes

    _run(scenario)


def test_full_tui_workflow_survives_restart_and_source_cleanup(tmp_path):
    workspace = tmp_path / "workspace.json"
    alpha = tmp_path / "alpha.txt"
    alpha.write_text("Alpha body has alphatoken.", encoding="utf-8")
    beta = tmp_path / "beta.txt"
    beta.write_text("Beta body has betatoken.", encoding="utf-8")

    index, sources = _load_or_create_workspace(workspace, create=True)
    app = ZMWikiApp(
        index,
        workspace_path=workspace,
        sources=sources,
        compiler_factory=lambda: FakeCompiler(
            title="Alpha Note", path="notes/alpha"
        ),
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, alpha)
            app.compiler_factory = lambda: FakeCompiler(
                title="Beta Note",
                path="notes/beta",
                candidate_links=("Alpha Note",),
            )
            await _import(pilot, app, beta)

            # Imported page is readable and searchable, links resolved.
            assert app.current_page.id == "page-local-beta"
            assert (
                "Beta body has betatoken."
                in app.query_one("#reader", Static).content
            )
            assert [
                r.page_id for r in wiki_search(app.index, "alphatoken")
            ] == ["page-local-alpha"]
            assert app.index.get("page-local-beta").links_to == (
                "page-local-alpha",
            )
            assert app.index.get("page-local-alpha").links_from == (
                "page-local-beta",
            )

            # Edit the open page, undo once, then edit it again for keeps.
            app.query_one("#pages", OptionList).focus()
            await pilot.press("e")
            await pilot.pause()
            await _save_editor_page(
                pilot,
                app,
                path="notes/beta",
                title="Beta Note v2",
                content="Beta body has betatoken v2.",
            )
            assert app.index.get("page-local-beta").title == "Beta Note v2"

            await pilot.press("ctrl+z")
            await pilot.pause()
            assert app.index.get("page-local-beta").title == "Beta Note"
            assert (
                app.index.get("page-local-beta").content
                == "Beta body has betatoken."
            )

            await pilot.press("e")
            await pilot.pause()
            await _save_editor_page(
                pilot,
                app,
                path="notes/beta",
                title="Beta Note v2",
                content="Beta body has betatoken v2.",
            )

            # Create a manual page; it has no source registration.
            await pilot.press("n")
            await pilot.pause()
            await _save_editor_page(
                pilot,
                app,
                page_id="gamma",
                path="notes/gamma",
                title="Gamma",
                content="Gamma body has gammatoken.",
            )
            assert app.index.get("gamma") is not None

            await pilot.press("s")
            await pilot.pause()
            assert "再次按 S" in app.query_one("#statusbar", Static).content
            await pilot.press("s")
            await pilot.pause()
            assert "已保存" in app.query_one("#statusbar", Static).content

        # Restart from disk.
        index2, sources2 = load_workspace_state(workspace)
        assert index2.count() == 3
        assert set(sources2) == {source_key(alpha), source_key(beta)}

        app2 = ZMWikiApp(
            index2, workspace_path=workspace, sources=sources2
        )
        async with app2.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app2.index.get("page-local-alpha").content == (
                "Alpha body has alphatoken."
            )
            beta_page = app2.index.get("page-local-beta")
            assert beta_page.title == "Beta Note v2"
            assert beta_page.content == "Beta body has betatoken v2."
            assert beta_page.links_to == ("page-local-alpha",)
            assert app2.index.get("page-local-alpha").links_from == (
                "page-local-beta",
            )
            assert app2.index.get("gamma").content == (
                "Gamma body has gammatoken."
            )
            assert app2.index.check_links().is_clean
            assert [
                r.page_id for r in wiki_search(app2.index, "betatoken")
            ] == ["page-local-beta"]
            assert [
                r.page_id for r in wiki_search(app2.index, "gammatoken")
            ] == ["gamma"]

            # Source cleanup after restart: only the managed page goes.
            alpha.unlink()
            app2.query_one("#pages", OptionList).focus()
            await pilot.press("u")
            await pilot.pause()

            status = app2.query_one("#statusbar", Static).content
            assert "清理 1 页" in status
            assert "已保存" in status
            assert app2.index.get("page-local-alpha") is None
            assert app2.index.get("page-local-beta") is not None
            assert app2.index.get("gamma") is not None
            assert app2.index.get("page-local-beta").links_to == ()
            assert app2.index.check_links().is_clean
            assert set(app2.sources) == {source_key(beta)}

        # Second restart sees the cleaned state.
        index3, sources3 = load_workspace_state(workspace)
        assert index3.count() == 2
        assert index3.get("page-local-alpha") is None
        assert index3.get("page-local-beta") is not None
        assert index3.get("gamma") is not None
        assert index3.get("page-local-beta").links_to == ()
        assert set(sources3) == {source_key(beta)}
        assert index3.check_links().is_clean
        assert [
            r.page_id for r in wiki_search(index3, "betatoken")
        ] == ["page-local-beta"]

    _run(scenario)


def test_failures_leave_a_restartable_workspace(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    alpha = tmp_path / "alpha.txt"
    alpha.write_text("Alpha body has alphatoken.", encoding="utf-8")
    beta = tmp_path / "beta.txt"
    beta.write_text("Beta body has betatoken.", encoding="utf-8")
    app = _app(
        tmp_path,
        FakeCompiler(title="Alpha Note", path="notes/alpha"),
        workspace_path=workspace,
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, alpha)
            assert app.index.count() == 1
            saved_bytes = workspace.read_bytes()

            def fail_save(*args, **kwargs):
                raise WorkspaceError("save boom")

            monkeypatch.setattr("zmwiki.tui.app.save_workspace", fail_save)

            # Failed import: rolled back in memory and on disk.
            app.compiler_factory = lambda: FakeCompiler(
                title="Beta Note", path="notes/beta"
            )
            await _import(pilot, app, beta)
            status = app.query_one("#statusbar", Static).content
            assert "已回滚" in status
            assert app.index.count() == 1
            assert app.index.get("page-local-beta") is None
            assert workspace.read_bytes() == saved_bytes

            # Failed source sync: page and registration restored.
            alpha.unlink()
            app.query_one("#pages", OptionList).focus()
            await pilot.press("u")
            await pilot.pause()
            status = app.query_one("#statusbar", Static).content
            assert "源同步失败" in status
            assert "已回滚" in status
            assert app.index.get("page-local-alpha") is not None
            assert app.sources[source_key(alpha)]["page_id"] == (
                "page-local-alpha"
            )
            assert workspace.read_bytes() == saved_bytes

    _run(scenario)

    # Restart from the untouched file: everything still loads and searches.
    index, sources = load_workspace_state(workspace)
    assert index.count() == 1
    assert index.get("page-local-alpha") is not None
    assert set(sources) == {source_key(alpha)}
    assert [
        r.page_id for r in wiki_search(index, "alphatoken")
    ] == ["page-local-alpha"]
    assert index.check_links().is_clean


def test_local_import_into_demo_workspace_stays_in_memory(
    tmp_path, monkeypatch
):
    demo = tmp_path / "demo_workspace.json"
    demo.write_text(
        json.dumps({"version": 1, "pages": []}), encoding="utf-8"
    )
    monkeypatch.setattr("zmwiki.tui.app._DEFAULT_WORKSPACE", demo)
    source = tmp_path / "alpha.txt"
    source.write_text("Alpha body.", encoding="utf-8")
    original = demo.read_bytes()
    app = ZMWikiApp(
        load_workspace(demo),
        workspace_path=demo,
        compiler_factory=lambda: FakeCompiler(
            title="Alpha Note", path="notes/alpha"
        ),
    )

    async def scenario():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await _import(pilot, app, source)
            assert app.index.get("page-local-alpha") is not None
            status = app.query_one("#statusbar", Static).content
            assert "已导入" in status
            assert "演示工作区只读" in status

    _run(scenario)
    assert demo.read_bytes() == original
