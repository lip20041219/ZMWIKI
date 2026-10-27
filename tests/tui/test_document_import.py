"""TUI local-document import tests.

A fake compiler stands in for the real LLM so the import plumbing,
conflict handling, link resolution, and persistence are deterministic.
"""

import asyncio
import json

from textual.widgets import Input, Static

from zmwiki.tui.app import ZMWikiApp
from zmwiki.web.workspace import WorkspaceError, load_workspace
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


def _workspace(tmp_path, pages=()):
    path = tmp_path / "workspace.json"
    path.write_text(
        json.dumps({"version": 1, "pages": [p.to_dict() for p in pages]}),
        encoding="utf-8",
    )
    return path


def _app(tmp_path, compiler, *, pages=(), workspace_path=None):
    path = _workspace(tmp_path, pages) if workspace_path is None else workspace_path
    index = load_workspace(path) if path.exists() else WikiIndex()
    return ZMWikiApp(
        index,
        workspace_path=path,
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
