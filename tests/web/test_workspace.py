"""Tests for the read-only JSON workspace loader."""

import json
import os
from pathlib import Path

import pytest

from zmwiki.web.workspace import (
    MAX_WORKSPACE_BYTES,
    WorkspaceError,
    load_workspace,
    save_workspace,
)
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


def _page_dict(page_id: str = "alice", **overrides) -> dict:
    data = WikiPage(
        id=page_id,
        path=f"people/{page_id}",
        title=f"Title {page_id}",
        content=f"Content for {page_id}.",
        aliases=(page_id.upper(),),
        tags=("person",),
        links_to=("project/phoenix",),
    ).to_dict()
    data.update(overrides)
    return data


def _write(tmp_path, payload) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_valid_workspace(tmp_path):
    path = _write(tmp_path, {
        "version": 1,
        "pages": [_page_dict("alice"), _page_dict("bob")],
    })

    index = load_workspace(path)

    assert index.count() == 2
    page = index.get("alice")
    assert page is not None
    assert page.title == "Title alice"
    assert page.to_dict()["links_to"] == ["project/phoenix"]
    assert page.to_dict()["aliases"] == ["ALICE"]


def test_empty_workspace_is_valid(tmp_path):
    path = _write(tmp_path, {"version": 1, "pages": []})
    assert load_workspace(path).count() == 0


def test_missing_file_has_safe_message(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(WorkspaceError, match="not found") as exc_info:
        load_workspace(path)
    assert str(path) not in str(exc_info.value)


def test_invalid_json(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="not valid JSON") as exc_info:
        load_workspace(path)
    assert str(path) not in str(exc_info.value)


def test_invalid_utf8(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(WorkspaceError, match="UTF-8"):
        load_workspace(path)


def test_missing_pages(tmp_path):
    path = _write(tmp_path, {"version": 1})
    with pytest.raises(WorkspaceError, match="'pages'"):
        load_workspace(path)


def test_pages_must_be_list(tmp_path):
    path = _write(tmp_path, {"version": 1, "pages": {}})
    with pytest.raises(WorkspaceError, match="must be a list"):
        load_workspace(path)


def test_page_element_must_be_object(tmp_path):
    path = _write(tmp_path, {"version": 1, "pages": ["nope"]})
    with pytest.raises(WorkspaceError, match="must be an object"):
        load_workspace(path)


def test_missing_required_page_field(tmp_path):
    page = _page_dict()
    del page["content"]
    path = _write(tmp_path, {"version": 1, "pages": [page]})
    with pytest.raises(WorkspaceError, match="invalid page at index 0"):
        load_workspace(path)


def test_duplicate_page_id_rejected(tmp_path):
    path = _write(tmp_path, {
        "version": 1,
        "pages": [
            _page_dict("alice"),
            _page_dict("alice", path="other/alice"),
        ],
    })
    with pytest.raises(WorkspaceError, match="duplicate page id"):
        load_workspace(path)


def test_wrong_version_rejected(tmp_path):
    path = _write(tmp_path, {"version": 2, "pages": []})
    with pytest.raises(WorkspaceError, match="version must be 1"):
        load_workspace(path)


def test_boolean_version_is_not_accepted(tmp_path):
    path = _write(tmp_path, {"version": True, "pages": []})
    with pytest.raises(WorkspaceError, match="version must be 1"):
        load_workspace(path)


def test_oversized_file_rejected_without_reading(tmp_path):
    path = tmp_path / "workspace.json"
    with path.open("wb") as handle:
        handle.truncate(MAX_WORKSPACE_BYTES + 1)
    with pytest.raises(WorkspaceError, match="10 MiB"):
        load_workspace(path)


# ---------------------------------------------------------------------------
# save_workspace tests
# ---------------------------------------------------------------------------


def _build_index(*page_ids: str) -> WikiIndex:
    """Build a WikiIndex with one page per id, carrying every optional field."""
    index = WikiIndex()
    for pid in page_ids:
        index.upsert(WikiPage(
            id=pid,
            path=f"topics/{pid}",
            title=f"Title of {pid}",
            content=f"Body content for {pid}.",
            aliases=(pid.upper(), f"{pid}-alias"),
            tags=("tag-a", "tag-b"),
            description=f"Description of {pid}",
            sources=(f"src-{pid}",),
            links_to=("other/page",),
            links_from=("another/page",),
            metadata={"custom": pid, "nested": {"k": "v"}},
        ))
    return index


def _assert_pages_equal(a: WikiPage, b: WikiPage) -> None:
    """Compare two WikiPages field by field (to_dict round-trip)."""
    assert a.to_dict() == b.to_dict()


class TestSaveWorkspace:
    """save_workspace: persistence, round-trip, safety, and failure recovery."""

    def test_single_page_save_and_reload(self, tmp_path):
        index = _build_index("alice")
        path = tmp_path / "ws.json"

        save_workspace(index, path)

        assert path.exists()
        reloaded = load_workspace(path)
        assert reloaded.count() == 1
        _assert_pages_equal(index.get("alice"), reloaded.get("alice"))

    def test_multiple_pages_save_and_reload(self, tmp_path):
        index = _build_index("alice", "bob", "carol")
        path = tmp_path / "ws.json"

        save_workspace(index, path)

        reloaded = load_workspace(path)
        assert reloaded.count() == 3
        for pid in ("alice", "bob", "carol"):
            _assert_pages_equal(index.get(pid), reloaded.get(pid))

    def test_all_optional_fields_preserved(self, tmp_path):
        """links_to, links_from, sources, metadata, aliases, tags must survive."""
        index = _build_index("alice")
        path = tmp_path / "ws.json"
        save_workspace(index, path)

        page = load_workspace(path).get("alice")
        assert page.aliases == ("ALICE", "alice-alias")
        assert page.tags == ("tag-a", "tag-b")
        assert page.description == "Description of alice"
        assert page.sources == ("src-alice",)
        assert page.links_to == ("other/page",)
        assert page.links_from == ("another/page",)
        assert page.metadata == {"custom": "alice", "nested": {"k": "v"}}

    def test_empty_workspace_save_and_reload(self, tmp_path):
        index = WikiIndex()
        path = tmp_path / "empty.json"

        save_workspace(index, path)

        assert path.exists()
        reloaded = load_workspace(path)
        assert reloaded.count() == 0

    def test_default_refuses_to_overwrite_existing_file(self, tmp_path):
        path = tmp_path / "ws.json"
        original = '{"version": 1, "pages": []}'
        path.write_text(original, encoding="utf-8")

        index = _build_index("alice")
        with pytest.raises(WorkspaceError, match="already exists"):
            save_workspace(index, path)

        # Original content must be untouched.
        assert path.read_text(encoding="utf-8") == original

    def test_explicit_overwrite_replaces_content(self, tmp_path):
        path = tmp_path / "ws.json"
        path.write_text('{"version": 1, "pages": []}', encoding="utf-8")

        index = _build_index("alice")
        save_workspace(index, path, overwrite=True)

        reloaded = load_workspace(path)
        assert reloaded.count() == 1
        assert reloaded.get("alice").title == "Title of alice"

    def test_missing_target_directory_raises(self, tmp_path):
        index = _build_index("alice")
        path = tmp_path / "nonexistent" / "ws.json"

        with pytest.raises(WorkspaceError, match="directory does not exist"):
            save_workspace(index, path)

        # No file should have been created anywhere.
        assert not path.exists()
        assert not (tmp_path / "nonexistent").exists()

    def test_serialization_failure_creates_no_file(self, tmp_path, monkeypatch):
        """If json.dumps raises, no temp file or target file is created."""
        index = _build_index("alice")
        path = tmp_path / "ws.json"

        def _boom(*args, **kwargs):
            raise TypeError("intentional serialization failure")

        monkeypatch.setattr("zmwiki.web.workspace.json.dumps", _boom)

        with pytest.raises(WorkspaceError, match="serialization failed"):
            save_workspace(index, path)

        assert not path.exists()
        # No leftover temp files in the directory.
        leftovers = list(tmp_path.glob(".workspace-*.tmp"))
        assert leftovers == []

    def test_write_failure_preserves_original_and_cleans_temp(
        self, tmp_path, monkeypatch,
    ):
        """If os.replace fails, the original file stays intact and temp is cleaned."""
        path = tmp_path / "ws.json"
        original = '{"version": 1, "pages": []}'
        path.write_text(original, encoding="utf-8")

        real_replace = os.replace
        temp_files_seen = []

        def _failing_replace(src, dst):
            # Record that a temp file was created, then simulate failure.
            temp_files_seen.append(Path(src))
            raise OSError("simulated replace failure")

        monkeypatch.setattr("zmwiki.web.workspace.os.replace", _failing_replace)

        index = _build_index("alice")
        with pytest.raises(WorkspaceError, match="cannot be written"):
            save_workspace(index, path, overwrite=True)

        # Original file content must be untouched.
        assert path.read_text(encoding="utf-8") == original
        # The temp file that was created must have been cleaned up.
        for tf in temp_files_seen:
            assert not tf.exists()
        # No leftover temp files in the directory.
        assert list(tmp_path.glob(".workspace-*.tmp")) == []

    def test_saved_json_has_correct_schema(self, tmp_path):
        """The saved file must use version=1 and a pages list, loadable by existing loader."""
        index = _build_index("alice")
        path = tmp_path / "ws.json"
        save_workspace(index, path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert isinstance(payload["pages"], list)
        assert len(payload["pages"]) == 1
        assert payload["pages"][0]["id"] == "alice"
        # All 12 WikiPage fields must be present.
        for key in ("id", "path", "title", "content", "aliases", "tags",
                     "description", "sources", "links_to", "links_from", "metadata"):
            assert key in payload["pages"][0]

    def test_save_does_not_mutate_index_or_pages(self, tmp_path):
        """Saving must not modify the in-memory WikiIndex or its WikiPages."""
        index = _build_index("alice", "bob")
        before = {pid: index.get(pid).to_dict() for pid in ("alice", "bob")}
        before_count = index.count()

        path = tmp_path / "ws.json"
        save_workspace(index, path)
        save_workspace(index, path, overwrite=True)  # twice to catch side effects

        assert index.count() == before_count
        for pid in ("alice", "bob"):
            assert index.get(pid).to_dict() == before[pid]

    def test_sorted_page_order_is_deterministic(self, tmp_path):
        """Pages must be saved in sorted page_id order regardless of insertion order."""
        index = WikiIndex()
        # Insert in reverse order.
        for pid in ("charlie", "bob", "alice"):
            index.upsert(WikiPage(
                id=pid, path=f"p/{pid}", title=pid, content=f"c {pid}",
            ))
        path = tmp_path / "ws.json"
        save_workspace(index, path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = [p["id"] for p in payload["pages"]]
        assert ids == ["alice", "bob", "charlie"]
