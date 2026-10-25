"""JSON workspace loading and saving for the local ZMWiki server.

The workspace is a single JSON document::

    {"version": 1, "pages": [<WikiPage.to_dict()>, ...]}

``load_workspace`` turns a validated file into an in-memory WikiIndex.
``save_workspace`` writes a WikiIndex back to disk using an atomic
temp-file-and-replace strategy, so a crash mid-write never leaves a
half-written JSON file in place of the original.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage

__all__ = ["MAX_WORKSPACE_BYTES", "WorkspaceError", "load_workspace", "save_workspace"]

SUPPORTED_VERSION = 1
MAX_WORKSPACE_BYTES = 10 * 1024 * 1024


class WorkspaceError(Exception):
    """The workspace file is missing, too large, or structurally invalid."""


def load_workspace(path: str | Path) -> WikiIndex:
    """Load one read-only workspace file into a WikiIndex."""
    workspace_path = Path(path)
    try:
        size = workspace_path.stat().st_size
    except FileNotFoundError:
        raise WorkspaceError("workspace file not found") from None
    except OSError:
        raise WorkspaceError("workspace file cannot be read") from None
    if size > MAX_WORKSPACE_BYTES:
        raise WorkspaceError("workspace file exceeds the 10 MiB limit")
    if not workspace_path.is_file():
        raise WorkspaceError("workspace path is not a regular file")

    try:
        raw = workspace_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise WorkspaceError("workspace file is not valid UTF-8") from None
    except OSError:
        raise WorkspaceError("workspace file cannot be read") from None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkspaceError(f"workspace file is not valid JSON: {exc.msg}") from None

    if not isinstance(payload, dict):
        raise WorkspaceError("workspace root must be a JSON object")
    version = payload.get("version")
    if isinstance(version, bool) or version != SUPPORTED_VERSION:
        raise WorkspaceError("workspace version must be 1")
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise WorkspaceError("workspace 'pages' must be a list")

    index = WikiIndex()
    seen_ids: set[str] = set()
    for position, item in enumerate(pages):
        if not isinstance(item, dict):
            raise WorkspaceError(
                f"workspace page at index {position} must be an object"
            )
        try:
            page = WikiPage.from_dict(item)
        except TypeError as exc:
            raise WorkspaceError(f"invalid page at index {position}: {exc}") from None
        if page.id in seen_ids:
            raise WorkspaceError(f"duplicate page id in workspace: {page.id!r}")
        seen_ids.add(page.id)
        try:
            index.upsert(page)
        except ValueError as exc:
            # Path collisions / paths that normalize to empty pass WikiPage
            # validation but are rejected by WikiIndex. Report them as a
            # workspace error so callers (CLI, TUI, server) never see a raw
            # ValueError or a traceback.
            raise WorkspaceError(
                f"invalid page at index {position}: {exc}"
            ) from None
    return index


def save_workspace(
    index: WikiIndex, path: str | Path, *, overwrite: bool = False,
) -> None:
    """Persist every page in ``index`` to a workspace JSON file.

    The output uses the same ``{"version": 1, "pages": [...]}`` schema as
    :func:`load_workspace`, so a saved file can be reloaded verbatim.

    Write safety:
      * The target directory must already exist — it is never created.
      * An existing target file is **not** overwritten unless ``overwrite=True``
        is passed explicitly.
      * Serialization and writes go to a temp file in the *same directory* as the
        target, then atomically replace the target via :func:`os.replace`. A crash
        mid-write therefore never leaves a half-written file at the target path.
      * If serialization or the temp-file write fails, the temp file is removed and
        the original target (if any) is left untouched.
      * The in-memory ``index`` and its pages are never modified.

    Raises :class:`WorkspaceError` for missing directory, existing-file collision
    (when ``overwrite`` is false), serialization failure, or write failure.
    """
    target = Path(path)
    target_dir = target.parent

    # Target directory must exist — never create arbitrary directories.
    if not target_dir.is_dir():
        raise WorkspaceError("workspace target directory does not exist")

    # Default: refuse to overwrite. Caller must opt in.
    if target.exists() and not overwrite:
        raise WorkspaceError(
            "workspace file already exists; pass overwrite=True to replace it"
        )

    # Serialize before touching the filesystem — if this fails, no file is created.
    pages = index.list_pages()
    payload = {
        "version": SUPPORTED_VERSION,
        "pages": [page.to_dict() for page in pages],
    }
    try:
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise WorkspaceError(f"workspace serialization failed: {exc}") from None

    # Atomic write: temp file in the same directory, then os.replace.
    # Same directory guarantees the rename is on one filesystem (atomic).
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=target_dir, prefix=".workspace-", suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
        # os.replace is atomic on POSIX and Windows; if target exists it is
        # replaced. If this raises, the target file is left intact.
        os.replace(tmp_path, target)
        tmp_path = None  # replace consumed the temp file
    except OSError as exc:
        # Best-effort cleanup of the temp file. Never touch the target.
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass  # cleanup failure must not mask the original error
        raise WorkspaceError("workspace file cannot be written") from None
