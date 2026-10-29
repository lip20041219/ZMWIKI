"""Local HTTP service and browser workbench for ZMWiki.

Start with::

    python -m zmwiki.web.server --workspace path/to/wiki.json --port 8000

The server binds 127.0.0.1 only. It serves the marketing page at ``/``, the
wiki workbench at ``/app``, read-only page/search endpoints, write endpoints
for page create/update/delete plus explicit workspace save, optional browser
document import, browser model setup with a connection test, and an optional
agent Q&A endpoint. It uses only the Python standard library; no third-party
dependencies. Model calls happen only when ``/api/ask`` or ``/api/import`` is
used. The API key is held server-side only and is never returned by any route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from zmwiki.file_ingestion import PlainTextFileIngestor
from zmwiki.schema import Document
from zmwiki.wiki.wiki_agent import AgentResult, AgentStatus, WikiAgent
from zmwiki.wiki.wiki_compiler import (
    CompilerResponseError,
    CompilerTransportError,
    CompilerValidationError,
    WikiCompiler,
    WikiCompilerError,
)
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_link_resolver import LinkResolutionReport, resolve_candidate_links
from zmwiki.wiki.wiki_retriever import wiki_search
from zmwiki.wiki.wiki_schema import WikiPage

from .model_settings import (
    DEFAULT_SETTINGS_PATH,
    ConfigError,
    ConnectionTestResult,
    ModelConfig,
    SettingsStore,
    make_config,
    parse_config_payload,
    test_model_connection,
)
from .workspace import (
    MAX_WORKSPACE_BYTES,
    SourceRegistry,
    WorkspaceError,
    load_workspace_state,
    save_workspace,
    source_key,
)

__all__ = [
    "HOST",
    "DEFAULT_PORT",
    "MAX_REQUEST_BODY_BYTES",
    "ZMWikiServer",
    "make_server",
    "main",
]

HOST = "127.0.0.1"
DEFAULT_PORT = 8000

MAX_QUERY_CHARS = 200
MAX_TOP_K = 50
DEFAULT_TOP_K = 5
MAX_QUERY_STRING_CHARS = 2048
MAX_PAGE_ID_CHARS = 512
MAX_QUESTION_CHARS = 4000
REQUEST_TIMEOUT_SECONDS = 10
MAX_REQUEST_BODY_BYTES = MAX_WORKSPACE_BYTES
MAX_IMPORT_BYTES = 2 * 1024 * 1024
MAX_IMPORT_FILENAME_CHARS = 200
UPLOAD_SOURCE_PREFIX = "upload:"

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app": "app.html",
    "/app.html": "app.html",
    "/app.css": "app.css",
    "/app.js": "app.js",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}

# The bundled demo workspace is treated as read-only, matching the TUI.
_DEMO_WORKSPACE = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "corpus"
    / "wiki_workspace.json"
)


class ZMWikiServer(ThreadingHTTPServer):
    """Threaded server holding the in-memory wiki shared by all requests."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        index: WikiIndex,
        *,
        workspace_path: str | Path | None = None,
        sources: SourceRegistry | None = None,
        read_only: bool = False,
        agent_factory: Callable[[WikiIndex], WikiAgent] | None = None,
        compiler_factory: Callable[[], WikiCompiler] | None = None,
        model_config: ModelConfig | None = None,
        settings_store: SettingsStore | None = None,
        connection_tester: Callable[[ModelConfig], ConnectionTestResult] | None = None,
        config_source: str = "none",
    ) -> None:
        self.index = index
        self.workspace_path = (
            Path(workspace_path) if workspace_path is not None else None
        )
        self.sources: SourceRegistry = dict(sources or {})
        self.read_only = bool(read_only)
        # Injected factories keep tests and embedding callers in control; when
        # absent the server builds agents/compilers from the active config,
        # which is either CLI-provided, loaded from the settings file, or set
        # through /api/config.
        self.agent_factory = agent_factory
        self.compiler_factory = compiler_factory
        self.model_config = model_config
        self.settings_store = settings_store
        self.connection_tester = connection_tester
        self.config_source = config_source
        # WikiIndex has no internal locking; serialize requests so readers and
        # writers never observe a partially-updated index.
        self.lock = threading.RLock()
        super().__init__(address, ZMWikiHandler)

    def agent_available(self) -> bool:
        return self.agent_factory is not None or self.model_config is not None

    def compiler_available(self) -> bool:
        return self.compiler_factory is not None or self.model_config is not None

    def build_agent(self, index: WikiIndex) -> WikiAgent | None:
        if self.agent_factory is not None:
            return self.agent_factory(index)
        config = self.model_config
        if config is None:
            return None
        return WikiAgent(
            index,
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
        )

    def build_compiler(self) -> WikiCompiler | None:
        if self.compiler_factory is not None:
            return self.compiler_factory()
        config = self.model_config
        if config is None:
            return None
        return WikiCompiler(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
        )


class ZMWikiHandler(BaseHTTPRequestHandler):
    """GET reads pages; POST/PUT/DELETE mutate pages and save the workspace."""

    server_version = "ZMWiki/0.2"
    protocol_version = "HTTP/1.1"
    timeout = REQUEST_TIMEOUT_SECONDS

    @property
    def index(self) -> WikiIndex:
        return self.server.index  # type: ignore[attr-defined]

    @property
    def workspace_path(self) -> Path | None:
        return self.server.workspace_path  # type: ignore[attr-defined]

    @property
    def sources(self) -> SourceRegistry:
        return self.server.sources  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        self._run(self._handle_get)

    def do_POST(self) -> None:
        target = self._write_target()
        if target is None:
            return
        if target.path == "/api/ask":
            # The agent call must not hold the index lock; it snapshots under
            # the lock and then runs the model loop outside it.
            self._run(self._handle_ask, lock=False)
        elif target.path == "/api/import":
            # ponytail: import keeps the index lock during the model call;
            # compile outside the lock if browsing must stay live meanwhile.
            self._run(self._handle_import, target.query)
        elif target.path == "/api/config":
            self._run(self._handle_save_config)
        elif target.path == "/api/config/test":
            # The connection test is a network call; keep the index lock free.
            self._run(self._handle_test_config, lock=False)
        elif target.path == "/api/pages":
            self._run(self._handle_create_page)
        elif target.path == "/api/workspace/save":
            self._run(self._handle_save_workspace)
        else:
            self.close_connection = True
            self._method_not_allowed()

    def do_PUT(self) -> None:
        self._run(self._handle_put)

    def do_DELETE(self) -> None:
        self._run(self._handle_delete)

    def _run(self, handler, *args, lock: bool = True) -> None:  # type: ignore[no-untyped-def]
        try:
            if lock:
                with self.server.lock:  # type: ignore[attr-defined]
                    handler(*args)
            else:
                handler(*args)
        except Exception as exc:
            # Never send exception text or a traceback to the client.
            self.log_error("unhandled error serving request: %s", type(exc).__name__)
            self.close_connection = True
            try:
                self._send_error(500, "internal_error", "internal server error")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # GET routes
    # ------------------------------------------------------------------

    def _handle_get(self) -> None:
        try:
            target = urlsplit(self.path)
        except ValueError:
            self._send_error(400, "invalid_request", "invalid request target")
            return

        if len(target.query) > MAX_QUERY_STRING_CHARS:
            self._send_error(400, "invalid_query", "query string is too long")
            return
        try:
            path = unquote(target.path, errors="strict")
            params = parse_qs(
                target.query, keep_blank_values=True, max_num_fields=10
            )
        except (UnicodeDecodeError, ValueError):
            self._send_error(400, "invalid_request", "invalid request target")
            return
        if "\x00" in path:
            self._send_error(404, "not_found", "not found")
            return

        if path == "/api/health":
            self._send_json(200, {"status": "ok", "pages": self.index.count()})
        elif path == "/api/capabilities":
            self._send_json(200, {
                "read_only": bool(self.server.read_only),  # type: ignore[attr-defined]
                "writable": self._workspace_writable(),
                "workspace_configured": self.workspace_path is not None,
                "agent_configured": self.server.agent_available(),  # type: ignore[attr-defined]
                "import_configured": self.server.compiler_available(),  # type: ignore[attr-defined]
            })
        elif path == "/api/config":
            self._handle_get_config()
        elif path == "/api/pages":
            self._handle_pages()
        elif path == "/api/page":
            self._handle_page(params)
        elif path == "/api/search":
            self._handle_search(params)
        elif path.startswith("/api/"):
            self._send_error(404, "not_found", "not found")
        else:
            self._handle_static(path)

    def _handle_pages(self) -> None:
        pages = [
            {
                "id": page.id,
                "path": page.path,
                "title": page.title,
                "aliases": list(page.aliases),
                "tags": list(page.tags),
                "description": page.description,
            }
            for page in self.index.list_pages()
        ]
        self._send_json(200, {"pages": pages})

    def _handle_page(self, params: dict[str, list[str]]) -> None:
        page_id = _single_param(params, "id")
        if page_id is None or not page_id.strip() or len(page_id) > MAX_PAGE_ID_CHARS:
            self._send_error(400, "invalid_id", "query parameter 'id' is required")
            return
        page = self.index.get(page_id)
        if page is None:
            self._send_error(404, "not_found", "page not found")
            return
        self._send_json(200, page.to_dict())

    def _handle_search(self, params: dict[str, list[str]]) -> None:
        query = _single_param(params, "q")
        if query is None or not query.strip():
            self._send_error(400, "invalid_query", "query parameter 'q' is required")
            return
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            self._send_error(
                400, "invalid_query",
                f"query parameter 'q' must be at most {MAX_QUERY_CHARS} characters",
            )
            return

        if "top_k" not in params:
            top_k = DEFAULT_TOP_K
        else:
            raw_top_k = _single_param(params, "top_k")
            try:
                top_k = int(raw_top_k) if raw_top_k is not None else 0
            except ValueError:
                self._send_error(400, "invalid_top_k", "top_k must be an integer")
                return
        if top_k < 1 or top_k > MAX_TOP_K:
            self._send_error(
                400, "invalid_top_k", f"top_k must be between 1 and {MAX_TOP_K}"
            )
            return

        results = wiki_search(self.index, query, top_k=top_k)
        self._send_json(200, {
            "query": query,
            "top_k": top_k,
            "results": [asdict(result) for result in results],
        })

    # ------------------------------------------------------------------
    # Write routes
    # ------------------------------------------------------------------

    def _handle_put(self) -> None:
        target = self._write_target()
        if target is None:
            return
        if target.path != "/api/page":
            self.close_connection = True
            self._method_not_allowed()
            return
        params = self._query_params(target.query)
        if params is None:
            return
        page_id = _single_param(params, "id")
        if not self._valid_page_id(page_id):
            return
        self._handle_update_page(page_id)  # type: ignore[arg-type]

    def _handle_delete(self) -> None:
        target = self._write_target()
        if target is None:
            return
        if target.path != "/api/page":
            self.close_connection = True
            self._method_not_allowed()
            return
        params = self._query_params(target.query)
        if params is None:
            return
        page_id = _single_param(params, "id")
        if not self._valid_page_id(page_id):
            return
        self._handle_delete_page(page_id)  # type: ignore[arg-type]

    def _handle_create_page(self) -> None:
        if not self._require_writable():
            return
        payload = self._read_json_object()
        if payload is None:
            return
        page, error = self._page_from_payload(payload)
        if page is None:
            self._send_error(400, "invalid_page", error or "invalid page")
            return
        if self.index.get(page.id) is not None:
            self._send_error(
                409, "page_exists", f"page already exists: {page.id!r}"
            )
            return
        if not self._check_path_conflict(page):
            return
        if not self._validate_page_links(page):
            return

        before = self._snapshot_pages()
        try:
            self.index.upsert(page)
            self._sync_links_from()
        except Exception:
            self._restore_pages(before)
            raise
        stored = self.index.get(page.id)
        self._send_json(201, {"page": stored.to_dict() if stored else page.to_dict()})

    def _handle_update_page(self, page_id: str) -> None:
        if not self._require_writable():
            return
        existing = self.index.get(page_id)
        if existing is None:
            self.close_connection = True
            self._send_error(404, "not_found", "page not found")
            return
        payload = self._read_json_object()
        if payload is None:
            return
        page, error = self._page_from_payload(payload, existing=existing, page_id=page_id)
        if page is None:
            self._send_error(400, "invalid_page", error or "invalid page")
            return
        if not self._check_path_conflict(page):
            return
        if not self._validate_page_links(page):
            return

        before = self._snapshot_pages()
        try:
            self.index.upsert(page)
            self._sync_links_from()
        except Exception:
            self._restore_pages(before)
            raise
        stored = self.index.get(page_id)
        self._send_json(200, {"page": stored.to_dict() if stored else page.to_dict()})

    def _handle_delete_page(self, page_id: str) -> None:
        if not self._require_writable():
            return
        page = self.index.get(page_id)
        if page is None:
            self._send_error(404, "not_found", "page not found")
            return

        before = self._snapshot_pages()
        try:
            for other in self.index.list_pages():
                if other.id == page_id:
                    continue
                links_to = tuple(t for t in other.links_to if t != page_id)
                links_from = tuple(s for s in other.links_from if s != page_id)
                if links_to != other.links_to or links_from != other.links_from:
                    self.index.upsert(
                        replace(other, links_to=links_to, links_from=links_from)
                    )
            self.index.delete(page_id)
        except Exception:
            self._restore_pages(before)
            raise

        self.server.sources = {  # type: ignore[attr-defined]
            key: entry
            for key, entry in self.sources.items()
            if entry["page_id"] != page_id
        }
        self._send_json(200, {"deleted": page_id})

    def _handle_save_workspace(self) -> None:
        if not self._require_writable():
            return
        # Save takes no JSON body. Close the connection if a body was sent so
        # an unread request body cannot poison keep-alive reuse.
        if self.headers.get("Content-Length") not in (None, "0"):
            self.close_connection = True
        try:
            save_workspace(
                self.index,
                self.workspace_path,  # type: ignore[arg-type]
                overwrite=True,
                sources=self.sources,
                backup=True,
            )
        except WorkspaceError as exc:
            self._send_error(500, "save_failed", str(exc))
            return
        except OSError:
            self._send_error(500, "save_failed", "workspace file cannot be written")
            return
        self._send_json(200, {"saved": True})

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    def _handle_get_config(self) -> None:
        self._send_json(200, self._public_config())

    def _public_config(self) -> dict[str, Any]:
        """Configuration as the browser may see it -- booleans, never the key."""
        config = self.server.model_config  # type: ignore[attr-defined]
        env_key = bool(os.environ.get("ZMWIKI_API_KEY", "").strip())
        stored_key = bool(config is not None and config.api_key)
        store = self.server.settings_store  # type: ignore[attr-defined]
        return {
            "configured": config is not None,
            "base_url": config.base_url if config is not None else None,
            "model": config.model if config is not None else None,
            "has_api_key": stored_key or env_key,
            "stored_api_key": stored_key,
            "key_from_environment": env_key and not stored_key,
            "source": self.server.config_source,  # type: ignore[attr-defined]
            "settings_file_present": bool(store is not None and store.persisted),
            "read_only": bool(self.server.read_only),  # type: ignore[attr-defined]
        }

    def _require_config_writable(self) -> bool:
        # Model settings are not workspace data, but read-only mode means "no
        # writes at all" so the boundary stays easy to reason about.
        if self.server.read_only:  # type: ignore[attr-defined]
            self.close_connection = True
            self._send_error(
                403,
                "read_only",
                "workspace is read-only; model settings cannot be changed",
            )
            return False
        return True

    def _handle_save_config(self) -> None:
        if not self._require_config_writable():
            return
        payload = self._read_json_object()
        if payload is None:
            return
        try:
            config = parse_config_payload(
                payload, existing=self.server.model_config  # type: ignore[attr-defined]
            )
        except ConfigError as exc:
            self._send_error(400, "invalid_config", str(exc))
            return
        store = self.server.settings_store  # type: ignore[attr-defined]
        if store is not None:
            try:
                store.save(config)
            except ConfigError as exc:
                self._send_error(500, "config_save_failed", str(exc))
                return
        self.server.model_config = config  # type: ignore[attr-defined]
        self.server.config_source = "ui"  # type: ignore[attr-defined]
        self._send_json(200, self._public_config())

    def _handle_test_config(self) -> None:
        payload = self._read_json_object()
        if payload is None:
            return
        try:
            config = parse_config_payload(
                payload, existing=self.server.model_config  # type: ignore[attr-defined]
            )
        except ConfigError as exc:
            self._send_json(200, {
                "ok": False, "code": "invalid_config", "message": str(exc),
            })
            return
        tester = (
            self.server.connection_tester  # type: ignore[attr-defined]
            or test_model_connection
        )
        try:
            result = tester(config)
        except Exception:
            result = ConnectionTestResult(
                False, "failed", "The connection test failed unexpectedly."
            )
        if not isinstance(result, ConnectionTestResult):
            result = ConnectionTestResult(
                False, "failed", "The connection test returned an unexpected result."
            )
        self._send_json(200, {
            "ok": result.ok,
            "code": result.code,
            "message": result.message,
        })

    def _handle_import(self, query: str) -> None:
        """Compile one uploaded UTF-8 ``.txt`` file into a wiki page.

        Request: ``POST /api/import?filename=name.txt`` with the raw file bytes
        as a ``text/plain`` body and ``X-ZMWiki-Import: 1``. The custom header
        keeps a cross-origin form/fetch (which would send a simple
        ``text/plain`` request without preflight) from importing attacker
        content into the workspace. The upload is never written under a
        client-supplied path: it is staged in the system temp directory only
        for :class:`PlainTextFileIngestor`, then removed. Re-uploading the same
        filename updates the page registered to it; a page that exists for any
        other reason is never overwritten.
        """
        if not self._require_writable():
            return
        if not self.server.compiler_available():  # type: ignore[attr-defined]
            self.close_connection = True
            self._send_error(
                503,
                "model_not_configured",
                "Document import needs a model. Open Model setup in the "
                "workbench, or start the server with --base-url and --model.",
            )
            return

        params = self._query_params(query)
        if params is None:
            return
        filename = _single_param(params, "filename")
        filename_error = _import_filename_error(filename)
        if filename_error is not None:
            self.close_connection = True
            self._send_error(400, "invalid_filename", filename_error)
            return
        assert filename is not None

        if (self.headers.get("X-ZMWiki-Import") or "").strip() != "1":
            self.close_connection = True
            self._send_error(
                403,
                "forbidden_request",
                "Import requires the X-ZMWiki-Import: 1 header",
            )
            return

        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0]
        if content_type.strip().lower() != "text/plain":
            self.close_connection = True
            self._send_error(
                415,
                "unsupported_media_type",
                "Content-Type must be text/plain for document import",
            )
            return
        data = self._read_raw_body(MAX_IMPORT_BYTES)
        if data is None:
            return
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            self.close_connection = True
            self._send_error(400, "invalid_encoding", "file must be UTF-8 plain text")
            return
        if not text.strip():
            self.close_connection = True
            self._send_error(400, "empty_document", "file is empty")
            return

        fingerprint = hashlib.sha256(data).hexdigest()
        document_id = f"local-{Path(filename).stem}"
        key = f"{UPLOAD_SOURCE_PREFIX}{filename}"

        document = self._ingest_upload(data, document_id)
        if document is None:
            return
        try:
            compiler = self.server.build_compiler()  # type: ignore[attr-defined]
        except Exception:
            compiler = None
        if compiler is None:
            self._send_error(
                503,
                "model_not_configured",
                "Document import needs a model. Open Model setup in the workbench.",
            )
            return
        try:
            compiled = compiler.compile(document)
        except WikiCompilerError as exc:
            self._send_error(502, "compile_failed", _compiler_error_message(exc))
            return
        page = compiled.page

        existing = self.index.get(page.id)
        registered = (
            self.sources.get(key, {}).get("page_id") == page.id
        )
        owned_elsewhere = any(
            entry["page_id"] == page.id
            for other_key, entry in self.sources.items()
            if other_key != key
        )
        updating = existing is not None and not owned_elsewhere and (
            registered or _page_imported_from(existing, page)
        )
        if existing is not None and not updating:
            detail = (
                "managed by another source file" if owned_elsewhere
                else "not imported from this source"
            )
            self._send_error(
                409,
                "page_exists",
                f"page already exists: {page.id!r} ({detail})",
            )
            return
        occupant = self.index.get_by_path(page.path)
        if occupant is not None and occupant.id != page.id:
            self._send_error(
                409,
                "path_conflict",
                f"path is already used by page {occupant.id!r}",
            )
            return

        report = self._apply_import(
            page,
            existing=existing,
            updating=updating,
            key=key,
            document_id=document_id,
            fingerprint=fingerprint,
        )
        if report is None:
            return
        stored = self.index.get(page.id) or page
        summary = {
            "id": stored.id,
            "path": stored.path,
            "title": stored.title,
        }
        self._send_json(200, {
            "status": "updated" if updating else "imported",
            "imported_pages": [] if updating else [summary],
            "updated_pages": [summary] if updating else [],
            "source": {
                "filename": filename,
                "key": key,
                "document_id": document_id,
                "fingerprint": fingerprint,
            },
            "pages_total": self.index.count(),
            "links": {
                "resolved": report.resolved_count,
                "ambiguous": report.ambiguous_count,
                "unresolved": report.unresolved_count,
            },
            "saved": True,
            "errors": [],
        })

    def _ingest_upload(self, data: bytes, document_id: str) -> Document | None:
        """Stage upload bytes for the existing file ingestor, then delete them."""
        staged: Path | None = None
        try:
            fd, name = tempfile.mkstemp(prefix="zmwiki-upload-", suffix=".txt")
            staged = Path(name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            return PlainTextFileIngestor().ingest(staged, document_id=document_id)
        except (OSError, TypeError, ValueError) as exc:
            self._send_error(400, "invalid_document", str(exc))
            return None
        finally:
            if staged is not None:
                try:
                    staged.unlink()
                except OSError:
                    pass

    def _apply_import(
        self,
        page: WikiPage,
        *,
        existing: WikiPage | None,
        updating: bool,
        key: str,
        document_id: str,
        fingerprint: str,
    ) -> LinkResolutionReport | None:
        """Resolve links, register the source, and save; roll back on failure."""
        before = self._snapshot_pages()
        before_sources = dict(self.sources)
        try:
            pages = [p for p in self.index.list_pages() if p.id != page.id]
            if updating and existing is not None:
                # Hand-added links are not in "generated_links"; carry them
                # over instead of silently dropping them on re-import.
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
                # Drop backlinks from the old version so links the edited
                # document no longer declares are not resurrected.
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
            resolved_for_page = sorted({
                link.target_page_id
                for link in report.resolved_links
                if link.source_page_id == page.id
            })
            for resolved in resolved_pages:
                if resolved.id == page.id:
                    metadata = dict(resolved.metadata)
                    metadata["generated_links"] = resolved_for_page
                    resolved = replace(resolved, metadata=metadata)
                self.index.upsert(resolved)
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
            save_workspace(
                self.index,
                self.workspace_path,  # type: ignore[arg-type]
                overwrite=True,
                sources=self.sources,
                backup=True,
            )
        except (WorkspaceError, OSError) as exc:
            self._restore_pages(before)
            self.server.sources = before_sources  # type: ignore[attr-defined]
            self._send_error(500, "save_failed", str(exc))
            return None
        except Exception:
            self._restore_pages(before)
            self.server.sources = before_sources  # type: ignore[attr-defined]
            raise
        return report

    def _handle_ask(self) -> None:
        if not self.server.agent_available():  # type: ignore[attr-defined]
            self.close_connection = True
            self._send_error(
                503,
                "model_not_configured",
                "Agent Q&A needs a model. Open Model setup in the workbench, "
                "or start the server with --base-url and --model.",
            )
            return

        payload = self._read_json_object()
        if payload is None:
            return
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            self._send_error(
                400, "invalid_question", "'question' must be a non-empty string"
            )
            return
        question = question.strip()
        if len(question) > MAX_QUESTION_CHARS:
            self._send_error(
                400,
                "invalid_question",
                f"'question' must be at most {MAX_QUESTION_CHARS} characters",
            )
            return

        # Snapshot the index briefly so the agent sees a consistent set of
        # pages. The model call runs outside the lock, so browsing and editing
        # remain responsive while the agent is thinking.
        with self.server.lock:  # type: ignore[attr-defined]
            snapshot = self._snapshot_index()
        try:
            agent = self.server.build_agent(snapshot)  # type: ignore[attr-defined]
        except Exception:
            self._send_error(502, "agent_failed", "agent execution failed")
            return
        if agent is None:
            self._send_error(
                503,
                "model_not_configured",
                "Agent Q&A needs a model. Open Model setup in the workbench.",
            )
            return
        try:
            result = agent.answer(question)
        except Exception:
            self._send_error(502, "agent_failed", "agent execution failed")
            return
        if not isinstance(result, AgentResult):
            self._send_error(502, "agent_failed", "agent returned an invalid result")
            return
        self._send_json(
            200, self._serialize_agent_result(question, result, snapshot)
        )

    def _snapshot_index(self) -> WikiIndex:
        snapshot = WikiIndex()
        for page in self.index.list_pages():
            snapshot.upsert(page)
        return snapshot

    @staticmethod
    def _serialize_agent_step(step) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        # Raw model output is intentionally omitted. Transport errors on
        # model/protocol failures are replaced by a safe terminal message.
        error = step.error
        if step.action in ("model_error", "protocol_error"):
            error = None
        return {
            "step": step.step,
            "action": step.action,
            "params": dict(step.params),
            "result_summary": step.result_summary,
            "error": error,
            "read_page_ids": list(step.read_page_ids),
            "search_hit_page_ids": list(step.search_hit_page_ids),
        }

    def _serialize_agent_result(
        self,
        question: str,
        result: AgentResult,
        snapshot: WikiIndex,
    ) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        for path in result.sources:
            page = snapshot.get_by_path(path)
            evidence.append({
                "id": page.id if page is not None else None,
                "path": page.path if page is not None else path,
                "title": page.title if page is not None else None,
                "sources": list(page.sources) if page is not None else [],
            })
        return {
            "question": question,
            "status": result.status,
            "answer": result.answer,
            "sources": list(result.sources),
            "evidence": evidence,
            "read_page_ids": list(result.read_page_ids),
            "pages_read": result.pages_read,
            "steps": [
                self._serialize_agent_step(step) for step in result.steps
            ],
            "error": self._agent_error_message(result.status),
        }

    @staticmethod
    def _agent_error_message(status: str) -> str | None:
        if status == AgentStatus.ANSWERED:
            return None
        if status == AgentStatus.INSUFFICIENT_EVIDENCE:
            return (
                "The wiki has no page with enough read evidence to answer. "
                "Import a relevant .txt document or ask about a page that exists."
            )
        if status == AgentStatus.BUDGET_EXHAUSTED:
            return (
                "The agent used its step budget without an answer. Narrow the "
                "question or ask about a more specific page."
            )
        if status == AgentStatus.MODEL_ERROR:
            return (
                "The model call failed. Open Model setup and run Test "
                "connection to check the Base URL, model and API key."
            )
        if status == AgentStatus.PROTOCOL_ERROR:
            return (
                "The model returned an unusable action. Try a model with "
                "reliable JSON output, or check the step diagnostics."
            )
        return "The agent did not produce an answer. Check the step diagnostics."

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _workspace_writable(self) -> bool:
        return (
            not self.server.read_only  # type: ignore[attr-defined]
            and self.workspace_path is not None
        )

    def _require_writable(self) -> bool:
        if self.server.read_only:  # type: ignore[attr-defined]
            self.close_connection = True
            self._send_error(403, "read_only", "workspace is read-only")
            return False
        if self.workspace_path is None:
            self.close_connection = True
            self._send_error(
                409, "workspace_not_configured", "workspace path is not configured"
            )
            return False
        return True

    def _write_target(self):  # type: ignore[no-untyped-def]
        try:
            target = urlsplit(self.path)
        except ValueError:
            self.close_connection = True
            self._send_error(400, "invalid_request", "invalid request target")
            return None
        if len(target.query) > MAX_QUERY_STRING_CHARS:
            self.close_connection = True
            self._send_error(400, "invalid_query", "query string is too long")
            return None
        return target

    def _query_params(self, query: str) -> dict[str, list[str]] | None:
        try:
            return parse_qs(query, keep_blank_values=True, max_num_fields=10)
        except (UnicodeDecodeError, ValueError):
            self.close_connection = True
            self._send_error(400, "invalid_query", "invalid query string")
            return None

    def _valid_page_id(self, page_id: str | None) -> bool:
        if page_id is None or not page_id.strip() or len(page_id) > MAX_PAGE_ID_CHARS:
            self.close_connection = True
            self._send_error(
                400, "invalid_id", "query parameter 'id' is required"
            )
            return False
        return True

    def parse_request(self) -> bool:
        # Per-request flag: finish() uses it to decide whether a rejected
        # request body still has to be drained before the socket closes.
        self._body_drained = False
        return super().parse_request()

    def finish(self) -> None:
        # Closing a socket with unread request bytes makes Windows send an
        # RST, which can discard the error response the client is waiting for.
        # Drain the declared body before closing so 4xx/5xx replies arrive.
        try:
            if self.close_connection and not getattr(self, "_body_drained", False):
                self._drain_request_body()
        except Exception:
            pass
        super().finish()

    def _drain_request_body(self) -> None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return
        try:
            length = int(raw_length)
        except ValueError:
            return
        if length <= 0 or length > MAX_REQUEST_BODY_BYTES:
            return
        try:
            self.rfile.read(length)
        except OSError:
            pass
        finally:
            self._body_drained = True

    def _read_raw_body(self, limit: int) -> bytes | None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.close_connection = True
            self._send_error(400, "invalid_body", "Content-Length is required")
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self.close_connection = True
            self._send_error(400, "invalid_body", "Content-Length must be an integer")
            return None
        if length < 0:
            self.close_connection = True
            self._send_error(400, "invalid_body", "Content-Length must be non-negative")
            return None
        if length > limit:
            self.close_connection = True
            self._send_error(
                413,
                "request_too_large",
                f"request body must be at most {limit} bytes",
            )
            return None
        if length == 0:
            self.close_connection = True
            self._send_error(400, "invalid_body", "request body is required")
            return None
        raw = self.rfile.read(length)
        self._body_drained = True
        if len(raw) != length:
            self.close_connection = True
            self._send_error(400, "invalid_body", "request body is incomplete")
            return None
        return raw

    def _read_json_object(self) -> dict[str, Any] | None:
        # Requiring a JSON content type makes the request non-simple in a
        # browser: without CORS approval the preflight fails, so a visited
        # website cannot drive these endpoints (including model settings,
        # where a forged Base URL could receive the stored API key).
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0]
        if content_type.strip().lower() != "application/json":
            self.close_connection = True
            self._send_error(
                415,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )
            return None
        raw = self._read_raw_body(MAX_REQUEST_BODY_BYTES)
        if raw is None:
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self.close_connection = True
            self._send_error(400, "invalid_body", "request body must be UTF-8")
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.close_connection = True
            self._send_error(400, "invalid_json", "request body must be valid JSON")
            return None
        if not isinstance(payload, dict):
            self.close_connection = True
            self._send_error(400, "invalid_body", "request body must be a JSON object")
            return None
        return payload

    def _page_from_payload(
        self,
        payload: dict[str, Any],
        *,
        existing: WikiPage | None = None,
        page_id: str | None = None,
    ) -> tuple[WikiPage | None, str | None]:
        data = existing.to_dict() if existing is not None else {}
        for field in (
            "id",
            "path",
            "title",
            "content",
            "aliases",
            "tags",
            "description",
            "sources",
            "links_to",
            "metadata",
        ):
            if field in payload:
                data[field] = payload[field]
        # links_from is derived from links_to; never accept it from a client.
        data.pop("links_from", None)
        if page_id is not None:
            body_id = payload.get("id")
            if body_id is not None and body_id != page_id:
                return None, "body 'id' must match the query id"
            data["id"] = page_id
        try:
            page = WikiPage.from_dict(data)
        except (TypeError, ValueError) as exc:
            return None, str(exc)
        if len(page.id) > MAX_PAGE_ID_CHARS:
            return None, f"'id' must be at most {MAX_PAGE_ID_CHARS} characters"
        if not page.path.strip().strip("/"):
            return None, "'path' must not be empty after normalization"
        return page, None

    def _check_path_conflict(self, page: WikiPage) -> bool:
        occupant = self.index.get_by_path(page.path)
        if occupant is not None and occupant.id != page.id:
            self._send_error(
                409,
                "path_conflict",
                f"path is already used by page {occupant.id!r}",
            )
            return False
        return True

    def _validate_page_links(self, page: WikiPage) -> bool:
        if page.id in page.links_to:
            self._send_error(
                400, "invalid_links", "a page cannot link to itself"
            )
            return False
        missing = [target for target in page.links_to if self.index.get(target) is None]
        if missing:
            self._send_error(
                400,
                "invalid_links",
                f"link targets not found: {', '.join(missing)}",
            )
            return False
        return True

    def _sync_links_from(self) -> None:
        """Recompute every page's backlinks from the authoritative links_to."""
        pages = self.index.list_pages()
        incoming: dict[str, list[str]] = {page.id: [] for page in pages}
        for page in pages:
            for target in page.links_to:
                if target in incoming:
                    incoming[target].append(page.id)
        for page in pages:
            desired = tuple(sorted(set(incoming[page.id])))
            if page.links_from != desired:
                self.index.upsert(replace(page, links_from=desired))

    def _snapshot_pages(self) -> dict[str, WikiPage]:
        return {page.id: page for page in self.index.list_pages()}

    def _restore_pages(self, before: dict[str, WikiPage]) -> None:
        for page_id in [page.id for page in self.index.list_pages()]:
            if page_id not in before:
                self.index.delete(page_id)
        for page in before.values():
            self.index.upsert(page)

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------

    def _handle_static(self, path: str) -> None:
        filename = STATIC_ROUTES.get(path)
        if filename is None:
            self._send_error(404, "not_found", "not found")
            return
        root = STATIC_DIR.resolve()
        candidate = (root / filename).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            self._send_error(404, "not_found", "not found")
            return
        try:
            body = candidate.read_bytes()
        except OSError:
            self._send_error(404, "not_found", "not found")
            return
        content_type = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
        self._send_bytes(200, body, content_type, cache_control="no-cache")

    # ------------------------------------------------------------------
    # Method handling
    # ------------------------------------------------------------------

    def _method_not_allowed(self) -> None:
        # Do not leave an unread request body on a keep-alive connection.
        self.close_connection = True
        self._send_error(
            405, "method_not_allowed", "method not allowed",
            headers={"Allow": "GET"},
        )

    def do_HEAD(self) -> None:
        self.close_connection = True
        self.send_response(405)
        self.send_header("Allow", "GET")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._method_not_allowed()

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_json(
        self, status: int, payload: dict[str, Any], *,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(
            status, body, "application/json; charset=utf-8",
            cache_control="no-store", headers=headers,
        )

    def _send_error(
        self, status: int, code: str, message: str, *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._send_json(status, {"error": code, "message": message}, headers=headers)

    def _send_bytes(
        self, status: int, body: bytes, content_type: str, *,
        cache_control: str, headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        # Keep request logs useful without recording query parameters.
        path = self.path.split("?", 1)[0]
        self.log_message('"%s %s" %s %s', self.command, path, code, size)


def _single_param(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    if not values or len(values) != 1:
        return None
    return values[0]


def _import_filename_error(filename: str | None) -> str | None:
    """Validate an uploaded base filename; return an error message or None.

    The name is never used as a filesystem path, but it becomes the source
    registry key and part of the page's document id, so separators, control
    characters and Windows-reserved characters are rejected outright.
    """
    if filename is None or not filename:
        return "query parameter 'filename' is required"
    if len(filename) > MAX_IMPORT_FILENAME_CHARS:
        return f"'filename' must be at most {MAX_IMPORT_FILENAME_CHARS} characters"
    if any(ch in filename for ch in '/\\:*?"<>|') or any(
        ord(ch) < 0x20 or ord(ch) == 0x7F for ch in filename
    ):
        return "'filename' must be a plain base name without path or control characters"
    name = Path(filename)
    if name.suffix.lower() != ".txt" or not name.stem:
        return "'filename' must be a non-empty .txt file name"
    return None


def _compiler_error_message(exc: WikiCompilerError) -> str:
    """Safe, actionable text for a failed compilation.

    Never echoes the upstream response body or raw model output: a broken or
    hostile endpoint could reflect the Authorization header into either.
    """
    if isinstance(exc, CompilerTransportError):
        return (
            "Could not reach the model endpoint. Open Model setup and run "
            "Test connection to check the Base URL and network."
        )
    if isinstance(exc, CompilerResponseError):
        marker = "model returned HTTP "
        text = str(exc)
        if marker in text:
            status = text.split(marker, 1)[1].split(":", 1)[0].strip()
            if status.isdigit() and len(status) == 3:
                return (
                    f"The model call failed (HTTP {status}). Open Model setup "
                    "and run Test connection to check the model and API key."
                )
        return (
            "The model call failed. Open Model setup and run Test connection "
            "to check the Base URL, model and API key."
        )
    if isinstance(exc, CompilerValidationError):
        return (
            "The model returned a response that is not a usable wiki page. "
            "Try a model with reliable JSON output."
        )
    return "Compilation failed. Open Model setup and run Test connection."


def _page_imported_from(existing: WikiPage, page: WikiPage) -> bool:
    """True when ``existing`` was compiled from the same source document."""
    source_id = page.metadata.get("source_document_id")
    if not isinstance(source_id, str) or not source_id:
        return False
    return (
        existing.metadata.get("source_document_id") == source_id
        or source_id in existing.sources
    )


def make_server(
    index: WikiIndex,
    port: int = DEFAULT_PORT,
    *,
    workspace_path: str | Path | None = None,
    sources: SourceRegistry | None = None,
    read_only: bool = False,
    agent_factory: Callable[[WikiIndex], WikiAgent] | None = None,
    compiler_factory: Callable[[], WikiCompiler] | None = None,
    model_config: ModelConfig | None = None,
    settings_store: SettingsStore | None = None,
    connection_tester: Callable[[ModelConfig], ConnectionTestResult] | None = None,
    config_source: str = "none",
) -> ZMWikiServer:
    """Create a loopback-only server. Tests pass port=0 for an ephemeral port."""
    return ZMWikiServer(
        (HOST, port),
        index,
        workspace_path=workspace_path,
        sources=sources,
        read_only=read_only,
        agent_factory=agent_factory,
        compiler_factory=compiler_factory,
        model_config=model_config,
        settings_store=settings_store,
        connection_tester=connection_tester,
        config_source=config_source,
    )


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
        prog="python -m zmwiki.web.server",
        description="Serve a local ZMWiki workspace and browser workbench.",
    )
    parser.add_argument("--workspace", required=True,
                        help="JSON workspace file")
    parser.add_argument("--create", action="store_true",
                        help="create a new empty workspace file (must not exist)")
    parser.add_argument("--settings", default=None,
                        help="model settings file "
                             f"(default: {DEFAULT_SETTINGS_PATH})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"loopback port (default: {DEFAULT_PORT})")
    parser.add_argument("--read-only", action="store_true",
                        help="Reject all write requests (the demo workspace is "
                             "read-only automatically)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL (enables /api/ask)")
    parser.add_argument("--model", default=None,
                        help="model identifier for /api/ask")
    parser.add_argument("--api-key", default=None,
                        help="server-side API key for /api/ask "
                             "(or set ZMWIKI_API_KEY; never sent to the browser)")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if (args.base_url is None) != (args.model is None):
        parser.error("--base-url and --model must be provided together")

    try:
        index, sources = _load_or_create_workspace(
            args.workspace, create=args.create
        )
    except WorkspaceError as exc:
        message = str(exc)
        if not args.create and message == "workspace file not found":
            message += " (pass --create to start a new empty workspace)"
        print(f"error: {message}", file=sys.stderr)
        return 2

    workspace_path = Path(args.workspace)
    read_only = args.read_only or (
        source_key(workspace_path) == source_key(_DEMO_WORKSPACE)
    )

    settings_store = SettingsStore(args.settings)
    model_config: ModelConfig | None = None
    config_source = "none"
    if args.base_url is not None:
        # Explicit CLI configuration wins over the saved settings file.
        try:
            model_config = make_config(args.base_url, args.model, args.api_key)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        config_source = "cli"
    else:
        model_config = settings_store.load()
        if model_config is not None:
            config_source = "settings"

    try:
        server = make_server(
            index,
            port=args.port,
            workspace_path=workspace_path,
            sources=sources,
            read_only=read_only,
            model_config=model_config,
            settings_store=settings_store,
            config_source=config_source,
        )
    except OSError as exc:
        detail = exc.strerror or str(exc)
        print(f"error: cannot bind {HOST}:{args.port}: {detail}", file=sys.stderr)
        return 2

    mode = "read-only" if read_only else "writable"
    model_mode = (
        f"model configured ({config_source})"
        if model_config is not None
        else "model not configured (open /app and use Model setup)"
    )
    print(
        f"ZMWiki serving {index.count()} pages at http://{HOST}:{args.port}/ "
        f"(workbench: /app, {mode}, {model_mode})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
