"""Minimal read-only local HTTP service for ZMWiki.

Start with::

    python -m zmwiki.web.server --workspace path/to/wiki.json --port 8000

The server binds 127.0.0.1 only. It serves the W1 static site and four
read-only JSON endpoints backed by an in-memory WikiIndex. There are no
write endpoints, no authentication, and no third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_retriever import wiki_search

from .workspace import WorkspaceError, load_workspace

__all__ = ["HOST", "DEFAULT_PORT", "ZMWikiServer", "make_server", "main"]

HOST = "127.0.0.1"
DEFAULT_PORT = 8000

MAX_QUERY_CHARS = 200
MAX_TOP_K = 50
DEFAULT_TOP_K = 5
MAX_QUERY_STRING_CHARS = 2048
MAX_PAGE_ID_CHARS = 512
REQUEST_TIMEOUT_SECONDS = 10

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class ZMWikiServer(ThreadingHTTPServer):
    """Threaded server holding the read-only index shared by all requests."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], index: WikiIndex) -> None:
        self.index = index
        super().__init__(address, ZMWikiHandler)


class ZMWikiHandler(BaseHTTPRequestHandler):
    """Handles GET routes only; every other method returns 405."""

    server_version = "ZMWiki/0.1"
    protocol_version = "HTTP/1.1"
    timeout = REQUEST_TIMEOUT_SECONDS

    @property
    def index(self) -> WikiIndex:
        return self.server.index  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except Exception as exc:
            # Never send exception text or a traceback to the client.
            self.log_error("unhandled error serving request: %s", type(exc).__name__)
            self.close_connection = True
            try:
                self._send_error(500, "internal_error", "internal server error")
            except Exception:
                pass

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

    # ------------------------------------------------------------------
    # API routes
    # ------------------------------------------------------------------

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
            405, "method_not_allowed", "only GET is supported",
            headers={"Allow": "GET"},
        )

    def do_HEAD(self) -> None:
        self.close_connection = True
        self.send_response(405)
        self.send_header("Allow", "GET")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed

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


def make_server(index: WikiIndex, port: int = DEFAULT_PORT) -> ZMWikiServer:
    """Create a loopback-only server. Tests pass port=0 for an ephemeral port."""
    return ZMWikiServer((HOST, port), index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zmwiki.web.server",
        description="Serve a read-only ZMWiki workspace on 127.0.0.1.",
    )
    parser.add_argument("--workspace", required=True,
                        help="JSON workspace file (read-only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"loopback port (default: {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        index = load_workspace(args.workspace)
    except WorkspaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        server = make_server(index, port=args.port)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        print(f"error: cannot bind {HOST}:{args.port}: {detail}", file=sys.stderr)
        return 2

    print(f"ZMWiki serving {index.count()} pages at http://{HOST}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
