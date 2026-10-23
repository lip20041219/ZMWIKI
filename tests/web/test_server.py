"""HTTP integration tests for the local read-only ZMWiki server."""

import http.client
import json
import threading
from contextlib import contextmanager
from urllib.parse import urlencode

import pytest

from zmwiki.web.server import make_server
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


def _build_index() -> WikiIndex:
    index = WikiIndex()
    index.upsert(WikiPage(
        id="alice", path="people/alice", title="Alice Smith",
        content="Alice leads engineering.",
        links_to=("project/phoenix",),
    ))
    index.upsert(WikiPage(
        id="project/phoenix", path="projects/phoenix", title="Project Phoenix",
        content="Phoenix budget is 500 thousand dollars.",
        links_from=("alice",),
    ))
    index.upsert(WikiPage(
        id="team/alice", path="teams/alice", title="Team Alice",
        content="The team page for Alice.",
    ))
    return index


@contextmanager
def _serving(index):
    server = make_server(index, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def server():
    with _serving(_build_index()) as httpd:
        yield httpd


def _request(server, path: str, method: str = "GET"):
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


def _json(body: bytes) -> dict:
    return json.loads(body.decode("utf-8"))


def test_health(server):
    response, body = _request(server, "/api/health")

    assert response.status == 200
    assert response.getheader("Content-Type") == "application/json; charset=utf-8"
    assert response.getheader("Cache-Control") == "no-store"
    assert int(response.getheader("Content-Length")) == len(body)
    assert response.getheader("Access-Control-Allow-Origin") is None
    assert _json(body) == {"status": "ok", "pages": 3}


def test_pages_list_is_metadata_only(server):
    response, body = _request(server, "/api/pages")
    payload = _json(body)

    assert response.status == 200
    assert [row["id"] for row in payload["pages"]] == [
        "alice", "project/phoenix", "team/alice",
    ]
    for row in payload["pages"]:
        assert set(row) == {"id", "path", "title", "aliases", "tags", "description"}
        assert "content" not in row


def test_page_detail(server):
    path = "/api/page?" + urlencode({"id": "project/phoenix"})
    response, body = _request(server, path)
    payload = _json(body)

    assert response.status == 200
    assert payload["id"] == "project/phoenix"
    assert payload["content"] == "Phoenix budget is 500 thousand dollars."
    assert payload["links_from"] == ["alice"]


def test_page_id_with_slash(server):
    path = "/api/page?" + urlencode({"id": "team/alice"})
    response, body = _request(server, path)

    assert response.status == 200
    assert _json(body)["id"] == "team/alice"


def test_missing_page_returns_404(server):
    response, body = _request(server, "/api/page?id=missing")
    payload = _json(body)

    assert response.status == 404
    assert payload["error"] == "not_found"


def test_search(server):
    path = "/api/search?" + urlencode({"q": "phoenix", "top_k": 5})
    response, body = _request(server, path)
    payload = _json(body)

    assert response.status == 200
    assert payload["query"] == "phoenix"
    assert payload["top_k"] == 5
    assert payload["results"]
    for row in payload["results"]:
        assert set(row) == {
            "page_id", "path", "title", "aliases", "tags", "description", "score",
        }
        assert "content" not in row


def test_search_top_k_is_respected(server):
    path = "/api/search?" + urlencode({"q": "alice", "top_k": 1})
    response, body = _request(server, path)
    payload = _json(body)

    assert response.status == 200
    assert payload["top_k"] == 1
    assert len(payload["results"]) <= 1


@pytest.mark.parametrize("path, error", [
    ("/api/search", "invalid_query"),
    ("/api/search?q=", "invalid_query"),
    ("/api/search?q=%20%20", "invalid_query"),
    ("/api/search?q=" + "x" * 201, "invalid_query"),
    ("/api/search?q=alice&top_k=abc", "invalid_top_k"),
    ("/api/search?q=alice&top_k=0", "invalid_top_k"),
    ("/api/search?q=alice&top_k=51", "invalid_top_k"),
])
def test_search_validation(server, path, error):
    response, body = _request(server, path)

    assert response.status == 400
    assert _json(body)["error"] == error


def test_unknown_routes_return_404(server):
    for path in ("/api/unknown", "/missing.txt"):
        response, body = _request(server, path)
        assert response.status == 404
        assert _json(body)["error"] == "not_found"


def test_non_get_returns_405(server):
    response, body = _request(server, "/api/search?q=alice", method="POST")

    assert response.status == 405
    assert response.getheader("Allow") == "GET"
    assert _json(body)["error"] == "method_not_allowed"


def test_static_site(server):
    for path in ("/", "/index.html"):
        response, body = _request(server, path)
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/html; charset=utf-8"
        assert b"ZMWiki" in body

    response, body = _request(server, "/styles.css")
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/css; charset=utf-8"
    assert b"--accent" in body


def test_path_traversal_is_rejected(server):
    for path in (
        "/../pyproject.toml",
        "/%2e%2e/pyproject.toml",
        "/..%2fpyproject.toml",
        "/%00",
    ):
        response, body = _request(server, path)
        assert response.status == 404
        assert _json(body)["error"] == "not_found"


def test_server_binds_loopback_only(server):
    assert server.server_address[0] == "127.0.0.1"


def test_internal_error_does_not_leak_details():
    class BrokenIndex:
        def count(self):
            raise RuntimeError("SENSITIVE-INTERNAL-DETAIL")

    with _serving(BrokenIndex()) as server:
        response, body = _request(server, "/api/health")

    text = body.decode("utf-8")
    assert response.status == 500
    assert _json(body)["error"] == "internal_error"
    assert "SENSITIVE-INTERNAL-DETAIL" not in text
    assert "RuntimeError" not in text
    assert "Traceback" not in text
