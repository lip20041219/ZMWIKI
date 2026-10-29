"""Workbench static assets and write-API tests for the local web server."""

import http.client
import json
import threading
from contextlib import contextmanager
from urllib.parse import urlencode

import pytest

import zmwiki.web.server as server_module
from zmwiki.web.server import make_server
from zmwiki.web.workspace import load_workspace_state, save_workspace
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage


@contextmanager
def _serving(index, **kwargs):
    server = make_server(index, port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(server, path, method="GET", *, payload=None, body=None, headers=None):
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **(headers or {})}
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


def _json(body: bytes) -> dict:
    return json.loads(body.decode("utf-8"))


def _workspace(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"version": 1, "pages": []}), encoding="utf-8")
    return path


def _page(**overrides):
    data = {
        "id": "alpha",
        "path": "pages/alpha",
        "title": "Alpha",
        "content": "alpha body",
    }
    data.update(overrides)
    return data


def test_workbench_static_assets():
    with _serving(WikiIndex()) as server:
        response, body = _request(server, "/app")
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/html; charset=utf-8"
        assert b"Workbench" in body
        assert b"app.css" in body
        assert b"app.js" in body
        assert b'id="ask-question"' in body
        assert b'id="ask-result"' in body
        assert b'id="ask-steps"' in body
        assert b'id="ask-evidence"' in body

        response, body = _request(server, "/app.html")
        assert response.status == 200
        assert b"ZMWiki" in body

        response, body = _request(server, "/app.css")
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/css; charset=utf-8"
        assert b"--accent" in body

        response, body = _request(server, "/app.js")
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/javascript; charset=utf-8"
        assert b"/api/pages" in body
        assert b"/api/ask" in body
        assert b"submitAsk" in body
        assert b"renderAskResult" in body
        assert b"renderAskSteps" in body
        assert b"renderAskEvidence" in body

        # The existing marketing homepage remains available.
        response, body = _request(server, "/")
        assert response.status == 200
        assert b"ZMWiki" in body


def test_create_search_update_and_delete_page(tmp_path):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, body = _request(
            server, "/api/pages", "POST", payload=_page(content="oldtoken")
        )
        assert response.status == 201
        assert _json(body)["page"]["id"] == "alpha"

        response, body = _request(server, "/api/pages")
        assert response.status == 200
        assert [row["id"] for row in _json(body)["pages"]] == ["alpha"]

        response, body = _request(server, "/api/page?id=alpha")
        assert response.status == 200
        assert _json(body)["content"] == "oldtoken"

        response, body = _request(server, "/api/search?q=oldtoken")
        assert response.status == 200
        assert [row["page_id"] for row in _json(body)["results"]] == ["alpha"]

        response, body = _request(
            server,
            "/api/page?" + urlencode({"id": "alpha"}),
            "PUT",
            payload={
                "id": "alpha",
                "path": "pages/alpha",
                "title": "Alpha Two",
                "content": "newtoken",
                "links_to": [],
            },
        )
        assert response.status == 200
        assert _json(body)["page"]["title"] == "Alpha Two"

        response, body = _request(server, "/api/search?q=oldtoken")
        assert [row["page_id"] for row in _json(body)["results"]] == []
        response, body = _request(server, "/api/search?q=newtoken")
        assert [row["page_id"] for row in _json(body)["results"]] == ["alpha"]

        response, body = _request(
            server, "/api/page?" + urlencode({"id": "alpha"}), "DELETE"
        )
        assert response.status == 200
        assert _json(body) == {"deleted": "alpha"}

        response, body = _request(server, "/api/page?id=alpha")
        assert response.status == 404
        response, body = _request(server, "/api/search?q=newtoken")
        assert _json(body)["results"] == []


def test_links_and_backlinks_stay_consistent(tmp_path):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, _ = _request(
            server, "/api/pages", "POST",
            payload=_page(id="b", path="pages/b", title="B", content="B"),
        )
        assert response.status == 201

        response, _ = _request(
            server, "/api/pages", "POST",
            payload=_page(
                id="a", path="pages/a", title="A", content="A", links_to=["b"]
            ),
        )
        assert response.status == 201

        response, body = _request(server, "/api/page?id=a")
        assert _json(body)["links_to"] == ["b"]
        response, body = _request(server, "/api/page?id=b")
        assert _json(body)["links_from"] == ["a"]

        response, body = _request(
            server,
            "/api/page?" + urlencode({"id": "a"}),
            "PUT",
            payload={
                "id": "a",
                "path": "pages/a",
                "title": "A",
                "content": "A",
                "links_to": [],
            },
        )
        assert response.status == 200
        response, body = _request(server, "/api/page?id=b")
        assert _json(body)["links_from"] == []

        # Re-add the link, then delete its target; both sides must be cleaned.
        response, _ = _request(
            server,
            "/api/page?" + urlencode({"id": "a"}),
            "PUT",
            payload={
                "id": "a",
                "path": "pages/a",
                "title": "A",
                "content": "A",
                "links_to": ["b"],
            },
        )
        assert response.status == 200
        response, _ = _request(
            server, "/api/page?" + urlencode({"id": "b"}), "DELETE"
        )
        assert response.status == 200
        response, body = _request(server, "/api/page?id=a")
        assert _json(body)["links_to"] == []
        assert server.index.check_links().is_clean


def test_slash_page_id_and_link_validation(tmp_path):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, _ = _request(
            server, "/api/pages", "POST",
            payload=_page(id="team/bob", path="team/bob", title="Bob", content="B"),
        )
        assert response.status == 201

        response, body = _request(
            server, "/api/page?" + urlencode({"id": "team/bob"})
        )
        assert response.status == 200
        assert _json(body)["id"] == "team/bob"

        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(
                id="team/alice", path="team/alice", title="Alice", content="A",
                links_to=["team/bob"],
            ),
        )
        assert response.status == 201

        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(
                id="bad-self", path="bad/self", title="Bad", content="B",
                links_to=["bad-self"],
            ),
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_links"

        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(
                id="bad-link", path="bad/link", title="Bad", content="B",
                links_to=["missing"],
            ),
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_links"


def test_update_preserves_sources_and_metadata(tmp_path):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, _ = _request(
            server, "/api/pages", "POST",
            payload=_page(
                sources=["doc-1"],
                metadata={"source_document_id": "doc-1", "custom": "kept"},
            ),
        )
        assert response.status == 201

        response, body = _request(
            server,
            "/api/page?" + urlencode({"id": "alpha"}),
            "PUT",
            payload={"content": "updated body"},
        )
        assert response.status == 200
        page = _json(body)["page"]
        assert page["sources"] == ["doc-1"]
        assert page["metadata"] == {
            "source_document_id": "doc-1",
            "custom": "kept",
        }
        assert page["content"] == "updated body"


def test_read_only_rejects_writes_and_keeps_file(tmp_path):
    workspace = _workspace(tmp_path)
    index = WikiIndex()
    index.upsert(WikiPage(
        id="alpha", path="pages/alpha", title="Alpha", content="alpha body"
    ))
    save_workspace(index, workspace, overwrite=True)
    before = workspace.read_bytes()

    with _serving(
        index, workspace_path=workspace, read_only=True
    ) as server:
        response, body = _request(server, "/api/capabilities")
        assert response.status == 200
        assert _json(body)["read_only"] is True
        assert _json(body)["writable"] is False

        for method, path, payload in (
            ("POST", "/api/pages", _page(id="new")),
            ("PUT", "/api/page?id=alpha", {"content": "changed"}),
            ("DELETE", "/api/page?id=alpha", None),
            ("POST", "/api/workspace/save", None),
        ):
            response, body = _request(
                server, path, method, payload=payload
            )
            assert response.status == 403
            assert _json(body)["error"] == "read_only"

        response, body = _request(server, "/api/page?id=alpha")
        assert response.status == 200
        assert _json(body)["content"] == "alpha body"
        assert workspace.read_bytes() == before


def test_save_and_reload_workspace(tmp_path):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, _ = _request(
            server, "/api/pages", "POST",
            payload=_page(id="b", path="pages/b", title="B", content="B"),
        )
        assert response.status == 201
        response, _ = _request(
            server, "/api/pages", "POST",
            payload=_page(
                id="a", path="pages/a", title="A", content="A", links_to=["b"]
            ),
        )
        assert response.status == 201

        response, body = _request(server, "/api/workspace/save", "POST")
        assert response.status == 200
        assert _json(body) == {"saved": True}

    index, sources = load_workspace_state(workspace)
    assert index.count() == 2
    assert index.get("a").links_to == ("b",)
    assert index.get("b").links_from == ("a",)
    assert index.check_links().is_clean

    # Recreate the server from the saved file, as a process restart would.
    with _serving(
        index, workspace_path=workspace, sources=sources
    ) as restarted:
        response, body = _request(restarted, "/api/page?id=a")
        assert response.status == 200
        assert _json(body)["links_to"] == ["b"]
        response, body = _request(restarted, "/api/page?id=b")
        assert _json(body)["links_from"] == ["a"]


def test_validation_errors_and_body_limit(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, body = _request(
            server, "/api/pages", "POST",
            payload={"id": "missing-body", "path": "x", "title": "X"},
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_page"

        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(),
        )
        assert response.status == 201
        response, body = _request(
            server, "/api/pages", "POST", payload=_page()
        )
        assert response.status == 409
        assert _json(body)["error"] == "page_exists"

        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(id="other", path="pages/alpha"),
        )
        assert response.status == 409
        assert _json(body)["error"] == "path_conflict"

        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(id="bad-path", path="///"),
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_page"

        response, body = _request(
            server,
            "/api/page?" + urlencode({"id": "alpha"}),
            "PUT",
            payload={"id": "wrong", "content": "changed"},
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_page"

        response, body = _request(
            server, "/api/page?id=missing", "PUT", payload={"content": "x"}
        )
        assert response.status == 404

        response, body = _request(
            server,
            "/api/pages",
            "POST",
            body=b"{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_json"

        monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 16)
        response, body = _request(
            server, "/api/pages", "POST",
            payload=_page(id="too-large", content="x" * 100),
        )
        assert response.status == 413
        assert _json(body)["error"] == "request_too_large"


def test_save_preserves_source_registry_and_delete_prunes_it(tmp_path):
    workspace = _workspace(tmp_path)
    index = WikiIndex()
    index.upsert(WikiPage(
        id="alpha", path="pages/alpha", title="Alpha", content="alpha body"
    ))
    sources = {
        "src/alpha.txt": {
            "page_id": "alpha",
            "document_id": "local-alpha",
            "fingerprint": "abc123",
        }
    }
    save_workspace(index, workspace, overwrite=True, sources=sources)

    with _serving(
        index, workspace_path=workspace, sources=sources
    ) as server:
        response, _ = _request(server, "/api/workspace/save", "POST")
        assert response.status == 200
        saved = json.loads(workspace.read_text(encoding="utf-8"))
        assert saved["sources"] == sources

        response, _ = _request(server, "/api/page?id=alpha", "DELETE")
        assert response.status == 200
        response, _ = _request(server, "/api/workspace/save", "POST")
        assert response.status == 200
        saved = json.loads(workspace.read_text(encoding="utf-8"))
        assert "sources" not in saved


def test_server_without_workspace_rejects_writes():
    with _serving(WikiIndex()) as server:
        response, body = _request(server, "/api/capabilities")
        assert _json(body) == {
            "read_only": False,
            "writable": False,
            "workspace_configured": False,
            "agent_configured": False,
            "import_configured": False,
        }
        response, body = _request(
            server, "/api/pages", "POST", payload=_page()
        )
        assert response.status == 409
        assert _json(body)["error"] == "workspace_not_configured"
