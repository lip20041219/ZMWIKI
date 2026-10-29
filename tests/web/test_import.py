"""POST /api/import tests using a fake compiler transport (no network)."""

import http.client
import json
import threading
from contextlib import contextmanager
from urllib.parse import quote

import zmwiki.web.server as server_module
from zmwiki.llm_generators import HttpResponse
from zmwiki.web.model_settings import ConnectionTestResult, SettingsStore
from zmwiki.web.server import make_server
from zmwiki.web.workspace import (
    WorkspaceError,
    load_workspace_state,
    save_workspace,
)
from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent
from zmwiki.wiki.wiki_compiler import WikiCompiler
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_schema import WikiPage

PAGE_ID = "page-local-notes"


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


def _request(server, path, method="GET", *, body=None, headers=None):
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


class _Transport:
    """Returns queued compiler payloads; an Exception entry is raised."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("fake transport exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, HttpResponse):
            return response
        return HttpResponse(
            status=200,
            body=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(response),
                    },
                }],
            }).encode("utf-8"),
        )


def _compiler(responses):
    transport = _Transport(responses)

    def factory():
        return WikiCompiler(
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key="sk-test-secret",
            transport=transport,
        )

    return factory, transport


def _agent_factory(responses):
    transport = _Transport(responses)

    def factory(index):
        return WikiAgent(
            index,
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key="sk-test-secret",
            transport=transport,
        )

    return factory


def _upload(
    server,
    body,
    filename="notes.txt",
    content_type="text/plain;charset=utf-8",
    *,
    csrf=True,
):
    path = "/api/import"
    if filename is not None:
        path += "?filename=" + quote(filename, safe="")
    headers = {}
    if content_type is not None:
        headers["Content-Type"] = content_type
    if csrf:
        headers["X-ZMWiki-Import"] = "1"
    return _request(server, path, "POST", body=body, headers=headers)


def test_import_creates_page_and_saves_workspace(tmp_path):
    workspace = _workspace(tmp_path)
    factory, transport = _compiler([
        {"title": "Release Notes", "content": "Version two changes."},
    ])
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"raw release notes")

        assert response.status == 200
        data = _json(body)
        assert data["status"] == "imported"
        assert data["imported_pages"] == [{
            "id": PAGE_ID,
            "path": "release-notes",
            "title": "Release Notes",
        }]
        assert data["updated_pages"] == []
        assert data["errors"] == []
        assert data["saved"] is True
        assert data["pages_total"] == 1
        assert data["links"] == {"resolved": 0, "ambiguous": 0, "unresolved": 0}
        assert data["source"]["filename"] == "notes.txt"
        assert data["source"]["key"] == "upload:notes.txt"
        assert data["source"]["document_id"] == "local-notes"
        assert len(data["source"]["fingerprint"]) == 64

        page = server.index.get(PAGE_ID)
        assert page.title == "Release Notes"
        assert page.content == "Version two changes."
        assert page.sources == ("local-notes",)
        # The uploaded text reached the compiler prompt.
        assert b"raw release notes" in transport.requests[0].body

    index, sources = load_workspace_state(workspace)
    assert index.get(PAGE_ID).title == "Release Notes"
    assert sources["upload:notes.txt"] == {
        "page_id": PAGE_ID,
        "document_id": "local-notes",
        "fingerprint": data["source"]["fingerprint"],
    }


def test_import_resolves_links_and_backlinks(tmp_path):
    workspace = _workspace(tmp_path)
    index = WikiIndex()
    index.upsert(WikiPage(
        id="beta", path="pages/beta", title="Beta", content="Beta body"
    ))
    factory, _ = _compiler([
        {"title": "Alpha", "content": "Alpha body", "links_to": ["Beta"]},
    ])
    with _serving(
        index, workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"alpha notes")

        assert response.status == 200
        assert _json(body)["links"] == {
            "resolved": 1, "ambiguous": 0, "unresolved": 0,
        }
        assert server.index.get(PAGE_ID).links_to == ("beta",)
        assert server.index.get("beta").links_from == (PAGE_ID,)
        assert server.index.check_links().is_clean


def test_reimport_same_filename_updates_in_place(tmp_path):
    workspace = _workspace(tmp_path)
    factory, _ = _compiler([
        {"title": "Notes", "content": "first body"},
        {"title": "Notes", "content": "second body"},
    ])
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, _ = _upload(server, b"first")
        assert response.status == 200
        first = json.loads(workspace.read_text(encoding="utf-8"))
        assert [page["id"] for page in first["pages"]] == [PAGE_ID]

        response, body = _upload(server, b"second")
        assert response.status == 200
        data = _json(body)
        assert data["status"] == "updated"
        assert data["updated_pages"] == [{
            "id": PAGE_ID, "path": "notes", "title": "Notes",
        }]
        assert data["imported_pages"] == []
        assert server.index.count() == 1
        assert server.index.get(PAGE_ID).content == "second body"


def test_import_never_overwrites_unrelated_page(tmp_path):
    workspace = _workspace(tmp_path)
    index = WikiIndex()
    index.upsert(WikiPage(
        id=PAGE_ID, path="manual", title="Manual", content="keep me"
    ))
    factory, _ = _compiler([{"title": "Notes", "content": "imported body"}])
    with _serving(
        index, workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 409
        assert _json(body)["error"] == "page_exists"
        assert server.index.get(PAGE_ID).content == "keep me"
        assert server.sources == {}


def test_import_rejects_path_conflict(tmp_path):
    workspace = _workspace(tmp_path)
    index = WikiIndex()
    index.upsert(WikiPage(
        id="other", path="notes", title="Other", content="other body"
    ))
    factory, _ = _compiler([{"title": "Notes", "content": "imported body"}])
    with _serving(
        index, workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 409
        assert _json(body)["error"] == "path_conflict"
        assert server.index.count() == 1


def test_import_rejects_invalid_requests(tmp_path):
    workspace = _workspace(tmp_path)
    factory, _ = _compiler([{"title": "Never", "content": "Never"}])
    cases = [
        ("notes.pdf", "text/plain", b"text", 400, "invalid_filename"),
        ("../notes.txt", "text/plain", b"text", 400, "invalid_filename"),
        ("C:notes.txt", "text/plain", b"text", 400, "invalid_filename"),
        (None, "text/plain", b"text", 400, "invalid_filename"),
        ("notes.txt", "application/json", b"{}", 415, "unsupported_media_type"),
        ("notes.txt", "text/plain", b"\xff\xfeabc", 400, "invalid_encoding"),
        ("notes.txt", "text/plain", b"  \n\t ", 400, "empty_document"),
        ("notes.txt", "text/plain", b"", 400, "invalid_body"),
    ]
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        for filename, content_type, payload, status, code in cases:
            response, body = _upload(
                server, payload, filename=filename, content_type=content_type
            )
            assert response.status == status, (filename, _json(body))
            assert _json(body)["error"] == code

        # A cross-origin simple request cannot import without the guard header.
        response, body = _upload(server, b"text", csrf=False)
        assert response.status == 403
        assert _json(body)["error"] == "forbidden_request"
        assert server.index.count() == 0
        assert server.sources == {}


def test_import_rejects_oversized_file(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    factory, _ = _compiler([{"title": "Never", "content": "Never"}])
    monkeypatch.setattr(server_module, "MAX_IMPORT_BYTES", 8)
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"x" * 9)

        assert response.status == 413
        assert _json(body)["error"] == "request_too_large"
        assert server.index.count() == 0


def test_import_rejected_when_read_only(tmp_path):
    workspace = _workspace(tmp_path)
    factory, _ = _compiler([{"title": "Never", "content": "Never"}])
    with _serving(
        WikiIndex(),
        workspace_path=workspace,
        read_only=True,
        compiler_factory=factory,
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 403
        assert _json(body)["error"] == "read_only"
        assert server.index.count() == 0


def test_import_requires_model_configuration(tmp_path):
    workspace = _workspace(tmp_path)
    with _serving(WikiIndex(), workspace_path=workspace) as server:
        response, body = _request(server, "/api/capabilities")
        assert _json(body)["import_configured"] is False

        response, body = _upload(server, b"notes")
        assert response.status == 503
        assert _json(body)["error"] == "model_not_configured"

    with _serving(WikiIndex()) as server:
        response, body = _upload(server, b"notes")
        assert response.status == 409
        assert _json(body)["error"] == "workspace_not_configured"


def test_compile_failure_leaves_workspace_unchanged(tmp_path):
    workspace = _workspace(tmp_path)
    before = workspace.read_bytes()
    factory, _ = _compiler([RuntimeError("network down")])
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 502
        assert _json(body)["error"] == "compile_failed"
        assert server.index.count() == 0
        assert server.sources == {}
        assert workspace.read_bytes() == before


def test_invalid_model_output_leaves_workspace_unchanged(tmp_path):
    workspace = _workspace(tmp_path)
    before = workspace.read_bytes()
    factory, _ = _compiler(["not a page object"])
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 502
        assert _json(body)["error"] == "compile_failed"
        assert b"not a page object" not in body
        assert server.index.count() == 0
        assert workspace.read_bytes() == before


def test_import_model_error_does_not_echo_upstream_body(tmp_path):
    workspace = _workspace(tmp_path)
    before = workspace.read_bytes()
    factory, _ = _compiler([
        HttpResponse(
            status=401,
            body=b'{"error": "sk-test-secret was rejected by upstream"}',
        ),
    ])
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 502
        data = _json(body)
        assert data["error"] == "compile_failed"
        assert "HTTP 401" in data["message"]
        assert b"sk-test-secret" not in body
        assert b"was rejected" not in body
        assert server.index.count() == 0
        assert workspace.read_bytes() == before


def test_save_failure_rolls_back_import(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    before = workspace.read_bytes()

    def boom(*args, **kwargs):
        raise WorkspaceError("disk full")

    monkeypatch.setattr(server_module, "save_workspace", boom)
    factory, _ = _compiler([{"title": "Notes", "content": "body"}])
    with _serving(
        WikiIndex(), workspace_path=workspace, compiler_factory=factory
    ) as server:
        response, body = _upload(server, b"notes")

        assert response.status == 500
        assert _json(body)["error"] == "save_failed"
        assert server.index.count() == 0
        assert server.sources == {}
        assert workspace.read_bytes() == before


def test_import_search_ask_save_and_restart(tmp_path):
    """End-to-end (mocked models): import TXT -> search -> links -> ask -> reload."""
    workspace = _workspace(tmp_path)
    index = WikiIndex()
    index.upsert(WikiPage(
        id="budget",
        path="pages/budget",
        title="Budget",
        content="The budget is two million.",
    ))
    save_workspace(index, workspace, overwrite=True)

    settings = SettingsStore(tmp_path / "settings" / "model.json")

    def tester(config):
        return ConnectionTestResult(True, "ok", "Connected.")

    compiler_factory, _ = _compiler([{
        "title": "Project Phoenix",
        "content": "Phoenix is the internal platform.",
        "links_to": ["Budget"],
    }])
    agent_factory = _agent_factory([
        {"action": "search", "query": "Phoenix"},
        {"action": "read", "paths": ["project-phoenix"]},
        {"action": "follow_links", "from_page": PAGE_ID},
        {
            "action": "answer",
            "text": "Phoenix is the internal platform.",
            "sources": [PAGE_ID, "budget"],
        },
    ])

    with _serving(
        index,
        workspace_path=workspace,
        compiler_factory=compiler_factory,
        agent_factory=agent_factory,
        settings_store=settings,
        connection_tester=tester,
    ) as server:
        # First-use step: configure the model from the browser API.
        secret = "sk-test-secret"
        response, body = _request(
            server,
            "/api/config",
            "POST",
            body=json.dumps({
                "base_url": "https://example.invalid/v1",
                "model": "test-model",
                "api_key": secret,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200
        assert _json(body)["configured"] is True
        assert secret.encode() not in body
        assert settings.load().api_key == secret

        response, body = _upload(server, b"Phoenix notes")
        assert response.status == 200
        assert _json(body)["status"] == "imported"
        assert server.index.get(PAGE_ID).links_to == ("budget",)

        response, body = _request(server, "/api/search?q=Phoenix")
        hits = [row["page_id"] for row in _json(body)["results"]]
        assert PAGE_ID in hits

        response, body = _request(server, f"/api/page?id={PAGE_ID}")
        page = _json(body)
        assert page["content"] == "Phoenix is the internal platform."
        assert page["links_to"] == ["budget"]
        response, body = _request(server, "/api/page?id=budget")
        assert _json(body)["links_from"] == [PAGE_ID]

        response, body = _request(
            server,
            "/api/ask",
            "POST",
            body=json.dumps({"question": "What is Phoenix?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200
        answer = _json(body)
        assert answer["status"] == AgentStatus.ANSWERED
        assert [step["action"] for step in answer["steps"]] == [
            "search", "read", "follow_links", "answer",
        ]
        assert answer["sources"] == ["project-phoenix", "pages/budget"]
        assert [item["id"] for item in answer["evidence"]] == [PAGE_ID, "budget"]
        assert answer["pages_read"] == 2
        assert "sk-test-secret" not in body.decode("utf-8")

    # A process restart must see the imported page, its links and its source.
    reloaded, sources = load_workspace_state(workspace)
    assert reloaded.get(PAGE_ID).links_to == ("budget",)
    assert reloaded.get("budget").links_from == (PAGE_ID,)
    assert sources["upload:notes.txt"]["page_id"] == PAGE_ID
    with _serving(
        reloaded,
        workspace_path=workspace,
        sources=sources,
        settings_store=settings,
        config_source="settings",
        model_config=settings.load(),
    ) as restarted:
        response, body = _request(restarted, f"/api/page?id={PAGE_ID}")
        assert response.status == 200
        assert _json(body)["title"] == "Project Phoenix"

        # The model configuration survives the restart without exposing the key.
        response, body = _request(restarted, "/api/config")
        assert response.status == 200
        config = _json(body)
        assert config["configured"] is True
        assert config["stored_api_key"] is True
        assert secret.encode() not in body


def test_workbench_exposes_import_controls():
    with _serving(WikiIndex()) as server:
        response, body = _request(server, "/app")
        assert response.status == 200
        assert b'id="import-file"' in body
        assert b'id="import-submit"' in body
        assert b'id="import-status"' in body

        response, body = _request(server, "/app.js")
        assert response.status == 200
        assert b"/api/import" in body
        assert b"submitImport" in body
