"""Model setup API: masking, persistence, validation and connection diagnostics."""

import http.client
import json
import os
import threading
from contextlib import contextmanager

from zmwiki.llm_generators import HttpResponse
from zmwiki.web.model_settings import (
    ConnectionTestResult,
    ModelConfig,
    SettingsStore,
    test_model_connection as run_connection_test,
)
from zmwiki.web.server import _load_or_create_workspace, make_server
from zmwiki.web.workspace import WorkspaceError
from zmwiki.wiki.wiki_index import WikiIndex

SECRET = "test-only-key-DO-NOT-LEAK-42"


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


def _post_json(server, path, payload):
    return _request(
        server,
        path,
        "POST",
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def _workspace(tmp_path):
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"version": 1, "pages": []}), encoding="utf-8")
    return path


def _settings(tmp_path):
    return SettingsStore(tmp_path / "settings" / "model.json")


def test_config_reports_unconfigured_state(tmp_path, monkeypatch):
    monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
    with _serving(WikiIndex(), workspace_path=_workspace(tmp_path)) as server:
        response, body = _request(server, "/api/config")
        assert response.status == 200
        data = _json(body)
        assert data == {
            "configured": False,
            "base_url": None,
            "model": None,
            "has_api_key": False,
            "stored_api_key": False,
            "key_from_environment": False,
            "source": "none",
            "settings_file_present": False,
            "read_only": False,
        }
        response, body = _request(server, "/api/capabilities")
        assert _json(body)["agent_configured"] is False
        assert _json(body)["import_configured"] is False


def test_save_config_masks_and_persists_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
    workspace = _workspace(tmp_path)
    store = _settings(tmp_path)
    with _serving(
        WikiIndex(), workspace_path=workspace,
        settings_store=store, config_source="none",
    ) as server:
        response, body = _post_json(server, "/api/config", {
            "base_url": "https://api.example.com/v1/",
            "model": "test-model",
            "api_key": SECRET,
        })
        assert response.status == 200
        data = _json(body)
        assert data["configured"] is True
        assert data["base_url"] == "https://api.example.com/v1"
        assert data["model"] == "test-model"
        assert data["has_api_key"] is True
        assert data["stored_api_key"] is True
        assert data["source"] == "ui"
        assert data["settings_file_present"] is True
        assert SECRET not in body.decode("utf-8")

        # The stored key lives server-side only.
        assert store.load().api_key == SECRET
        assert _secret_only_on_disk(store)

        response, body = _request(server, "/api/config")
        assert SECRET not in body.decode("utf-8")
        assert _json(body)["has_api_key"] is True

        response, body = _request(server, "/api/capabilities")
        assert _json(body)["agent_configured"] is True
        assert _json(body)["import_configured"] is True

    # A restart with the same settings file keeps the configuration (``main``
    # loads the file and passes it in, exactly as here).
    with _serving(
        WikiIndex(), workspace_path=workspace,
        settings_store=store, config_source="settings",
        model_config=store.load(),
    ) as restarted:
        response, body = _request(restarted, "/api/config")
        data = _json(body)
        assert data["configured"] is True
        assert data["source"] == "settings"
        assert data["stored_api_key"] is True
        assert SECRET not in body.decode("utf-8")


def _secret_only_on_disk(store):
    """The secret is on disk (server-side) but never in an object repr."""
    assert SECRET in store.path.read_text(encoding="utf-8")
    config = store.load()
    assert SECRET not in repr(config)
    if os.name != "nt":
        assert (store.path.stat().st_mode & 0o777) == 0o600
    return True


def test_blank_key_keeps_stored_key_and_clear_removes_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
    store = _settings(tmp_path)
    seen = []

    def tester(config):
        seen.append(config)
        return ConnectionTestResult(True, "ok", "Connected.")

    with _serving(
        WikiIndex(), workspace_path=_workspace(tmp_path),
        settings_store=store, connection_tester=tester,
    ) as server:
        _post_json(server, "/api/config", {
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
            "api_key": SECRET,
        })

        # The browser never receives the key, so a blank field must keep it.
        response, body = _post_json(server, "/api/config/test", {
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
        })
        assert response.status == 200
        assert _json(body)["ok"] is True
        assert seen[-1].api_key == SECRET
        assert SECRET not in body.decode("utf-8")

        response, body = _post_json(server, "/api/config", {
            "base_url": "https://api.example.com/v1",
            "model": "test-model",
        })
        assert _json(body)["stored_api_key"] is True

        response, body = _post_json(server, "/api/config", {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
            "clear_api_key": True,
        })
        assert _json(body)["stored_api_key"] is False
        assert _json(body)["has_api_key"] is False
        assert store.load().api_key is None


def test_config_validation_errors_are_safe(tmp_path, monkeypatch):
    monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)
    with _serving(
        WikiIndex(), workspace_path=_workspace(tmp_path),
        settings_store=_settings(tmp_path),
    ) as server:
        cases = [
            {"model": "m"},
            {"base_url": "https://api.example.com/v1"},
            {"base_url": "ftp://api.example.com/v1", "model": "m"},
            {"base_url": "https://user:pass@api.example.com/v1", "model": "m"},
            {"base_url": "https://api.example.com/v1?x=1", "model": "m"},
            {"base_url": "https://api.example.com/v1", "model": "bad\x00name"},
            {"base_url": "https://api.example.com/v1", "model": "m",
             "api_key": "has\nnewline"},
            {"base_url": "https://api.example.com/v1", "model": "m",
             "api_key": "x" * 5000},
            {"base_url": "https://api.example.com/v1", "model": "m",
             "clear_api_key": "yes"},
        ]
        for payload in cases:
            response, body = _post_json(server, "/api/config", payload)
            assert response.status == 400, (payload, _json(body))
            assert _json(body)["error"] == "invalid_config"
            assert SECRET not in body.decode("utf-8")

        response, body = _request(server, "/api/config")
        assert _json(body)["configured"] is False


def test_config_is_blocked_in_read_only_mode_but_test_is_allowed(tmp_path):
    def tester(config):
        return ConnectionTestResult(False, "auth_failed", "Authentication failed.")

    with _serving(
        WikiIndex(), workspace_path=_workspace(tmp_path),
        read_only=True, connection_tester=tester,
    ) as server:
        response, body = _request(server, "/api/config")
        assert response.status == 200
        assert _json(body)["read_only"] is True

        response, body = _post_json(server, "/api/config", {
            "base_url": "https://api.example.com/v1", "model": "m",
        })
        assert response.status == 403
        assert _json(body)["error"] == "read_only"

        response, body = _post_json(server, "/api/config/test", {
            "base_url": "https://api.example.com/v1", "model": "m",
        })
        assert response.status == 200
        assert _json(body)["code"] == "auth_failed"


def test_config_endpoints_reject_non_json_content_type(tmp_path):
    calls = []

    def tester(config):
        calls.append(config)
        return ConnectionTestResult(True, "ok", "Connected.")

    with _serving(
        WikiIndex(), workspace_path=_workspace(tmp_path),
        settings_store=_settings(tmp_path), connection_tester=tester,
    ) as server:
        # A cross-origin simple request cannot reconfigure the endpoint, so it
        # cannot redirect the stored key to an attacker-controlled URL.
        payload = json.dumps({
            "base_url": "http://attacker.invalid/v1", "model": "m",
        }).encode("utf-8")
        for path in ("/api/config", "/api/config/test"):
            response, body = _request(
                server, path, "POST", body=payload,
                headers={"Content-Type": "text/plain"},
            )
            assert response.status == 415
            assert _json(body)["error"] == "unsupported_media_type"
        assert calls == []
        response, body = _request(server, "/api/config")
        assert _json(body)["configured"] is False


def test_connection_test_endpoint_maps_safe_messages(tmp_path):
    results = {
        "ok": ConnectionTestResult(True, "ok", "Connected."),
        "auth_failed": ConnectionTestResult(False, "auth_failed", "Auth failed."),
        "connection_failed": ConnectionTestResult(
            False, "connection_failed", "Unreachable."
        ),
    }

    def tester_for(code):
        def tester(config):
            return results[code]
        return tester

    for code in results:
        with _serving(
            WikiIndex(), workspace_path=_workspace(tmp_path),
            connection_tester=tester_for(code),
        ) as server:
            response, body = _post_json(server, "/api/config/test", {
                "base_url": "https://api.example.com/v1", "model": "m",
            })
            assert response.status == 200
            assert _json(body)["code"] == code
            assert SECRET not in body.decode("utf-8")


def test_real_connection_tester_maps_http_outcomes_without_leaking():
    config = ModelConfig("https://api.example.com/v1", "m", SECRET)
    key_request = {}

    def respond(status, payload=None):
        def transport(request):
            key_request["headers"] = request.headers
            body = json.dumps(payload if payload is not None else {}).encode()
            return HttpResponse(status=status, body=body)
        return transport

    def raises(request):
        raise RuntimeError(f"{SECRET} exploded")

    cases = [
        (respond(200, {"choices": [{"message": {"content": "ok"}}]}), True, "ok"),
        (respond(401), False, "auth_failed"),
        (respond(403), False, "auth_failed"),
        (respond(404), False, "endpoint_not_found"),
        (respond(400), False, "bad_request"),
        (respond(429), False, "rate_limited"),
        (respond(500), False, "service_error"),
        (respond(418), False, "failed"),
        (respond(200, {"unexpected": True}), False, "invalid_response"),
        (raises, False, "connection_failed"),
    ]
    for transport, ok, code in cases:
        result = run_connection_test(config, transport=transport)
        assert result.ok is ok
        assert result.code == code
        assert SECRET not in result.message
        assert SECRET not in repr(result)

    # The key must reach the endpoint as the Authorization header.
    assert key_request["headers"]["Authorization"] == f"Bearer {SECRET}"


def test_api_key_is_not_logged(tmp_path, capsys):
    with _serving(
        WikiIndex(), workspace_path=_workspace(tmp_path),
        settings_store=_settings(tmp_path),
    ) as server:
        response, _ = _post_json(server, "/api/config", {
            "base_url": "https://api.example.com/v1",
            "model": "m",
            "api_key": SECRET,
        })
        assert response.status == 200
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err


def test_settings_store_ignores_corrupt_files(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    assert store.load() is None

    for payload in (
        "{not json",
        json.dumps([]),
        json.dumps({"version": 2, "base_url": "https://x/v1", "model": "m"}),
        json.dumps({"version": True, "base_url": "https://x/v1", "model": "m"}),
        json.dumps({"version": 1, "model": "m"}),
        json.dumps({"version": 1, "base_url": "https://x/v1"}),
        json.dumps({"version": 1, "base_url": "file:///etc", "model": "m"}),
    ):
        path.write_text(payload, encoding="utf-8")
        assert store.load() is None


def test_create_workspace_for_first_use(tmp_path):
    target = tmp_path / "new.json"
    index, sources = _load_or_create_workspace(target, create=True)
    assert index.count() == 0
    assert sources == {}
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "version": 1, "pages": [],
    }

    try:
        _load_or_create_workspace(target, create=True)
    except WorkspaceError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("--create must not overwrite an existing file")

    try:
        _load_or_create_workspace(tmp_path / "missing.json")
    except WorkspaceError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("a missing workspace must be reported")


def test_workbench_exposes_setup_wizard():
    with _serving(WikiIndex()) as server:
        response, body = _request(server, "/app")
        assert response.status == 200
        for marker in (
            b'id="setup"', b'id="setup-form"', b'id="setup-base-url"',
            b'id="setup-model"', b'id="setup-api-key"', b'id="setup-test"',
            b'id="setup-save"', b'id="setup-open"',
        ):
            assert marker in body, marker

        response, body = _request(server, "/app.js")
        assert b"/api/config" in body
        assert b"testSetupConnection" in body
        assert b"saveSetup" in body
        assert b"maybeOpenSetup" in body
