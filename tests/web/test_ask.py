"""POST /api/ask tests using a fake WikiAgent transport (no network)."""

import http.client
import json
import threading
import time
from contextlib import contextmanager

from zmwiki.llm_generators import HttpResponse
from zmwiki.web.server import make_server
from zmwiki.wiki.wiki_agent import AgentResult, AgentStatus, WikiAgent
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


class SequentialFakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, request):
        if not self.responses:
            raise RuntimeError("fake transport exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _completion(action: dict) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps(action),
                },
            }],
        }).encode("utf-8"),
    )


def _factory(responses, *, api_key="sk-test-secret"):
    def build(index):
        return WikiAgent(
            index,
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key=api_key,
            transport=SequentialFakeTransport(responses),
        )

    return build


def _index() -> WikiIndex:
    index = WikiIndex()
    index.upsert(WikiPage(
        id="alice",
        path="people/alice",
        title="Alice Smith",
        content="Alice leads engineering.",
        links_to=("bob",),
        sources=("doc-alice",),
    ))
    index.upsert(WikiPage(
        id="bob",
        path="people/bob",
        title="Bob Jones",
        content="Bob is Director of Engineering.",
        links_from=("alice",),
        sources=("doc-bob",),
    ))
    return index


def test_ask_returns_answer_steps_and_evidence():
    responses = [
        _completion({"action": "search", "query": "Alice"}),
        _completion({"action": "read", "paths": ["people/alice"]}),
        _completion({"action": "follow_links", "from_page": "alice"}),
        _completion({
            "action": "answer",
            "text": "Alice leads engineering and Bob is Director of Engineering.",
            "sources": ["alice", "bob"],
        }),
    ]
    with _serving(
        _index(), agent_factory=_factory(responses)
    ) as server:
        response, caps_body = _request(server, "/api/capabilities")
        assert _json(caps_body)["agent_configured"] is True

        response, body = _request(
            server,
            "/api/ask",
            "POST",
            payload={"question": "Who leads engineering and who is Bob?"},
        )

    assert response.status == 200
    payload = _json(body)
    assert payload["status"] == AgentStatus.ANSWERED
    assert payload["question"] == "Who leads engineering and who is Bob?"
    assert "Alice leads engineering" in payload["answer"]
    assert payload["sources"] == ["people/alice", "people/bob"]
    assert payload["read_page_ids"] == ["alice", "bob"]
    assert payload["pages_read"] == 2
    assert payload["error"] is None

    assert [step["action"] for step in payload["steps"]] == [
        "search", "read", "follow_links", "answer",
    ]
    assert payload["steps"][0]["search_hit_page_ids"]
    assert payload["steps"][1]["read_page_ids"] == ["alice"]
    assert payload["steps"][2]["read_page_ids"] == ["bob"]
    assert "raw_output" not in payload["steps"][2]

    evidence = {item["id"]: item for item in payload["evidence"]}
    assert evidence["alice"]["title"] == "Alice Smith"
    assert evidence["alice"]["path"] == "people/alice"
    assert evidence["alice"]["sources"] == ["doc-alice"]
    assert evidence["bob"]["sources"] == ["doc-bob"]
    assert "sk-test-secret" not in body.decode("utf-8")


def test_ask_reports_insufficient_evidence():
    responses = [
        _completion({"action": "search", "query": "Alice"}),
        _completion({"action": "read", "paths": ["people/alice"]}),
        _completion({
            "action": "answer",
            "text": "I am not sure.",
            "sources": [],
        }),
    ]
    with _serving(
        _index(), agent_factory=_factory(responses)
    ) as server:
        response, body = _request(
            server, "/api/ask", "POST", payload={"question": "What is the answer?"}
        )

    assert response.status == 200
    payload = _json(body)
    assert payload["status"] == AgentStatus.INSUFFICIENT_EVIDENCE
    assert payload["answer"] == "I am not sure."
    assert payload["sources"] == []
    assert payload["evidence"] == []
    assert "enough read evidence" in payload["error"]


def test_ask_requires_model_configuration():
    with _serving(_index()) as server:
        response, body = _request(server, "/api/capabilities")
        assert _json(body)["agent_configured"] is False

        response, body = _request(
            server, "/api/ask", "POST", payload={"question": "Alice?"}
        )
        assert response.status == 503
        assert _json(body)["error"] == "model_not_configured"


def test_ask_validates_question_and_json():
    with _serving(_index(), agent_factory=_factory([])) as server:
        for payload in (
            {},
            {"question": ""},
            {"question": "   "},
            {"question": 123},
            {"question": "x" * 4001},
        ):
            response, body = _request(server, "/api/ask", "POST", payload=payload)
            assert response.status == 400
            assert _json(body)["error"] == "invalid_question"

        response, body = _request(
            server,
            "/api/ask",
            "POST",
            body=b"{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert _json(body)["error"] == "invalid_json"


def test_ask_model_failure_is_safe():
    responses = [RuntimeError("sk-test-secret exploded")]
    with _serving(
        _index(), agent_factory=_factory(responses)
    ) as server:
        response, body = _request(
            server, "/api/ask", "POST", payload={"question": "Alice?"}
        )

    assert response.status == 200
    payload = _json(body)
    assert payload["status"] == AgentStatus.MODEL_ERROR
    assert payload["answer"] is None
    assert payload["error"]
    assert "sk-test-secret" not in body.decode("utf-8")
    assert "exploded" not in body.decode("utf-8")
    assert "raw_output" not in body.decode("utf-8")
    model_step = payload["steps"][0]
    assert model_step["action"] == "model_error"
    assert model_step["error"] is None


def test_ask_unexpected_failure_is_safe():
    def broken_factory(index):
        raise RuntimeError("sk-test-secret failed")

    with _serving(_index(), agent_factory=broken_factory) as server:
        response, body = _request(
            server, "/api/ask", "POST", payload={"question": "Alice?"}
        )

    assert response.status == 502
    assert _json(body)["error"] == "agent_failed"
    assert "sk-test-secret" not in body.decode("utf-8")


def test_ask_works_in_read_only_workspace():
    responses = [
        _completion({"action": "search", "query": "Alice"}),
        _completion({"action": "read", "paths": ["people/alice"]}),
        _completion({
            "action": "answer",
            "text": "Alice leads engineering.",
            "sources": ["alice"],
        }),
    ]
    with _serving(
        _index(), read_only=True, agent_factory=_factory(responses)
    ) as server:
        response, body = _request(
            server, "/api/ask", "POST", payload={"question": "Who leads engineering?"}
        )

    assert response.status == 200
    assert _json(body)["status"] == AgentStatus.ANSWERED


def test_ask_does_not_hold_index_lock_during_model_call():
    started = threading.Event()
    release = threading.Event()

    class BlockingAgent:
        def answer(self, question):
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test release timeout")
            return AgentResult(
                answer="ok",
                sources=(),
                status=AgentStatus.ANSWERED,
                steps=(),
                pages_read=0,
            )

    def factory(index):
        return BlockingAgent()

    with _serving(_index(), agent_factory=factory) as server:
        result = {}

        def ask():
            result["response"], result["body"] = _request(
                server, "/api/ask", "POST", payload={"question": "block"}
            )

        thread = threading.Thread(target=ask)
        thread.start()
        try:
            assert started.wait(timeout=5), "agent did not start"
            began = time.monotonic()
            response, body = _request(server, "/api/pages")
            elapsed = time.monotonic() - began
            assert response.status == 200
            assert elapsed < 2.0, f"reads blocked for {elapsed:.2f}s"
        finally:
            release.set()
            thread.join(timeout=5)

    assert result["response"].status == 200
    assert _json(result["body"])["status"] == AgentStatus.ANSWERED
