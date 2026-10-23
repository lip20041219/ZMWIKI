"""LLM generator tests: the OpenAI-compatible adapter over an injected transport.

Everything here is hermetic: the HTTP boundary is a fake transport, no socket is
opened, no key is needed, and nothing reaches a provider. These tests pin the
adapter's *protocol* -- the request it builds, the headers and auth it sends,
the parsing rules it enforces, and how it fails -- not urllib's behaviour.
"""

import ast
import json
import pathlib
import urllib.error
import urllib.request

import pytest

from zmwiki import (
    Chunk,
    Context,
    Generator,
    GeneratorConfigError,
    GeneratorError,
    GeneratorResponseError,
    GeneratorTransportError,
    HttpRequest,
    HttpResponse,
    OpenAICompatibleGenerator,
    build_prompt,
    urllib_transport,
)

BASE_URL = "https://provider.invalid/v1"
MODEL = "test-model"
ANSWER = "grounded answer"


# --- fakes: the injectable HTTP boundary ---------------------------------------


class FakeTransport:
    """Records every request and returns a preset response.

    Implements the ``Transport`` contract (``HttpRequest -> HttpResponse``), so
    the adapter is exercised end to end with no network involved.
    """

    def __init__(self, response=None, error=None):
        self.response = (
            response if response is not None else json_response(_completion(ANSWER))
        )
        self.error = error
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response

    @property
    def last(self):
        return self.requests[-1]

    def payload(self):
        return json.loads(self.last.body.decode("utf-8"))


def _completion(content, **extra):
    payload = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    payload.update(extra)
    return payload


def json_response(payload, status=200):
    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def raw_response(body, status=200):
    data = body if isinstance(body, bytes) else body.encode("utf-8")
    return HttpResponse(status=status, body=data)


def _chunk(cid, text):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


def _context(*entries):
    chunks = tuple(_chunk(cid, text) for cid, text in entries)
    text = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    return Context(text=text, chunks=chunks)


def _generator(*, transport=None, **overrides):
    kwargs = {"base_url": BASE_URL, "model": MODEL}
    kwargs.update(overrides)
    if transport is not None:
        kwargs["transport"] = transport
    return OpenAICompatibleGenerator(**kwargs)


def _generate(query="what is it?", context=None, *, transport=None, **overrides):
    gen = _generator(transport=transport, **overrides)
    return gen.generate(query, context if context is not None else _context(("A", "ctx")))


@pytest.fixture(autouse=True)
def _no_ambient_api_key(monkeypatch):
    """No test depends on the machine's environment (or leaks its key)."""
    monkeypatch.delenv("ZMWIKI_API_KEY", raising=False)


# --- 1-8. configuration --------------------------------------------------------


def test_constructor_validates_types():
    with pytest.raises(TypeError):
        _generator(base_url=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _generator(model=42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _generator(api_key=42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _generator(timeout="30")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _generator(temperature="0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _generator(max_tokens="10")  # type: ignore[arg-type]


def test_base_url_must_be_absolute_http():
    for bad in ("", "   ", "provider.invalid/v1", "/v1", "ftp://provider.invalid"):
        with pytest.raises(GeneratorConfigError):
            _generator(base_url=bad)
    with pytest.raises(GeneratorConfigError):
        _generator(base_url="https://")  # scheme but no host


def test_base_url_trailing_slash_is_absorbed():
    assert _generator(base_url="https://provider.invalid/v1/").base_url == BASE_URL
    assert _generator().base_url == BASE_URL


def test_model_must_be_a_non_empty_str():
    for bad in ("", "   ", "\n"):
        with pytest.raises(GeneratorConfigError):
            _generator(model=bad)
    assert _generator(model=MODEL).model == MODEL


def test_api_key_argument_is_used_and_never_exposed():
    gen = _generator(api_key="sk-secret-value")
    assert gen.has_api_key is True
    assert "sk-secret-value" not in repr(gen)
    assert repr(gen).endswith("api_key=set)")


def test_api_key_falls_back_to_the_zmwiki_environment_variable(monkeypatch):
    monkeypatch.setenv("ZMWIKI_API_KEY", "env-key")
    assert _generator().has_api_key is True
    # An explicit argument wins over the environment.
    assert _generator(api_key="arg-key")._api_key == "arg-key"
    monkeypatch.setenv("ZMWIKI_API_KEY", "   ")
    assert _generator().has_api_key is False  # blank env is not a key


def test_api_key_is_optional_and_not_read_from_vendor_variables(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "vendor-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "vendor-key")
    assert _generator().has_api_key is False  # only ZMWIKI_API_KEY is consulted


def test_api_key_must_not_be_blank_when_given():
    with pytest.raises(GeneratorConfigError):
        _generator(api_key="")
    with pytest.raises(GeneratorConfigError):
        _generator(api_key="   ")


def test_timeout_validation():
    for bad in (0, -1, 0.0, True):
        with pytest.raises((TypeError, GeneratorConfigError)):
            _generator(timeout=bad)
    assert _generator().timeout == 30.0
    assert _generator(timeout=5).timeout == 5.0
    assert isinstance(_generator(timeout=5).timeout, float)


def test_temperature_validation():
    for bad in (-0.1, 2.1, True):
        with pytest.raises((TypeError, GeneratorConfigError)):
            _generator(temperature=bad)
    assert _generator().temperature == 0.0  # greedy by default
    assert _generator(temperature=1).temperature == 1.0
    assert _generator(temperature=2.0).temperature == 2.0


def test_max_tokens_validation():
    for bad in (0, -5, True):
        with pytest.raises((TypeError, GeneratorConfigError)):
            _generator(max_tokens=bad)
    assert _generator().max_tokens is None  # omitted, not defaulted
    assert _generator(max_tokens=256).max_tokens == 256


def test_transport_must_be_callable():
    with pytest.raises(TypeError):
        _generator(transport="not a transport")


# --- 9-12. request construction ------------------------------------------------


def test_request_url_method_and_timeout():
    transport = FakeTransport()
    _generate(transport=transport, timeout=12)
    assert transport.last.url == f"{BASE_URL}/chat/completions"
    assert transport.last.method == "POST"
    assert transport.last.timeout == 12.0
    assert isinstance(transport.last, HttpRequest)


def test_request_headers_and_authorization():
    transport = FakeTransport()
    _generate(transport=transport, api_key="sk-test")
    assert transport.last.headers["Content-Type"] == "application/json"
    assert transport.last.headers["Accept"] == "application/json"
    assert transport.last.headers["Authorization"] == "Bearer sk-test"


def test_no_authorization_header_without_a_key():
    transport = FakeTransport()
    _generate(transport=transport)
    assert "Authorization" not in transport.last.headers


def test_request_payload_shape():
    transport = FakeTransport()
    _generate("what is it?", _context(("A", "ctx")), transport=transport, max_tokens=64)
    payload = transport.payload()
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 64
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_max_tokens_is_omitted_when_not_configured():
    transport = FakeTransport()
    _generate(transport=transport)
    assert "max_tokens" not in transport.payload()


def test_prompt_carries_the_shared_prompt_and_a_system_instruction():
    context = _context(("A", "ctx"), ("B", "more"))
    transport = FakeTransport()
    _generate("what is it?", context, transport=transport)
    messages = transport.payload()["messages"]
    assert messages[1]["content"] == build_prompt("what is it?", context)
    assert context.text in messages[1]["content"]
    assert "what is it?" in messages[1]["content"]
    system = messages[0]["content"]
    assert "context" in system.lower()
    assert "insufficient" in system.lower()  # no inventing facts


def test_request_body_is_utf8_json_bytes():
    transport = FakeTransport()
    _generate("¿qué es? 苹果", transport=transport)
    assert isinstance(transport.last.body, bytes)
    assert "苹果" in transport.payload()["messages"][1]["content"]


# --- 13-20. response parsing ---------------------------------------------------


def test_successful_response_returns_the_message_content():
    transport = FakeTransport(json_response(_completion("the answer")))
    assert _generate(transport=transport) == "the answer"


def test_empty_content_is_returned_verbatim():
    # "" is a valid str: the adapter does not invent a fallback sentence.
    assert _generate(transport=FakeTransport(json_response(_completion("")))) == ""


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"choices": []}, "empty"),
        ({"choices": None}, "no 'choices' list"),
        ({"id": "x"}, "no 'choices' list"),
        ({"choices": [{"message": {}}]}, "content must be a str"),
        ({"choices": [{"message": {"content": 42}}]}, "content must be a str"),
        ({"choices": [{"message": {"content": None}}]}, "content must be a str"),
        ({"choices": [{"message": "not an object"}]}, "no 'message' object"),
        ({"choices": [{"delta": {}}]}, "no 'message' object"),
        ({"choices": "not a list"}, "no 'choices' list"),
        ([{"choices": []}], "must be an object"),
        ("a string", "must be an object"),
    ],
)
def test_unusable_payloads_fail_explicitly(payload, match):
    transport = FakeTransport(json_response(payload))
    with pytest.raises(GeneratorResponseError, match=match):
        _generate(transport=transport)
    assert len(transport.requests) == 1  # it really was sent, then rejected


def test_malformed_json_fails_explicitly():
    transport = FakeTransport(raw_response("{not json at all"))
    with pytest.raises(GeneratorResponseError, match="not valid JSON"):
        _generate(transport=transport)


def test_non_utf8_body_fails_explicitly():
    transport = FakeTransport(raw_response(b"\xff\xfe\x00invalid"))
    with pytest.raises(GeneratorResponseError, match="not valid UTF-8"):
        _generate(transport=transport)


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_http_error_status_fails_explicitly(status):
    transport = FakeTransport(raw_response("upstream is unwell", status=status))
    with pytest.raises(GeneratorResponseError) as excinfo:
        _generate(transport=transport)
    assert str(status) in str(excinfo.value)
    assert "upstream is unwell" in str(excinfo.value)  # provider detail preserved


def test_http_error_body_is_not_parsed_as_json():
    # A non-2xx is reported as a status error even when the body is JSON.
    transport = FakeTransport(json_response({"detail": "nope"}, status=422))
    with pytest.raises(GeneratorResponseError, match="HTTP 422"):
        _generate(transport=transport)


def test_provider_error_payload_fails_explicitly():
    payload = {"error": {"message": "model not found", "type": "invalid_request"}}
    transport = FakeTransport(json_response(payload))  # HTTP 200 + error body
    with pytest.raises(GeneratorResponseError, match="model not found"):
        _generate(transport=transport)


def test_provider_error_payload_without_a_message_is_reported():
    transport = FakeTransport(json_response({"error": {"code": "nope"}}))
    with pytest.raises(GeneratorResponseError, match="provider error payload"):
        _generate(transport=transport)


def test_long_provider_text_is_bounded_in_the_error_message():
    transport = FakeTransport(raw_response("x" * 5000, status=500))
    with pytest.raises(GeneratorResponseError) as excinfo:
        _generate(transport=transport)
    assert len(str(excinfo.value)) < 1000
    assert "chars total" in str(excinfo.value)


# --- 21. transport failures ----------------------------------------------------


def test_transport_error_propagates_unchanged():
    error = GeneratorTransportError("connection refused")
    transport = FakeTransport(error=error)
    with pytest.raises(GeneratorTransportError) as excinfo:
        _generate(transport=transport)
    assert excinfo.value is error


def test_unexpected_transport_exception_is_not_swallowed():
    transport = FakeTransport(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _generate(transport=transport)


def test_transport_must_return_an_http_response():
    transport = FakeTransport(response="not a response")
    with pytest.raises(TypeError, match="must return an HttpResponse"):
        _generate(transport=transport)


def test_urllib_transport_maps_http_error_to_a_response(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 429, "Too Many Requests", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    response = urllib_transport(
        HttpRequest(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=b"{}",
            timeout=5.0,
        )
    )
    assert isinstance(response, HttpResponse)
    assert response.status == 429  # returned, not raised


def test_urllib_transport_maps_unreachable_provider_to_a_transport_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(GeneratorTransportError) as excinfo:
        urllib_transport(
            HttpRequest(
                url=f"{BASE_URL}/chat/completions",
                method="POST",
                headers={},
                body=b"{}",
                timeout=5.0,
            )
        )
    assert isinstance(excinfo.value.__cause__, urllib.error.URLError)


def test_urllib_transport_sends_method_headers_body_and_timeout(monkeypatch):
    captured = {}

    class FakeRawResponse:
        status = 200

        def read(self):
            return b'{"choices": []}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeRawResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    response = urllib_transport(
        HttpRequest(
            url=f"{BASE_URL}/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer k"},
            body=b'{"a": 1}',
            timeout=7.0,
        )
    )
    assert response == HttpResponse(status=200, body=b'{"choices": []}')
    assert captured["url"] == f"{BASE_URL}/chat/completions"
    assert captured["method"] == "POST"
    assert captured["body"] == b'{"a": 1}'
    assert captured["timeout"] == 7.0
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Authorization"] == "Bearer k"


# --- 22-23. inputs are read, never modified ------------------------------------


def test_query_is_not_modified_and_is_sent_verbatim():
    query = "  what   is it?  "
    transport = FakeTransport()
    _generate(query, transport=transport)
    assert query == "  what   is it?  "  # str is immutable; no rebinding here
    assert query in transport.payload()["messages"][1]["content"]


def test_context_and_its_chunks_are_never_modified():
    context = _context(("A", "first"), ("B", "second"))
    before = (context.text, context.chunks)
    chunks_before = [(c.id, c.text, c.position, dict(c.metadata)) for c in context.chunks]

    _generate("what is it?", context, transport=FakeTransport())

    assert (context.text, context.chunks) == before
    assert [
        (c.id, c.text, c.position, dict(c.metadata)) for c in context.chunks
    ] == chunks_before


def test_invalid_inputs_fail_before_any_request_is_sent():
    transport = FakeTransport()
    gen = _generator(transport=transport)
    for bad in ("", "   ", 42, None):
        with pytest.raises(TypeError):
            gen.generate(bad, _context(("A", "ctx")))  # type: ignore[arg-type]
    for bad in (None, "ctx", 42, ("A", "ctx")):
        with pytest.raises(TypeError):
            gen.generate("q", bad)  # type: ignore[arg-type]
    assert transport.requests == []  # nothing was sent


def test_empty_context_still_produces_a_well_formed_request():
    transport = FakeTransport()
    answer = _generate("q", Context(text="", chunks=()), transport=transport)
    assert answer == ANSWER
    assert "<context>\n\n</context>" in transport.payload()["messages"][1]["content"]


# --- 24. behaviour across calls ------------------------------------------------


def test_each_call_sends_exactly_one_request():
    transport = FakeTransport()
    gen = _generator(transport=transport)
    context = _context(("A", "ctx"))
    assert gen.generate("first", context) == ANSWER
    assert gen.generate("second", context) == ANSWER
    assert len(transport.requests) == 2
    assert transport.requests[0].body != transport.requests[1].body


def test_requests_are_deterministic_across_instances():
    context = _context(("A", "ctx"))
    first = FakeTransport()
    second = FakeTransport()
    _generate("q", context, transport=first)
    _generate("q", context, transport=second)
    assert first.last == second.last  # dataclass value equality
    assert first.last.body == second.last.body


def test_generator_is_reusable_after_a_failure():
    failing = FakeTransport(error=GeneratorTransportError("down"))
    gen = _generator(transport=failing)
    with pytest.raises(GeneratorTransportError):
        gen.generate("q", _context(("A", "ctx")))
    # Nothing is cached or latched: the next call sends again.
    gen.transport = FakeTransport()
    assert gen.generate("q", _context(("A", "ctx"))) == ANSWER


def test_the_api_key_never_appears_in_an_error_message():
    key = "sk-do-not-leak-me"
    transport = FakeTransport(raw_response("unauthorized", status=401))
    gen = _generator(transport=transport, api_key=key)
    with pytest.raises(GeneratorResponseError) as excinfo:
        gen.generate("q", _context(("A", "ctx")))
    assert key not in str(excinfo.value)
    assert key not in repr(gen)
    assert key not in repr(transport.last)


def test_errors_share_one_base_class():
    for error in (
        GeneratorConfigError,
        GeneratorTransportError,
        GeneratorResponseError,
    ):
        assert issubclass(error, GeneratorError)
    assert issubclass(GeneratorConfigError, ValueError)  # builtin catches still work


# --- 25. public API ------------------------------------------------------------


def test_public_import():
    from zmwiki import __all__

    expected = {
        "OpenAICompatibleGenerator",
        "HttpRequest",
        "HttpResponse",
        "Transport",
        "urllib_transport",
        "GeneratorError",
        "GeneratorConfigError",
        "GeneratorTransportError",
        "GeneratorResponseError",
    }
    assert expected <= set(__all__)


def test_adapter_is_a_generator_and_the_contract_is_unchanged():
    gen = _generator(transport=FakeTransport())
    assert isinstance(gen, Generator)
    assert gen.generate("q", _context(("A", "ctx"))) == ANSWER


def test_a_plain_callable_is_a_valid_transport():
    # The HTTP boundary is duck-typed: no client class, no SDK, no subclassing.
    seen = []

    def transport(request):
        seen.append(request)
        return json_response(_completion(ANSWER))

    assert _generate(transport=transport) == ANSWER
    assert len(seen) == 1


def test_module_imports_are_stdlib_only():
    import zmwiki.llm_generators as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    absolute = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:  # skip `.context`
            absolute.add((node.module or "").split(".")[0])
    assert absolute <= {"__future__", "json", "os", "urllib", "collections", "dataclasses"}
    for banned in ("openai", "anthropic", "requests", "httpx", "aiohttp"):
        assert banned not in absolute
