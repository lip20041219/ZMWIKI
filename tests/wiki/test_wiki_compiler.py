"""WikiCompiler unit tests.

Uses a FakeTransport (record-and-replay) so no real LLM call is made.
Covers successful compilation, index integration, source preservation,
model output failures, transport failures, deterministic id/path, and
regression guards.
"""

import json

import pytest

from zmwiki import Document
from zmwiki.wiki.wiki_compiler import (
    CompiledPage,
    CompilerResponseError,
    CompilerTransportError,
    CompilerValidationError,
    WikiCompiler,
)
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_retriever import wiki_read, wiki_search
from zmwiki.wiki.wiki_schema import WikiPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id="d1", content="A B-tree index keeps keys sorted and balanced."):
    return Document(id=doc_id, content=content)


def _completion(content: str, **extra) -> dict:
    payload = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    payload.update(extra)
    return payload


def _json_response(payload: dict, status: int = 200):
    from zmwiki.llm_generators import HttpResponse

    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


class FakeTransport:
    """Records every request and returns a preset response or raises."""

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else _json_response(
            _completion('{"title": "Test", "content": "test"}')
        )
        self.error = error
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def _compiler(transport=None) -> WikiCompiler:
    return WikiCompiler(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key="test-key",
        transport=transport or FakeTransport(),
    )


# ---------------------------------------------------------------------------
# Successful compilation
# ---------------------------------------------------------------------------


class TestSuccessfulCompilation:
    def test_compile_returns_compiled_page(self):
        model_json = json.dumps({
            "title": "B-Tree Index",
            "aliases": ["BTree", "balanced tree"],
            "tags": ["database", "index"],
            "description": "A balanced tree for database indexes.",
            "content": "A B-tree keeps keys sorted and balanced, enabling efficient search.",
        })
        transport = FakeTransport(response=_json_response(_completion(model_json)))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())

        assert isinstance(result, CompiledPage)
        assert isinstance(result.page, WikiPage)
        assert result.page.title == "B-Tree Index"
        assert result.page.aliases == ("BTree", "balanced tree")
        assert result.page.tags == ("database", "index")
        assert result.page.description == "A balanced tree for database indexes."
        assert result.page.content == "A B-tree keeps keys sorted and balanced, enabling efficient search."

    def test_sources_preserves_document_id(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "T", "content": "c"}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc(doc_id="my-doc-42"))
        assert result.page.sources == ("my-doc-42",)

    def test_page_id_stable_and_derived_from_document(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "Any Title", "content": "c"}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc(doc_id="doc123"))
        assert result.page.id == "page-doc123"

    def test_path_slugified_from_title(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "B-Tree Index!", "content": "c"}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())
        assert result.page.path == "b-tree-index"

    def test_path_fallback_when_title_slugifies_empty(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "!!!", "content": "c"}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc(doc_id="d7"))
        # Title "!!!" slugifies to empty, falls back to page-{doc_id}
        assert result.page.path == "page-d7"

    def test_links_to_empty_in_compiled_page(self):
        """Model-suggested links are NOT written to page.links_to."""
        model_json = json.dumps({
            "title": "Cache",
            "content": "LRU eviction.",
            "links_to": ["Memory Hierarchy", "Page Replacement"],
        })
        transport = FakeTransport(response=_json_response(_completion(model_json)))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())
        assert result.page.links_to == ()
        assert result.page.links_from == ()
        # Candidate links preserved in result and metadata.
        assert result.candidate_links == ("Memory Hierarchy", "Page Replacement")
        assert result.page.metadata["candidate_links"] == ["Memory Hierarchy", "Page Replacement"]

    def test_optional_fields_default_when_absent(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "Minimal", "content": "body"}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())
        assert result.page.aliases == ()
        assert result.page.tags == ()
        assert result.page.description is None
        assert result.candidate_links == ()

    def test_markdown_code_fence_stripped(self):
        fenced = '```json\n{"title": "Fenced", "content": "works"}\n```'
        transport = FakeTransport(response=_json_response(_completion(fenced)))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())
        assert result.page.title == "Fenced"
        assert result.page.content == "works"

    def test_raw_response_preserved_for_debugging(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "T", "content": "c"}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())
        assert '"title"' in result.raw_response
        assert '"content"' in result.raw_response

    def test_compiler_sends_correct_endpoint(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "T", "content": "c"}'
        )))
        compiler = _compiler(transport)
        compiler.compile(_doc())
        assert len(transport.requests) == 1
        req = transport.requests[0]
        assert req.url == "https://api.example.com/v1/chat/completions"
        assert req.method == "POST"
        body = json.loads(req.body)
        assert body["model"] == "test-model"
        assert body["temperature"] == 0.0
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert "d1" in body["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Index integration
# ---------------------------------------------------------------------------


class TestIndexIntegration:
    def test_compiled_page_registers_and_searchable(self):
        model_json = json.dumps({
            "title": "B-Tree Index",
            "content": "A B-tree keeps keys sorted and balanced for database indexes.",
            "tags": ["database"],
        })
        transport = FakeTransport(response=_json_response(_completion(model_json)))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())

        idx = WikiIndex()
        idx.upsert(result.page)
        assert idx.count() == 1

        search_results = wiki_search(idx, "B-Tree")
        assert len(search_results) >= 1
        assert search_results[0].page_id == result.page.id
        assert search_results[0].title == "B-Tree Index"

    def test_compiled_page_readable_by_path(self):
        model_json = json.dumps({
            "title": "Cache Eviction",
            "content": "LRU evicts least recently used entries.",
        })
        transport = FakeTransport(response=_json_response(_completion(model_json)))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())

        idx = WikiIndex()
        idx.upsert(result.page)

        pages = wiki_read(idx, "cache-eviction")
        assert len(pages) == 1
        assert pages[0].id == result.page.id
        assert pages[0].content == "LRU evicts least recently used entries."
        assert pages[0].sources == ("d1",)

    def test_compiled_page_passes_link_check(self):
        """A freshly compiled page has empty links_to/links_from, so check_links
        must report clean."""
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "T", "content": "c", "links_to": ["Other Page"]}'
        )))
        compiler = _compiler(transport)
        result = compiler.compile(_doc())

        idx = WikiIndex()
        idx.upsert(result.page)
        link_result = idx.check_links()
        assert link_result.is_clean
        assert link_result.broken_links == ()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_document_same_id_and_path_across_runs(self):
        model_json = json.dumps({"title": "Stable Title", "content": "c"})
        transport1 = FakeTransport(response=_json_response(_completion(model_json)))
        transport2 = FakeTransport(response=_json_response(_completion(model_json)))
        r1 = _compiler(transport1).compile(_doc(doc_id="doc-x"))
        r2 = _compiler(transport2).compile(_doc(doc_id="doc-x"))
        assert r1.page.id == r2.page.id
        assert r1.page.path == r2.page.path

    def test_different_documents_different_ids(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "Same Title", "content": "c"}'
        )))
        r1 = _compiler(transport).compile(_doc(doc_id="a"))
        r2 = _compiler(transport).compile(_doc(doc_id="b"))
        assert r1.page.id != r2.page.id


# ---------------------------------------------------------------------------
# Model output failures
# ---------------------------------------------------------------------------


class TestModelOutputFailures:
    def test_missing_title_raises_validation_error(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"content": "no title here"}'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError, match="title"):
            compiler.compile(_doc())

    def test_missing_content_raises_validation_error(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "No Content"}'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError, match="content"):
            compiler.compile(_doc())

    def test_empty_title_raises(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "   ", "content": "c"}'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError):
            compiler.compile(_doc())

    def test_invalid_json_raises_validation_error(self):
        transport = FakeTransport(response=_json_response(_completion(
            'this is not json {{{'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError):
            compiler.compile(_doc())

    def test_json_array_not_object_raises(self):
        transport = FakeTransport(response=_json_response(_completion(
            '["not", "an", "object"]'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError):
            compiler.compile(_doc())

    def test_wrong_field_type_raises(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": 123, "content": "c"}'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError):
            compiler.compile(_doc())

    def test_aliases_wrong_type_raises(self):
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "T", "content": "c", "aliases": "not-a-list"}'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError):
            compiler.compile(_doc())

    def test_no_fake_success_on_invalid_output(self):
        """Invalid model output must never produce a WikiPage."""
        transport = FakeTransport(response=_json_response(_completion(
            '{"title": "", "content": ""}'
        )))
        compiler = _compiler(transport)
        with pytest.raises(CompilerValidationError):
            compiler.compile(_doc())


# ---------------------------------------------------------------------------
# Transport / HTTP failures
# ---------------------------------------------------------------------------


class TestTransportFailures:
    def test_transport_exception_wrapped_as_transport_error(self):
        transport = FakeTransport(error=ConnectionError("connection refused"))
        compiler = _compiler(transport)
        with pytest.raises(CompilerTransportError, match="connection refused"):
            compiler.compile(_doc())

    def test_http_non_2xx_raises_response_error(self):
        transport = FakeTransport(response=_json_response(
            {"error": {"message": "rate limited"}}, status=429
        ))
        compiler = _compiler(transport)
        with pytest.raises(CompilerResponseError, match="429"):
            compiler.compile(_doc())

    def test_empty_choices_raises_response_error(self):
        transport = FakeTransport(response=_json_response({"choices": []}))
        compiler = _compiler(transport)
        with pytest.raises(CompilerResponseError):
            compiler.compile(_doc())

    def test_missing_content_raises_response_error(self):
        transport = FakeTransport(response=_json_response(
            {"choices": [{"message": {"role": "assistant"}}]}
        ))
        compiler = _compiler(transport)
        with pytest.raises(CompilerResponseError):
            compiler.compile(_doc())


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_non_document_rejected(self):
        compiler = _compiler()
        with pytest.raises(TypeError):
            compiler.compile("not a document")  # type: ignore[arg-type]

    def test_empty_document_content_rejected_at_document_level(self):
        """Document construction already rejects empty content, so compiler
        never sees it."""
        with pytest.raises(TypeError):
            Document(id="d", content="")


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_wikipage_contract_unchanged(self):
        page = WikiPage(id="p", path="p", title="t", content="c")
        assert page.id == "p"
        assert page.sources == ()

    def test_wikiindex_contract_unchanged(self):
        idx = WikiIndex()
        assert idx.count() == 0
        idx.upsert(WikiPage(id="p", path="p", title="t", content="c"))
        assert idx.count() == 1

    def test_compiler_not_in_top_level_exports(self):
        import zmwiki

        for name in ("WikiCompiler", "CompiledPage", "WikiCompilerError"):
            assert not hasattr(zmwiki, name)
