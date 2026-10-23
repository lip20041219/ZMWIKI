"""WikiAgent unit tests.

Uses a SequentialFakeTransport that returns a preset sequence of model
responses, so multi-turn Agent loops can be exercised with no real LLM.
Multi-hop tests use hand-built WikiPages where the answer requires
information from at least two pages.
"""

import json

import pytest

from zmwiki.wiki.wiki_agent import (
    AgentProtocolError,
    AgentResult,
    AgentStatus,
    AgentStep,
    WikiAgent,
)
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_retriever import wiki_read, wiki_search
from zmwiki.wiki.wiki_schema import WikiPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _json_response(payload: dict, status: int = 200):
    from zmwiki.llm_generators import HttpResponse

    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


class SequentialFakeTransport:
    """Returns responses from a preset list, one per call. Records requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self._idx >= len(self._responses):
            raise RuntimeError(
                f"SequentialFakeTransport exhausted after {len(self._responses)} calls"
            )
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def _agent(index, responses, **kwargs) -> WikiAgent:
    return WikiAgent(
        index,
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key="test-key",
        transport=SequentialFakeTransport(responses),
        **kwargs,
    )


def _page(page_id, path, title, content, links_to=(), links_from=(), **kwargs):
    return WikiPage(
        id=page_id, path=path, title=title, content=content,
        links_to=tuple(links_to), links_from=tuple(links_from), **kwargs,
    )


# ---------------------------------------------------------------------------
# Single-page question
# ---------------------------------------------------------------------------


class TestSinglePageQuestion:
    def test_search_read_answer_success(self):
        idx = WikiIndex()
        idx.upsert(_page(
            "btree", "databases/btree", "B-Tree Index",
            "A B-tree is a balanced tree data structure used for database indexes.",
        ))
        # Turn 1: model chooses search. Turn 2: model chooses read. Turn 3: answer.
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "B-tree"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["databases/btree"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "A B-tree is a balanced tree for database indexes.",
                "sources": ["btree"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("What is a B-tree?")

        assert isinstance(result, AgentResult)
        assert result.status == AgentStatus.ANSWERED
        assert result.answer == "A B-tree is a balanced tree for database indexes."
        assert result.sources == ("databases/btree",)
        assert result.pages_read == 1
        assert len(result.steps) == 3
        assert result.steps[0].action == "search"
        assert result.steps[1].action == "read"
        assert result.steps[2].action == "answer"

    def test_search_candidates_not_treated_as_evidence(self):
        """If the model answers after only searching (no read), status must be
        insufficient_evidence, not answered."""
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "Page"}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "I know it.", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.pages_read == 0
        assert result.sources == ()

    def test_answer_without_sources_is_insufficient(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "answer without sources", "sources": [],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.pages_read == 1
        assert result.sources == ()


# ---------------------------------------------------------------------------
# Multi-hop question
# ---------------------------------------------------------------------------


class TestMultiHopQuestion:
    def _multi_hop_index(self) -> WikiIndex:
        """Build a small wiki where the answer requires two pages.

        Page 'alice' mentions her colleague Bob (link to bob).
        Page 'bob' contains his job title.
        Question: "What is Alice's colleague Bob's job title?"
        """
        idx = WikiIndex()
        idx.upsert(_page(
            "alice", "people/alice", "Alice Smith",
            "Alice Smith is a senior engineer. Her colleague is Bob Jones, who works in the same team.",
            links_to=["bob"],
        ))
        idx.upsert(_page(
            "bob", "people/bob", "Bob Jones",
            "Bob Jones is the Director of Engineering at Acme Corp.",
            links_from=["alice"],
        ))
        return idx

    def test_follow_links_multi_hop(self):
        """Agent searches Alice, reads her page, follows link to Bob, reads Bob,
        then answers using both pages."""
        idx = self._multi_hop_index()
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "Alice colleague"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["people/alice"]}))),
            _json_response(_completion(json.dumps({"action": "follow_links", "from_page": "alice"}))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "Bob Jones is the Director of Engineering at Acme Corp.",
                "sources": ["alice", "bob"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("What is Alice's colleague Bob's job title?")

        assert result.status == AgentStatus.ANSWERED
        assert result.pages_read == 2
        assert "people/alice" in result.sources
        assert "people/bob" in result.sources
        # Verify follow_links step actually read bob.
        actions = [s.action for s in result.steps]
        assert "follow_links" in actions

    def test_research_then_read_multi_hop(self):
        """Agent searches, reads one page, then searches again for the second
        page, reads it, and answers."""
        idx = self._multi_hop_index()
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "Alice"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["people/alice"]}))),
            _json_response(_completion(json.dumps({"action": "search", "query": "Bob Jones title"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["people/bob"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "Bob is Director of Engineering.",
                "sources": ["alice", "bob"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("What is Bob's job title?")

        assert result.status == AgentStatus.ANSWERED
        assert result.pages_read == 2
        assert len(result.steps) == 5


# ---------------------------------------------------------------------------
# Protocol errors
# ---------------------------------------------------------------------------


class TestProtocolErrors:
    def test_invalid_json_does_not_execute(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion("this is not json")),
            _json_response(_completion(json.dumps({"action": "search", "query": "Page"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        # First step is protocol_error, then recovers.
        assert result.steps[0].action == "protocol_error"
        assert result.status == AgentStatus.ANSWERED

    def test_unknown_action_rejected(self):
        idx = WikiIndex()
        responses = [
            _json_response(_completion(json.dumps({"action": "delete_everything"}))),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.steps[0].action == "protocol_error"
        assert "delete_everything" in result.steps[0].error

    def test_consecutive_protocol_errors_stops(self):
        idx = WikiIndex()
        responses = [
            _json_response(_completion("bad json 1")),
            _json_response(_completion("bad json 2")),
        ]
        agent = _agent(idx, responses, max_consecutive_errors=2)
        result = agent.answer("test")

        assert result.status == AgentStatus.PROTOCOL_ERROR
        assert len(result.steps) == 2
        assert all(s.action == "protocol_error" for s in result.steps)

    def test_protocol_error_records_raw_output(self):
        idx = WikiIndex()
        bad = '{"action": "search", "query": "x"}{"action": "search", "query": "y"}'
        responses = [
            _json_response(_completion(bad)),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        result = _agent(idx, responses).answer("test")

        assert result.steps[0].action == "protocol_error"
        assert result.steps[0].raw_output == bad

    def test_raw_output_truncated_and_marked(self):
        idx = WikiIndex()
        bad = "x" * 2500
        result = _agent(
            idx, [_json_response(_completion(bad))], max_consecutive_errors=1,
        ).answer("test")

        assert result.status == AgentStatus.PROTOCOL_ERROR
        assert result.steps[0].raw_output == "x" * 2000 + "\n...[truncated]"

    def test_successful_steps_have_no_raw_output(self):
        idx = WikiIndex()
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "x"}))),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        result = _agent(idx, responses).answer("test")

        assert [s.action for s in result.steps] == ["search", "answer"]
        assert all(s.raw_output is None for s in result.steps)

    def test_missing_search_query_rejected(self):
        idx = WikiIndex()
        responses = [
            _json_response(_completion(json.dumps({"action": "search"}))),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")
        assert result.steps[0].action == "protocol_error"

    def test_missing_read_paths_rejected(self):
        idx = WikiIndex()
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": []}))),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")
        assert result.steps[0].action == "protocol_error"


# ---------------------------------------------------------------------------
# Budget & duplicate control
# ---------------------------------------------------------------------------


class TestBudgetControl:
    def test_max_steps_stops_agent(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        # Model keeps searching, never answers.
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": f"q{i}"})))
            for i in range(10)
        ]
        agent = _agent(idx, responses, max_steps=3)
        result = agent.answer("test")

        assert result.status == AgentStatus.BUDGET_EXHAUSTED
        assert len(result.steps) == 3
        assert result.error is not None
        assert "max_steps" in result.error

    def test_duplicate_search_ignored(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "Page"}))),
            _json_response(_completion(json.dumps({"action": "search", "query": "Page"}))),  # duplicate
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        actions = [s.action for s in result.steps]
        assert actions.count("search") == 1
        assert "duplicate" in actions

    def test_duplicate_read_ignored(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),  # dup
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        actions = [s.action for s in result.steps]
        assert actions.count("read") == 1
        assert "duplicate" in actions


# ---------------------------------------------------------------------------
# Read failures
# ---------------------------------------------------------------------------


class TestReadFailures:
    def test_nonexistent_page_not_fabricated(self):
        idx = WikiIndex()
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["nonexistent/page"]}))),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.steps[0].action == "read"
        assert result.steps[0].error is not None
        assert "nonexistent" in result.steps[0].error
        assert result.pages_read == 0

    def test_follow_links_from_unread_page_rejected(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content", links_to=["p2"]))
        idx.upsert(_page("p2", "a/p2", "Other", "other content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "follow_links", "from_page": "p1"}))),
            _json_response(_completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.steps[0].action == "follow_links"
        assert result.steps[0].error is not None
        assert "not been read" in result.steps[0].error

    def test_follow_links_with_no_links(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))  # no links_to
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({"action": "follow_links", "from_page": "p1"}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.steps[1].action == "follow_links"
        assert result.steps[1].result_summary == "no new links to follow"


# ---------------------------------------------------------------------------
# Model call failures
# ---------------------------------------------------------------------------


class TestModelFailures:
    def test_transport_exception_returns_model_error(self):
        idx = WikiIndex()
        responses = [ConnectionError("connection refused")]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.MODEL_ERROR
        assert "connection refused" in result.error
        assert result.answer is None

    def test_http_non_2xx_returns_model_error(self):
        idx = WikiIndex()
        responses = [_json_response({"error": "rate limited"}, status=429)]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.MODEL_ERROR
        assert result.answer is None


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------


class TestSourceTracking:
    def test_sources_only_include_read_pages(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page1", "content1"))
        idx.upsert(_page("p2", "a/p2", "Page2", "content2"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok",
                "sources": ["p1", "p2", "nonexistent"],  # p2 not read, nonexistent not real
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        # Only p1 was read, so only p1's path appears in sources.
        assert result.sources == ("a/p1",)

    def test_steps_record_full_action_history(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "Page"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert len(result.steps) == 3
        for step in result.steps:
            assert isinstance(step, AgentStep)
            assert step.step >= 0
        assert result.steps[0].params == {"query": "Page"}
        assert result.steps[1].params == {"paths": ["a/p1"]}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_question_rejected(self):
        idx = WikiIndex()
        agent = _agent(idx, [])
        with pytest.raises(TypeError):
            agent.answer("")

    def test_non_index_rejected(self):
        with pytest.raises(TypeError):
            WikiAgent(
                "not an index",  # type: ignore[arg-type]
                base_url="https://api.example.com/v1", model="m",
            )


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_wiki_search_unchanged(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Unique Title XYZ", "content"))
        results = wiki_search(idx, "Unique Title XYZ")
        assert len(results) == 1
        assert results[0].page_id == "p1"

    def test_wiki_read_unchanged(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        pages = wiki_read(idx, "a/p1")
        assert pages[0].id == "p1"

    def test_wikiindex_unchanged(self):
        idx = WikiIndex()
        assert idx.count() == 0
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        assert idx.count() == 1
        assert idx.check_links().is_clean

    def test_agent_not_in_top_level_exports(self):
        import zmwiki

        for name in ("WikiAgent", "AgentResult", "AgentStatus"):
            assert not hasattr(zmwiki, name)
