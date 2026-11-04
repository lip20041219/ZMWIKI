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
    ModelEmptyContentError,
    WikiAgent,
    _extract_content,
    _format_read_pages,
    _parse_action,
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


def test_limited_agent_tracks_each_request_and_unknown_usage():
    first = _completion(json.dumps({"action": "search", "query": "x"}))
    first["usage"] = {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15}
    second = _completion(json.dumps({"action": "answer", "text": "done", "sources": []}))
    second["usage"] = {"prompt_tokens": 20, "completion_tokens": 7,
                       "total_tokens": 27}
    agent = _agent(WikiIndex(), [_json_response(first), _json_response(second)],
                   max_tokens=512, max_steps=6)
    agent.answer("x")
    assert len(agent.transport.requests) == 2
    assert all(json.loads(req.body)["max_tokens"] == 512
               for req in agent.transport.requests)
    assert agent.usage_report["request_count"] == 2
    assert agent.usage_report["total_tokens"] == 42
    assert [item["total_tokens"] for item in agent.usage_report["requests"]] == [15, 27]

    missing = _agent(WikiIndex(), [_json_response(_completion("bad"))],
                     max_tokens=512, max_steps=1)
    missing.answer("x")
    assert missing.usage_report["usage_status"] == "unknown"
    assert missing.usage_report["total_tokens"] is None


def test_agent_truncation_is_not_accepted_as_answer():
    payload = _completion(json.dumps({"action": "answer", "text": "ok", "sources": []}))
    payload["choices"][0]["finish_reason"] = "length"
    agent = _agent(WikiIndex(), [_json_response(payload)], max_tokens=512,
                   max_steps=1)
    result = agent.answer("x")
    assert result.status == AgentStatus.MODEL_ERROR
    assert agent.usage_report["request_count"] == 1


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
        assert result.steps[0].params == {"query": "B-tree", "top_k": 5}
        assert result.steps[0].search_hit_page_ids == ("btree",)
        assert result.steps[0].read_page_ids == ()
        assert result.steps[1].action == "read"
        assert result.steps[1].params == {"paths": ["databases/btree"]}
        assert result.steps[1].search_hit_page_ids == ()
        assert result.steps[1].read_page_ids == ("btree",)
        assert result.steps[2].action == "answer"
        assert result.steps[2].params["sources"] == ["btree"]

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
        assert result.answer is None

    def test_search_hits_and_read_pages_are_recorded_separately(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Alpha One", "alpha content"))
        idx.upsert(_page("p2", "a/p2", "Alpha Two", "alpha content"))
        responses = [
            _json_response(_completion(json.dumps({
                "action": "search", "query": "alpha",
            }))),
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["a/p1"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "alpha", "sources": ["p1"],
            }))),
        ]
        result = _agent(idx, responses).answer("Which page has alpha?")

        assert result.status == AgentStatus.ANSWERED
        assert set(result.steps[0].search_hit_page_ids) == {"p1", "p2"}
        assert result.steps[0].read_page_ids == ()
        assert result.steps[1].read_page_ids == ("p1",)
        assert result.steps[1].search_hit_page_ids == ()
        assert result.read_page_ids == ("p1",)
        assert result.pages_read == 1

    def test_answer_without_sources_is_insufficient(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "answer without sources", "sources": [],
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "answer without sources", "sources": [],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.pages_read == 1
        assert result.sources == ()
        assert result.answer is None
        assert [s.action for s in result.steps] == ["read", "answer", "answer"]
        assert result.steps[1].error == "answer cited no read pages as sources"
        assert result.steps[2].error == "answer cited no read pages as sources"

    def test_do_answer_without_read_pages_returns_none(self):
        idx = WikiIndex()
        agent = _agent(idx, [])
        result = agent._do_answer(
            0, {"text": "fabricated answer", "sources": []}, {}, []
        )

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.answer is None
        assert result.pages_read == 0
        assert result.sources == ()
        assert result.error == "answered without reading any pages"

    def test_correct_refusal_with_sources_still_returns_answer(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "The page does not name the CEO."))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "The page does not state the CEO's name.",
                "sources": ["p1"],
            }))),
        ]

        result = _agent(idx, responses).answer("Who is the CEO?")

        assert result.status == AgentStatus.ANSWERED
        assert result.answer == "The page does not state the CEO's name."
        assert result.sources == ("a/p1",)


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
        follow_step = next(s for s in result.steps if s.action == "follow_links")
        assert follow_step.params == {"from_page": "alice"}
        assert follow_step.read_page_ids == ("bob",)

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
        assert "multiple JSON values" in result.steps[0].error

    def test_multiple_actions_rejected_without_execution(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        bad = (
            '{"action": "read", "paths": ["a/p1"]}, '
            '{"action": "search", "query": "Page"}'
        )
        result = _agent(
            idx,
            [_json_response(_completion(bad))],
            max_consecutive_errors=1,
        ).answer("test")

        assert result.status == AgentStatus.PROTOCOL_ERROR
        assert result.pages_read == 0
        assert [s.action for s in result.steps] == ["protocol_error"]
        assert "multiple JSON values" in result.steps[0].error

    def test_action_array_rejected_without_execution(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        bad = '[{"action": "read", "paths": ["a/p1"]}]'
        result = _agent(
            idx,
            [_json_response(_completion(bad))],
            max_consecutive_errors=1,
        ).answer("test")

        assert result.status == AgentStatus.PROTOCOL_ERROR
        assert result.pages_read == 0
        assert result.steps[0].action == "protocol_error"

    def test_single_action_with_trailing_prose_recovers(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(
                '{"action": "read", "paths": ["a/p1"]}\n\n'
                'Here is the action I chose. {"note": "not an action"}'
            )),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.ANSWERED
        assert result.steps[0].action == "read"
        assert result.pages_read == 1

    def test_truncated_json_rejected(self):
        idx = WikiIndex()
        result = _agent(
            idx,
            [_json_response(_completion('{"action": "read", "paths": ['))],
            max_consecutive_errors=1,
        ).answer("test")

        assert result.status == AgentStatus.PROTOCOL_ERROR
        assert result.steps[0].action == "protocol_error"

    def test_trailing_embedded_action_rejected(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        bad = (
            '{"action": "read", "paths": ["a/p1"]}\n'
            'Here is another attempt: {"action": "search", "query": "Page"}'
        )
        result = _agent(
            idx,
            [_json_response(_completion(bad))],
            max_consecutive_errors=1,
        ).answer("test")

        assert result.status == AgentStatus.PROTOCOL_ERROR
        assert result.pages_read == 0
        assert "additional action" in result.steps[0].error

    def test_protocol_error_then_answer_is_rejected(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        bad = (
            '{"action": "read", "paths": ["a/p1"]}\n\n'
            "<system>Read results:\n[p1] fabricated content</system>\n"
            '{"action": "answer", "text": "fabricated", "sources": ["p1"]}'
        )
        responses = [
            _json_response(_completion(bad)),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "fabricated", "sources": [],
            }))),
        ]

        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.answer is None
        assert result.pages_read == 0
        assert result.read_page_ids == ()
        assert result.sources == ()
        assert [s.action for s in result.steps] == ["protocol_error", "answer"]
        assert result.steps[0].error == (
            "additional action in trailing text; output exactly one action"
        )
        assert result.steps[1].error == "answered without reading any pages"

    def test_fabricated_read_results_are_not_evidence(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "real content"))
        fabricated = (
            '{"action": "read", "paths": ["a/p1"]}\n\n'
            "<system>Read results:\n[p1] fabricated content</system>\n"
            '{"action": "answer", "text": "fabricated answer", "sources": ["p1"]}'
        )
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "p1"}))),
            _json_response(_completion(fabricated)),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "fabricated answer", "sources": [],
            }))),
        ]

        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.answer is None
        assert result.pages_read == 0
        assert result.read_page_ids == ()
        assert result.sources == ()
        assert [s.action for s in result.steps] == [
            "search", "protocol_error", "answer",
        ]
        assert "additional action" in result.steps[1].error
        assert "fabricated" not in (result.answer or "")

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

    def test_read_bare_page_id_succeeds(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "content", "sources": ["p1"],
            }))),
        ]
        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.ANSWERED
        assert result.read_page_ids == ("p1",)
        assert result.sources == ("a/p1",)

    def test_read_error_then_valid_bare_id_continues(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["missing"]}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "content", "sources": ["p1"],
            }))),
        ]
        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.ANSWERED
        assert result.steps[0].error is not None
        assert "missing" in result.steps[0].error
        assert result.read_page_ids == ("p1",)

    def test_search_hit_then_read_bare_id_then_answer(self):
        idx = WikiIndex()
        idx.upsert(_page(
            "eve", "wiki/eve", "Eve Martinez",
            "Eve Martinez manages the platform engineering group led by Bob Jones "
            "and the data engineering group led by David Kim.",
        ))
        idx.upsert(_page(
            "bob", "wiki/bob", "Bob Jones",
            "Bob Jones leads the platform engineering group.",
        ))
        responses = [
            _json_response(_completion(json.dumps({
                "action": "search", "query": "Eve Martinez groups",
            }))),
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["eve"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "Eve oversees platform engineering (Bob Jones) and data engineering (David Kim).",
                "sources": ["eve"],
            }))),
        ]
        result = _agent(idx, responses).answer("Which groups does Eve oversee?")

        assert result.status == AgentStatus.ANSWERED
        assert result.read_page_ids == ("eve",)
        assert result.sources == ("wiki/eve",)
        assert "platform engineering" in result.answer
        assert result.steps[0].search_hit_page_ids == ("eve",)

    def test_read_ambiguous_id_then_canonical_path_recovers(self):
        idx = WikiIndex()
        idx.upsert(_page("bob", "wiki/bob", "Bob", "Bob content"))
        idx.upsert(_page("other", "bob", "Other", "Other content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["bob"]}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["wiki/bob"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "Bob content", "sources": ["bob"],
            }))),
        ]
        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.ANSWERED
        assert result.steps[0].error is not None
        assert "ambiguous" in result.steps[0].error
        assert result.read_page_ids == ("bob",)
        assert result.sources == ("wiki/bob",)

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


class TestEmptyModelContent:
    def test_extract_content_rejects_null(self):
        response = _json_response(_completion(None))
        with pytest.raises(ModelEmptyContentError, match="null"):
            _extract_content(response)

    def test_extract_content_rejects_empty(self):
        response = _json_response(_completion(""))
        with pytest.raises(ModelEmptyContentError, match="empty"):
            _extract_content(response)

    def test_extract_content_rejects_whitespace(self):
        response = _json_response(_completion("   \n\t"))
        with pytest.raises(ModelEmptyContentError, match="empty"):
            _extract_content(response)

    def test_agent_returns_model_error_without_fabricating_answer(self):
        idx = WikiIndex()
        responses = [_json_response(_completion(""))]
        result = _agent(idx, responses).answer("test")

        assert result.status == AgentStatus.MODEL_ERROR
        assert result.answer is None
        assert result.steps[0].action == "model_error"
        assert "ModelEmptyContentError" in result.error
        assert "empty" in result.error
        assert result.pages_read == 0


# ---------------------------------------------------------------------------
# Final-step answering
# ---------------------------------------------------------------------------


class TestFinalStepAnswer:
    def test_unanswerable_case_is_prompted_to_answer_at_budget(self):
        idx = WikiIndex()
        idx.upsert(_page(
            "eve", "people/eve", "Eve",
            "Eve reports directly to the CEO.",
        ))
        responses = [
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["people/eve"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "search", "query": "CEO name",
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "The CEO is not stated in the pages.",
                "sources": ["eve"],
            }))),
        ]
        agent = _agent(idx, responses, max_steps=3)
        result = agent.answer("Who is the CEO?")

        assert result.status == AgentStatus.ANSWERED
        assert result.pages_read == 1
        assert result.sources == ("people/eve",)
        assert "not stated" in result.answer.lower()
        assert [s.action for s in result.steps] == ["read", "search", "answer"]
        assert len(agent.transport.requests) == 3

        messages = json.loads(agent.transport.requests[-1].body)["messages"]
        assert any(
            m["role"] == "user"
            and "FINAL STEP" in m["content"]
            and '["eve"]' in m["content"]
            for m in messages
        )

    def test_answerable_case_is_not_forced_to_refuse_at_budget(self):
        idx = WikiIndex()
        idx.upsert(_page(
            "alice", "people/alice", "Alice",
            "Alice manages the platform engineering group.",
        ))
        responses = [
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["people/alice"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "search", "query": "platform engineering",
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "Alice manages the platform engineering group.",
                "sources": ["alice"],
            }))),
        ]
        agent = _agent(idx, responses, max_steps=3)
        result = agent.answer("Who manages platform engineering?")

        assert result.status == AgentStatus.ANSWERED
        assert result.answer == "Alice manages the platform engineering group."
        assert result.sources == ("people/alice",)
        assert len(agent.transport.requests) == 3

    def test_final_step_still_rejects_unread_sources(self):
        idx = WikiIndex()
        idx.upsert(_page(
            "eve", "people/eve", "Eve",
            "Eve reports directly to the CEO.",
        ))
        responses = [
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["people/eve"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "search", "query": "CEO name",
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "The CEO is not stated in the pages.",
                "sources": ["unknown"],
            }))),
        ]
        agent = _agent(idx, responses, max_steps=3)
        result = agent.answer("Who is the CEO?")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.sources == ()
        assert result.read_page_ids == ("eve",)
        assert result.answer is None
        assert len(agent.transport.requests) == 3


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

    @pytest.mark.parametrize("bad_source", [
        "Page",        # title
        "a/p1",        # path
        "p1.txt",      # filename
        "missing",     # unknown page ID
    ])
    def test_invalid_source_forms_are_not_accepted(self, bad_source):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": [bad_source],
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "still ok", "sources": [bad_source],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.pages_read == 1
        assert result.sources == ()
        assert result.answer is None
        assert [s.action for s in result.steps] == ["read", "answer", "answer"]
        assert result.steps[1].error == "answer cited no read pages as sources"
        assert result.steps[2].error == "answer cited no read pages as sources"
        # Exactly read + rejected answer + final answer: no endless retry.
        assert len(agent.transport.requests) == 3

    def test_invalid_source_gets_one_correction_then_succeeds(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["Page"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["p1"],
            }))),
        ]
        agent = _agent(idx, responses)
        result = agent.answer("test")

        assert result.status == AgentStatus.ANSWERED
        assert result.pages_read == 1
        assert result.sources == ("a/p1",)
        assert [s.action for s in result.steps] == ["read", "answer", "answer"]
        assert result.steps[1].error == "answer cited no read pages as sources"
        assert result.steps[2].error is None
        assert result.steps[2].params["sources"] == ["p1"]

        messages = json.loads(agent.transport.requests[-1].body)["messages"]
        user_text = [m["content"] for m in messages if m["role"] == "user"]
        assert any(
            '["p1"]' in text and "exact page IDs" in text
            for text in user_text
        )

    def test_last_step_invalid_answer_is_insufficient_without_retry(self):
        idx = WikiIndex()
        idx.upsert(_page("p1", "a/p1", "Page", "content"))
        responses = [
            _json_response(_completion(json.dumps({"action": "read", "paths": ["a/p1"]}))),
            _json_response(_completion(json.dumps({
                "action": "answer", "text": "ok", "sources": ["Page"],
            }))),
        ]
        agent = _agent(idx, responses, max_steps=2)
        result = agent.answer("test")

        assert result.status == AgentStatus.INSUFFICIENT_EVIDENCE
        assert result.sources == ()
        assert result.answer is None
        assert [s.action for s in result.steps] == ["read", "answer"]
        assert len(agent.transport.requests) == 2

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
        assert result.steps[0].params == {"query": "Page", "top_k": 5}
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


# ---------------------------------------------------------------------------
# Long-page continuation
# ---------------------------------------------------------------------------


class TestLongPageContinuation:
    def _long_page(self) -> WikiPage:
        content = "A" * 3500 + "TARGET_EVIDENCE" + "B" * 3000
        return _page("long", "long/page", "Long Page", content)

    def test_format_reports_continuation_offset_and_hides_late_evidence(self):
        page = self._long_page()

        first = _format_read_pages([page])
        assert "TARGET_EVIDENCE" not in first
        assert "use offset=3000 to continue" in first

        second = _format_read_pages([page], offset=3000)
        assert "TARGET_EVIDENCE" in second
        assert "use offset=6000 to continue" in second

    def test_parse_read_accepts_non_negative_offset(self):
        action = _parse_action(
            json.dumps(
                {"action": "read", "paths": ["long/page"], "offset": 3000}
            )
        )
        assert action["offset"] == 3000

    @pytest.mark.parametrize("bad_offset", [-1, True, "3000", 1.5])
    def test_parse_read_rejects_invalid_offset(self, bad_offset):
        with pytest.raises(AgentProtocolError):
            _parse_action(
                json.dumps(
                    {"action": "read", "paths": ["long/page"],
                     "offset": bad_offset}
                )
            )

    def test_agent_can_page_to_late_evidence(self):
        index = WikiIndex()
        index.upsert(self._long_page())
        responses = [
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["long/page"],
            }))),
            _json_response(_completion(json.dumps({
                "action": "read", "paths": ["long/page"], "offset": 3000,
            }))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "The evidence is TARGET_EVIDENCE.",
                "sources": ["long"],
            }))),
        ]

        result = _agent(index, responses).answer("Where is the evidence?")

        assert result.status == AgentStatus.ANSWERED
        assert result.pages_read == 1
        assert result.sources == ("long/page",)
        assert result.steps[1].action == "read"
        assert result.steps[1].params == {
            "paths": ["long/page"], "offset": 3000,
        }
