"""Tests for wiki_link_resolver: candidate link resolution and bidirectional
link construction. Uses hand-built WikiPages, no LLM calls.
"""

import json
import sys
from pathlib import Path

import pytest

# benchmark/ must be importable for MonitoringWikiIndex.
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from zmwiki.wiki.wiki_agent import AgentStatus, WikiAgent
from zmwiki.wiki.wiki_index import WikiIndex
from zmwiki.wiki.wiki_link_resolver import (
    LinkResolutionReport,
    ResolvedLink,
    resolve_candidate_links,
)
from zmwiki.wiki.wiki_schema import WikiPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page(page_id, path, title, content, aliases=(), links_to=(), links_from=(),
          sources=(), candidate_links=None, metadata=None):
    md = dict(metadata) if metadata else {}
    if candidate_links is not None:
        md["candidate_links"] = list(candidate_links)
    return WikiPage(
        id=page_id, path=path, title=title, content=content,
        aliases=tuple(aliases), links_to=tuple(links_to),
        links_from=tuple(links_from), sources=tuple(sources), metadata=md,
    )


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _json_response(payload: dict, status: int = 200):
    from zmwiki.llm_generators import HttpResponse
    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


class SequentialFakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self._idx >= len(self._responses):
            raise RuntimeError("SequentialFakeTransport exhausted")
        resp = self._responses[self._idx]
        self._idx += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


# ---------------------------------------------------------------------------
# Matching by identifier
# ---------------------------------------------------------------------------


class TestMatchingByIdentifier:
    def test_resolve_by_page_id(self):
        pages = [
            _page("alice", "wiki/alice", "Alice Smith", "content",
                  candidate_links=["bob"]),
            _page("bob", "wiki/bob", "Bob Jones", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 1
        assert report.resolved_links[0].match_method == "page_id"
        assert report.resolved_links[0].target_page_id == "bob"
        alice = next(p for p in resolved if p.id == "alice")
        assert "bob" in alice.links_to
        bob = next(p for p in resolved if p.id == "bob")
        assert "alice" in bob.links_from

    def test_resolve_by_path(self):
        pages = [
            _page("alice", "wiki/alice", "Alice Smith", "content",
                  candidate_links=["wiki/bob"]),
            _page("bob", "wiki/bob", "Bob Jones", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 1
        assert report.resolved_links[0].match_method == "path"
        alice = next(p for p in resolved if p.id == "alice")
        assert "bob" in alice.links_to

    def test_resolve_by_title(self):
        pages = [
            _page("alice", "wiki/alice", "Alice Smith", "content",
                  candidate_links=["Bob Jones"]),
            _page("bob", "wiki/bob", "Bob Jones", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 1
        assert report.resolved_links[0].match_method == "title"

    def test_resolve_by_alias(self):
        pages = [
            _page("alice", "wiki/alice", "Alice Smith", "content",
                  candidate_links=["Bobby"]),
            _page("bob", "wiki/bob", "Bob Jones", "content", aliases=["Bobby"]),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 1
        assert report.resolved_links[0].match_method == "alias"


# ---------------------------------------------------------------------------
# Unresolved & ambiguous
# ---------------------------------------------------------------------------


class TestUnresolvedAndAmbiguous:
    def test_nonexistent_target_is_unresolved(self):
        pages = [
            _page("alice", "wiki/alice", "Alice", "content",
                  candidate_links=["nonexistent"]),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.unresolved_count == 1
        assert report.resolved_count == 0
        assert ("alice", "nonexistent") in report.unresolved_candidates
        # No broken link created.
        alice = next(p for p in resolved if p.id == "alice")
        assert alice.links_to == ()

    def test_ambiguous_candidate_reported(self):
        pages = [
            _page("alice", "wiki/alice", "Alice", "content",
                  candidate_links=["Bob"]),
            _page("bob1", "wiki/bob1", "Bob", "content"),
            _page("bob2", "wiki/bob2", "Bob", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.ambiguous_count == 1
        assert report.resolved_count == 0
        amb = report.ambiguous_candidates[0]
        assert amb[0] == "alice"
        assert amb[1] == "Bob"
        assert set(amb[2]) == {"bob1", "bob2"}
        # No link created for ambiguous candidate.
        alice = next(p for p in resolved if p.id == "alice")
        assert alice.links_to == ()

    def test_self_link_skipped(self):
        pages = [
            _page("alice", "wiki/alice", "Alice", "content",
                  candidate_links=["alice"]),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.self_link_skipped == 1
        assert report.resolved_count == 0
        alice = next(p for p in resolved if p.id == "alice")
        assert "alice" not in alice.links_to


# ---------------------------------------------------------------------------
# Bidirectional links & check_links
# ---------------------------------------------------------------------------


class TestBidirectionalLinks:
    def test_bidirectional_links_built(self):
        pages = [
            _page("a", "wiki/a", "A", "content", candidate_links=["b"]),
            _page("b", "wiki/b", "B", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        a = next(p for p in resolved if p.id == "a")
        b = next(p for p in resolved if p.id == "b")
        assert "b" in a.links_to
        assert "a" in b.links_from

    def test_check_links_clean_after_resolution(self):
        pages = [
            _page("a", "wiki/a", "A", "content", candidate_links=["b", "c"]),
            _page("b", "wiki/b", "B", "content"),
            _page("c", "wiki/c", "C", "content"),
        ]
        resolved, _ = resolve_candidate_links(pages)
        idx = WikiIndex()
        for p in resolved:
            idx.upsert(p)
        report = idx.check_links()
        assert report.is_clean
        assert report.broken_links == ()
        assert report.inconsistent_links == ()

    def test_existing_valid_links_preserved(self):
        pages = [
            _page("a", "wiki/a", "A", "content", links_to=["b"], links_from=[]),
            _page("b", "wiki/b", "B", "content", links_to=[], links_from=["a"]),
        ]
        resolved, report = resolve_candidate_links(pages)
        a = next(p for p in resolved if p.id == "a")
        assert "b" in a.links_to
        assert report.broken_existing_links == 0
        assert report.inconsistent_existing_links == 0

    def test_existing_broken_link_reported_and_cleaned(self):
        pages = [
            _page("a", "wiki/a", "A", "content", links_to=["nonexistent"]),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.broken_existing_links == 1
        assert ("a", "nonexistent") in report.broken_existing
        a = next(p for p in resolved if p.id == "a")
        assert "nonexistent" not in a.links_to

    def test_existing_inconsistent_link_fixed(self):
        # A links to B, but B doesn't link from A.
        pages = [
            _page("a", "wiki/a", "A", "content", links_to=["b"]),
            _page("b", "wiki/b", "B", "content", links_from=()),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.inconsistent_existing_links == 1
        b = next(p for p in resolved if p.id == "b")
        assert "a" in b.links_from  # fixed
        idx = WikiIndex()
        for p in resolved:
            idx.upsert(p)
        assert idx.check_links().is_clean


# ---------------------------------------------------------------------------
# Idempotency & order independence
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_repeat_resolution_no_duplicate_links(self):
        pages = [
            _page("a", "wiki/a", "A", "content", candidate_links=["b"]),
            _page("b", "wiki/b", "B", "content"),
        ]
        resolved1, _ = resolve_candidate_links(pages)
        resolved2, report2 = resolve_candidate_links(resolved1)
        a = next(p for p in resolved2 if p.id == "a")
        assert a.links_to.count("b") == 1  # not duplicated
        # Second pass still resolves the candidate (it's still in metadata),
        # but no duplicate link is added.
        assert report2.resolved_count == 1

    def test_order_independent_link_set(self):
        p_a = _page("a", "wiki/a", "A", "content", candidate_links=["b"])
        p_b = _page("b", "wiki/b", "B", "content", candidate_links=["a"])
        forward, _ = resolve_candidate_links([p_a, p_b])
        reverse, _ = resolve_candidate_links([p_b, p_a])
        forward_links = {(p.id, p.links_to, p.links_from) for p in forward}
        reverse_links = {(p.id, p.links_to, p.links_from) for p in reverse}
        assert forward_links == reverse_links

    def test_original_pages_unmodified(self):
        original_a = _page("a", "wiki/a", "A", "content", candidate_links=["b"])
        original_b = _page("b", "wiki/b", "B", "content")
        original_a_links = original_a.links_to
        original_b_links = original_b.links_from
        resolve_candidate_links([original_a, original_b])
        assert original_a.links_to == original_a_links
        assert original_b.links_from == original_b_links
        assert original_a.links_to == ()  # still empty in original

    def test_page_id_path_sources_content_unchanged(self):
        pages = [
            _page("a", "wiki/a", "A", "orig content", candidate_links=["b"],
                  sources=("doc_a",)),
            _page("b", "wiki/b", "B", "other content", sources=("doc_b",)),
        ]
        resolved, _ = resolve_candidate_links(pages)
        a = next(p for p in resolved if p.id == "a")
        assert a.id == "a"
        assert a.path == "wiki/a"
        assert a.title == "A"
        assert a.content == "orig content"
        assert a.sources == ("doc_a",)


# ---------------------------------------------------------------------------
# WikiIndex & WikiAgent integration
# ---------------------------------------------------------------------------


class TestIntegrationWithIndexAndAgent:
    def test_resolved_pages_register_and_searchable(self):
        pages = [
            _page("alice", "wiki/alice", "Alice Smith", "Alice works on Project Phoenix.",
                  candidate_links=["Project Phoenix"]),
            _page("project_phoenix", "wiki/project-phoenix", "Project Phoenix",
                  "Project Phoenix is a migration initiative."),
        ]
        resolved, _ = resolve_candidate_links(pages)
        idx = WikiIndex()
        for p in resolved:
            idx.upsert(p)
        assert idx.count() == 2
        assert idx.check_links().is_clean
        from zmwiki.wiki.wiki_retriever import wiki_search
        results = wiki_search(idx, "Project Phoenix")
        assert any(r.page_id == "project_phoenix" for r in results)

    def test_follow_links_actually_reads_target_page(self):
        """End-to-end: compile pages with candidate_links -> resolve -> register
        -> Agent reads source page -> follow_links -> target page is actually
        read (verified via MonitoringWikiIndex)."""
        from zmwiki.wiki.wiki_link_resolver import resolve_candidate_links as resolve
        from run_multihop_evaluation import MonitoringWikiIndex

        pages = [
            _page("alice", "wiki/alice", "Alice Smith",
                  "Alice reports to Bob Jones.", candidate_links=["Bob Jones"]),
            _page("bob", "wiki/bob", "Bob Jones",
                  "Bob Jones is the Director of Engineering."),
        ]
        resolved, report = resolve(pages)
        assert report.resolved_count == 1

        inner_idx = WikiIndex()
        for p in resolved:
            inner_idx.upsert(p)
        assert inner_idx.check_links().is_clean

        monitor = MonitoringWikiIndex(inner_idx)

        # Agent: search -> read alice -> follow_links -> answer
        responses = [
            _json_response(_completion(json.dumps({"action": "search", "query": "Alice manager"}))),
            _json_response(_completion(json.dumps({"action": "read", "paths": ["wiki/alice"]}))),
            _json_response(_completion(json.dumps({"action": "follow_links", "from_page": "alice"}))),
            _json_response(_completion(json.dumps({
                "action": "answer",
                "text": "Bob Jones is the Director of Engineering.",
                "sources": ["alice", "bob"],
            }))),
        ]
        agent = WikiAgent(
            monitor, base_url="https://api.example.com/v1", model="test",
            api_key="key", transport=SequentialFakeTransport(responses),
        )
        result = agent.answer("Who is Alice's manager?")
        assert result.status == AgentStatus.ANSWERED
        # The critical assertion: bob was actually read via follow_links.
        assert "bob" in monitor.accessed_page_ids
        assert "alice" in monitor.accessed_page_ids
        assert result.pages_read == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_page_list(self):
        resolved, report = resolve_candidate_links([])
        assert resolved == []
        assert report.pages_processed == 0
        assert report.candidates_total == 0

    def test_no_candidate_links(self):
        pages = [
            _page("a", "wiki/a", "A", "content"),
            _page("b", "wiki/b", "B", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 0
        assert report.candidates_total == 0
        # Pages returned unchanged.
        assert resolved[0].links_to == ()
        assert resolved[1].links_from == ()

    def test_candidate_links_not_a_list_ignored(self):
        pages = [
            _page("a", "wiki/a", "A", "content", metadata={"candidate_links": "not a list"}),
            _page("b", "wiki/b", "B", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 0

    def test_empty_string_candidate_ignored(self):
        pages = [
            _page("a", "wiki/a", "A", "content", candidate_links=["", "  ", "b"]),
            _page("b", "wiki/b", "B", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 1  # only "b"
        assert report.candidates_total == 1  # empty strings not counted

    def test_non_string_candidate_ignored(self):
        pages = [
            _page("a", "wiki/a", "A", "content",
                  metadata={"candidate_links": [123, None, "b"]}),
            _page("b", "wiki/b", "B", "content"),
        ]
        resolved, report = resolve_candidate_links(pages)
        assert report.resolved_count == 1

    def test_non_wikipage_rejected(self):
        with pytest.raises(TypeError):
            resolve_candidate_links(["not a page"])
        with pytest.raises(TypeError):
            resolve_candidate_links("not a list")  # type: ignore[arg-type]

    def test_report_is_clean_property(self):
        pages = [
            _page("a", "wiki/a", "A", "content", candidate_links=["b"]),
            _page("b", "wiki/b", "B", "content"),
        ]
        _, report = resolve_candidate_links(pages)
        assert report.is_clean is True

        pages2 = [
            _page("a", "wiki/a", "A", "content", candidate_links=["nonexistent"]),
        ]
        _, report2 = resolve_candidate_links(pages2)
        assert report2.is_clean is False


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_wikipage_contract_unchanged(self):
        p = WikiPage(id="x", path="wiki/x", title="X", content="c")
        assert p.id == "x"
        assert p.links_to == ()
        assert p.metadata == {}

    def test_wikiindex_upsert_unchanged(self):
        idx = WikiIndex()
        idx.upsert(WikiPage(id="x", path="wiki/x", title="X", content="c"))
        assert idx.count() == 1
        assert idx.get("x") is not None

    def test_resolver_not_in_top_level_exports(self):
        import zmwiki
        for name in ("resolve_candidate_links", "LinkResolutionReport"):
            assert not hasattr(zmwiki, name)
