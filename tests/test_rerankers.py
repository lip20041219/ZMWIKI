"""Reranker tests: Reranker contract + DeterministicReranker re-ordering."""

import pytest

from zmrag import (
    Chunk,
    DeterministicReranker,
    Reranker,
    SearchResult,
    SimpleTokenizer,
)


def _chunk(cid, text):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


def _hits(*entries):
    """entries of (cid, text, score) -> SearchResult list (input order kept)."""
    return [SearchResult(chunk=_chunk(cid, text), score=score) for cid, text, score in entries]


def _rerank(query, candidates, top_k=10, **kw):
    return DeterministicReranker(**kw).rerank(query, candidates, top_k=top_k)


# --- contract shape ---


def test_reranker_is_abstract():
    with pytest.raises(TypeError):
        Reranker()  # type: ignore[abstract]
    assert isinstance(DeterministicReranker(), Reranker)


def test_deterministic_defaults_and_tokenizer_injection():
    rr = DeterministicReranker()
    assert isinstance(rr.tokenizer, SimpleTokenizer)
    custom = DeterministicReranker(tokenizer=SimpleTokenizer())
    assert custom.tokenizer is not SimpleTokenizer()
    with pytest.raises(TypeError):
        DeterministicReranker(tokenizer="nope")  # type: ignore[arg-type]


def test_rerank_signature_default_top_k():
    # default top_k=10; 2 candidates -> both returned
    out = _rerank("pie", _hits(("A", "pie", 1.0), ("B", "cake", 1.0)))
    assert len(out) == 2


# --- validation ---


@pytest.mark.parametrize("bad", ["", "   \n\t", 42, None, ["q"]])
def test_invalid_query_rejected(bad):
    with pytest.raises(TypeError):
        _rerank(bad, _hits(("A", "pie", 1.0)))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, "x", (), {"A": 1.0}, (r for r in [])])
def test_candidates_must_be_list(bad):
    with pytest.raises(TypeError):
        _rerank("q", bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, "chunk", _chunk("A", "pie"), 42])
def test_candidate_element_must_be_searchresult(bad):
    with pytest.raises(TypeError):
        _rerank("q", [bad])  # type: ignore[list-item]


def test_duplicate_chunk_id_rejected():
    dup = _hits(("A", "pie", 1.0), ("A", "pie", 1.0))
    with pytest.raises(ValueError):
        _rerank("pie", dup)
    # same id, different text/object still a duplicate -> reject
    diff = _hits(("A", "pie", 1.0), ("A", "cake", 2.0))
    with pytest.raises(ValueError):
        _rerank("pie", diff)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "10", True])
def test_top_k_validation(bad):
    hits = _hits(("A", "pie", 1.0))
    with pytest.raises((ValueError, TypeError)):
        _rerank("pie", hits, top_k=bad)  # type: ignore[arg-type]


def test_empty_candidates_returns_empty():
    assert _rerank("pie", []) == []


# --- recall boundary / impartiality ---


def test_output_ids_subset_of_input_ids():
    hits = _hits(("A", "pie apple", 0.2), ("B", "apple", 0.5), ("C", "banana", 0.9))
    out = _rerank("pie", hits)
    in_ids = {r.chunk.id for r in hits}
    assert {r.chunk.id for r in out} <= in_ids
    assert len(out) == len(hits)  # nothing dropped below top_k


def test_returns_same_chunk_objects_no_new_chunks():
    hits = _hits(("A", "pie", 0.1), ("B", "apple", 0.9))
    out = _rerank("pie", hits)
    chunk_objs = {id(r.chunk) for r in hits}
    assert all(id(r.chunk) in chunk_objs for r in out)


def test_candidates_and_chunks_not_mutated():
    hits = _hits(("A", "pie apple", 0.2), ("B", "banana", 0.9))
    before = [(r.chunk.id, r.chunk.text, r.score) for r in hits]
    _rerank("pie", hits)
    after = [(r.chunk.id, r.chunk.text, r.score) for r in hits]
    assert before == after


def test_outputs_are_fresh_searchresults():
    hits = _hits(("A", "pie", 0.1), ("B", "apple", 0.9))
    out = _rerank("pie", hits)
    ids_out = {id(r) for r in out}
    ids_in = {id(r) for r in hits}
    assert not (ids_out & ids_in)  # new objects, never the input instances


def test_same_query_repeated_is_deterministic():
    hits = _hits(("A", "apple pie", 0.4), ("B", "apple", 0.7), ("C", "pie", 0.1))
    rr = DeterministicReranker()
    first = [(r.chunk.id, r.score) for r in rr.rerank("apple pie", hits)]
    second = [(r.chunk.id, r.score) for r in rr.rerank("apple pie", hits)]
    assert first == second


def test_two_instances_agree():
    hits = _hits(("A", "apple pie", 0.4), ("B", "apple", 0.7))
    a = DeterministicReranker().rerank("apple", hits)
    b = DeterministicReranker().rerank("apple", hits)
    assert [(r.chunk.id, r.score) for r in a] == [(r.chunk.id, r.score) for r in b]


# --- scoring & ordering ---


def test_overlap_ranking_full_partial_none():
    hits = _hits(
        ("full", "apple pie crumble", 0.1),
        ("partial", "apple sauce", 0.2),
        ("none", "banana split", 0.9),
    )
    out = _rerank("apple pie", hits)
    assert [r.chunk.id for r in out] == ["full", "partial", "none"]


def test_score_is_token_overlap_relevance():
    # "apple pie pie" has pie twice: score = 1(apple) + 2(pie) = 3
    hits = _hits(
        ("once", "apple pie", 5.0),
        ("twice", "apple pie pie", 1.0),
        ("zero", "banana", 0.5),
    )
    out = _rerank("apple pie", hits)
    by_id = {r.chunk.id: r.score for r in out}
    assert by_id["twice"] == 3.0
    assert by_id["once"] == 2.0
    assert by_id["zero"] == 0.0


def test_score_overwrites_incoming_retrieval_score():
    # no token overlap -> 0.0 even though retrieval score was high
    out = _rerank("pie", _hits(("A", "banana smoothie", 999.0)))
    assert out[0].chunk.id == "A"
    assert out[0].score == 0.0


def test_duplicate_query_terms_not_double_counted():
    hits = _hits(("A", "apple pie", 1.0), ("B", "apple", 1.0))
    once = {r.chunk.id: r.score for r in _rerank("apple pie", hits)}
    twice = {r.chunk.id: r.score for r in _rerank("apple apple pie pie", hits)}
    assert once == twice


def test_case_insensitive_matching():
    hits = _hits(("A", "Apple Pie", 1.0), ("B", "apple crumble", 1.0))
    out = _rerank("APPLE", hits)
    assert [r.chunk.id for r in out] == ["A", "B"]
    assert all(r.score > 0.0 for r in out)


def test_order_is_score_desc_then_id_asc():
    hits = _hits(
        ("z1", "apple apple", 0.1),  # score 2
        ("a1", "apple", 0.9),        # score 1
        ("m1", "apple", 0.3),        # score 1 -> tie with a1, id asc
        ("z0", "banana", 0.2),       # score 0
    )
    out = _rerank("apple", hits)
    assert [r.chunk.id for r in out] == ["z1", "a1", "m1", "z0"]


def test_tie_break_independent_of_input_order():
    hits_a = _hits(("B", "apple", 0.2), ("A", "apple", 0.8))
    hits_b = _hits(("A", "apple", 0.8), ("B", "apple", 0.2))
    out_a = _rerank("apple", hits_a)
    out_b = _rerank("apple", hits_b)
    assert [r.chunk.id for r in out_a] == [r.chunk.id for r in out_b] == ["A", "B"]


def test_rerank_handles_all_zero_overlap_deterministically():
    hits = _hits(("c", "x", 1.0), ("a", "y", 1.0), ("b", "z", 1.0))
    out = _rerank("qqqq", hits)  # matches nothing
    assert [r.chunk.id for r in out] == ["a", "b", "c"]  # all 0 -> id asc


# --- top_k behaviour ---


def test_top_k_truncates_to_first_k():
    hits = _hits(
        ("A", "apple pie", 1.0),
        ("B", "apple", 1.0),
        ("C", "pie", 1.0),
        ("D", "banana", 1.0),
    )
    out = _rerank("apple pie", hits, top_k=2)
    assert len(out) == 2
    assert [r.chunk.id for r in out] == ["A", "B"]


def test_top_k_beyond_count_returns_all():
    hits = _hits(("A", "apple", 1.0), ("B", "banana", 1.0))
    out = _rerank("apple", hits, top_k=50)
    assert len(out) == 2


def test_public_import():
    from zmrag import __all__

    assert {"Reranker", "DeterministicReranker"} <= set(__all__)
    assert isinstance(DeterministicReranker(), Reranker)
