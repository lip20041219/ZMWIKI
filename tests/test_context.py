"""Context Builder tests: Context contract + DeterministicContextBuilder packing."""

import dataclasses

import pytest

from zmrag import (
    Chunk,
    Context,
    ContextBuilder,
    DeterministicContextBuilder,
    SearchResult,
)


def _chunk(cid, text):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


def _hits(*entries):
    """entries of (cid, text, score) -> SearchResult list (input order kept)."""
    return [SearchResult(chunk=_chunk(cid, text), score=score) for cid, text, score in entries]


def _build(results, max_chars):
    return DeterministicContextBuilder().build(results, max_chars)


# --- Contract ---


def test_context_is_frozen():
    ctx = Context(text="x", chunks=(_chunk("A", "apple"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.text = "y"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.chunks = ()  # type: ignore[misc]


def test_context_chunks_is_tuple_and_no_append():
    chunks = (_chunk("A", "apple"), _chunk("B", "beta"))
    ctx = Context(text="ab", chunks=chunks)
    assert isinstance(ctx.chunks, tuple)
    assert not hasattr(ctx.chunks, "append")


def test_context_field_validation():
    with pytest.raises(TypeError):
        Context(text=42, chunks=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Context(text="x", chunks=[_chunk("A", "a")])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Context(text="x", chunks=("not a chunk",))  # type: ignore[assignment]


def test_context_builder_is_abstract():
    with pytest.raises(TypeError):
        ContextBuilder()  # type: ignore[abstract]
    assert isinstance(DeterministicContextBuilder(), ContextBuilder)


def test_public_import():
    from zmrag import __all__

    assert {"Context", "ContextBuilder", "DeterministicContextBuilder"} <= set(__all__)


# --- Basic ---


def test_empty_results():
    ctx = _build([], max_chars=100)
    assert ctx == Context(text="", chunks=())
    assert ctx.text == ""
    assert ctx.chunks == ()


def test_one_result():
    ctx = _build(_hits(("A", "apple", 0.9)), max_chars=100)
    assert ctx.text == "[1] apple"
    assert [c.id for c in ctx.chunks] == ["A"]


def test_multiple_results_all_fit():
    ctx = _build(
        _hits(("A", "apple", 0.3), ("B", "beta", 0.6), ("C", "cake", 0.9)),
        max_chars=100,
    )
    assert ctx.text == "[1] apple\n\n[2] beta\n\n[3] cake"
    assert [c.id for c in ctx.chunks] == ["A", "B", "C"]


def test_input_order_preserved_not_resorted():
    # high score and small id both arrive last; order must follow the list
    results = _hits(("z", "apple", 0.1), ("a", "beta cake", 0.99))
    ctx = _build(results, max_chars=100)
    assert [c.id for c in ctx.chunks] == ["z", "a"]
    assert ctx.text.startswith("[1] apple")


# --- Budget ---


def test_max_chars_truncates_tail():
    # "[1] aaaa" = 8 chars (4 for text) ... craft so only A fits
    results = _hits(("A", "aaaa", 0.5), ("B", "bbbb", 0.5))
    ctx = _build(results, max_chars=8)
    assert [c.id for c in ctx.chunks] == ["A"]
    assert ctx.text == "[1] aaaa"


def test_greedy_packing_in_order():
    # A(10) B(8): 14 + 2 + 12 = 28 <= 30; adding C -> +2+14 = 44 > 30 -> stop
    results = _hits(
        ("A", "a" * 10, 0.9),
        ("B", "b" * 8, 0.8),
        ("C", "c" * 10, 0.7),
        ("D", "d" * 2, 0.6),
    )
    ctx = _build(results, max_chars=30)
    assert [c.id for c in ctx.chunks] == ["A", "B"]
    assert len(ctx.text) <= 30


def test_separator_counts_toward_budget():
    # A alone = 8 chars. A+B without separator = 15 <= 16, but with "\n\n" = 17.
    # So max_chars=16 must still exclude B -> separator is charged.
    results = _hits(("A", "alpha", 0.5), ("B", "beta", 0.5))
    ctx = _build(results, max_chars=16)
    assert [c.id for c in ctx.chunks] == ["A"]
    assert ctx.text == "[1] alpha"


def test_first_chunk_over_budget_is_kept_untouched():
    chunk = _chunk("A", "a" * 50)
    ctx = _build([SearchResult(chunk=chunk, score=1.0)], max_chars=10)
    assert [c.id for c in ctx.chunks] == ["A"]
    assert ctx.text == "[1] " + "a" * 50  # not truncated, over budget
    assert len(ctx.text) > 10


def test_first_over_budget_blocks_later_chunks():
    results = _hits(("A", "a" * 50, 0.9), ("B", "b", 0.1))
    ctx = _build(results, max_chars=10)
    assert [c.id for c in ctx.chunks] == ["A"]  # first kept; B never fits


def test_later_chunk_over_budget_stops_packing():
    results = _hits(
        ("A", "a" * 10, 0.9),  # entry 14
        ("B", "b" * 60, 0.8),  # huge entry -> would exceed
        ("C", "c", 0.7),       # tiny, but greedy already stopped
    )
    ctx = _build(results, max_chars=30)
    assert [c.id for c in ctx.chunks] == ["A"]


def test_exact_boundary_includes_chunk_and_stops():
    results = _hits(("A", "a" * 10, 0.9), ("B", "b" * 8, 0.8), ("C", "c", 0.7))
    ctx = _build(results, max_chars=28)  # 14 + 2 + 12 == 28 exactly
    assert [c.id for c in ctx.chunks] == ["A", "B"]
    assert len(ctx.text) == 28


@pytest.mark.parametrize("bad", [0, -1, 1.5, "100", True, None])
def test_max_chars_validation(bad):
    with pytest.raises((TypeError, ValueError)):
        _build(_hits(("A", "apple", 0.5)), max_chars=bad)  # type: ignore[arg-type]


# --- Validation ---


@pytest.mark.parametrize("bad", [None, "x", (), {"A": 1.0}, (r for r in [])])
def test_results_must_be_list(bad):
    with pytest.raises(TypeError):
        _build(bad, max_chars=100)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, "chunk", _chunk("A", "apple"), 42])
def test_element_must_be_searchresult(bad):
    with pytest.raises(TypeError):
        _build([bad], max_chars=100)  # type: ignore[list-item]


def test_duplicate_chunk_id_rejected():
    with pytest.raises(ValueError):
        _build(_hits(("A", "apple", 0.5), ("A", "apple", 0.5)), max_chars=100)


# --- Immutability ---


def test_input_list_unchanged():
    results = _hits(("A", "apple", 0.5), ("B", "beta", 0.5))
    before = list(results)
    _build(results, max_chars=100)
    assert results == before


def test_searchresults_unchanged():
    results = _hits(("A", "apple", 0.5), ("B", "beta", 0.9))
    before = [(r.chunk.id, r.score) for r in results]
    _build(results, max_chars=100)
    assert [(r.chunk.id, r.score) for r in results] == before


def test_chunks_unchanged():
    results = _hits(("A", "apple", 0.5), ("B", "beta", 0.9))
    before = [(c.id, c.text, dict(c.metadata)) for c in (r.chunk for r in results)]
    _build(results, max_chars=100)
    after = [(c.id, c.text, dict(c.metadata)) for c in (r.chunk for r in results)]
    assert before == after


def test_same_chunk_objects_retained():
    results = _hits(("A", "apple", 0.5), ("B", "beta", 0.9))
    ctx = _build(results, max_chars=100)
    for ctx_chunk, result in zip(ctx.chunks, results):
        assert ctx_chunk is result.chunk


# --- Consistency ---


def test_chunks_exactly_correspond_to_text():
    results = _hits(("A", "apple", 0.9), ("B", "beta", 0.8), ("C", "cake", 0.7))
    ctx = _build(results, max_chars=100)
    expected = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(ctx.chunks, start=1))
    assert ctx.text == expected


def test_excluded_chunks_absent_from_text_and_chunks():
    results = _hits(("A", "aaaa", 0.5), ("B", "bbbb", 0.5))
    ctx = _build(results, max_chars=8)
    assert [c.id for c in ctx.chunks] == ["A"]
    assert "bbbb" not in ctx.text


def test_repeated_execution_deterministic():
    results = _hits(("A", "apple", 0.9), ("B", "beta", 0.1), ("C", "cake", 0.5))
    one = _build(results, max_chars=20)
    two = _build(results, max_chars=20)
    assert one == two
    assert one.text == two.text and one.chunks == two.chunks


def test_input_order_reflected_exactly():
    ab = _hits(("A", "apple", 0.5), ("B", "beta", 0.5))
    ba = _hits(("B", "beta", 0.5), ("A", "apple", 0.5))
    assert _build(ab, max_chars=100).text == "[1] apple\n\n[2] beta"
    assert _build(ba, max_chars=100).text == "[1] beta\n\n[2] apple"


# --- Text ---


def test_text_formatting_deterministic():
    results = _hits(("A", "apple", 0.1), ("B", "beta", 0.9))
    ctx = _build(results, max_chars=100)
    assert ctx.text == "[1] apple\n\n[2] beta"


def test_original_chunk_text_preserved_verbatim():
    results = _hits(("A", "apple pie is good", 0.5), ("B", "second chunk here", 0.5))
    ctx = _build(results, max_chars=100)
    for c in ctx.chunks:
        assert c.text in ctx.text


def test_score_not_inserted_into_context():
    chunk_a = _chunk("A", "alpha beta gamma delta")
    chunk_b = _chunk("B", "epsilon zeta eta theta")
    results = [
        SearchResult(chunk=chunk_a, score=0.1234567),
        SearchResult(chunk=chunk_b, score=0.9876543),
    ]
    ctx = _build(results, max_chars=100)
    assert "0.1234567" not in ctx.text and "0.9876543" not in ctx.text
    assert "score" not in ctx.text
