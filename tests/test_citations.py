"""Attribution tests: Citation, AnswerResult, citations_from_context."""

import pytest

from zmwiki import (
    AnswerResult,
    Chunk,
    Citation,
    Context,
    citations_from_context,
)


def _chunk(cid, text=None, doc="doc", position=0, metadata=None):
    return Chunk(
        id=cid,
        document_id=doc,
        text=text or cid,
        position=position,
        metadata=metadata or {},
    )


def _context(*chunks):
    text = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    return Context(text=text, chunks=tuple(chunks))


# --- Citation shape ---


def test_from_chunk_copies_both_identity_levels():
    chunk = _chunk("doc::3", text="apple pie", doc="doc", position=3)
    citation = Citation.from_chunk(chunk)
    assert citation.document_id == "doc"  # source-level identity
    assert citation.chunk_id == "doc::3"  # chunk-level identity
    assert citation.position == 3
    assert citation.text == "apple pie"


def test_from_chunk_keeps_text_verbatim():
    text = "  Mixed CASE, punctuation!  " * 3
    citation = Citation.from_chunk(_chunk("c", text=text))
    assert citation.text == text  # no truncation, no normalization, no summary


def test_from_chunk_copies_metadata_without_aliasing():
    chunk = _chunk("c", metadata={"filename": "a.txt"})
    citation = Citation.from_chunk(chunk)
    assert citation.metadata == {"filename": "a.txt"}
    chunk.metadata["later"] = "mutation"  # shallowly-mutable dict on a frozen Chunk
    assert citation.metadata == {"filename": "a.txt"}
    citation.metadata["x"] = 1
    assert chunk.metadata == {"filename": "a.txt", "later": "mutation"}


def test_from_chunk_does_not_mutate_the_chunk():
    chunk = _chunk("c", text="apple", position=2, metadata={"k": "v"})
    before = (chunk.id, chunk.document_id, chunk.text, chunk.position, dict(chunk.metadata))
    Citation.from_chunk(chunk)
    assert (
        chunk.id,
        chunk.document_id,
        chunk.text,
        chunk.position,
        dict(chunk.metadata),
    ) == before


def test_from_chunk_rejects_non_chunk():
    for bad in (None, "doc::0", 42, {"id": "c"}):
        with pytest.raises(TypeError):
            Citation.from_chunk(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["document_id", "chunk_id", "text"])
def test_blank_string_fields_rejected(field_name):
    fields = {"document_id": "d", "chunk_id": "c", "text": "t"}
    fields[field_name] = "   \n\t"
    with pytest.raises(TypeError):
        Citation(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 1.0, "0", True, None])
def test_position_validation(bad):
    with pytest.raises(TypeError):
        Citation(document_id="d", chunk_id="c", position=bad, text="t")  # type: ignore[arg-type]


def test_metadata_must_be_a_dict():
    with pytest.raises(TypeError):
        Citation(document_id="d", chunk_id="c", position=0, text="t", metadata="nope")  # type: ignore[arg-type]


def test_citation_is_frozen():
    citation = Citation(document_id="d", chunk_id="c", position=0, text="t")
    with pytest.raises(Exception):
        citation.chunk_id = "other"  # type: ignore[misc]


# --- citations_from_context ---


def test_citations_follow_context_order():
    ctx = _context(
        _chunk("doc::0", position=0),
        _chunk("doc::1", position=1),
        _chunk("doc::2", position=2),
    )
    assert [c.chunk_id for c in citations_from_context(ctx)] == [
        "doc::0",
        "doc::1",
        "doc::2",
    ]


def test_one_citation_per_packed_chunk_and_none_beyond():
    chunks = (_chunk("doc::0", position=0), _chunk("doc::1", position=1))
    ctx = _context(*chunks)  # text mentions both; only these two were packed
    citations = citations_from_context(ctx)
    assert len(citations) == 2
    assert [c.chunk_id for c in citations] == [c.id for c in ctx.chunks]


def test_empty_context_yields_no_sources():
    assert citations_from_context(Context(text="", chunks=())) == ()


def test_duplicate_chunk_ids_collapse_to_the_first_occurrence():
    chunk = _chunk("doc::0", text="apple", position=0)
    ctx = _context(chunk, chunk, _chunk("doc::1", text="banana", position=1))
    citations = citations_from_context(ctx)
    assert [c.chunk_id for c in citations] == ["doc::0", "doc::1"]
    assert citations[0].text == "apple"


def test_multiple_documents_keep_their_own_identities():
    ctx = _context(
        _chunk("a::0", text="alpha", doc="a", position=0),
        _chunk("b::0", text="beta", doc="b", position=0),
        _chunk("a::1", text="gamma", doc="a", position=1),
    )
    citations = citations_from_context(ctx)
    assert [(c.document_id, c.chunk_id, c.position) for c in citations] == [
        ("a", "a::0", 0),
        ("b", "b::0", 0),
        ("a", "a::1", 1),
    ]


def test_citations_are_deterministic_across_calls():
    ctx = _context(_chunk("doc::0"), _chunk("doc::1", position=1))
    assert citations_from_context(ctx) == citations_from_context(ctx)


def test_citations_from_context_rejects_non_context():
    for bad in (None, "ctx", (), [_chunk("c")]):
        with pytest.raises(TypeError):
            citations_from_context(bad)  # type: ignore[arg-type]


# --- AnswerResult shape ---


def test_answer_result_requires_str_and_citation_tuple():
    with pytest.raises(TypeError):
        AnswerResult(answer=None, sources=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AnswerResult(answer="a", sources=[Citation("d", "c", 0, "t")])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AnswerResult(answer="a", sources=("not a citation",))  # type: ignore[arg-type]


def test_answer_result_rejects_duplicate_chunk_ids():
    citation = Citation("d", "c", 0, "t")
    with pytest.raises(ValueError):
        AnswerResult(answer="a", sources=(citation, citation))


def test_answer_result_allows_empty_answer_and_empty_sources():
    result = AnswerResult(answer="", sources=())
    assert result.answer == ""
    assert result.sources == ()


def test_answer_result_is_frozen_and_comparable_by_value():
    citation = Citation("d", "c", 0, "t")
    a = AnswerResult(answer="a", sources=(citation,))
    b = AnswerResult(answer="a", sources=(citation,))
    assert a == b  # value equality, so determinism is assertable directly
    with pytest.raises(Exception):
        a.answer = "other"  # type: ignore[misc]


def test_public_import():
    from zmwiki import __all__

    assert {"Citation", "AnswerResult", "citations_from_context"} <= set(__all__)
