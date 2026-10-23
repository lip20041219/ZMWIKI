"""VectorStore / InMemoryVectorStore unit tests."""

import pytest

from zmwiki import (
    Chunk,
    Embedding,
    InMemoryVectorStore,
    SearchResult,
    VectorStore,
)
from zmwiki.stores import cosine_similarity


def _chunk(cid, text="t"):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


def _vec(*values):
    return Embedding(vector=values)


# --- cosine similarity (pure) ---


def test_cos_identical_vectors():
    assert cosine_similarity((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_cos_orthogonal_vectors():
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cos_opposite_vectors():
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_cos_non_unit_lengths_normalize():
    # (2,2) is parallel to (1,1): cos = 1 despite different norms
    assert cosine_similarity((2.0, 2.0), (1.0, 1.0)) == pytest.approx(1.0)


def test_cos_zero_vector_scores_zero():
    assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0
    assert cosine_similarity((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_cos_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 0.0), (1.0,))


# --- basic store behaviour ---


def test_empty_store():
    store = InMemoryVectorStore()
    assert store.count() == 0
    assert store.dimension is None
    assert store.get("x") is None
    assert store.delete("x") is False
    assert store.search(_vec(0.1, 0.2)) == []


def test_upsert_and_count():
    store = InMemoryVectorStore()
    store.upsert(_chunk("c1"), _vec(1.0, 0.0))
    store.upsert(_chunk("c2"), _vec(0.0, 1.0))
    assert store.count() == 2
    assert store.dimension == 2


def test_get_existing_and_missing():
    store = InMemoryVectorStore()
    chunk, emb = _chunk("c1"), _vec(1.0, 2.0)
    store.upsert(chunk, emb)
    got = store.get("c1")
    assert got == (chunk, emb)
    assert store.get("nope") is None


# --- search ---


def test_search_returns_identical_top_hit():
    store = InMemoryVectorStore()
    store.upsert(_chunk("c1"), _vec(1.0, 0.0, 0.0))
    hits = store.search(_vec(1.0, 0.0, 0.0), top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk.id == "c1"
    assert hits[0].score == pytest.approx(1.0)


def test_search_ranks_by_cosine_descending():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0, 0.0))  # cos 1
    store.upsert(_chunk("c"), _vec(1.0, 1.0, 0.0))  # cos ~0.7071
    store.upsert(_chunk("b"), _vec(0.0, 1.0, 0.0))  # cos 0 (orthogonal)
    hits = store.search(_vec(1.0, 0.0, 0.0), top_k=3)
    assert [h.chunk.id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.70710678)
    assert hits[2].score == pytest.approx(0.0)
    # scores non-increasing
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_top_k_limits_results():
    store = InMemoryVectorStore()
    for i, cid in enumerate(["a", "b", "c"]):
        store.upsert(_chunk(cid), _vec(float(i), 0.0))
    assert len(store.search(_vec(0.0, 1.0), top_k=2)) == 2
    assert len(store.search(_vec(0.0, 1.0), top_k=1)) == 1


def test_top_k_greater_than_count_returns_all():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0))
    hits = store.search(_vec(1.0, 0.0), top_k=100)
    assert len(hits) == 1


def test_top_k_equals_count_returns_all():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0))
    store.upsert(_chunk("b"), _vec(0.0, 1.0))
    assert len(store.search(_vec(1.0, 0.0), top_k=2)) == 2


def test_empty_store_search_returns_empty():
    assert InMemoryVectorStore().search(_vec(0.5, 0.5), top_k=5) == []


# --- dimension consistency ---


def test_first_insert_establishes_dimension():
    store = InMemoryVectorStore()
    store.upsert(_chunk("c"), _vec(1.0, 2.0, 3.0))
    assert store.dimension == 3


def test_dimension_mismatch_on_insert_rejected():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0))
    with pytest.raises(ValueError):
        store.upsert(_chunk("b"), _vec(1.0, 0.0, 0.0))


def test_dimension_mismatch_on_query_rejected():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0))
    with pytest.raises(ValueError):
        store.search(_vec(1.0, 0.0, 0.0))


# --- upsert semantics ---


def test_upsert_same_id_overwrites_without_growth():
    store = InMemoryVectorStore()
    old = _chunk("c", text="old"), _vec(1.0, 0.0)  # far from query below
    store.upsert(old[0], old[1])
    new = _chunk("c", text="new"), _vec(0.0, 1.0)  # close to query below
    store.upsert(new[0], new[1])
    assert store.count() == 1
    assert store.get("c") == new  # latest record seen
    hits = store.search(_vec(0.0, 1.0), top_k=1)
    assert hits[0].chunk.text == "new"
    assert hits[0].score == pytest.approx(1.0)


# --- delete ---


def test_delete_existing_id():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0))
    assert store.delete("a") is True
    assert store.count() == 0
    assert store.get("a") is None


def test_delete_missing_id_returns_false():
    store = InMemoryVectorStore()
    assert store.delete("ghost") is False


def test_search_after_delete_excludes_removed():
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), _vec(1.0, 0.0, 0.0))
    store.upsert(_chunk("b"), _vec(0.9, 0.1, 0.0))
    store.delete("a")
    hits = store.search(_vec(1.0, 0.0, 0.0), top_k=5)
    assert [h.chunk.id for h in hits] == ["b"]
    assert store.dimension == 3  # established dimension persists


# --- determinism / tie-break ---


def test_tie_scores_order_by_chunk_id():
    store = InMemoryVectorStore()
    store.upsert(_chunk("b"), _vec(0.5, 0.5))
    store.upsert(_chunk("a"), _vec(0.5, 0.5))  # same vector -> same score
    hits = store.search(_vec(0.5, 0.5), top_k=2)
    assert [h.chunk.id for h in hits] == ["a", "b"]
    assert hits[0].score == hits[1].score == pytest.approx(1.0)


def test_repeated_search_is_stable():
    store = InMemoryVectorStore()
    for cid, v in [("b", (1.0, 0.0)), ("a", (1.0, 0.0)), ("c", (0.0, 1.0))]:
        store.upsert(_chunk(cid), _vec(*v))
    first = [h.chunk.id for h in store.search(_vec(1.0, 0.0), top_k=3)]
    second = [h.chunk.id for h in store.search(_vec(1.0, 0.0), top_k=3)]
    assert first == second == ["a", "b", "c"]


# --- validation ---


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_invalid_top_k_value_rejected(bad):
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.search(_vec(1.0, 0.0), top_k=bad)


@pytest.mark.parametrize("bad", [True, False, 2.5, "3"])
def test_invalid_top_k_type_rejected(bad):
    store = InMemoryVectorStore()
    with pytest.raises(TypeError):
        store.search(_vec(1.0, 0.0), top_k=bad)


def test_invalid_query_type_rejected():
    store = InMemoryVectorStore()
    with pytest.raises(TypeError):
        store.search("not an embedding")  # type: ignore[arg-type]


def test_invalid_chunk_on_upsert_rejected():
    store = InMemoryVectorStore()
    with pytest.raises(TypeError):
        store.upsert("c", _vec(1.0, 0.0))  # type: ignore[arg-type]


def test_invalid_embedding_on_upsert_rejected():
    store = InMemoryVectorStore()
    with pytest.raises(TypeError):
        store.upsert(_chunk("c"), (1.0, 0.0))  # type: ignore[arg-type]


def test_store_abstract_and_public_import():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]
    from zmwiki import __all__

    assert {"VectorStore", "InMemoryVectorStore", "SearchResult"} <= set(__all__)


def test_searchresult_returns_chunk_objects():
    store = InMemoryVectorStore()
    chunk = _chunk("c1")
    store.upsert(chunk, _vec(1.0, 0.0))
    hit = store.search(_vec(1.0, 0.0))[0]
    assert isinstance(hit, SearchResult)
    assert isinstance(hit.chunk, Chunk)
    assert hit.chunk == chunk
