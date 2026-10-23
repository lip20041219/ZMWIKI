"""Retriever / DenseRetriever unit tests (orchestration, not algorithm)."""

import pytest

from zmwiki import (
    Chunk,
    DenseRetriever,
    DeterministicEmbedder,
    Embedding,
    EmbeddingProvider,
    InMemoryVectorStore,
    Retriever,
    SearchResult,
    VectorStore,
)


# --- test doubles ---


class RecordingEmbedder(EmbeddingProvider):
    def __init__(self, dimension=4, error=None):
        self._dimension = dimension
        self.error = error
        self.calls = 0
        self.last_text = None

    @property
    def dimension(self):
        return self._dimension

    def embed(self, text):
        self.calls += 1
        self.last_text = text
        if self.error is not None:
            raise self.error
        return Embedding(vector=(0.25,) * self._dimension)


class PresetStore(VectorStore):
    """Fake store returning preset results; records search calls."""

    def __init__(self, preset=None, error=None):
        self.preset = preset or []
        self.error = error
        self.search_calls = 0
        self.last_embedding = None
        self.last_top_k = None

    @property
    def dimension(self):
        return 4

    def upsert(self, chunk, embedding):
        raise NotImplementedError

    def get(self, chunk_id):
        raise NotImplementedError

    def delete(self, chunk_id):
        raise NotImplementedError

    def count(self):
        return len(self.preset)

    def search(self, query, top_k=10):
        self.search_calls += 1
        self.last_embedding = query
        self.last_top_k = top_k
        if self.error is not None:
            raise self.error
        return list(self.preset)


def _chunk(cid, text="t"):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


# --- basics ---


def test_retriever_is_abstract():
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]


def test_dense_retriever_init_and_di():
    retriever = DenseRetriever(
        embedder=DeterministicEmbedder(dimension=3),
        store=InMemoryVectorStore(),
    )
    assert retriever.embedder is not None
    assert retriever.store is not None


def test_init_rejects_non_provider_or_store():
    with pytest.raises(TypeError):
        DenseRetriever(embedder="nope", store=InMemoryVectorStore())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DenseRetriever(embedder=DeterministicEmbedder(), store="nope")  # type: ignore[arg-type]


# --- query validation ---


@pytest.mark.parametrize("bad", ["", "   \n\t"])
def test_empty_and_whitespace_query_rejected(bad):
    retriever = DenseRetriever(DeterministicEmbedder(), InMemoryVectorStore())
    with pytest.raises(TypeError):
        retriever.retrieve(bad)


def test_non_string_query_rejected():
    retriever = DenseRetriever(DeterministicEmbedder(), InMemoryVectorStore())
    with pytest.raises(TypeError):
        retriever.retrieve(42)  # type: ignore[arg-type]


# --- end-to-end over deterministic stack ---


def test_retrieve_returns_searchresults_ranked():
    embedder = DeterministicEmbedder(dimension=6)
    store = InMemoryVectorStore()
    retriever = DenseRetriever(embedder, store)
    chunks = [_chunk("needle", text="needle"), _chunk("other", text="other")]
    for c in chunks:
        store.upsert(c, embedder.embed(c.text))
    hits = retriever.retrieve("needle", top_k=2)
    assert isinstance(hits, list) and len(hits) == 2
    assert all(isinstance(h, SearchResult) for h in hits)
    assert hits[0].chunk.text == "needle"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score >= hits[1].score


def test_default_top_k_passthrough():
    store = PresetStore()
    retriever = DenseRetriever(RecordingEmbedder(), store)
    retriever.retrieve("q")
    assert store.last_top_k == 10


def test_custom_top_k_passthrough():
    store = PresetStore()
    retriever = DenseRetriever(RecordingEmbedder(), store)
    retriever.retrieve("q", top_k=3)
    assert store.last_top_k == 3


def test_empty_store_returns_empty():
    retriever = DenseRetriever(DeterministicEmbedder(dimension=4), InMemoryVectorStore())
    assert retriever.retrieve("anything") == []


def test_top_k_beyond_count_returns_all():
    embedder = DeterministicEmbedder(dimension=4)
    store = InMemoryVectorStore()
    store.upsert(_chunk("a"), embedder.embed("a"))
    retriever = DenseRetriever(embedder, store)
    assert len(retriever.retrieve("a", top_k=50)) == 1


# --- orchestration: no algorithm re-implemented ---


def test_retriever_returns_store_results_unchanged():
    # Preset results are deliberately NOT sorted by similarity. If the
    # retriever re-ranked or re-scored, this order would change.
    a = _chunk("a", "far")
    b = _chunk("b", "close")
    preset = [SearchResult(chunk=a, score=0.1), SearchResult(chunk=b, score=0.9)]
    embedder = RecordingEmbedder()
    store = PresetStore(preset=preset)
    retriever = DenseRetriever(embedder, store)

    out = retriever.retrieve("q", top_k=5)

    assert out == preset
    assert [r.chunk.id for r in out] == ["a", "b"]  # untouched ordering


def test_embedder_and_store_called_exactly_once():
    embedder = RecordingEmbedder()
    store = PresetStore()
    retriever = DenseRetriever(embedder, store)
    retriever.retrieve("hello", top_k=4)
    assert embedder.calls == 1
    assert store.search_calls == 1
    assert embedder.last_text == "hello"
    assert isinstance(store.last_embedding, Embedding)
    assert store.last_top_k == 4


# --- exception propagation ---


def test_embedder_exception_propagates_unchanged():
    boom = RuntimeError("embedder boom")
    retriever = DenseRetriever(RecordingEmbedder(error=boom), PresetStore())
    with pytest.raises(RuntimeError) as exc:
        retriever.retrieve("q")
    assert exc.value is boom


def test_store_exception_propagates_unchanged():
    boom = ValueError("store boom")
    retriever = DenseRetriever(RecordingEmbedder(), PresetStore(error=boom))
    with pytest.raises(ValueError) as exc:
        retriever.retrieve("q")
    assert exc.value is boom


def test_top_k_validation_is_delegated_to_store():
    # Store rejects invalid top_k before any dim/empty concerns; retriever does
    # not duplicate validation and lets it propagate.
    retriever = DenseRetriever(DeterministicEmbedder(dimension=3), InMemoryVectorStore())
    with pytest.raises(ValueError):
        retriever.retrieve("q", top_k=0)
    with pytest.raises(ValueError):
        retriever.retrieve("q", top_k=-1)
    with pytest.raises(TypeError):
        retriever.retrieve("q", top_k=True)
    with pytest.raises(TypeError):
        retriever.retrieve("q", top_k=2.5)


def test_public_import():
    from zmwiki import __all__

    assert {"Retriever", "DenseRetriever"} <= set(__all__)
