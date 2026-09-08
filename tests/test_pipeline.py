"""Pipeline orchestration tests: DI, indexing/query coordination, determinism.

Covers the Phase 7 DeterministicRAGPipeline: dependency injection (no internal
construction), exact embed counts, lexical-index sync + idempotent re-ingest,
strict query stage ordering/parameter passing, a real end-to-end ingest-then-
query, determinism across runs and pipeline instances, and documented partial-
failure (non-transactional indexing) behaviour.
"""

import pytest

from zmrag import (
    Chunk,
    Chunker,
    Context,
    ContextBuilder,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    Embedding,
    EmbeddingProvider,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalIndex,
    LexicalRetriever,
    RAGPipeline,
    Reranker,
    Retriever,
    SearchResult,
    SimpleTokenizer,
    VectorStore,
    WeightedScoreFusion,
)


def _chunk(cid, text=None, doc="doc"):
    return Chunk(
        id=cid, document_id=doc, text=text or cid, position=int(cid.rsplit("::", 1)[-1])
    )


# --- recording fakes: prove the pipeline delegates and never re-implements ---


class FakeChunker(Chunker):
    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])
        self.calls = 0
        self.last_document = None

    def chunk(self, document):
        self.calls += 1
        self.last_document = document
        return list(self.chunks)


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, vector=(0.25, 0.5, 0.75, 1.0)):
        self._vector = vector
        self.calls = 0
        self.texts = []

    @property
    def dimension(self):
        return len(self._vector)

    def embed(self, text):
        self.calls += 1
        self.texts.append(text)
        return Embedding(self._vector)


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.records: dict[str, tuple[Chunk, Embedding]] = {}
        self.upserts = []

    @property
    def dimension(self):
        return None

    def upsert(self, chunk, embedding):
        self.upserts.append((chunk, embedding))
        self.records[chunk.id] = (chunk, embedding)

    def get(self, chunk_id):
        return self.records.get(chunk_id)

    def delete(self, chunk_id):
        return self.records.pop(chunk_id, None) is not None

    def count(self):
        return len(self.records)

    def search(self, query, top_k=10):
        return [SearchResult(chunk=c, score=1.0) for c, _ in self.records.values()][:top_k]


class FakeLexicalIndex(LexicalIndex):
    def __init__(self):
        self.records: dict[str, Chunk] = {}
        self.upserts = []

    def upsert(self, chunk):
        self.upserts.append(chunk)
        self.records[chunk.id] = chunk

    def get(self, chunk_id):
        return self.records.get(chunk_id)

    def delete(self, chunk_id):
        return self.records.pop(chunk_id, None) is not None

    def count(self):
        return len(self.records)

    def search(self, query, top_k=10):
        return [SearchResult(chunk=c, score=1.0) for c in self.records.values()][:top_k]


class FakeRetriever(Retriever):
    """Hybrid-shaped retriever: records the (top_k, candidate_k) it is asked for."""

    def __init__(self, results=None):
        self.preset = list(results or [])
        self.calls = 0
        self.last_query = None
        self.last_top_k = None
        self.last_candidate_k = None

    def retrieve(self, query, top_k=10, candidate_k=50):
        self.calls += 1
        self.last_query = query
        self.last_top_k = top_k
        self.last_candidate_k = candidate_k
        return list(self.preset)


class FakeReranker(Reranker):
    def __init__(self, results=None):
        self.preset = list(results or [])
        self.calls = 0
        self.last_query = None
        self.last_candidates = None
        self.last_top_k = None

    def rerank(self, query, candidates, top_k=10):
        self.calls += 1
        self.last_query = query
        self.last_candidates = candidates
        self.last_top_k = top_k
        return list(self.preset)


class FakeContextBuilder(ContextBuilder):
    def __init__(self, context=None):
        self.preset = context if context is not None else Context(text="", chunks=())
        self.calls = 0
        self.last_results = None
        self.last_max_chars = None

    def build(self, results, max_chars):
        self.calls += 1
        self.last_results = results
        self.last_max_chars = max_chars
        return self.preset


class CountingEmbedder(DeterministicEmbedder):
    """DeterministicEmbedder that also records how often embed() is called."""

    def __init__(self, dimension=8):
        super().__init__(dimension=dimension)
        self.calls = 0
        self.texts = []

    def embed(self, text):
        self.calls += 1
        self.texts.append(text)
        return super().embed(text)


def _fake_pipeline(**overrides):
    """DeterministicRAGPipeline over fakes; override any collaborator."""
    deps = {
        "chunker": FakeChunker(),
        "embedder": FakeEmbedder(),
        "vector_store": FakeVectorStore(),
        "lexical_index": FakeLexicalIndex(),
        "retriever": FakeRetriever(),
        "reranker": FakeReranker(),
        "context_builder": FakeContextBuilder(),
    }
    deps.update(overrides)
    return DeterministicRAGPipeline(**deps), deps


# --- A. Dependency injection ---


def test_pipeline_uses_injected_components_and_never_constructs_its_own():
    chunk = _chunk("doc::0", text="alpha")
    doc = Document(id="doc", content="alpha")
    hit = SearchResult(chunk=chunk, score=1.0)
    reranked = SearchResult(chunk=chunk, score=2.0)
    ctx = Context(text="[1] alpha", chunks=(chunk,))

    pipe, deps = _fake_pipeline(
        chunker=FakeChunker(chunks=[chunk]),
        retriever=FakeRetriever(results=[hit]),
        reranker=FakeReranker(results=[reranked]),
        context_builder=FakeContextBuilder(context=ctx),
    )
    chunker, embedder, vstore, lindex = (
        deps["chunker"],
        deps["embedder"],
        deps["vector_store"],
        deps["lexical_index"],
    )
    retriever, reranker, builder = deps["retriever"], deps["reranker"], deps["context_builder"]

    pipe.ingest(doc)
    assert chunker.calls == 1 and chunker.last_document is doc
    assert embedder.calls == 1 and embedder.texts == ["alpha"]
    assert len(vstore.upserts) == 1 and vstore.upserts[0][0] is chunk
    assert len(lindex.upserts) == 1 and lindex.upserts[0] is chunk

    out = pipe.query("alpha", top_k=1, candidate_k=2, max_chars=100)
    assert out is ctx
    assert retriever.calls == 1 and retriever.last_query == "alpha"
    assert reranker.calls == 1 and reranker.last_candidates == [hit]
    assert builder.calls == 1 and builder.last_results == [reranked]
    assert builder.last_max_chars == 100


def test_pipeline_constructor_rejects_wrong_typed_collaborator():
    good = _fake_pipeline()[1]
    for name in (
        "chunker",
        "embedder",
        "vector_store",
        "lexical_index",
        "retriever",
        "reranker",
        "context_builder",
    ):
        bad = dict(good)
        bad[name] = "not a stage"  # type: ignore[assignment]
        with pytest.raises(TypeError):
            DeterministicRAGPipeline(**bad)


def test_pipeline_requires_all_collaborators():
    with pytest.raises(TypeError):
        DeterministicRAGPipeline()  # type: ignore[call-arg]


def test_pipeline_abstract_not_instantiable():
    with pytest.raises(TypeError):
        RAGPipeline()  # type: ignore[abstract]


# --- B + C. Index pipeline: chunk/embed/upsert counts, objects unchanged ---


def _three_chunk_doc():
    return Document(id="doc", content="abcdefghij" * 3)  # 30 chars -> 3 chunks @ size 10


def test_ingest_chunks_embeds_upserts_and_lexical_syncs_each_chunk_once():
    doc = _three_chunk_doc()
    chunker = FixedSizeChunker(chunk_size=10)
    embedder = CountingEmbedder(dimension=8)
    vstore = InMemoryVectorStore()
    lindex = InMemoryLexicalIndex()
    pipe = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        context_builder=FakeContextBuilder(),
    )

    before = Document(id=doc.id, content=doc.content, metadata=dict(doc.metadata))
    pipe.ingest(doc)

    expected = [c.text for c in FixedSizeChunker(chunk_size=10).chunk(doc)]
    assert len(expected) == 3
    assert chunker.chunk(doc) == [  # chunker output objects are untouched
        c for c in (vstore.get(f"doc::{i}")[0] for i in range(3)) if c is not None
    ]
    assert embedder.calls == 3  # exactly once per chunk, never 6/9
    assert embedder.texts == expected
    assert vstore.count() == 3 and lindex.count() == 3
    # no embedding leaked into the lexical index; Chunk/Document never mutated
    assert doc == before
    for i in range(3):
        chunk = vstore.get(f"doc::{i}")[0]
        assert lindex.get(f"doc::{i}") == chunk
        assert chunk.text == expected[i]


def test_ingest_embeds_each_chunk_exactly_once_even_with_lexical_sync():
    doc = _three_chunk_doc()
    embedder = CountingEmbedder()
    pipe, _ = _fake_pipeline(
        chunker=FixedSizeChunker(chunk_size=10),
        embedder=embedder,
    )
    pipe.ingest(doc)
    assert embedder.calls == 3


# --- D. Lexical index sync + idempotent re-ingest ---


def test_reingest_same_document_overwrites_not_grows():
    doc = _three_chunk_doc()
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    pipe = DeterministicRAGPipeline(
        chunker=FixedSizeChunker(chunk_size=10),
        embedder=DeterministicEmbedder(dimension=8),
        vector_store=vstore,
        lexical_index=lindex,
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        context_builder=FakeContextBuilder(),
    )
    pipe.ingest(doc)
    pipe.ingest(doc)  # same chunk ids -> overwrite
    assert vstore.count() == 3
    assert lindex.count() == 3
    assert all(lindex.get(f"doc::{i}") is not None for i in range(3))


# --- E. Query pipeline: strict order, single call, params forwarded ---


def test_query_threads_results_through_stages_without_reordering_or_filtering():
    pool = [
        SearchResult(chunk=_chunk("doc::0", text="b"), score=0.1),
        SearchResult(chunk=_chunk("doc::1", text="a"), score=0.9),
        SearchResult(chunk=_chunk("doc::2", text="c"), score=0.5),
    ]
    reranked_preset = [pool[1], pool[0]]  # a narrower, reordered answer
    ctx = Context(text="[1] a", chunks=(pool[1].chunk,))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=reranked_preset),
        context_builder=FakeContextBuilder(context=ctx),
    )
    retriever, reranker, builder = deps["retriever"], deps["reranker"], deps["context_builder"]

    out = pipe.query("q", top_k=1, candidate_k=3, max_chars=500)

    # retriever gets a candidate_k-wide recall request; the full pool goes on
    assert retriever.calls == 1
    assert retriever.last_query == "q"
    assert retriever.last_top_k == 3 and retriever.last_candidate_k == 3
    # reranker receives the retriever's output untouched (no re-sort/filter here)
    assert reranker.calls == 1
    assert reranker.last_query == "q"
    assert reranker.last_candidates == pool
    assert reranker.last_top_k == 1
    # context builder receives the reranker output untouched, with max_chars
    assert builder.calls == 1
    assert builder.last_results == reranked_preset
    assert builder.last_max_chars == 500
    assert out is ctx


# --- F. End-to-end with real implementations ---


def _real_pipeline(doc, *, fusion=None, chunk_size=20, dimension=8):
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    embedder = DeterministicEmbedder(dimension=dimension)
    dense = DenseRetriever(embedder, vstore)
    lexical = LexicalRetriever(lindex)
    hybrid = HybridRetriever(dense, lexical, fusion or WeightedScoreFusion())
    pipe = DeterministicRAGPipeline(
        chunker=FixedSizeChunker(chunk_size=chunk_size),
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    pipe.ingest(doc)
    return pipe


def test_end_to_end_ingest_then_query_returns_context_from_real_data():
    # ~8 chunks of chunk_size=20; top_k/candidate_k (10) >= chunk count, so the
    # whole ingested corpus reaches context and the "needle" chunk is present.
    doc = Document(
        id="needle-doc",
        content=(
            "The first line talks about weather patterns. "
            "A needle is hidden within this haystack somewhere. "
            "More filler ordinary sentence words. "
            "The trailing line ends here with final words."
        ),
    )
    pipe = _real_pipeline(doc)
    ctx = pipe.query("needle", top_k=10, candidate_k=10, max_chars=600)
    assert ctx.text
    assert ctx.chunks
    assert all(c.document_id == "needle-doc" for c in ctx.chunks)
    assert "needle" in ctx.text  # context text is real ingested text


def test_end_to_end_lexical_dominant_query_ranks_needle_chunk_first():
    # Content starts with "needle ", so chunk 0 is the only chunk containing the
    # token; pure-BM25 fusion (lexical_weight=1.0) must rank it first.
    doc = Document(
        id="needle-doc",
        content="needle " + "filler " * 18 + "trailing ordinary words",
    )
    pipe = _real_pipeline(doc, fusion=WeightedScoreFusion(dense_weight=0.0, lexical_weight=1.0))
    ctx = pipe.query("needle", top_k=1, candidate_k=10, max_chars=2000)
    assert ctx.chunks
    assert "needle" in ctx.chunks[0].text


# --- Determinism ---


def test_repeated_query_returns_equal_context():
    doc = _three_chunk_doc()
    pipe = _real_pipeline(doc)
    first = pipe.query("abcdef", top_k=2, candidate_k=10, max_chars=2000)
    second = pipe.query("abcdef", top_k=2, candidate_k=10, max_chars=2000)
    assert first == second
    assert first.text == second.text


def test_equivalent_pipeline_instance_returns_equal_context():
    doc = _three_chunk_doc()
    ctx_a = _real_pipeline(doc).query("abcdef", top_k=2, candidate_k=10, max_chars=2000)
    ctx_b = _real_pipeline(doc).query("abcdef", top_k=2, candidate_k=10, max_chars=2000)
    assert ctx_a == ctx_b
    assert ctx_a.text == ctx_b.text


# --- Partial failure (documented: indexing is NOT transactional) ---


def test_embedding_failure_propagates_and_leaves_earlier_chunks_indexed():
    class BoomEmbedder(DeterministicEmbedder):
        def __init__(self):
            super().__init__(dimension=4)
            self.n = 0

        def embed(self, text):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("embed boom")
            return super().embed(text)

    doc = _three_chunk_doc()
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    pipe = DeterministicRAGPipeline(
        chunker=FixedSizeChunker(chunk_size=10),
        embedder=BoomEmbedder(),
        vector_store=vstore,
        lexical_index=lindex,
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        context_builder=FakeContextBuilder(),
    )
    with pytest.raises(RuntimeError):
        pipe.ingest(doc)
    # chunk 0 fully indexed; chunk 1 failed mid-document; chunk 2 never reached
    assert vstore.count() == 1 and lindex.count() == 1
    assert vstore.get("doc::0") is not None


def test_ingest_rejects_non_document_without_touching_stages():
    pipe, deps = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.ingest("not a document")  # type: ignore[arg-type]
    assert deps["chunker"].calls == 0


def test_query_rejects_empty_query():
    pipe, deps = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.query("   \n\t")  # type: ignore[arg-type]


# --- G. retrieve(): the shared evaluation-facing retrieval path ---


def test_retrieve_returns_reranker_output_without_fabricated_order_or_scores():
    pool = [
        SearchResult(chunk=_chunk("doc::0", text="b"), score=0.1),
        SearchResult(chunk=_chunk("doc::1", text="a"), score=0.9),
        SearchResult(chunk=_chunk("doc::2", text="c"), score=0.5),
    ]
    reranked = [pool[2], pool[0]]  # a narrower, reordered answer
    pipe, _ = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=reranked),
    )
    out = pipe.retrieve("q", top_k=2, candidate_k=5)
    # Real SearchResults, exactly the reranker's order and scores -- no
    # fabricated 1.0s, no position/hash guessing, no re-sort to "score order".
    assert isinstance(out, list) and all(isinstance(r, SearchResult) for r in out)
    assert out == reranked
    assert [r.chunk.id for r in out] == ["doc::2", "doc::0"]
    assert [r.score for r in out] == [0.5, 0.1]


def test_retrieve_forwards_candidate_k_to_retriever_and_top_k_to_reranker():
    pool = [SearchResult(chunk=_chunk("doc::0"), score=0.3)]
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=pool),
    )
    retriever, reranker = deps["retriever"], deps["reranker"]

    pipe.retrieve("q", top_k=2, candidate_k=7)

    # Wide recall pool from the retriever; the reranker narrows to top_k.
    assert retriever.calls == 1
    assert retriever.last_query == "q"
    assert retriever.last_top_k == 7 and retriever.last_candidate_k == 7
    assert reranker.calls == 1
    assert reranker.last_query == "q"
    assert reranker.last_candidates == pool  # untouched retriever output
    assert reranker.last_top_k == 2


def test_retrieve_never_touches_context_builder_or_ingest():
    chunk = _chunk("doc::0", text="alpha")
    doc = Document(id="doc", content="alpha")
    hit = SearchResult(chunk=chunk, score=1.0)
    pipe, deps = _fake_pipeline(
        chunker=FakeChunker(chunks=[chunk]),
        retriever=FakeRetriever(results=[hit]),
        reranker=FakeReranker(results=[hit]),
    )
    builder = deps["context_builder"]
    out = pipe.retrieve("q", top_k=1, candidate_k=3)
    assert out == [hit]
    assert builder.calls == 0  # retrieve is retrieval-only, no context packing
    assert deps["embedder"].calls == 0  # and never ingests / embeds


def test_query_is_exactly_retrieve_plus_context_build_one_path():
    pool = [SearchResult(chunk=_chunk("doc::0"), score=0.2)]
    ctx = Context(text="[1] doc::0", chunks=(pool[0].chunk,))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=pool),
        context_builder=FakeContextBuilder(context=ctx),
    )
    builder = deps["context_builder"]
    pipe.query("q", top_k=1, candidate_k=4, max_chars=200)
    # retrieve() called with identical params yields exactly what builder saw.
    assert builder.last_results == pipe.retrieve("q", top_k=1, candidate_k=4)
    assert [r.chunk.id for r in builder.last_results] == ["doc::0"]


def test_query_does_not_retrieve_twice_just_because_retrieve_exists():
    pool = [SearchResult(chunk=_chunk("doc::0"), score=0.4)]
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=pool),
    )
    pipe.query("q", top_k=1, candidate_k=3, max_chars=100)
    assert deps["retriever"].calls == 1  # one retrieval pass, never duplicated
    assert deps["reranker"].calls == 1


def test_retrieve_rejects_blank_query_without_calling_retriever():
    pipe, deps = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.retrieve("   \n\t")  # type: ignore[arg-type]
    assert deps["retriever"].calls == 0


def test_real_pipeline_retrieve_and_query_share_the_same_ranking():
    doc = Document(
        id="needle-doc",
        content="needle " + "filler " * 18 + "trailing ordinary words",
    )
    pipe = _real_pipeline(doc)
    results = pipe.retrieve("needle", top_k=1, candidate_k=10)
    ctx = pipe.query("needle", top_k=1, candidate_k=10, max_chars=2000)
    # Context packs the same ranked list -- chunk order never diverges.
    assert results
    assert [c.id for c in ctx.chunks] == [r.chunk.id for r in results]
    assert "needle" in results[0].chunk.text  # real reranker score ordering


def test_public_import():
    from zmrag import __all__

    assert {"RAGPipeline", "DeterministicRAGPipeline"} <= set(__all__)
