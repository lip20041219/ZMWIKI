"""Pipeline orchestration tests: DI, indexing/query coordination, determinism.

Covers the Phase 7 DeterministicRAGPipeline: dependency injection (no internal
construction), exact embed counts, lexical-index sync + idempotent re-ingest,
strict query stage ordering/parameter passing, a real end-to-end ingest-then-
query, determinism across runs and pipeline instances, and documented partial-
failure (non-transactional indexing) behaviour.
"""

import json

import pytest

from zmwiki import (
    AnswerResult,
    Chunk,
    Chunker,
    Context,
    ContextBuilder,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicGenerator,
    DeterministicQueryRewriter,
    DeterministicRAGPipeline,
    DeterministicReranker,
    DenseRetriever,
    Document,
    Embedding,
    EmbeddingProvider,
    FixedSizeChunker,
    Generator,
    GeneratorResponseError,
    HybridRetriever,
    HttpResponse,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    LexicalIndex,
    LexicalRetriever,
    OpenAICompatibleGenerator,
    QueryRewriter,
    RAGPipeline,
    Reranker,
    Retriever,
    SearchResult,
    SimpleTokenizer,
    VectorStore,
    WeightedScoreFusion,
    citations_from_context,
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


class FakeGenerator(Generator):
    def __init__(self, answer="ANSWER"):
        self.preset = answer
        self.calls = 0
        self.last_query = None
        self.last_context = None

    def generate(self, query, context):
        self.calls += 1
        self.last_query = query
        self.last_context = context
        return self.preset


class FakeRewriter(QueryRewriter):
    """Records the query it was handed and returns a preset retrieval query."""

    def __init__(self, rewritten="REWRITTEN", error=None):
        self.preset = rewritten
        self.error = error
        self.calls = 0
        self.last_query = None

    def rewrite(self, query):
        self.calls += 1
        self.last_query = query
        if self.error is not None:
            raise self.error
        return self.preset


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
    for optional in ("reranker", "generator", "query_rewriter"):
        # explicit None -> omitted -> pipeline default: that stage is disabled
        if deps.get(optional) is None and optional in deps:
            del deps[optional]
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


def _real_pipeline(
    doc,
    *,
    fusion=None,
    chunk_size=20,
    dimension=8,
    rerank=True,
    generate=True,
    rewriter=None,
    generator=None,
):
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    embedder = DeterministicEmbedder(dimension=dimension)
    dense = DenseRetriever(embedder, vstore)
    lexical = LexicalRetriever(lindex)
    hybrid = HybridRetriever(dense, lexical, fusion or WeightedScoreFusion())
    optional = {}
    if rerank:
        optional["reranker"] = DeterministicReranker()
    if generator is not None:
        optional["generator"] = generator
    elif generate:
        optional["generator"] = DeterministicGenerator()
    if rewriter is not None:
        optional["query_rewriter"] = rewriter
    pipe = DeterministicRAGPipeline(
        chunker=FixedSizeChunker(chunk_size=chunk_size),
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        context_builder=DeterministicContextBuilder(),
        **optional,
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
    from zmwiki import __all__

    assert {"RAGPipeline", "DeterministicRAGPipeline"} <= set(__all__)


# --- H. optional reranker: enabled (default) vs disabled ---


def _scored(*entries):
    """SearchResult pool from (chunk_id, score) pairs, in the given order."""
    return [SearchResult(chunk=_chunk(cid), score=score) for cid, score in entries]


def test_reranker_omitted_disables_the_stage():
    pool = _scored(("doc::0", 0.1), ("doc::1", 0.9))
    pipe, deps = _fake_pipeline(retriever=FakeRetriever(results=pool), reranker=None)
    assert pipe.reranker is None  # stage off, not a null-object stand-in
    assert "reranker" not in deps  # nothing was constructed
    assert pipe.retrieve("q", top_k=1, candidate_k=2) == pool[:1]


def test_reranker_wrong_type_still_rejected():
    with pytest.raises(TypeError):
        _fake_pipeline(reranker="not a stage")


def test_disabled_reranker_keeps_retriever_order_scores_and_objects():
    # Deliberately NOT score-ordered: a disabled stage must not re-sort.
    pool = _scored(("doc::0", 0.1), ("doc::1", 0.9), ("doc::2", 0.5))
    pipe, _ = _fake_pipeline(retriever=FakeRetriever(results=pool), reranker=None)

    out = pipe.retrieve("q", top_k=2, candidate_k=9)

    assert out == pool[:2]
    assert [r.chunk.id for r in out] == ["doc::0", "doc::1"]  # input order kept
    assert [r.score for r in out] == [0.1, 0.9]  # retrieval scores, not re-scored
    assert [id(r) for r in out] == [id(r) for r in pool[:2]]  # same instances


def test_disabled_reranker_candidate_pool_wider_than_top_k():
    pool = _scored(("doc::0", 1.0), ("doc::1", 1.0), ("doc::2", 1.0), ("doc::3", 1.0))
    pipe, deps = _fake_pipeline(retriever=FakeRetriever(results=pool), reranker=None)

    out = pipe.retrieve("q", top_k=2, candidate_k=4)

    assert [r.chunk.id for r in out] == ["doc::0", "doc::1"]
    assert deps["retriever"].last_candidate_k == 4  # recall width still honoured
    assert deps["retriever"].last_top_k == 4


def test_disabled_reranker_fewer_candidates_than_top_k_returns_all():
    pool = _scored(("doc::0", 0.3))
    pipe, _ = _fake_pipeline(retriever=FakeRetriever(results=pool), reranker=None)
    assert pipe.retrieve("q", top_k=10, candidate_k=10) == pool


def test_disabled_reranker_empty_candidates_returns_empty():
    pipe, deps = _fake_pipeline(retriever=FakeRetriever(results=[]), reranker=None)
    assert pipe.retrieve("q", top_k=5, candidate_k=5) == []
    assert deps["retriever"].calls == 1  # retrieved, just nothing to rank


def test_disabled_reranker_still_validates_query_and_top_k():
    pipe, deps = _fake_pipeline(retriever=FakeRetriever(), reranker=None)
    with pytest.raises(TypeError):
        pipe.retrieve("   \n\t")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pipe.retrieve("q", top_k=0)
    with pytest.raises((TypeError, ValueError)):
        pipe.retrieve("q", top_k="2")  # type: ignore[arg-type]
    assert deps["retriever"].calls == 0  # rejected before any work


def test_query_without_reranker_packs_the_same_cut_list():
    pool = _scored(("doc::0", 0.4), ("doc::1", 0.3), ("doc::2", 0.2))
    pipe, deps = _fake_pipeline(retriever=FakeRetriever(results=pool), reranker=None)
    pipe.query("q", top_k=2, candidate_k=3, max_chars=100)
    assert deps["context_builder"].last_results == pool[:2]


def test_real_pipeline_reranker_narrows_the_candidate_pool():
    # 10 chunks @ chunk_size=20; query "needle" hits chunk 0 only.
    doc = Document(
        id="needle-doc",
        content="needle " + "filler " * 18 + "trailing ordinary words",
    )
    pipe = _real_pipeline(doc)
    assert pipe.reranker is not None
    raw = pipe.retriever.retrieve("needle", top_k=10, candidate_k=10)

    out = pipe.retrieve("needle", top_k=2, candidate_k=10)

    assert len(raw) > len(out) == 2  # candidate_k > top_k, really narrowed
    assert {r.chunk.id for r in out} <= {r.chunk.id for r in raw}  # no recall
    # Pipeline forwards the pool verbatim into the real reranker.
    assert out == DeterministicReranker().rerank("needle", raw, top_k=2)
    assert out[0].chunk.id == raw[0].chunk.id  # needle chunk wins on overlap


def test_real_pipeline_without_reranker_keeps_raw_retrieval_ranking():
    doc = Document(
        id="needle-doc",
        content="needle " + "filler " * 18 + "trailing ordinary words",
    )
    pipe = _real_pipeline(doc, rerank=False)
    assert pipe.reranker is None
    raw = pipe.retriever.retrieve("needle", top_k=10, candidate_k=10)

    out = pipe.retrieve("needle", top_k=2, candidate_k=10)

    assert out == raw[:2]
    assert [r.chunk.id for r in out] == [r.chunk.id for r in raw[:2]]
    ctx = pipe.query("needle", top_k=2, candidate_k=10, max_chars=2000)
    assert [c.id for c in ctx.chunks] == [r.chunk.id for r in raw[:2]]


def test_real_pipeline_with_reranker_is_deterministic():
    doc = _three_chunk_doc()
    a = _real_pipeline(doc).retrieve("abcdef", top_k=1, candidate_k=10)
    b = _real_pipeline(doc).retrieve("abcdef", top_k=1, candidate_k=10)
    assert [(r.chunk.id, r.score) for r in a] == [(r.chunk.id, r.score) for r in b]
    assert len(a) == 1  # candidate_k (10) > corpus (3) > top_k (1)
    assert len(_real_pipeline(doc, rerank=False).retrieve("abcdef", top_k=1, candidate_k=10)) == 1


# --- I. optional generator: generation off (default) vs on ---


def _needle_doc():
    return Document(
        id="needle-doc",
        content="needle " + "filler " * 18 + "trailing ordinary words",
    )


def test_generator_omitted_keeps_retrieval_and_context_behavior():
    pool = _scored(("doc::0", 0.2), ("doc::1", 0.5))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool), reranker=None, generator=None
    )
    assert pipe.generator is None  # stage off, nothing constructed
    assert "generator" not in deps

    # The pre-generation chain is byte-for-byte what it always was.
    assert pipe.retrieve("q", top_k=2, candidate_k=2) == pool[:2]
    assert pipe.query("q", top_k=1, candidate_k=2, max_chars=100) is deps[
        "context_builder"
    ].preset
    with pytest.raises(ValueError):
        pipe.answer("q", top_k=1, candidate_k=2, max_chars=100)  # no silent fallback


def test_generator_wrong_type_rejected():
    with pytest.raises(TypeError):
        _fake_pipeline(generator="not a stage")


def test_answer_forwards_the_query_and_the_built_context():
    pool = _scored(("doc::0", 0.2))
    ctx = Context(text="[1] doc::0", chunks=(pool[0].chunk,))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=pool),
        context_builder=FakeContextBuilder(context=ctx),
        generator=FakeGenerator(answer="the answer"),
    )
    generator = deps["generator"]

    out = pipe.answer("what is dp?", top_k=1, candidate_k=3, max_chars=200)

    assert out == "the answer"  # the generator's own return value, verbatim
    assert generator.calls == 1
    assert generator.last_query == "what is dp?"
    assert generator.last_context is ctx
    # Exactly the Context query() returns -- one retrieval path, one pack path.
    assert generator.last_context == pipe.query(
        "what is dp?", top_k=1, candidate_k=3, max_chars=200
    )
    assert deps["context_builder"].last_max_chars == 200


def test_answer_runs_the_chain_once_in_stage_order():
    hits = _scored(("doc::0", 1.0))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=hits),
        reranker=FakeReranker(results=hits),
        generator=FakeGenerator(),
    )
    pipe.answer("q", top_k=1, candidate_k=5, max_chars=100)
    assert deps["retriever"].calls == 1  # retrieval never duplicated
    assert deps["reranker"].calls == 1
    assert deps["context_builder"].calls == 1
    assert deps["generator"].calls == 1
    assert deps["reranker"].last_candidates == hits  # rerank before generation
    assert deps["generator"].last_context is deps["context_builder"].preset


def test_answer_works_without_a_reranker():
    pool = _scored(("doc::0", 0.4), ("doc::1", 0.1))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=None,
        context_builder=DeterministicContextBuilder(),
        generator=FakeGenerator(answer="ok"),
    )
    assert pipe.answer("q", top_k=1, candidate_k=2, max_chars=100) == "ok"
    assert [c.id for c in deps["generator"].last_context.chunks] == ["doc::0"]


def test_real_pipeline_full_chain_query_to_answer():
    pipe = _real_pipeline(_needle_doc())
    assert isinstance(pipe.generator, DeterministicGenerator)

    answer = pipe.answer("needle", top_k=2, candidate_k=10, max_chars=2000)
    context = pipe.query("needle", top_k=2, candidate_k=10, max_chars=2000)

    assert answer == DeterministicGenerator().generate("needle", context)
    assert "needle" in answer
    assert answer in [c.text for c in context.chunks]  # extractive: a real chunk
    results = pipe.retrieve("needle", top_k=2, candidate_k=10)
    assert answer == DeterministicGenerator().generate(
        "needle", pipe.context_builder.build(results, max_chars=2000)
    )


def test_answer_over_empty_retrieval_returns_empty_string():
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=[]),
        generator=DeterministicGenerator(),
        context_builder=DeterministicContextBuilder(),
    )
    assert pipe.answer("nothing matches", top_k=3, candidate_k=5, max_chars=200) == ""
    assert deps["retriever"].calls == 1  # empty pool: ran, found nothing
    # Context is empty, so the extractive generator has nothing to copy out.
    assert pipe.query("nothing matches", top_k=3, candidate_k=5, max_chars=200).chunks == ()


def test_real_pipeline_answer_is_deterministic_across_instances():
    a = _real_pipeline(_three_chunk_doc()).answer(
        "abcdef", top_k=2, candidate_k=10, max_chars=2000
    )
    b = _real_pipeline(_three_chunk_doc()).answer(
        "abcdef", top_k=2, candidate_k=10, max_chars=2000
    )
    assert a == b
    assert a  # non-empty context -> non-empty extractive answer


def test_generator_errors_propagate_unchanged():
    class Boom(Exception):
        pass

    class ExplodingGenerator(Generator):
        def generate(self, query, context):
            raise Boom("model backend unavailable")

    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=_scored(("doc::0", 1.0))),
        generator=ExplodingGenerator(),
    )
    with pytest.raises(Boom, match="model backend unavailable"):
        pipe.answer("q", top_k=1, candidate_k=3, max_chars=100)
    # Failures are never swallowed into a fallback answer, and retrieval that
    # already happened is not rolled back (same posture as indexing).
    assert deps["retriever"].calls == 1


def test_answer_validates_query_before_any_stage_runs():
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(), generator=FakeGenerator(), reranker=None
    )
    with pytest.raises(TypeError):
        pipe.answer("   \n\t")  # type: ignore[arg-type]
    assert deps["retriever"].calls == 0
    assert deps["generator"].calls == 0


# --- J. answer_with_sources: attribution of the very Context that was used ---


def test_answer_with_sources_returns_generator_output_and_context_sources():
    results = _scored(("doc::0", 0.9), ("doc::1", 0.5))
    ctx = DeterministicContextBuilder().build(results, max_chars=2000)
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=results),
        reranker=FakeReranker(results=results),
        context_builder=FakeContextBuilder(context=ctx),
        generator=FakeGenerator(answer="generated answer"),
    )

    result = pipe.answer_with_sources("q", top_k=2, candidate_k=5, max_chars=2000)

    assert isinstance(result, AnswerResult)
    assert result.answer == "generated answer"  # verbatim, never post-edited
    assert [s.chunk_id for s in result.sources] == ["doc::0", "doc::1"]
    assert [s.text for s in result.sources] == [r.chunk.text for r in results]
    assert all(s.document_id == "doc" for s in result.sources)
    assert deps["generator"].last_context is ctx  # cited the Context it was given


def test_sources_are_the_retrieval_results_that_participated():
    pipe = _real_pipeline(_needle_doc())

    result = pipe.answer_with_sources("needle", top_k=2, candidate_k=10, max_chars=2000)
    ranked = pipe.retrieve("needle", top_k=2, candidate_k=10)

    # Same chunks, same order as the ranked results that built the Context.
    assert [s.chunk_id for s in result.sources] == [r.chunk.id for r in ranked]
    assert [s.text for s in result.sources] == [r.chunk.text for r in ranked]
    assert all(s.document_id == "needle-doc" for s in result.sources)
    assert all(s.position >= 0 for s in result.sources)
    assert len({s.chunk_id for s in result.sources}) == len(result.sources)  # unique


def test_sources_never_exceed_the_packed_context():
    results = _scored(("doc::0", 0.9), ("doc::1", 0.5), ("doc::2", 0.4))
    # Budget fits only the first entry: the other two never reached the answer.
    ctx = DeterministicContextBuilder().build(results, max_chars=len("[1] doc::0"))
    pipe, _ = _fake_pipeline(
        retriever=FakeRetriever(results=results),
        reranker=FakeReranker(results=results),
        context_builder=FakeContextBuilder(context=ctx),
        generator=FakeGenerator(),
    )
    result = pipe.answer_with_sources("q", top_k=3, candidate_k=3, max_chars=9)
    assert [s.chunk_id for s in result.sources] == ["doc::0"]  # not "doc::1/2"
    assert result.sources == citations_from_context(ctx)


def test_answer_with_sources_runs_the_chain_once():
    results = _scored(("doc::0", 1.0))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=results),
        reranker=FakeReranker(results=results),
        generator=FakeGenerator(),
    )
    pipe.answer_with_sources("q", top_k=1, candidate_k=5, max_chars=100)
    assert deps["retriever"].calls == 1  # sources are not re-retrieved
    assert deps["reranker"].calls == 1
    assert deps["context_builder"].calls == 1
    assert deps["generator"].calls == 1


def test_answer_is_answer_with_sources_without_the_attribution():
    pipe = _real_pipeline(_needle_doc())
    kwargs = {"top_k": 2, "candidate_k": 10, "max_chars": 2000}
    assert pipe.answer("needle", **kwargs) == pipe.answer_with_sources(
        "needle", **kwargs
    ).answer


def test_answer_with_sources_raises_without_a_generator():
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=_scored(("doc::0", 1.0))), generator=None
    )
    with pytest.raises(ValueError):
        pipe.answer_with_sources("q", top_k=1, candidate_k=2, max_chars=100)
    assert deps["retriever"].calls == 0  # refused before any work


def test_empty_retrieval_gives_empty_answer_and_no_sources():
    pipe, _ = _fake_pipeline(
        retriever=FakeRetriever(results=[]),
        generator=DeterministicGenerator(),
        context_builder=DeterministicContextBuilder(),
    )
    result = pipe.answer_with_sources("nothing", top_k=3, candidate_k=5, max_chars=200)
    assert result.answer == ""
    assert result.sources == ()  # no fabricated source to justify an empty answer


def test_citation_does_not_mutate_context_or_chunks():
    pipe = _real_pipeline(_needle_doc())
    context = pipe.query("needle", top_k=2, candidate_k=10, max_chars=2000)
    before = (context.text, context.chunks)
    chunks_before = [(c.id, c.text, c.position, dict(c.metadata)) for c in context.chunks]

    pipe.answer_with_sources("needle", top_k=2, candidate_k=10, max_chars=2000)

    assert (context.text, context.chunks) == before  # Context untouched
    assert [
        (c.id, c.text, c.position, dict(c.metadata)) for c in context.chunks
    ] == chunks_before  # and every Chunk in it still identical


def test_real_pipeline_answer_with_sources_is_deterministic():
    kwargs = {"top_k": 2, "candidate_k": 10, "max_chars": 2000}
    a = _real_pipeline(_three_chunk_doc()).answer_with_sources("abcdef", **kwargs)
    b = _real_pipeline(_three_chunk_doc()).answer_with_sources("abcdef", **kwargs)
    assert a == b  # value equality of answer + sources
    assert a.answer and a.sources


def test_public_import_answer_result():
    from zmwiki import __all__

    assert {"Citation", "AnswerResult", "citations_from_context"} <= set(__all__)
    assert "answer_with_sources" in dir(DeterministicRAGPipeline)


# --- K. optional query rewriter: off (default) vs on ---


def _rewriter_pipeline(*, rewriter, results=(), generator=None, context=None):
    """Pipeline over fakes with a rewriter injected, plus its recording deps."""
    kwargs = {
        "retriever": FakeRetriever(results=list(results)),
        "query_rewriter": rewriter,
    }
    if generator is not None:
        kwargs["generator"] = generator
        kwargs["context_builder"] = FakeContextBuilder(context=context)
    return _fake_pipeline(**kwargs)


def test_query_rewriter_omitted_disables_the_stage():
    pipe, deps = _fake_pipeline(retriever=FakeRetriever(results=_scored(("doc::0", 1.0))))

    assert pipe.query_rewriter is None  # stage off, not a null-object stand-in
    assert "query_rewriter" not in deps  # nothing was constructed
    pipe.retrieve("  raw   query  ", top_k=1, candidate_k=1)
    assert deps["retriever"].last_query == "  raw   query  "  # searched verbatim


def test_query_rewriter_wrong_type_rejected():
    with pytest.raises(TypeError):
        _fake_pipeline(query_rewriter="not a stage")


def test_disabled_rewriter_leaves_query_and_answer_unchanged():
    # Baseline equivalence: with no rewriter the caller's query reaches both the
    # retriever and the generator exactly as passed, untrimmed.
    chunk = _chunk("doc::0")
    ctx = Context(text="[1] doc::0", chunks=(chunk,))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=[SearchResult(chunk=chunk, score=1.0)]),
        generator=FakeGenerator(),
        context_builder=FakeContextBuilder(context=ctx),
    )

    pipe.query("  spaced  query  ", top_k=1, candidate_k=1)
    assert deps["retriever"].last_query == "  spaced  query  "
    assert pipe.answer("  spaced  query  ", top_k=1, candidate_k=1) == "ANSWER"
    assert deps["generator"].last_query == "  spaced  query  "


def test_enabled_rewriter_decides_what_the_retriever_searches_for():
    rewriter = FakeRewriter(rewritten="rewritten query")
    pipe, deps = _rewriter_pipeline(rewriter=rewriter, results=_scored(("doc::0", 1.0)))

    pipe.retrieve("original   query", top_k=1, candidate_k=1)

    assert deps["retriever"].last_query == "rewritten query"
    assert rewriter.last_query == "original   query"  # its input, never its own output
    assert rewriter.calls == 1  # exactly once per retrieval


def test_rewriter_runs_before_the_retriever():
    order = []

    class OrderedRetriever(FakeRetriever):
        def retrieve(self, query, top_k=10, candidate_k=50):
            order.append(("retrieve", query))
            return super().retrieve(query, top_k=top_k, candidate_k=candidate_k)

    class OrderedRewriter(FakeRewriter):
        def rewrite(self, query):
            order.append(("rewrite", query))
            return super().rewrite(query)

    pipe, _ = _fake_pipeline(
        retriever=OrderedRetriever(), query_rewriter=OrderedRewriter(rewritten="rw")
    )
    pipe.retrieve("orig", top_k=1, candidate_k=1)

    assert order == [("rewrite", "orig"), ("retrieve", "rw")]


def test_reranker_scores_against_the_retrieval_query():
    pool = _scored(("doc::0", 1.0))
    pipe, deps = _fake_pipeline(
        retriever=FakeRetriever(results=pool),
        reranker=FakeReranker(results=pool),
        query_rewriter=FakeRewriter(rewritten="rw"),
    )

    pipe.retrieve("orig", top_k=1, candidate_k=1)

    # The candidates were recalled with the retrieval query, so the whole
    # retrieval path scores against it.
    assert deps["reranker"].last_query == "rw"


def test_generator_receives_the_original_query_not_the_retrieval_query():
    original = "  How does Python work?  "
    chunk = _chunk("doc::0")
    ctx = Context(text="[1] doc::0", chunks=(chunk,))
    pipe, deps = _rewriter_pipeline(
        rewriter=FakeRewriter(rewritten="How does Python work?"),
        results=[SearchResult(chunk=chunk, score=1.0)],
        generator=FakeGenerator(),
        context=ctx,
    )

    result = pipe.answer_with_sources(original, top_k=1, candidate_k=1)

    assert deps["retriever"].last_query == "How does Python work?"  # looked up rewritten
    assert deps["generator"].last_query == original  # answered for the original, verbatim
    assert result.answer == "ANSWER"
    assert [c.chunk_id for c in result.sources] == ["doc::0"]  # attribution unaffected


def test_rewriter_is_called_once_per_answer_not_once_per_stage():
    chunk = _chunk("doc::0")
    rewriter = FakeRewriter(rewritten="rw")
    pipe, deps = _rewriter_pipeline(
        rewriter=rewriter,
        results=[SearchResult(chunk=chunk, score=1.0)],
        generator=FakeGenerator(),
        context=Context(text="[1] doc::0", chunks=(chunk,)),
    )

    pipe.answer_with_sources("orig", top_k=1, candidate_k=1)

    assert rewriter.calls == 1
    assert deps["retriever"].calls == 1


def test_rewriter_exception_propagates_unchanged():
    class Boom(Exception):
        pass

    boom = Boom("rewriter exploded")
    pipe, deps = _rewriter_pipeline(rewriter=FakeRewriter(error=boom))

    with pytest.raises(Boom) as excinfo:
        pipe.retrieve("q", top_k=1, candidate_k=1)

    assert excinfo.value is boom
    assert deps["retriever"].calls == 0  # nothing downstream ran


def test_retrieval_exception_propagates_unchanged_with_a_rewriter():
    class Boom(Exception):
        pass

    boom = Boom("retriever exploded")

    class ExplodingRetriever(Retriever):
        def retrieve(self, query, top_k=10, candidate_k=50):
            raise boom

    pipe, deps = _fake_pipeline(
        retriever=ExplodingRetriever(),
        generator=FakeGenerator(),
        query_rewriter=FakeRewriter(rewritten="rw"),
    )
    with pytest.raises(Boom) as excinfo:
        pipe.answer("q", top_k=1, candidate_k=1, max_chars=100)

    assert excinfo.value is boom
    assert deps["generator"].calls == 0  # the chain stopped where it broke


@pytest.mark.parametrize("bad", [42, None, "", "   ", ["q"]])
def test_invalid_rewriter_output_fails_instead_of_falling_back(bad):
    pipe, deps = _rewriter_pipeline(rewriter=FakeRewriter(rewritten=bad))

    with pytest.raises(TypeError):
        pipe.retrieve("q", top_k=1, candidate_k=1)

    assert deps["retriever"].calls == 0  # never silently searched the original


def test_query_validation_happens_before_the_rewriter_runs():
    rewriter = FakeRewriter()
    pipe, deps = _rewriter_pipeline(rewriter=rewriter)

    with pytest.raises(TypeError):
        pipe.retrieve("   \n\t")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pipe.retrieve("q", top_k=0)

    assert rewriter.calls == 0
    assert deps["retriever"].calls == 0


def test_end_to_end_rewriter_reaches_the_retriever_but_not_the_generator():
    original = "  How   does Python work?  "

    class RecordingRetriever(Retriever):
        def __init__(self, inner):
            self.inner = inner
            self.last_query = None

        def retrieve(self, query, top_k=10, candidate_k=50):
            self.last_query = query
            return self.inner.retrieve(query, top_k=top_k, candidate_k=candidate_k)

    class RecordingGenerator(Generator):
        def __init__(self, inner):
            self.inner = inner
            self.last_query = None

        def generate(self, query, context):
            self.last_query = query
            return self.inner.generate(query, context)

    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    embedder = DeterministicEmbedder(dimension=8)
    retriever = RecordingRetriever(
        HybridRetriever(
            DenseRetriever(embedder, vstore),
            LexicalRetriever(lindex),
            WeightedScoreFusion(),
        )
    )
    generator = RecordingGenerator(DeterministicGenerator())
    pipe = DeterministicRAGPipeline(
        chunker=FixedSizeChunker(chunk_size=40),
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=retriever,
        context_builder=DeterministicContextBuilder(),
        reranker=DeterministicReranker(),
        generator=generator,
        query_rewriter=DeterministicQueryRewriter(),
    )
    pipe.ingest(
        Document(id="py-doc", content="Python is a programming language. " * 4)
    )

    result = pipe.answer_with_sources(original, top_k=3, candidate_k=10, max_chars=500)

    assert retriever.last_query == "How does Python work?"  # the rewritten query
    assert generator.last_query == original  # the caller's query, character for character
    assert original == "  How   does Python work?  "  # the caller's own string, untouched
    assert result.answer  # a real extractive answer from real retrieved chunks
    assert result.sources
    assert all(s.chunk_id and s.text for s in result.sources)


def test_deterministic_query_rewriter_is_stable_across_pipelines():
    doc = _needle_doc()
    kwargs = {"top_k": 2, "candidate_k": 10, "max_chars": 2000}

    def run():
        pipe = _real_pipeline(doc, rewriter=DeterministicQueryRewriter())
        return pipe.answer_with_sources("  needle   haystack  ", **kwargs)

    assert run() == run()  # same answer, same sources, across instances


# --- L. real LLM generator through the pipeline (fake transport, no network) ---


def _llm_transport(answer="LLM ANSWER", status=200, calls=None):
    """A canned OpenAI-compatible transport: records requests, never dials out."""
    requests = calls if calls is not None else []
    payload = {"choices": [{"message": {"role": "assistant", "content": answer}}]}

    def transport(request):
        requests.append(request)
        body = payload if status == 200 else {"error": {"message": "provider down"}}
        return HttpResponse(status=status, body=json.dumps(body).encode("utf-8"))

    transport.requests = requests
    return transport


def _llm_generator(transport, **kwargs):
    settings = {"base_url": "https://provider.invalid/v1", "model": "test-model"}
    settings.update(kwargs)
    return OpenAICompatibleGenerator(transport=transport, **settings)


def test_llm_generator_answer_becomes_answer_result_answer():
    doc = _needle_doc()
    transport = _llm_transport("A needle is hidden in this haystack.")
    pipe = _real_pipeline(doc, generator=_llm_generator(transport))
    kwargs = {"top_k": 2, "candidate_k": 10, "max_chars": 2000}

    result = pipe.answer_with_sources("needle", **kwargs)

    assert result.answer == "A needle is hidden in this haystack."
    # Sources stay a projection of the Context the generator was given.
    context = pipe.query("needle", **kwargs)
    assert result.sources == citations_from_context(context)
    assert [s.chunk_id for s in result.sources] == [c.id for c in context.chunks]


def test_pipeline_sends_the_original_query_and_the_packed_context():
    doc = _needle_doc()
    transport = _llm_transport()
    pipe = _real_pipeline(
        doc, generator=_llm_generator(transport), rewriter=DeterministicQueryRewriter()
    )
    original = "  needle   haystack  "
    kwargs = {"top_k": 2, "candidate_k": 10, "max_chars": 2000}

    pipe.answer_with_sources(original, **kwargs)

    assert len(transport.requests) == 1  # exactly one generation
    messages = json.loads(transport.requests[0].body.decode("utf-8"))["messages"]
    user_message = messages[1]["content"]
    context = pipe.query(original, **kwargs)
    assert original in user_message  # the caller's query, not a stripped copy
    assert context.text in user_message  # the packed Context, verbatim
    assert context.chunks  # there was something to ground on


def test_pipeline_retrieves_once_and_generates_once():
    doc = _needle_doc()
    transport = _llm_transport()
    pipe = _real_pipeline(doc, generator=_llm_generator(transport))
    retriever = pipe.retriever

    class CountingRetriever(Retriever):
        calls = 0

        def retrieve(self, query, top_k=10, candidate_k=50):
            CountingRetriever.calls += 1
            return retriever.retrieve(query, top_k=top_k, candidate_k=candidate_k)

    pipe.retriever = CountingRetriever()
    pipe.answer_with_sources("needle", top_k=2, candidate_k=10, max_chars=2000)

    assert CountingRetriever.calls == 1  # one retrieval
    assert len(transport.requests) == 1  # one generation


def test_answer_still_returns_a_plain_str_with_an_llm_generator():
    pipe = _real_pipeline(_needle_doc(), generator=_llm_generator(_llm_transport("plain")))
    assert pipe.answer("needle", top_k=2, candidate_k=10, max_chars=2000) == "plain"


def test_llm_generator_failures_propagate_through_the_pipeline():
    doc = _needle_doc()
    transport = _llm_transport(status=500)
    pipe = _real_pipeline(doc, generator=_llm_generator(transport))

    with pytest.raises(GeneratorResponseError, match="HTTP 500"):
        pipe.answer("needle", top_k=2, candidate_k=10, max_chars=2000)

    # Nothing is caught or retried: one request was made, retrieval already
    # happened and is not rolled back (same posture as indexing failures).
    assert len(transport.requests) == 1


def test_pipeline_does_not_parse_citations_out_of_the_answer():
    # A model that writes a chunk id into its prose must not create a source.
    doc = _needle_doc()
    transport = _llm_transport("According to doc_a::0 and missing::9, ...")
    pipe = _real_pipeline(doc, generator=_llm_generator(transport))
    kwargs = {"top_k": 2, "candidate_k": 10, "max_chars": 2000}

    result = pipe.answer_with_sources("needle", **kwargs)

    assert "doc_a::0" in result.answer
    assert [s.chunk_id for s in result.sources] == [
        c.id for c in pipe.query("needle", **kwargs).chunks
    ]
    assert "missing::9" not in [s.chunk_id for s in result.sources]


def test_llm_generator_is_not_constructed_by_the_pipeline():
    # Dependency injection still holds: the adapter is the caller's object.
    transport = _llm_transport()
    generator = _llm_generator(transport)
    pipe = _real_pipeline(_needle_doc(), generator=generator)
    assert pipe.generator is generator
    assert pipe.generator.transport is transport
