"""Pipeline orchestration: wire injected stages into a queryable RAG index.

The pipeline owns *coordination only*. Every algorithm -- chunking, embedding,
vector similarity, BM25, fusion, reranking, context packing -- lives in an
injected collaborator and is never reimplemented here. All collaborators are
constructor-injected (dependency injection); the pipeline never constructs a
concrete embedder / store / retriever / reranker / context builder internally.

Entry points:

- ``ingest(document)`` -- chunk -> embed (exactly once per chunk) -> vector
  store + lexical index.
- ``retrieve(query, *, top_k, candidate_k)`` -- the shared retrieval path
  (hybrid retrieval over a wide candidate pool -> reranking to ``top_k``) as a
  real ``list[SearchResult]``. Evaluation-facing: scores are the stages' own,
  never fabricated.
- ``query(...)`` -- ``retrieve(...)`` -> context packing -> ``Context``; the
  user-facing text answer. It reuses ``retrieve``, so retrieval is never
  duplicated.

Indexing is deliberately **not** transactional: if embedding/upserting chunk N
fails, chunks 0..N-1 already sit in the stores. Failures propagate unchanged;
nothing is silently skipped or rolled back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .chunkers import Chunker
from .context import Context, ContextBuilder
from .embedders import EmbeddingProvider
from .lexical import LexicalIndex
from .rerankers import Reranker
from .retrievers import Retriever
from .schema import Document, _nonempty_str
from .stores import SearchResult, VectorStore

__all__ = ["RAGPipeline", "DeterministicRAGPipeline"]

#: Default final result count a query returns.
_DEFAULT_TOP_K = 5
#: Default width of the candidate pool the retriever hands the reranker.
_DEFAULT_CANDIDATE_K = 20
#: Default character budget for the assembled Context.text.
_DEFAULT_MAX_CHARS = 2000


class RAGPipeline(ABC):
    """End-to-end pipeline contract: ingest Documents, retrieve, query.

    ``ingest`` indexes one Document (chunk -> embed -> vector + lexical).
    ``retrieve`` runs the shared retrieval path (hybrid retrieval -> reranking)
    and returns the real ranked ``list[SearchResult]``. ``query`` calls
    ``retrieve`` and packs the results into a :class:`Context`; no answer
    generation happens here by design.
    """

    @abstractmethod
    def ingest(self, document: Document) -> None:
        """Chunk, embed once per chunk, and index ``document``."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int,
    ) -> list[SearchResult]:
        """Return the ranked ``SearchResult`` list for ``query`` (at most ``top_k``).

        This is the evaluation-facing retrieval entry point. It uses exactly
        the stages ``query`` uses -- the retriever asked for a
        ``candidate_k``-wide pool, then reranked to ``top_k`` -- and returns
        real scores (no fabricated ranking evidence), so downstream Evaluation
        can build ``SearchResult``-based predictions directly. The order and
        scores are the stages' own; nothing is re-sorted or recomputed here.
        """

    @abstractmethod
    def query(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int,
        max_chars: int,
    ) -> Context:
        """Return a Context assembled from the best ``top_k`` chunks for ``query``."""


class DeterministicRAGPipeline(RAGPipeline):
    """Dependency-injected orchestration of the ZMRAG retrieval stages.

    Every collaborator is required and keyword-injected; the pipeline builds
    nothing itself. It is deterministic whenever its injected stages are (all
    shipped implementations are), so the same Documents + configuration +
    query always yield the same Context.

    Wiring contract: ``retriever`` must expose hybrid semantics
    ``retrieve(query, top_k, candidate_k)`` (e.g. :class:`HybridRetriever`), and
    must search the same ``vector_store`` / ``lexical_index`` that ``ingest``
    writes to. The caller owns that wiring; this class only coordinates.

    Query semantics: the retriever is asked for a ``candidate_k``-wide pool
    (recall); the reranker narrows it to the best ``top_k``; ContextBuilder
    packs whatever fits ``max_chars``. If ``top_k > candidate_k`` the pool
    simply cannot fill ``top_k`` -- every stage here returns "at most k".

    ``retrieve`` exposes that same retrieve-then-rerank stage pair (before any
    Context packing) as the evaluation-facing API; ``query`` is exactly
    ``retrieve`` + ``ContextBuilder.build``, so the two never diverge.
    """

    def __init__(
        self,
        *,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
    ) -> None:
        for name, value, cls in (
            ("chunker", chunker, Chunker),
            ("embedder", embedder, EmbeddingProvider),
            ("vector_store", vector_store, VectorStore),
            ("lexical_index", lexical_index, LexicalIndex),
            ("retriever", retriever, Retriever),
            ("reranker", reranker, Reranker),
            ("context_builder", context_builder, ContextBuilder),
        ):
            if not isinstance(value, cls):
                raise TypeError(
                    f"{name} must be a {cls.__name__}, got {type(value).__name__}"
                )
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder

    def ingest(self, document: Document) -> None:
        if not isinstance(document, Document):
            raise TypeError(
                f"document must be a Document, got {type(document).__name__}"
            )
        for chunk in self.chunker.chunk(document):
            embedding = self.embedder.embed(chunk.text)  # exactly once per chunk
            self.vector_store.upsert(chunk, embedding)
            self.lexical_index.upsert(chunk)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        candidate_k: int = _DEFAULT_CANDIDATE_K,
    ) -> list[SearchResult]:
        _nonempty_str(query, "query")
        candidates = self.retriever.retrieve(
            query, top_k=candidate_k, candidate_k=candidate_k
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def query(
        self,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        candidate_k: int = _DEFAULT_CANDIDATE_K,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> Context:
        # query() IS retrieve() followed by packing -- one retrieval path.
        results = self.retrieve(query, top_k=top_k, candidate_k=candidate_k)
        return self.context_builder.build(results, max_chars=max_chars)
