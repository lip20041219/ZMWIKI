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
  (query rewriter when configured -> hybrid retrieval over a wide candidate
  pool -> reranking to ``top_k`` when a reranker is configured) as a real
  ``list[SearchResult]``. Evaluation-facing: scores are the stages' own, never
  fabricated.
- ``query(...)`` -- ``retrieve(...)`` -> context packing -> ``Context``; the
  user-facing text answer. It reuses ``retrieve``, so retrieval is never
  duplicated.
- ``answer(...)`` -- ``query(...)`` -> injected Generator -> ``str``. Only
  available when a generator is configured; it reuses ``query``, so retrieval
  and packing are never duplicated either.
- ``answer_with_sources(...)`` -- ``answer(...)`` plus the citations for that
  same Context -> ``AnswerResult`` (answer + sources).

Indexing is deliberately **not** transactional: if embedding/upserting chunk N
fails, chunks 0..N-1 already sit in the stores. Failures propagate unchanged;
nothing is silently skipped or rolled back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .chunkers import Chunker
from .citations import AnswerResult, citations_from_context
from .context import Context, ContextBuilder
from .embedders import EmbeddingProvider
from .generators import Generator
from .lexical import LexicalIndex
from .rerankers import Reranker, _check_top_k
from .retrievers import Retriever
from .rewriters import QueryRewriter, _check_rewritten
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
    """End-to-end pipeline contract: ingest Documents, retrieve, query, answer.

    ``ingest`` indexes one Document (chunk -> embed -> vector + lexical).
    ``retrieve`` runs the shared retrieval path (optional query rewriting ->
    hybrid retrieval -> optional reranking) and returns the real ranked
    ``list[SearchResult]``. ``query`` calls ``retrieve`` and packs the results
    into a :class:`Context`; ``answer`` calls ``query`` and hands that Context
    to the injected Generator. Query rewriting, reranking, *and* generation are
    all optional: with no rewriter the query is searched verbatim, with no
    reranker the retriever's own ranking stands, and generation lives entirely
    in the generator stage, never in the pipeline.
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
        the stages ``query`` uses -- the query rewritten when a rewriter is
        configured, the retriever asked for a ``candidate_k``-wide pool, then
        reranked to ``top_k`` when a reranker is configured -- and returns real
        scores (no fabricated ranking evidence), so downstream Evaluation can
        build ``SearchResult``-based predictions directly. The order and scores
        are the stages' own; nothing is re-sorted or recomputed here.
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

    def answer(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int,
        max_chars: int,
    ) -> str:
        """Return the generated answer text for ``query``.

        Concrete (not ``@abstractmethod``) so that pipeline implementations
        predating the generation stage stay valid: the default says plainly
        that this pipeline cannot generate. Pipelines that own a Generator
        override it. Generation is opt-in, so a pipeline without one never
        silently returns its raw context where an answer was expected.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement answer(); "
            "generation is only available on pipelines configured with a generator"
        )

    def answer_with_sources(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int,
        max_chars: int,
    ) -> AnswerResult:
        """Return the answer together with the sources it was grounded in.

        Concrete for the same reason as :meth:`answer`: pipelines predating
        attribution stay valid.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement answer_with_sources(); "
            "generation is only available on pipelines configured with a generator"
        )


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
    (recall); the reranker -- when one is injected -- narrows it to the best
    ``top_k``; ContextBuilder packs whatever fits ``max_chars``. If
    ``top_k > candidate_k`` the pool simply cannot fill ``top_k`` -- every
    stage here returns "at most k".

    ``query_rewriter``, ``reranker`` and ``generator`` are the optional
    collaborators (``None`` = stage disabled, and the default for all three):
    with no rewriter the query reaches the retriever verbatim; with no reranker,
    retrieval keeps the retriever's own order and scores, cut to ``top_k``; with
    no generator, the chain ends at ``query``'s Context and ``answer`` reports
    that generation is unavailable. ``candidate_k`` (recall width) and
    ``top_k`` (final result count) are configured independently either way;
    without a reranker the wider pool is simply not needed to answer, so no
    re-ranking is simulated. Injecting ``None`` is how a caller turns a stage
    off.

    Query responsibility is split deliberately: a configured ``query_rewriter``
    decides what the *retrieval* path searches for (it is called once, inside
    ``retrieve``, and its output goes to the retriever and reranker), while the
    query the caller passed in is the one that reaches the Generator. The
    rewriter can change how a question is looked up; it can never change the
    question the answer is written for.

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
        context_builder: ContextBuilder,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        stages: list[tuple[str, object, type]] = [
            ("chunker", chunker, Chunker),
            ("embedder", embedder, EmbeddingProvider),
            ("vector_store", vector_store, VectorStore),
            ("lexical_index", lexical_index, LexicalIndex),
            ("retriever", retriever, Retriever),
            ("context_builder", context_builder, ContextBuilder),
        ]
        # Optional stages: None means "disabled", so they are only validated
        # (and only ever constructed by the caller) when actually injected.
        if reranker is not None:
            stages.append(("reranker", reranker, Reranker))
        if generator is not None:
            stages.append(("generator", generator, Generator))
        if query_rewriter is not None:
            stages.append(("query_rewriter", query_rewriter, QueryRewriter))
        for name, value, cls in stages:
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
        self.generator = generator
        self.query_rewriter = query_rewriter
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
        _check_top_k(top_k)
        # Rewriting happens after argument validation: a bad call fails before
        # any stage, model, or network is touched.
        retrieval_query = self._retrieval_query(query)
        candidates = self.retriever.retrieve(
            retrieval_query, top_k=candidate_k, candidate_k=candidate_k
        )
        if self.reranker is None:
            # Stage disabled: the retriever's own order and scores stand; top_k
            # is a plain cut. Still a subset-of-candidates, best-first list.
            return candidates[:top_k]
        return self.reranker.rerank(retrieval_query, candidates, top_k=top_k)

    def _retrieval_query(self, query: str) -> str:
        """The query the retrieval path searches with: rewritten, or verbatim.

        The reranker scores against this same query, not the original one: the
        candidates were recalled with it, so the whole retrieval path works on
        one query, and the caller's own query is left untouched throughout.
        """
        if self.query_rewriter is None:
            return query
        # The rewriter's contract is enforced, never repaired -- a broken
        # rewrite raises instead of quietly falling back to the original query,
        # which would hide the bug behind plausible-looking results.
        return _check_rewritten(self.query_rewriter.rewrite(query))

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

    def answer(
        self,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        candidate_k: int = _DEFAULT_CANDIDATE_K,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> str:
        # answer() IS answer_with_sources() without the attribution -- one
        # retrieval path, one packing path, one generation call.
        return self.answer_with_sources(
            query, top_k=top_k, candidate_k=candidate_k, max_chars=max_chars
        ).answer

    def answer_with_sources(
        self,
        query: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        candidate_k: int = _DEFAULT_CANDIDATE_K,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> AnswerResult:
        if self.generator is None:
            raise ValueError(
                "no generator configured: this pipeline was built without a "
                "generator, so it can retrieve and pack context but not answer; "
                "pass generator=... to enable generation"
            )
        # query() IS retrieve() + packing -- so the Context a generator sees,
        # and the Context the citations are derived from, are the same object.
        # Generator errors propagate unchanged (nothing is caught, retried, or
        # turned into a fallback answer).
        #
        # The generator is handed the caller's *original* query, never the
        # retrieval query a rewriter produced: rewriting decides what to look
        # up, not what the user asked. The rewriter also runs exactly once per
        # call, because query() reaches it only by way of retrieve().
        context = self.query(
            query, top_k=top_k, candidate_k=candidate_k, max_chars=max_chars
        )
        return AnswerResult(
            answer=self.generator.generate(query, context),
            sources=citations_from_context(context),
        )
