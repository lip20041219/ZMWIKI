"""Retrieval orchestration: query -> Retriever -> EmbeddingProvider -> VectorStore.

``Retriever`` is the orchestration layer only. It embeds the query with an
injected ``EmbeddingProvider`` and searches an injected ``VectorStore``; it
never computes similarity, sorts, reranks, mutates results, or calls an LLM.
Only a dense retriever is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .embedders import EmbeddingProvider
from .fusion import FusionStrategy
from .lexical import LexicalIndex
from .schema import _nonempty_str
from .stores import SearchResult, VectorStore

__all__ = ["Retriever", "DenseRetriever", "LexicalRetriever", "HybridRetriever"]


class Retriever(ABC):
    """Turns a query string into ordered retrieval results.

    Responsibilities: orchestrating query -> embedding -> vector search.
    All low-level behaviour lives in the injected provider/store, never here.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return the ``top_k`` most relevant SearchResults for ``query``."""


class DenseRetriever(Retriever):
    """Dense (embedding) retriever over an injected provider + store.

    Exact pipeline:
        query -> embedder.embed(query) -> store.search(embedding, top_k)
        -> list[SearchResult]

    Both collaborators are dependency-injected; this class never constructs a
    concrete embedder/store, reads keys, or touches a database. top_k and
    dimension consistency are the store's contract and propagate here as-is.
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        if not isinstance(embedder, EmbeddingProvider):
            raise TypeError(
                f"embedder must be an EmbeddingProvider, got {type(embedder).__name__}"
            )
        if not isinstance(store, VectorStore):
            raise TypeError(f"store must be a VectorStore, got {type(store).__name__}")
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        _nonempty_str(query, "query")
        embedding = self.embedder.embed(query)
        return self.store.search(embedding, top_k=top_k)


class LexicalRetriever(Retriever):
    """Lexical retriever over an injected LexicalIndex.

    Exact pipeline: query -> index.search(query, top_k) -> list[SearchResult].
    Tokenization, BM25 scoring, and sorting all live in the index; this class
    only validates the query string and delegates. It never depends on an
    EmbeddingProvider.
    """

    def __init__(self, index: LexicalIndex) -> None:
        if not isinstance(index, LexicalIndex):
            raise TypeError(f"index must be a LexicalIndex, got {type(index).__name__}")
        self.index = index

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        _nonempty_str(query, "query")
        return self.index.search(query, top_k=top_k)


class HybridRetriever(Retriever):
    """Combines dense and lexical retrieval via an injected FusionStrategy.

        query
         |-- dense_retriever.retrieve(query, candidate_k)
         |-- lexical_retriever.retrieve(query, candidate_k)
         `-> fusion.fuse(dense, lexical, top_k) -> final SearchResult[]

    Each underlying retriever fetches ``candidate_k`` candidates (>= top_k) so
    the fusion gets a wider pool than the final count. The fusion deduplicates
    by ``chunk.id`` and ranks by hybrid score. Both retrievers and the fusion
    are dependency-injected; nothing is constructed or fetched internally.
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        lexical_retriever: Retriever,
        fusion: FusionStrategy,
    ) -> None:
        for name, value in (
            ("dense_retriever", dense_retriever),
            ("lexical_retriever", lexical_retriever),
        ):
            if not isinstance(value, Retriever):
                raise TypeError(
                    f"{name} must be a Retriever, got {type(value).__name__}"
                )
        if not isinstance(fusion, FusionStrategy):
            raise TypeError(f"fusion must be a FusionStrategy, got {type(fusion).__name__}")
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.fusion = fusion

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> list[SearchResult]:
        _nonempty_str(query, "query")
        _check_positive_int(top_k, "top_k")
        _check_positive_int(candidate_k, "candidate_k")
        if candidate_k < top_k:
            raise ValueError(f"candidate_k ({candidate_k}) must be >= top_k ({top_k})")
        dense = self.dense_retriever.retrieve(query, top_k=candidate_k)
        lexical = self.lexical_retriever.retrieve(query, top_k=candidate_k)
        return self.fusion.fuse(dense, lexical, top_k)


def _check_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
