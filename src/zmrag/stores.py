"""VectorStore contract: Chunk + Embedding -> store -> similarity search.

Only an in-memory implementation is shipped. The store is decoupled from any
embedding provider, database, or vector library. It receives ``Embedding``
objects -- it never embeds text itself, never chunks, never orchestrates
retrieval, and never calls an LLM.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .embedders import Embedding
from .schema import Chunk, _nonempty_str

__all__ = ["VectorStore", "InMemoryVectorStore", "SearchResult", "cosine_similarity"]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """cos(a,b) = (a.b) / (||a|| ||b||).

    Vectors must have equal length. A zero vector (no direction) scores 0.0
    against anything. Pure stdlib; no numpy.
    """
    if len(a) != len(b):
        raise ValueError(
            f"cosine_similarity dimension mismatch: {len(a)} != {len(b)}"
        )
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class SearchResult:
    """One hit from a search: the stored Chunk and its relevance score.

    ``score`` is a retrieval-relevance score -- higher means more relevant.
    Its range and meaning are backend-specific: cosine similarity in [-1, 1]
    for dense VectorStore search, unbounded non-negative BM25 for lexical
    search. It is never normalized across backends. No redundant data is
    carried: fetch anything else through the store/index.
    """

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """Stores chunk+embedding pairs and serves cosine similarity search.

    Responsibilities: storing vectors and searching them. It does **not**
    embed queries, chunk documents, orchestrate retrieval, rerank, build
    context, or call an LLM -- ``search`` takes an ``Embedding``, not text.

    Dimension rule: the first upsert establishes the store's dimension; any
    later upsert or search whose embedding dimension differs is rejected
    (never silently truncated, padded, or projected).

    ``top_k`` must be an int >= 1 (bool excluded). When no vectors match the
    dimension store is empty, ``search`` returns ``[]`` rather than raising.
    """

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Established embedding dimension, or None while the store is empty."""

    @abstractmethod
    def upsert(self, chunk: Chunk, embedding: Embedding) -> None:
        """Insert or replace the record keyed by ``chunk.id``."""

    @abstractmethod
    def get(self, chunk_id: str) -> tuple[Chunk, Embedding] | None:
        """Return the stored (Chunk, Embedding) for ``chunk_id``, or None."""

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """Remove the record for ``chunk_id``; return True if one was removed."""

    @abstractmethod
    def count(self) -> int:
        """Number of records currently stored."""

    @abstractmethod
    def search(self, query: Embedding, top_k: int = 10) -> list[SearchResult]:
        """Return the ``top_k`` stored records closest to ``query`` by cosine
        similarity, sorted best-first with a stable tie-breaker."""


def _check_top_k(top_k: object) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")


class InMemoryVectorStore(VectorStore):
    """Stdlib-only, in-memory VectorStore.

    Internally a dict keyed by ``chunk.id`` -> ``(Chunk, Embedding)``. One
    chunk_id per record: re-upserting an existing id overwrites it (count does
    not grow). Deterministic: results are sorted by (-score, chunk_id), never
    by dict iteration order.

    Trade-off (ponytail): plain brute-force scan per search -- fine for tests
    and small local indexes; swap in an ANN/external backend behind the same
    interface if throughput ever matters.
    """

    def __init__(self) -> None:
        self._records: dict[str, tuple[Chunk, Embedding]] = {}
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def upsert(self, chunk: Chunk, embedding: Embedding) -> None:
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunk must be a Chunk, got {type(chunk).__name__}")
        if not isinstance(embedding, Embedding):
            raise TypeError(
                f"embedding must be an Embedding, got {type(embedding).__name__}"
            )
        self._check_dimension(embedding.dimension)
        self._records[chunk.id] = (chunk, embedding)

    def get(self, chunk_id: str) -> tuple[Chunk, Embedding] | None:
        _nonempty_str(chunk_id, "chunk_id")
        return self._records.get(chunk_id)

    def delete(self, chunk_id: str) -> bool:
        _nonempty_str(chunk_id, "chunk_id")
        if chunk_id in self._records:
            del self._records[chunk_id]
            return True
        return False

    def count(self) -> int:
        return len(self._records)

    def search(self, query: Embedding, top_k: int = 10) -> list[SearchResult]:
        _check_top_k(top_k)
        if not isinstance(query, Embedding):
            raise TypeError(f"query must be an Embedding, got {type(query).__name__}")
        if not self._records:
            return []
        if query.dimension != self._dimension:
            raise ValueError(
                f"query dimension {query.dimension} != store dimension {self._dimension}"
            )
        scored = (
            (SearchResult(chunk=chunk, score=cosine_similarity(query.vector, emb.vector)))
            for chunk, emb in self._records.values()
        )
        ordered = sorted(scored, key=lambda r: (-r.score, r.chunk.id))
        return ordered[:top_k]

    def _check_dimension(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError(
                f"embedding dimension {dimension} != store dimension {self._dimension}"
            )
