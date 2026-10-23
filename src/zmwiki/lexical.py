"""Lexical retrieval: Chunk -> LexicalIndex (BM25) -> SearchResult[].

Independent of the dense/embedding path: no EmbeddingProvider is used here.
BM25 scoring lives inside ``InMemoryLexicalIndex`` -- it is an implementation
detail and is intentionally not part of the public surface. Scores are
standard non-negative BM25, *not* cosine, and are never normalized.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .schema import Chunk
from .stores import SearchResult, _check_top_k
from .tokenizers import SimpleTokenizer, Tokenizer

__all__ = ["LexicalIndex", "InMemoryLexicalIndex"]

# Classic BM25 defaults.
_K1 = 1.2
_B = 0.75


class LexicalIndex(ABC):
    """Stores Chunks and serves lexical (BM25) search over ``chunk.text``.

    Responsibilities mirror VectorStore but are purely lexical: text is
    tokenized and scored; no embeddings involved. ``search`` takes a query
    string. ``score`` in results is BM25 relevance (higher = more relevant).
    """

    @abstractmethod
    def upsert(self, chunk: Chunk) -> None:
        """Insert or replace the record keyed by ``chunk.id``."""

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """Return the stored Chunk for ``chunk_id``, or None."""

    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """Remove the record for ``chunk_id``; True if one was removed."""

    @abstractmethod
    def count(self) -> int:
        """Number of records currently stored."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return the ``top_k`` records most lexically relevant to ``query``."""


class InMemoryLexicalIndex(LexicalIndex):
    """Stdlib-only, in-memory BM25 lexical index.

    BM25(query tokens, doc) = sum_t idf(t) * (tf(t,d) * (k1+1)) /
    (tf(t,d) + k1 * (1 - b + b * dl/avgdl)), with Robertson's smoothed
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)). Query terms are deduplicated
    for scoring (repeated query tokens are not double-counted). k1 and b are
    configurable; default k1=1.2, b=0.75.

    Consistency strategy (ponytail): every mutation re-tokenizes all stored
    chunks from scratch (``_rebuild``), so postings can never go stale. That is
    O(total tokens) per write -- fine at this scale; an incremental inverted
    index belongs to a later phase if throughput ever matters.

    Score semantics are plain BM25 -- never normalized to a cosine-like range.
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        k1: float = _K1,
        b: float = _B,
    ) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError(
                f"tokenizer must be a Tokenizer, got {type(self.tokenizer).__name__}"
            )
        self.k1 = _check_nonneg(k1, "k1")
        self.b = _check_unit(b)
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, dict[str, int]] = {}  # term -> {chunk_id: count}
        self._doc_len: dict[str, int] = {}  # chunk_id -> token count

    # --- mutations ---

    def upsert(self, chunk: Chunk) -> None:
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunk must be a Chunk, got {type(chunk).__name__}")
        self._chunks[chunk.id] = chunk
        self._rebuild()

    def delete(self, chunk_id: str) -> bool:
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
            self._rebuild()
            return True
        return False

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)

    # --- search ---

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        _check_top_k(top_k)
        terms = self._unique(self.tokenizer.tokenize(query))
        if not terms or not self._chunks:
            return []
        n = len(self._chunks)
        avgdl = sum(self._doc_len.values()) / n
        k1, b = self.k1, self.b
        scores: dict[str, float] = {}
        for term in terms:
            postings = self._tf.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for chunk_id, tf in postings.items():
                dl = self._doc_len[chunk_id]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl) if avgdl else 1.0
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (k1 + 1.0)) / denom
        results = [
            SearchResult(chunk=self._chunks[cid], score=scores[cid])
            for cid in scores
        ]
        results.sort(key=lambda r: (-r.score, r.chunk.id))
        return results[:top_k]

    # --- internals ---

    def _rebuild(self) -> None:
        tf: dict[str, dict[str, int]] = {}
        doc_len: dict[str, int] = {}
        for chunk_id, chunk in self._chunks.items():
            counts: dict[str, int] = {}
            for token in self.tokenizer.tokenize(chunk.text):
                counts[token] = counts.get(token, 0) + 1
            doc_len[chunk_id] = sum(counts.values())
            for token, count in counts.items():
                tf.setdefault(token, {})[chunk_id] = count
        self._tf = tf
        self._doc_len = doc_len

    @staticmethod
    def _unique(tokens: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                out.append(token)
        return out


def _check_nonneg(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _check_unit(value: object) -> float:
    value = _check_nonneg(value, "b")
    if value > 1.0:
        raise ValueError(f"b must be in [0, 1], got {value}")
    return value
