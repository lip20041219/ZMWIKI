"""Score/rank fusion of dense and lexical retrieval results.

Fuses two independently-scaled result lists into one ``SearchResult`` list.
Dense scores (cosine, ~[-1, 1]) and lexical scores (BM25, [0, +inf)) are on
different scales and are **never added raw**; every strategy here resolves
that explicitly (Weighted normalizes query-locally; RRF uses ranks only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .stores import SearchResult

__all__ = ["FusionStrategy", "WeightedScoreFusion", "RRFFusion"]


class FusionStrategy(ABC):
    """Merges ordered dense and lexical SearchResult lists into one ranked list.

    ``dense_results`` / ``lexical_results`` are each already ranked by their
    source retriever (best first). Implementations deduplicate by ``chunk.id``,
    assign a hybrid score, sort by (-score, chunk.id), and return the top
    ``top_k``.
    """

    @abstractmethod
    def fuse(
        self,
        dense_results: list[SearchResult],
        lexical_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Return at most ``top_k`` fused results, best first."""


class WeightedScoreFusion(FusionStrategy):
    """Query-local min-max normalization + weighted linear combination.

    Each source list is normalized *within the current query's candidates*:
    norm = (score - min) / (max - min) -> [0, 1]. If a source list is empty or
    all its scores are equal (max == min), its normalized values are 1.0. A
    chunk absent from a source contributes 0 for that source.

        final = dense_weight * norm_dense + lexical_weight * norm_lexical

    Weights must be numbers >= 0 and must not both be 0. They are *not*
    required to sum to 1: scaling both by a constant does not change ranking.
    """

    def __init__(self, dense_weight: float = 0.5, lexical_weight: float = 0.5) -> None:
        self.dense_weight = _check_weight(dense_weight, "dense_weight")
        self.lexical_weight = _check_weight(lexical_weight, "lexical_weight")
        if self.dense_weight == 0.0 and self.lexical_weight == 0.0:
            raise ValueError("dense_weight and lexical_weight cannot both be 0")

    def fuse(
        self,
        dense_results: list[SearchResult],
        lexical_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        dense_norm = _minmax_normalize(dense_results)
        lex_norm = _minmax_normalize(lexical_results)
        chunks: dict[str, object] = {}
        for result in dense_results:
            chunks.setdefault(result.chunk.id, result.chunk)
        for result in lexical_results:
            chunks.setdefault(result.chunk.id, result.chunk)
        fused = [
            SearchResult(
                chunk=chunks[cid],  # type: ignore[arg-type]
                score=self.dense_weight * dense_norm.get(cid, 0.0)
                + self.lexical_weight * lex_norm.get(cid, 0.0),
            )
            for cid in set(dense_norm) | set(lex_norm)
        ]
        fused.sort(key=lambda r: (-r.score, r.chunk.id))
        return fused[:top_k]


class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion -- rank-based, scale-free.

        rrf(d) = sum over sources of 1 / (rrf_k + rank(d))

    ``rank`` is 1-based position within that source's ordered list. A document
    retrieved by both sources therefore scores higher than one retrieved by a
    single source, with no score normalization needed (raw dense/lexical scores
    are ignored).
    """

    def __init__(self, rrf_k: int = 60) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
            raise TypeError(f"rrf_k must be an int, got {type(rrf_k).__name__}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
        self.k = rrf_k

    def fuse(
        self,
        dense_results: list[SearchResult],
        lexical_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        scores: dict[str, float] = {}
        chunks: dict[str, object] = {}
        for source in (dense_results, lexical_results):
            for rank, result in enumerate(source, start=1):
                cid = result.chunk.id
                chunks.setdefault(cid, result.chunk)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
        fused = [
            SearchResult(chunk=chunks[cid], score=scores[cid])  # type: ignore[arg-type]
            for cid in scores
        ]
        fused.sort(key=lambda r: (-r.score, r.chunk.id))
        return fused[:top_k]


def _check_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _minmax_normalize(results: list[SearchResult]) -> dict[str, float]:
    """Query-local min-max to [0, 1]. Empty list -> {}; max == min -> all 1.0."""
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {r.chunk.id: 1.0 for r in results}
    return {r.chunk.id: (r.score - lo) / (hi - lo) for r in results}
