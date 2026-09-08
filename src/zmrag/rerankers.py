"""Reranking stage: re-score and re-order candidate SearchResults.

A Reranker is separate from a Retriever: Retriever recalls/generates
candidates, Reranker only re-orders the candidates it is given. It never
queries a VectorStore/LexicalIndex, never embeds, never fuses, and never
recalls new documents.

Only a deterministic, test-only reranker ships here. It is **not** a
cross-encoder, transformer reranker, semantic model, or production reranker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .schema import _nonempty_str
from .stores import SearchResult
from .tokenizers import SimpleTokenizer, Tokenizer

__all__ = ["Reranker", "DeterministicReranker"]


class Reranker(ABC):
    """Re-scores and re-orders candidate SearchResults for one query.

    Inputs are the candidate list returned by a (hybrid) retriever; the output
    is a new, ranked ``list[SearchResult]`` whose ``score`` reflects this
    reranking stage. Reranking never introduces documents absent from the
    input candidates (no recall).
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Return at most ``top_k`` re-ranked SearchResults, best first."""


class DeterministicReranker(Reranker):
    """Test-only / local-development reranker. NOT a semantic reranker.

    Scoring is a simple, interpretable lexical-overlap relevance: for each
    unique query token, add that token's frequency in the candidate's
    ``chunk.text``. No BM25, no IDF, no learned model -- purely deterministic
    so it can stand in as a swappable reranking stage for pipeline tests.

    The output score replaces the candidate's incoming retrieval score: it now
    means "reranker relevance for this stage". Original Chunk / SearchResult /
    candidate list objects are never mutated; fresh SearchResults are created.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError(
                f"tokenizer must be a Tokenizer, got {type(self.tokenizer).__name__}"
            )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        _nonempty_str(query, "query")
        _check_top_k(top_k)
        if not isinstance(candidates, list):
            raise TypeError(
                f"candidates must be a list, got {type(candidates).__name__}"
            )
        seen: set[str] = set()
        for result in candidates:
            if not isinstance(result, SearchResult):
                raise TypeError(
                    f"each candidate must be a SearchResult, got {type(result).__name__}"
                )
            if result.chunk.id in seen:
                raise ValueError(f"duplicate chunk.id in candidates: {result.chunk.id!r}")
            seen.add(result.chunk.id)
        if not candidates:
            return []

        query_terms = _unique(self.tokenizer.tokenize(query))
        reranked: list[SearchResult] = []
        for result in candidates:
            token_counts: dict[str, int] = {}
            for token in self.tokenizer.tokenize(result.chunk.text):
                token_counts[token] = token_counts.get(token, 0) + 1
            score = sum(token_counts.get(term, 0) for term in query_terms)
            # Same chunk object; new score. Candidate object itself untouched.
            reranked.append(SearchResult(chunk=result.chunk, score=float(score)))
        reranked.sort(key=lambda r: (-r.score, r.chunk.id))
        return reranked[:top_k]


def _unique(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _check_top_k(top_k: object) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
