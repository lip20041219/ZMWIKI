"""Optional semantic cross-encoder reranking behind the existing Reranker contract.

zmwiki core ships ``DeterministicReranker`` (in ``rerankers.py``) -- a lexical
token-overlap stand-in for local development. This module adds a *real*,
optional cross-encoder reranker for Phase 12's two-stage retrieval profile. It
mirrors ``semantic.py``'s posture exactly:

- It is **not** part of the top-level public API and is never imported by the
  default import graph (``__init__.py`` does not import it); callers opt in with
  ``from zmwiki.semantic_rerank import CrossEncoderReranker``.
- Importing this module is always safe: ``torch`` / ``sentence-transformers``
  are imported only inside ``_create_backend`` at instantiation time, so
  ``import zmwiki`` (or ``import zmwiki.semantic_rerank``) never loads them.
- The heavy backend is hidden behind a narrow seam: :class:`CrossEncoderReranker`
  depends only on ``score_pairs(query, texts) -> list[float]``, never on the
  concrete CrossEncoder API. Tests inject a fake by monkeypatching
  ``_create_backend``.

Deterministic profile vs semantic rerank profile (Phase 12 boundary):

- Deterministic profile: stdlib-only, byte-reproducible, stays canonical; its
  reranker (``DeterministicReranker``) and evidence are untouched.
- Semantic rerank profile (this module): requires the ``semantic`` optional
  dependency and model weights; reproducible within one pinned model +
  dependency set, never byte-reproducible across versions, and never merged or
  labelled as deterministic evidence.

Reranker contract this class must honor (from ``Reranker`` in ``rerankers.py``):
reranking is *subset-only* (output chunk set is always a subset of the input
candidates; it never re-recalls, never queries a store/retriever/index), re-scores
the candidates, and returns at most ``top_k`` best-first. The CrossEncoder may
therefore change *ranking only* -- never recall, candidate generation,
``candidate_k``, retrieval, or fusion.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

from .rerankers import Reranker, _check_top_k
from .schema import _nonempty_str
from .stores import SearchResult

__all__ = ["CrossEncoderReranker"]


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranker over the existing ``Reranker`` contract.

    Backend: sentence-transformers ``CrossEncoder`` (the ``semantic`` optional
    dependency), loaded lazily. Constructing a reranker with the dependency
    absent raises an actionable ``ImportError``; merely importing this module
    never does.

    Exactly one model source is required (keyword-only):

    - ``model_name``: a sentence-transformers cross-encoder model id (weights
      are downloaded on first use unless already cached).
    - ``local_model_path``: a directory holding a locally stored cross-encoder
      model, for offline use.

    ``device`` is passed through to the backend (``None`` = backend default).

    ``rerank`` never mutates its inputs and never re-recalls: it scores exactly
    the ``chunk.text`` of the supplied candidates, emits fresh ``SearchResult``
    objects whose ``score`` is the cross-encoder score, sorts by
    ``(-score, chunk.id)`` (deterministic tie-break even though the model is a
    non-deterministic optional backend), and returns at most ``top_k``.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        local_model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        # Validation happens before any backend work / ML import.
        if (model_name is None) == (local_model_path is None):
            raise ValueError(
                "exactly one of 'model_name' or 'local_model_path' is required"
            )
        if model_name is not None:
            _nonempty_str(model_name, "model_name")
            model_source: str = model_name
        else:
            model_source = os.fspath(local_model_path)
            _nonempty_str(model_source, "local_model_path")
        self._backend = _create_backend(model_source=model_source, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Re-rank ``candidates`` by cross-encoder score; return at most ``top_k``.

        Contract (mirrors ``DeterministicReranker``): ``[] -> []``; every
        candidate must be a ``SearchResult`` with a unique ``chunk.id``;
        ``top_k`` is an int >= 1 (bool rejected) and returning more than
        ``len(candidates)`` returns everything. Output chunks are a strict
        subset of the input candidates -- never re-recalled, never fetched from
        a store/retriever/index.
        """
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

        # One batched backend call over every candidate's text.
        scores = self._backend.score_pairs(
            query, [result.chunk.text for result in candidates]
        )
        if len(scores) != len(candidates):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(candidates)} "
                "candidates; scores must align one-to-one (no truncation/padding)"
            )

        reranked: list[SearchResult] = []
        for result, value in zip(candidates, scores):
            # Same chunk; new score. Candidate object itself untouched.
            reranked.append(
                SearchResult(chunk=result.chunk, score=_finite_float(value))
            )
        # -score desc; ties broken by chunk.id asc (deterministic tie-break).
        reranked.sort(key=lambda r: (-r.score, r.chunk.id))
        return reranked[:top_k]


def _finite_float(value: Any) -> float:
    """Coerce a scorer value to a finite Python float; reject NaN/Inf and bad types."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"scorer scores must be int/float, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"scorer returned a non-finite score: {value!r}")
    return number


def _create_backend(*, model_source: str, device: str | None) -> "_SentenceTransformerCrossEncoderBackend":
    """Lazily build the sentence-transformers CrossEncoder backend adapter.

    The heavy import happens *here*, never at module scope. With the optional
    dependency absent this raises a clear, actionable ImportError.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # optional dependency missing
        raise ImportError(
            "CrossEncoderReranker requires the 'semantic' optional "
            "dependency (sentence-transformers + torch). Install it with: "
            'pip install "zmwiki[semantic]"'
        ) from exc
    return _SentenceTransformerCrossEncoderBackend(
        CrossEncoder(model_source, device=device)
    )


class _SentenceTransformerCrossEncoderBackend:
    """Narrow adapter: ``score_pairs(query, texts) -> list[float]``.

    The only place sentence-transformers' return types are touched. Builds
    ``[(query, text1), (query, text2), ...]`` pairs, scores them in one call,
    and returns plain Python ``float`` values -- numpy / torch never leak into
    the reranker layer.
    """

    def __init__(self, model) -> None:
        self._model = model

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)
        return [float(x) for x in scores]
