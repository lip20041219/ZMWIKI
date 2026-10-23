"""Evaluation: offline, deterministic retrieval metric contract.

Evaluation is an independent *consumer* of retrieval output. A pipeline /
retriever runs first and produces ``SearchResult`` lists; the evaluator then
computes deterministic aggregate metrics over those already-obtained results.
It never retrieves, chunks, embeds, reranks, builds context, generates, or
calls an LLM -- and it never imports or drives the pipeline.

Phase 8A metrics are the minimal, reliable retrieval set:

- **Hit Rate** -- fraction of queries whose retrieval surfaced at least one
  ground-truth chunk.
- **Recall** -- macro average of |relevant ∩ retrieved| / |relevant|.
- **Precision** -- macro average of |relevant ∩ retrieved| / |retrieved|.

Phase 8C adds the classic ranking metrics -- **MRR / MAP / NDCG**, each
optionally at a cutoff ``k`` (``MRR@K`` / ``MAP@K`` / ``NDCG@K``). They honor
the exact same rules: binary relevance decided solely by ``chunk.id``
membership in ``relevant_chunk_ids``, query-level macro aggregation, and the
*prediction's given order is authoritative* -- evaluation re-ranks nothing.

Aggregation is a **macro average over queries** (each query is scored first,
then averaged); micro pooling is deliberately not used. The only identity is
``chunk.id`` -- text, object identity, metadata, and ``SearchResult.score`` are
never compared or recomputed. Without a cutoff, ``K`` is the depth of the
predictions supplied: whatever the retrieval stage actually returned is the
candidate set; there is no second, evaluator-side truncation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .schema import _nonempty_str
from .stores import SearchResult

__all__ = [
    "EvaluationCase",
    "EvaluationPrediction",
    "EvaluationResult",
    "Evaluator",
    "RetrievalEvaluator",
]


@dataclass(frozen=True)
class EvaluationCase:
    """One evaluation sample: a query plus its ground-truth relevant chunk ids.

    ``relevant_chunk_ids`` is a stable set of ``chunk.id`` strings, stored as a
    canonical sorted tuple so equality and metrics never depend on input order,
    set iteration, or ``hash()``. An empty set (or a duplicated id) is a
    malformed sample and is rejected -- recall needs a non-empty denominator.
    """

    query: str
    relevant_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_str(self.query, "query")
        object.__setattr__(self, "relevant_chunk_ids", _normalize_ids(self.relevant_chunk_ids))


@dataclass(frozen=True)
class EvaluationPrediction:
    """The retrieval output for one query, already produced by an upstream stage.

    ``results`` is the ordered ``SearchResult`` list the retriever / reranker /
    pipeline returned for ``query``. Duplicate ``chunk.id`` is rejected: metrics
    must be defined over an unambiguous Top-K candidate set, and the evaluator
    never silently deduplicates. Results are matched to cases **by position**
    (``cases[i]`` <-> ``predictions[i]``); ``query`` here is self-description.
    """

    query: str
    results: tuple[SearchResult, ...]

    def __post_init__(self) -> None:
        _nonempty_str(self.query, "query")
        object.__setattr__(self, "results", _normalize_results(self.results))


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate macro retrieval metrics over all evaluated queries.

    The five Hit Rate / Recall / Precision rates are floats in [0, 1] (never
    rounded); ``mrr`` / ``map`` / ``ndcg`` are the Phase 8C ranking metrics.
    ``evaluated_cases`` equals ``total_cases`` whenever cases and predictions
    matched one-to-one. The three ranking fields default to ``0.0`` so
    positional constructions with the original five arguments keep working.
    """

    hit_rate: float
    recall: float
    precision: float
    total_cases: int
    evaluated_cases: int
    mrr: float = 0.0
    map: float = 0.0
    ndcg: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Stable, JSON-friendly dict representation.

        Plain built-ins only (int/float), fixed key order, no object
        references, no timestamp, no paths -- safe to serialize anywhere.
        """
        return {
            "hit_rate": self.hit_rate,
            "recall": self.recall,
            "precision": self.precision,
            "mrr": self.mrr,
            "map": self.map,
            "ndcg": self.ndcg,
            "total_cases": self.total_cases,
            "evaluated_cases": self.evaluated_cases,
        }

    def summary(self) -> str:
        """Fixed-format, human-readable one-liner summary.

        Deterministic across calls and instances: no timestamp, no path, no
        dict/set iteration order.
        """
        return (
            "## Evaluation Result\n\n"
            f"Queries: {self.total_cases}\n"
            f"Hit Rate: {self.hit_rate:.6f}\n"
            f"Recall: {self.recall:.6f}\n"
            f"Precision: {self.precision:.6f}\n"
            f"MRR: {self.mrr:.6f}\n"
            f"MAP: {self.map:.6f}\n"
            f"NDCG: {self.ndcg:.6f}\n"
        )


class Evaluator(ABC):
    """Computes deterministic metrics from already-produced retrieval output."""

    @abstractmethod
    def evaluate(
        self,
        cases: list[EvaluationCase],
        predictions: list[EvaluationPrediction],
    ) -> EvaluationResult:
        """Score ``cases`` against ``predictions`` (matched by position)."""


class RetrievalEvaluator(Evaluator):
    """Macro-average Hit Rate / Recall / Precision / MRR / MAP / NDCG.

    Pure and stateless: no randomness, no mutable state, no rounding. The same
    ``(cases, predictions)`` always yields an equal ``EvaluationResult``.

    ``k`` is an optional ranking cutoff for the Phase 8C metrics only: when
    given, MRR / MAP / NDCG are computed over the first ``k`` results of each
    prediction (``MRR@K`` / ``MAP@K`` / ``NDCG@K``). ``k=None`` (the default)
    uses every supplied result. Hit Rate / Recall / Precision always use the
    full prediction regardless of ``k`` -- they are Phase 8A's whole-depth
    metrics and never become ``@K`` variants here.
    """

    def evaluate(
        self,
        cases: list[EvaluationCase],
        predictions: list[EvaluationPrediction],
        *,
        k: int | None = None,
    ) -> EvaluationResult:
        _require_list(cases, "cases")
        _require_list(predictions, "predictions")
        _check_k(k)
        if len(cases) != len(predictions):
            raise ValueError(
                f"cases ({len(cases)}) and predictions ({len(predictions)}) "
                "must match one-to-one by position"
            )
        for i, case in enumerate(cases):
            if not isinstance(case, EvaluationCase):
                raise TypeError(
                    f"each case must be an EvaluationCase, got {type(case).__name__}"
                )
        for i, prediction in enumerate(predictions):
            if not isinstance(prediction, EvaluationPrediction):
                raise TypeError(
                    f"each prediction must be an EvaluationPrediction, "
                    f"got {type(prediction).__name__}"
                )

        total = len(cases)
        if total == 0:
            # Explicit empty contract: zero queries -> zero aggregates, no error.
            return EvaluationResult(
                hit_rate=0.0, recall=0.0, precision=0.0, total_cases=0, evaluated_cases=0
            )

        hits = recall_sum = precision_sum = 0.0
        mrr_sum = map_sum = ndcg_sum = 0.0
        for case, prediction in zip(cases, predictions):
            relevant = set(case.relevant_chunk_ids)  # canonical, non-empty
            retrieved_ids = [result.chunk.id for result in prediction.results]
            overlap = sum(1 for cid in retrieved_ids if cid in relevant)
            hits += 1.0 if overlap > 0 else 0.0
            recall_sum += overlap / len(relevant)
            # Precision over an empty candidate set is defined as 0.0.
            precision_sum += overlap / len(retrieved_ids) if retrieved_ids else 0.0

            # Ranking metrics respect the optional cutoff; a fresh prefix slice
            # never mutates ``prediction.results``. No @K semantics for the
            # Phase 8A metrics above.
            ranked_ids = retrieved_ids if k is None else retrieved_ids[:k]
            mrr_sum += _reciprocal_rank(ranked_ids, relevant)
            map_sum += _average_precision(ranked_ids, relevant)
            ndcg_sum += _ndcg(ranked_ids, relevant)

        n = float(total)
        return EvaluationResult(
            hit_rate=hits / n,
            recall=recall_sum / n,
            precision=precision_sum / n,
            total_cases=total,
            evaluated_cases=total,
            mrr=mrr_sum / n,
            map=map_sum / n,
            ndcg=ndcg_sum / n,
        )


def _normalize_ids(ids: object) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple, set, frozenset)):
        raise TypeError(
            "relevant_chunk_ids must be a list/tuple/set of chunk-id strings, "
            f"got {type(ids).__name__}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for cid in ids:
        if not isinstance(cid, str):
            raise TypeError(
                f"each relevant chunk id must be a str, got {type(cid).__name__}"
            )
        if not cid.strip():
            raise ValueError(f"relevant chunk ids must be non-empty strings, got {cid!r}")
        if cid in seen:
            raise ValueError(f"duplicate relevant chunk id: {cid!r}")
        seen.add(cid)
        out.append(cid)
    if not out:
        raise ValueError("relevant_chunk_ids must not be empty")
    return tuple(sorted(out))


def _normalize_results(results: object) -> tuple[SearchResult, ...]:
    if isinstance(results, (str, bytes)) or not isinstance(results, (list, tuple)):
        raise TypeError(
            f"results must be a list/tuple of SearchResult, got {type(results).__name__}"
        )
    seen: set[str] = set()
    out: list[SearchResult] = []
    for result in results:
        if not isinstance(result, SearchResult):
            raise TypeError(
                f"each result must be a SearchResult, got {type(result).__name__}"
            )
        if result.chunk.id in seen:
            raise ValueError(f"duplicate chunk.id in results: {result.chunk.id!r}")
        seen.add(result.chunk.id)
        out.append(result)
    return tuple(out)


def _require_list(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")


def _check_k(k: object) -> None:
    """Validate an optional ranking cutoff: None or an int >= 1 (no bool)."""
    if k is None:
        return
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int or None, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _reciprocal_rank(ids: list[str], relevant: set[str]) -> float:
    """RR for one query: 1 / (rank of first relevant result), rank from 1.

    0.0 when no supplied result is relevant. ``ids`` is the authoritative,
    already-ordered candidate list -- never re-sorted here.
    """
    for i, cid in enumerate(ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def _average_precision(ids: list[str], relevant: set[str]) -> float:
    """AP for one query over the ranked ``ids``.

    Sum over each rank where the result is relevant of Precision@rank, divided
    by the number of ground-truth relevant ids (retrieved or not) -- so an
    unretrieved relevant id still dilutes the denominator. 0.0 when none of
    ``ids`` is relevant.
    """
    hits = 0.0
    total = 0.0
    for i, cid in enumerate(ids, start=1):
        if cid in relevant:
            hits += 1.0
            total += hits / i
    return total / len(relevant)


def _ndcg(ids: list[str], relevant: set[str]) -> float:
    """Binary-relevance NDCG for one query over the ranked ``ids``.

    DCG = sum of (2^rel - 1) / log2(rank + 1); rel is 1 for a relevant id else
    0, so each relevant result at rank ``r`` contributes ``1 / log2(r + 1)``.
    IDCG is DCG of the ideal ranking (all relevant ids first). An empty
    candidate list makes both zero; the ratio is then defined as 0.0.
    """
    dcg = 0.0
    for i, cid in enumerate(ids, start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, min(len(relevant), len(ids)) + 1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
