"""Retrieval evaluation integration: run a pipeline over EvaluationCases.

A very thin orchestration adapter. The runner owns no retrieval and no metric
math: for each :class:`EvaluationCase` it asks the injected :class:`RAGPipeline`
for its ranked ``SearchResult`` list (``pipeline.retrieve``), wraps that list in
an :class:`EvaluationPrediction` (keeping the case's query and the results'
order verbatim), and hands the whole ``(cases, predictions)`` pair to the
injected :class:`EvaluationHarness`, returning whatever ``EvaluationResult`` the
harness/evaluator produce.

Layering (kept distinct, exactly as elsewhere in the library):

- ``Pipeline`` -- real retrieval.
- ``RetrievalEvaluationRunner`` -- evaluation orchestration only.
- ``EvaluationHarness`` -- input validation and evaluator delegation.
- ``Evaluator`` -- metric calculation.

The runner never computes a metric, never constructs an ``EvaluationResult``
itself, never sorts or truncates results, never fabricates a score, and never
mutates a case / prediction / result / chunk. It constructs nothing internally:
both collaborators are constructor-injected.
"""

from __future__ import annotations

from .evaluation import EvaluationCase, EvaluationPrediction, EvaluationResult
from .evaluation_harness import EvaluationHarness
from .pipeline import RAGPipeline

__all__ = ["RetrievalEvaluationRunner"]


class RetrievalEvaluationRunner:
    """Runs EvaluationCases through an injected pipeline and harness.

    ``pipeline`` and ``harness`` are keyword-injected and validated against
    their contracts; nothing is constructed internally.

    ``run`` validates ``cases`` (a list of :class:`EvaluationCase`) and the
    ``top_k`` / ``candidate_k`` cutoffs, then, for each case in order, calls
    ``pipeline.retrieve(case.query, top_k=..., candidate_k=...)`` exactly once
    and builds an ``EvaluationPrediction`` from the returned
    ``SearchResult`` list -- query copied verbatim, results kept in the exact
    order and with the exact scores the pipeline returned. It finally calls
    ``harness.run(cases, predictions)`` exactly once and returns its
    ``EvaluationResult`` unchanged. Empty ``cases`` still flows through
    ``harness.run([], [])`` so the evaluator's empty-run contract holds; the
    runner never builds an ``EvaluationResult`` itself.
    """

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        harness: EvaluationHarness,
    ) -> None:
        if not isinstance(pipeline, RAGPipeline):
            raise TypeError(
                f"pipeline must be a RAGPipeline, got {type(pipeline).__name__}"
            )
        if not isinstance(harness, EvaluationHarness):
            raise TypeError(
                f"harness must be an EvaluationHarness, got {type(harness).__name__}"
            )
        self.pipeline = pipeline
        self.harness = harness

    def run(
        self,
        cases: list[EvaluationCase],
        *,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> EvaluationResult:
        _require_list(cases, "cases")
        for case in cases:
            if not isinstance(case, EvaluationCase):
                raise TypeError(
                    f"each case must be an EvaluationCase, got {type(case).__name__}"
                )
        _check_cutoff(top_k, "top_k")
        _check_cutoff(candidate_k, "candidate_k")
        if candidate_k < top_k:
            raise ValueError(
                f"candidate_k ({candidate_k}) must be >= top_k ({top_k})"
            )

        predictions: list[EvaluationPrediction] = []
        for case in cases:
            results = self.pipeline.retrieve(
                case.query, top_k=top_k, candidate_k=candidate_k
            )
            predictions.append(
                EvaluationPrediction(query=case.query, results=results)
            )
        return self.harness.run(cases, predictions)


def _require_list(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")


def _check_cutoff(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
