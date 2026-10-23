"""Benchmark runner: offline retrieval evaluation execution layer.

Phase 9D composes the already-built evaluation stages into one minimal,
deterministic, offline execution entry point. A :class:`BenchmarkRunner` owns
*orchestration only*: it receives an :class:`EvaluationCaseSet`, validates the
call arguments, converts the immutable ``cases`` tuple into the ``list`` the
Phase 9A :class:`RetrievalEvaluationRunner` already requires, delegates exactly
once, and returns that runner's :class:`EvaluationResult` unchanged.

```
EvaluationCaseSet
   ↓  BenchmarkRunner.run(cases, *, top_k, candidate_k)
   ↓    list(cases.cases)  ->  RetrievalEvaluationRunner.run(...)
   ↓      per case: Pipeline.retrieve -> EvaluationPrediction[]
   ↓        EvaluationHarness.run(cases, predictions) -> RetrievalEvaluator
EvaluationResult   (returned verbatim, never rebuilt)
```

The benchmark runner never retrieves, reranks, computes a metric, loads a
dataset, reads a file, or saves a result. It constructs nothing internally:
pipeline / harness / evaluator wiring is folded into the injected
:class:`RetrievalEvaluationRunner`, which the caller owns. Dataset loading is a
separate concern -- :class:`JsonEvaluationDatasetLoader` maps a JSON file to an
``EvaluationCaseSet`` *before* the benchmark runner is involved.

Deterministic and pure: no ``random`` / ``uuid`` / ``time`` / ``hash()``, no
hidden state, no global cache, and no mutation of the input CaseSet (an empty
CaseSet is delegated to the runner, never short-circuited into a fabricated
result). This module is deliberately not a benchmark framework, CLI, web UI,
database, or result-store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .evaluation import EvaluationResult
from .evaluation_dataset import EvaluationCaseSet
from .evaluation_runner import RetrievalEvaluationRunner

__all__ = ["BenchmarkRunner", "DeterministicBenchmarkRunner"]


class BenchmarkRunner(ABC):
    """Runs an EvaluationCaseSet through an injected RetrievalEvaluationRunner."""

    @abstractmethod
    def run(
        self,
        cases: EvaluationCaseSet,
        *,
        top_k: int,
        candidate_k: int,
    ) -> EvaluationResult:
        """Evaluate every case in ``cases`` and return the EvaluationResult.

        ``top_k`` / ``candidate_k`` are ints >= 1 (bool/float/str/None/0/
        negative rejected) and ``candidate_k >= top_k`` is required. All
        retrieval and metric work is delegated exactly-once to the injected
        :class:`RetrievalEvaluationRunner`; the returned ``EvaluationResult``
        is passed through unchanged, never constructed or modified here.
        """


class DeterministicBenchmarkRunner(BenchmarkRunner):
    """Offline evaluation orchestration over an injected evaluation runner.

    ``evaluation_runner`` is keyword-injected and type-validated; nothing is
    constructed internally -- no pipeline, harness, evaluator, or runner is
    ever built here, and no file / environment / API is read. ``run``
    validates its arguments *before* delegating, converts ``cases.cases`` to a
    fresh list (order preserved, contents untouched), calls
    ``evaluation_runner.run(list(cases.cases), top_k=..., candidate_k=...)``
    exactly once, and returns its ``EvaluationResult`` verbatim. An empty
    CaseSet still reaches the runner (``list([])``), so the evaluator's
    empty-run contract -- never a locally fabricated result -- is preserved.
    """

    def __init__(self, *, evaluation_runner: RetrievalEvaluationRunner) -> None:
        if not isinstance(evaluation_runner, RetrievalEvaluationRunner):
            raise TypeError(
                "evaluation_runner must be a RetrievalEvaluationRunner, "
                f"got {type(evaluation_runner).__name__}"
            )
        self.evaluation_runner = evaluation_runner

    def run(
        self,
        cases: EvaluationCaseSet,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> EvaluationResult:
        if not isinstance(cases, EvaluationCaseSet):
            raise TypeError(
                f"cases must be an EvaluationCaseSet, got {type(cases).__name__}"
            )
        _check_cutoff(top_k, "top_k")
        _check_cutoff(candidate_k, "candidate_k")
        if candidate_k < top_k:
            raise ValueError(
                f"candidate_k ({candidate_k}) must be >= top_k ({top_k})"
            )
        # Exactly-once delegation; the runner owns all retrieval + metrics.
        return self.evaluation_runner.run(
            list(cases.cases), top_k=top_k, candidate_k=candidate_k
        )


def _check_cutoff(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
