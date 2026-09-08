"""Evaluation harness: execution / validation / reporting around an Evaluator.

The harness owns orchestration only. It takes a prepared ``(cases,
predictions)`` pair, validates that they are positionally consistent, delegates
metric computation to an injected :class:`Evaluator`, and hands back the
resulting :class:`EvaluationResult` untouched. It never computes metrics
itself, never reads files or a database, never retrieves, and never imports or
drives the pipeline.

Layering (kept distinct):

- ``Pipeline`` -- produces retrieval output.
- ``Evaluator`` -- computes metrics.
- ``EvaluationHarness`` -- executes a run, validates inputs, reports results.

Machine-readable (``EvaluationResult.to_dict``) and human-readable
(``EvaluationResult.summary``) representations live on the result value, not
here.
"""

from __future__ import annotations

from .evaluation import EvaluationCase, EvaluationPrediction, EvaluationResult, Evaluator

__all__ = ["EvaluationHarness"]


class EvaluationHarness:
    """Runs a prepared (cases, predictions) pair through an injected Evaluator.

    ``evaluator`` is constructor-injected and never constructed internally.
    ``run`` validates consistency *before* delegating -- both lists are
    required, equal in length, well-typed per element, and each prediction's
    ``query`` must equal its case's ``query`` (a mismatch raises ``ValueError``;
    queries are never auto-corrected). It then calls
    ``evaluator.evaluate(cases, predictions)`` exactly once and returns that
    ``EvaluationResult`` unchanged; Hit Rate / Recall / Precision are never
    computed here.

    Inputs are never mutated (no sort/pop/clear/append, no field writes) and no
    state is cached between calls. An empty run ``([], [])`` is delegated to
    the evaluator, whose empty-run contract is therefore preserved.
    """

    def __init__(self, evaluator: Evaluator) -> None:
        if not isinstance(evaluator, Evaluator):
            raise TypeError(
                f"evaluator must be an Evaluator, got {type(evaluator).__name__}"
            )
        self.evaluator = evaluator

    def run(
        self,
        cases: list[EvaluationCase],
        predictions: list[EvaluationPrediction],
    ) -> EvaluationResult:
        _require_list(cases, "cases")
        _require_list(predictions, "predictions")
        if len(cases) != len(predictions):
            raise ValueError(
                f"cases ({len(cases)}) and predictions ({len(predictions)}) "
                "must match one-to-one by position"
            )
        for case in cases:
            if not isinstance(case, EvaluationCase):
                raise TypeError(
                    f"each case must be an EvaluationCase, got {type(case).__name__}"
                )
        for prediction in predictions:
            if not isinstance(prediction, EvaluationPrediction):
                raise TypeError(
                    "each prediction must be an EvaluationPrediction, "
                    f"got {type(prediction).__name__}"
                )
        for i, (case, prediction) in enumerate(zip(cases, predictions)):
            if prediction.query != case.query:
                raise ValueError(
                    f"query mismatch at index {i}: case {case.query!r} != "
                    f"prediction {prediction.query!r}"
                )
        return self.evaluator.evaluate(cases, predictions)


def _require_list(value: object, name: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list, got {type(value).__name__}")
