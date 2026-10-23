"""Evaluation case set: an ordered, immutable collection of EvaluationCases.

Phase 9B is the *case organization contract* of offline retrieval evaluation.
:class:`EvaluationCaseSet` groups :class:`EvaluationCase` samples the same way
EvaluationCase already groups a query with its ground-truth chunk ids, and
nothing more. It guarantees three properties:

- **Order** -- the case order is exactly the caller's input order; it is never
  sorted, filtered, or reordered, because ``EvaluationCaseSet[i]`` maps to
  ``EvaluationPrediction[i]`` by position downstream.
- **Duplicates** -- duplicate cases (and duplicate case objects) are legal and
  preserved: a CaseSet is an ordered sequence, not a math set. No ``set()`` /
  ``hash()`` / deduplication is applied.
- **Empty is legal** -- ``EvaluationCaseSet(cases=())`` is a valid empty
  evaluation dataset, never an error.

A CaseSet is a pure, frozen value: it holds only ``cases`` -- no id, name,
version, timestamp, hash, checksum, path, source, split, or metadata -- and it
never touches the filesystem, a database, or the network. It never calls a
pipeline, retriever, harness, or evaluator, and it computes no retrieval metric:
it only *organizes* cases so a :class:`RetrievalEvaluationRunner` can consume
them positionally.

Phase 9B introduces no benchmark framework or dataset (no BEIR / MS MARCO /
HuggingFace datasets, no loaders, no downloaders). ``evaluation_dataset.py``
implements only ``EvaluationCaseSet``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evaluation import EvaluationCase

__all__ = ["EvaluationCaseSet"]


@dataclass(frozen=True)
class EvaluationCaseSet:
    """An ordered, immutable sequence of EvaluationCases.

    ``cases`` must be a tuple of :class:`EvaluationCase` and is kept in its
    exact input order -- never sorted, deduplicated, or reordered
    (``EvaluationCase[i]`` aligns to ``EvaluationPrediction[i]`` by position).
    Duplicate cases are legal and preserved. An empty tuple is a valid empty
    evaluation dataset. Because ``cases`` is a tuple, the caller's mutable list
    can never leak in, and the stored sequence can never be mutated through
    this object -- the same immutable-tuple contract ``Context.chunks`` uses.

    EvaluationCaseSet performs no EvaluationCase validation of its own: query
    and ``relevant_chunk_ids`` rules stay where ``EvaluationCase`` already owns
    them (its canonical sorted relevant-id tuple is left untouched). CaseSet
    only asserts that this is a well-formed, ordered sequence of cases.
    """

    cases: tuple[EvaluationCase, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cases, tuple):
            raise TypeError(
                f"cases must be a tuple of EvaluationCase, got {type(self.cases).__name__}"
            )
        for case in self.cases:
            if not isinstance(case, EvaluationCase):
                raise TypeError(
                    f"each case must be an EvaluationCase, got {type(case).__name__}"
                )

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvaluationCase:
        return self.cases[index]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dict of the cases in their exact order.

        Only evaluation-case data is serialized -- each ``query`` and the
        relevant chunk ids reproduced from ``EvaluationCase``'s canonical sorted
        tuple as a JSON-friendly list (a CaseSet never re-sorts them). Fixed key
        structure (``{"cases": [{"query", "relevant_chunk_ids"}, ...]}``), no
        reprs, object addresses, type names, timestamps, paths, or hashes.
        Deterministic across repeated calls and across equivalent instances.
        """
        return {
            "cases": [
                {
                    "query": case.query,
                    "relevant_chunk_ids": list(case.relevant_chunk_ids),
                }
                for case in self.cases
            ]
        }
