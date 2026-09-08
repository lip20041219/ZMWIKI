"""Controlled retrieval experiments: Dense / Lexical / Hybrid x Weighted / RRF x reranker.

Phase 10C measures the existing retrieval system under one fixed environment
(corpus / dataset / chunking / embedding / top_k / candidate_k / evaluator) by
varying only the retrieval configuration. It reuses the canonical corpus
ingestion + boundary checks from the Phase 10B baseline runner (byte-identical
input), then builds each configuration as a *direct composition* of existing
public components. ``DeterministicRAGPipeline`` is deliberately not used:
single-source and no-reranker configurations are not expressible through it
(``retrieve`` always asks the retriever for ``candidate_k`` and always reranks);
see docs/architecture.md "Controlled Retrieval Experiments".

The locked matrix (fixed in Step 1, never tuned against results):

    E1 Dense-only            E3 Hybrid + Weighted            E5 Hybrid + Weighted + Reranker
    E2 Lexical-only          E4 Hybrid + RRF                 E6 Hybrid + RRF + Reranker

E0 / the Phase 10B baseline (``Hybrid + WeightedScoreFusion +
DeterministicReranker``) is E5's configuration and is never regenerated: E5 is
a *recompute guard* that must reproduce the frozen baseline metrics exactly,
else the script stops.

For every configuration E1..E6 the script:

- rebuilds all components from scratch (no cache, no singleton, no reading of a
  prior retrieval result);
- runs the real chain twice on one instance and once on a fully fresh instance,
  and refuses to write evidence unless same-instance and cross-instance runs are
  identical in every view (``result`` / ``to_dict`` / ``summary``);
- verifies the E5 recompute against the committed baseline and stops otherwise;
- writes one machine-readable evidence file whose metrics come verbatim from
  ``EvaluationResult.to_dict()``. The frozen baseline file is never overwritten.

Run from the repository root:

    python benchmark/run_retrieval_experiments.py

An optional positional output path overrides the default evidence location
(used by the reproducibility tests so they never overwrite the committed
evidence file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrapping for the uninstalled src/ layout (mirrors the pyproject
# ``pythonpath = ["src"]`` pytest setting), so the script runs as-is from the
# repo root without ``pip install``. Stdlib only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zmrag import (
    DenseRetriever,
    DeterministicEmbedder,
    DeterministicReranker,
    EvaluationHarness,
    EvaluationPrediction,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    RetrievalEvaluator,
    RRFFusion,
    WeightedScoreFusion,
)

#: Sibling Phase 10B runner: the canonical corpus ingestion + boundary checks.
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmark" / "datasets" / "retrieval_small.json"
BASELINE_PATH = ROOT / "benchmark" / "results" / "retrieval_small_baseline.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_small_experiments.json"

#: Stable relative identifiers used in the evidence (never an absolute path).
DATASET_ID = "benchmark/datasets/retrieval_small.json"
EXPERIMENT_ID = "retrieval_small_controlled_experiments"

#: Fixed experiment environment -- identical to the frozen Phase 10B baseline.
CHUNK_SIZE = 120
OVERLAP = 20
TOP_K = 33
CANDIDATE_K = 33
EXPECTED_DOCUMENTS = 6
EXPECTED_CHUNKS = 33
EXPECTED_CASES = 8

#: The locked experiment matrix -- the *only* thing changed across configs.
CONFIGS = (
    {"id": "E1", "retriever": "dense", "fusion": None, "reranker": None},
    {"id": "E2", "retriever": "lexical", "fusion": None, "reranker": None},
    {"id": "E3", "retriever": "hybrid", "fusion": "weighted", "reranker": None},
    {"id": "E4", "retriever": "hybrid", "fusion": "rrf", "reranker": None},
    {"id": "E5", "retriever": "hybrid", "fusion": "weighted", "reranker": "deterministic"},
    {"id": "E6", "retriever": "hybrid", "fusion": "rrf", "reranker": "deterministic"},
)

#: Metric/count keys shared by EvaluationResult.to_dict and both evidence files.
RESULT_FIELDS = (
    "total_cases",
    "evaluated_cases",
    "hit_rate",
    "recall",
    "precision",
    "mrr",
    "map",
    "ndcg",
)


class RetrievalPath:
    """One freshly built, self-contained retrieval configuration.

    Every instance reads the corpus, chunks + embeds + indexes it into brand-new
    stores, and wires only the public components its ``spec`` names. Nothing is
    shared between instances and nothing is read from a prior result.
    """

    def __init__(self, spec: dict) -> None:
        chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        embedder = DeterministicEmbedder()
        self.vector_store = InMemoryVectorStore()
        self.lexical_index = InMemoryLexicalIndex()

        documents = _baseline.ingest_corpus()
        _baseline.verify_corpus(documents)
        for document in documents:
            for chunk in chunker.chunk(document):
                self.vector_store.upsert(chunk, embedder.embed(chunk.text))
                self.lexical_index.upsert(chunk)
        if not (
            self.vector_store.count() == EXPECTED_CHUNKS
            and self.lexical_index.count() == EXPECTED_CHUNKS
        ):
            raise RuntimeError(
                f"indexed {self.vector_store.count()}/{self.lexical_index.count()} "
                f"chunks, expected {EXPECTED_CHUNKS}"
            )

        dense = DenseRetriever(embedder, self.vector_store)
        lexical = LexicalRetriever(self.lexical_index)
        self._reranker = (
            DeterministicReranker() if spec["reranker"] is not None else None
        )

        if spec["retriever"] == "dense":
            self._retrieve = dense.retrieve
        elif spec["retriever"] == "lexical":
            self._retrieve = lexical.retrieve
        else:  # hybrid
            fusion = (
                WeightedScoreFusion() if spec["fusion"] == "weighted" else RRFFusion()
            )
            self._hybrid = HybridRetriever(dense, lexical, fusion)

    def run_case(self, query: str) -> list:
        """Ordered ``SearchResult`` list for one query under this configuration.

        Single-source retrievers are asked for the full corpus depth (top_k=33,
        which equals the corpus size). Hybrid retrieval widens to ``candidate_k``
        then optionally reranks. Results are used verbatim -- never padded,
        never re-sorted, never capped beyond the stage's own "at most k".
        """
        if getattr(self, "_hybrid", None) is not None:
            results = self._hybrid.retrieve(
                query, top_k=TOP_K, candidate_k=CANDIDATE_K
            )
        else:
            results = self._retrieve(query, top_k=TOP_K)
        if self._reranker is not None:
            return self._reranker.rerank(query, results, top_k=TOP_K)
        return results


def _evaluate(retrieval: RetrievalPath, case_set) -> object:
    """Run every case through ``retrieval`` and delegate to the real evaluator.

    Predictions are built per case in dataset order from the retrieval output
    and scored by ``EvaluationHarness`` -> ``RetrievalEvaluator``. No metric is
    computed here; the returned ``EvaluationResult`` is the evaluator's own.
    """
    predictions = []
    for case in case_set:
        predictions.append(
            EvaluationPrediction(query=case.query, results=retrieval.run_case(case.query))
        )
    harness = EvaluationHarness(RetrievalEvaluator())
    return harness.run(list(case_set.cases), predictions)


def _assert_equal(label: str, a, b) -> None:
    """Stop-and-report guard: two runs must be identical in every view."""
    checks = {
        "result equality": a == b,
        "to_dict equality": a.to_dict() == b.to_dict(),
        "summary equality": a.summary() == b.summary(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"{label}: determinism failed on {', '.join(failed)}")


def _check_e5_against_baseline(row: dict) -> None:
    """E5 recompute guard: E5 must equal the frozen Phase 10B baseline metrics.

    Reads the baseline file read-only; never regenerates or overwrites it.
    """
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    mismatches = [
        field
        for field in RESULT_FIELDS
        if row[field] != baseline[field]
    ]
    if mismatches:
        raise RuntimeError(
            "BLOCKED -- baseline recompute mismatch on: "
            + ", ".join(f"{f} ({row[f]} != {baseline[f]})" for f in mismatches)
        )


def build_row(spec: dict, result) -> dict:
    """Assemble one evidence row: fixed config description + real metrics."""
    metrics = result.to_dict()
    row = {
        "id": spec["id"],
        "retriever": spec["retriever"],
        "fusion": spec["fusion"],
        "reranker": spec["reranker"],
    }
    row.update(metrics)
    row["deterministic"] = True
    return row


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the controlled matrix, verify determinism + the E5 guard, write JSON."""
    output = Path(out_path)

    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(
            f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}"
        )

    rows = []
    for spec in CONFIGS:
        retrieval = RetrievalPath(spec)
        result1 = _evaluate(retrieval, case_set)
        # Same-instance determinism: a second run over the same retrieval path.
        result2 = _evaluate(retrieval, case_set)
        _assert_equal("same-instance", result1, result2)

        # Cross-instance determinism: an entirely fresh path + fresh dataset load.
        fresh = RetrievalPath(spec)
        case_set_fresh = JsonEvaluationDatasetLoader().load(DATASET_PATH)
        result3 = _evaluate(fresh, case_set_fresh)
        _assert_equal("cross-instance", result1, result3)

        row = build_row(spec, result1)
        rows.append(row)
        if spec["id"] == "E5":
            # The E5 recompute guard is checked before E6 runs.
            _check_e5_against_baseline(row)

    evidence = {
        "experiment": EXPERIMENT_ID,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_CHUNKS,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "total_cases": rows[0]["total_cases"],
        "evaluated_cases": rows[0]["evaluated_cases"],
        "configs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVIDENCE_PATH
    evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
