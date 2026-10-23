"""Controlled truncation experiments: Dense / Lexical / Hybrid x Weighted / RRF under candidate_k / top_k cutoffs.

Phase 10D measures how candidate_k / top_k truncation changes recall and
ranking for each retrieval configuration. Phase 10C ran everything at the
full-pool depth top_k = candidate_k = 33 (where hit_rate / recall saturate at
7/8 because the whole 33-chunk corpus is returned), so candidate truncation
was unobservable. This script fixes that single variable: it reuses the
canonical Phase 10B corpus ingestion and Phase 10C evaluation chain unchanged
and varies only the two cutoff knobs.

The locked matrix (fixed in Step 1, never tuned against results):

    D1 Dense-only             top_k in {4, 8}                 (candidate_k = null)
    D2 Lexical-only           top_k in {4, 8}                 (candidate_k = null)
    D3 Hybrid + Weighted      candidate_k in {4,8,16,33}, top_k in {4,8}, top_k <= candidate_k
    D4 Hybrid + RRF           same 7 cells as D3

No reranker is used (Phase 10C established the DeterministicReranker is
subset-only and therefore has no recall role under truncation). No pipeline:
``DeterministicRAGPipeline`` always asks for a candidate_k pool and always
reranks, so single-source and no-reranker configurations are not expressible
through it; each cell is composed directly from existing public components,
exactly like Phase 10C.

The full-pool control (candidate_k = top_k = 33) is *not* regenerated as new
rows: Phase 10C already provides it (E1/E2/E3/E4). Instead an E3 cross-phase
guard recomputes the Phase 10C E3 configuration (Hybrid + Weighted, 33/33)
through this script's own path and requires it to reproduce the frozen E3 row
exactly -- proving this composition path equals the proven Phase 10C path --
before any evidence is written.

For every cell the script:
- rebuilds all components from scratch (no cache, no singleton, no reading of a
  prior retrieval result);
- runs the real chain twice on one instance and once on a fully fresh instance,
  and refuses to write evidence unless same-instance and cross-instance runs are
  identical in every view (``result`` / ``to_dict`` / ``summary``);
- runs the E3 cross-phase recompute guard and stops otherwise;
- writes one machine-readable evidence file whose metrics come verbatim from
  ``EvaluationResult.to_dict()``. The frozen Phase 10B / Phase 10C evidence
  files are never overwritten.

Run from the repository root:

    python benchmark/run_retrieval_truncation.py

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

from zmwiki import (
    DenseRetriever,
    DeterministicEmbedder,
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
EXPERIMENTS_PATH = ROOT / "benchmark" / "results" / "retrieval_small_experiments.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_truncation.json"

#: Stable identifiers used in the evidence (never an absolute path).
DATASET_ID = "benchmark/datasets/retrieval_small.json"
EXPERIMENT_ID = "retrieval_truncation"
REFERENCE_EXPERIMENT = "retrieval_small_controlled_experiments"

#: Fixed experiment environment -- identical to Phase 10B / Phase 10C.
CHUNK_SIZE = 120
OVERLAP = 20
EXPECTED_DOCUMENTS = 6
EXPECTED_CHUNKS = 33
EXPECTED_CASES = 8

#: The full-pool control (Hybrid + Weighted, candidate_k = top_k = 33) as
#: frozen in Phase 10C's E3 row -- recomputed, never regenerated as a row.
E3_CONFIG = {"retriever": "hybrid", "fusion": "weighted", "candidate_k": 33, "top_k": 33}

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

#: Metric keys that appear in each evidence row (counts live at the top level).
ROW_METRIC_FIELDS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")

#: The locked 18-cell matrix, in fixed order (never reordered, never sorted).
ROWS = (
    # D1 Dense-only: candidate_k = null (DenseRetriever has no candidate_k).
    {"id": "D1-t4", "retriever": "dense", "fusion": None, "candidate_k": None, "top_k": 4},
    {"id": "D1-t8", "retriever": "dense", "fusion": None, "candidate_k": None, "top_k": 8},
    # D2 Lexical-only: candidate_k = null (LexicalRetriever has no candidate_k).
    {"id": "D2-t4", "retriever": "lexical", "fusion": None, "candidate_k": None, "top_k": 4},
    {"id": "D2-t8", "retriever": "lexical", "fusion": None, "candidate_k": None, "top_k": 8},
    # D3 Hybrid + Weighted: 7 cells, top_k <= candidate_k.
    {"id": "D3-c4-t4", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 4, "top_k": 4},
    {"id": "D3-c8-t4", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 8, "top_k": 4},
    {"id": "D3-c16-t4", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 16, "top_k": 4},
    {"id": "D3-c33-t4", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 33, "top_k": 4},
    {"id": "D3-c8-t8", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 8, "top_k": 8},
    {"id": "D3-c16-t8", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 16, "top_k": 8},
    {"id": "D3-c33-t8", "retriever": "hybrid", "fusion": "weighted", "candidate_k": 33, "top_k": 8},
    # D4 Hybrid + RRF: identical 7 cells.
    {"id": "D4-c4-t4", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 4, "top_k": 4},
    {"id": "D4-c8-t4", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 8, "top_k": 4},
    {"id": "D4-c16-t4", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 16, "top_k": 4},
    {"id": "D4-c33-t4", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 33, "top_k": 4},
    {"id": "D4-c8-t8", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 8, "top_k": 8},
    {"id": "D4-c16-t8", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 16, "top_k": 8},
    {"id": "D4-c33-t8", "retriever": "hybrid", "fusion": "rrf", "candidate_k": 33, "top_k": 8},
)

#: Each row's config identity (for grouping a single path build per config).
#: A path does not depend on top_k / candidate_k, only on retriever + fusion.
def _group_key(row: dict) -> str:
    return row["retriever"] + "|" + str(row["fusion"])


class RetrievalPath:
    """One freshly built, self-contained retrieval configuration (no reranker).

    Every instance reads the corpus, chunks + embeds + indexes it into brand-new
    stores, and wires only the public components its retriever/fusion name.
    Nothing is shared between instances and nothing is read from a prior result.
    """

    def __init__(self, retriever: str, fusion: str | None) -> None:
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

        self._dense = DenseRetriever(embedder, self.vector_store)
        self._lexical = LexicalRetriever(self.lexical_index)
        self._kind = retriever
        if retriever == "hybrid":
            if fusion == "weighted":
                self._fusion = WeightedScoreFusion()
            elif fusion == "rrf":
                self._fusion = RRFFusion()  # default rrf_k = 60 -- never tuned
            else:
                raise ValueError(f"unknown hybrid fusion: {fusion!r}")
            self._hybrid = HybridRetriever(self._dense, self._lexical, self._fusion)

    def retrieve(self, query: str, top_k: int, candidate_k: int | None) -> list:
        """Ordered ``SearchResult`` list for one query under this configuration.

        Single-source retrievers are asked for ``top_k`` only (their API has no
        ``candidate_k``). Hybrid retrieval widens each source to ``candidate_k``
        and fuses down to ``top_k``. Results are used verbatim -- never padded,
        never re-sorted, never capped beyond the stage's own "at most k".
        """
        if self._kind == "dense":
            return self._dense.retrieve(query, top_k=top_k)
        if self._kind == "lexical":
            return self._lexical.retrieve(query, top_k=top_k)
        return self._hybrid.retrieve(query, top_k=top_k, candidate_k=candidate_k)


def _evaluate(retrieval: RetrievalPath, case_set, top_k: int, candidate_k: int | None) -> object:
    """Run every case through ``retrieval`` and delegate to the real evaluator.

    Predictions are built per case in dataset order from the retrieval output
    and scored by ``EvaluationHarness`` -> ``RetrievalEvaluator``. No metric is
    computed here; the returned ``EvaluationResult`` is the evaluator's own.
    """
    predictions = []
    for case in case_set:
        results = retrieval.retrieve(case.query, top_k=top_k, candidate_k=candidate_k)
        predictions.append(EvaluationPrediction(query=case.query, results=results))
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


def _check_e3_against_experiments(result) -> None:
    """E3 cross-phase guard: recompute must equal the frozen Phase 10C E3 row.

    Reads the Phase 10C experiments file read-only; never regenerates or
    overwrites it, and never copies E3 into the Phase 10D evidence.
    """
    frozen = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
    e3 = next(row for row in frozen["configs"] if row["id"] == "E3")
    if (e3["retriever"], e3["fusion"], e3["reranker"]) != ("hybrid", "weighted", None):
        raise RuntimeError(f"frozen E3 configuration is not hybrid+weighted: {e3!r}")
    got = result.to_dict()
    mismatches = [field for field in RESULT_FIELDS if got[field] != e3[field]]
    if mismatches:
        raise RuntimeError(
            "BLOCKED -- E3 cross-phase recompute mismatch on: "
            + ", ".join(f"{f} ({got[f]} != {e3[f]})" for f in mismatches)
        )


def build_row(spec: dict, result) -> dict:
    """Assemble one evidence row: fixed config description + real metrics."""
    metrics = result.to_dict()
    row = {
        "id": spec["id"],
        "retriever": spec["retriever"],
        "fusion": spec["fusion"],
        "reranker": None,
        "candidate_k": spec["candidate_k"],
        "top_k": spec["top_k"],
    }
    for field in ROW_METRIC_FIELDS:
        row[field] = metrics[field]
    row["deterministic"] = True
    return row


def _build_evidence(rows: list, result0) -> dict:
    """Assemble the fixed top-level evidence schema (counts come from a result)."""
    counts = result0.to_dict()
    return {
        "experiment": EXPERIMENT_ID,
        "reference": REFERENCE_EXPERIMENT,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_CHUNKS,
        "total_cases": counts["total_cases"],
        "evaluated_cases": counts["evaluated_cases"],
        "results": rows,
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the 18-cell truncation matrix, verify determinism + the E3 guard, write JSON."""
    output = Path(out_path)

    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}")

    # Group cells by config so each config's stores are built once; the path is
    # identical across that config's cutoffs (cutoffs are only retrieve args).
    groups: dict[str, list[dict]] = {}
    for row in ROWS:
        groups.setdefault(_group_key(row), []).append(row)

    rows: list[dict] = []
    counts_result = None
    for key, group in groups.items():
        # First instance: build once, run each cell twice (same-instance guard).
        first = RetrievalPath(group[0]["retriever"], group[0]["fusion"])
        first_results = {
            spec["id"]: _evaluate(first, case_set, spec["top_k"], spec["candidate_k"])
            for spec in group
        }
        for spec in group:
            again = _evaluate(first, case_set, spec["top_k"], spec["candidate_k"])
            _assert_equal(f"same-instance {spec['id']}", first_results[spec["id"]], again)

        # Cross-instance determinism: an entirely fresh path + fresh dataset load.
        fresh_documents = _baseline.ingest_corpus()
        _baseline.verify_corpus(fresh_documents)
        fresh = RetrievalPath(group[0]["retriever"], group[0]["fusion"])
        case_set_fresh = JsonEvaluationDatasetLoader().load(DATASET_PATH)
        for spec in group:
            third = _evaluate(fresh, case_set_fresh, spec["top_k"], spec["candidate_k"])
            _assert_equal(
                f"cross-instance {spec['id']}", first_results[spec["id"]], third
            )
            rows.append(build_row(spec, first_results[spec["id"]]))
        if counts_result is None:
            counts_result = first_results[group[0]["id"]]

    # E3 cross-phase guard: recompute Hybrid + Weighted @ 33/33 through this
    # script's own path and compare to the frozen Phase 10C E3 row.
    e3_path = RetrievalPath(E3_CONFIG["retriever"], E3_CONFIG["fusion"])
    e3_result = _evaluate(
        e3_path, case_set, E3_CONFIG["top_k"], E3_CONFIG["candidate_k"]
    )
    e3_again = _evaluate(
        e3_path, case_set, E3_CONFIG["top_k"], E3_CONFIG["candidate_k"]
    )
    _assert_equal("E3 guard same-instance", e3_result, e3_again)
    _check_e3_against_experiments(e3_result)

    if counts_result is None:
        raise RuntimeError("no cells evaluated -- internal error")
    evidence = _build_evidence(rows, counts_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVIDENCE_PATH
    evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
