"""Semantic rerank profile: real CrossEncoder two-stage retrieval, E2E.

Phase 12 Step 3 proves the optional :class:`CrossEncoderReranker`
(``zmrag/semantic_rerank.py``) drives a real two-stage retrieval pass through
the existing stack, end to end, on the *same* locked environment Phase 11 used:

- corpus / dataset / chunking / evaluation are the *same canonical ones* used
  by the frozen Phase 10B/10C/10D runs and Phase 11's semantic profile
  (``run_retrieval_small_baseline`` ingestion +
  ``benchmark/datasets/retrieval_small.json``, 6 documents / 33 chunks / 8
  cases);
- the embedder is ``SemanticEmbeddingProvider`` over
  ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim), exactly the Phase 11
  locked embedding -- no substitution;
- the reranker is a real sentence-transformers ``CrossEncoder`` behind
  ``CrossEncoderReranker`` -- one locked model,
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``, used identically by all three
  rows;
- retrieval components are the existing public ones, unchanged and unpatched;
- two-stage wiring, per row: first stage recalls a candidate pool of up to 33
  (``candidate_k``), then ``CrossEncoderReranker.rerank`` re-scores that pool
  and returns at most ``top_k=8``. The pool is *never* truncated to 8 before
  the reranker -- otherwise there is no 33 -> 8 rerank stage::

      R1 Dense 33 -> CrossEncoder -> 8
      R2 Hybrid(Weighted) 33 -> CrossEncoder -> 8
      R3 Hybrid(RRF) 33 -> CrossEncoder -> 8

- metrics come verbatim from ``EvaluationResult.to_dict()`` via the real
  evaluator (``EvaluationHarness`` -> ``RetrievalEvaluator``); nothing is
  hand-computed;
- the Phase 11 semantic evidence is a read-only delta baseline. Each row
  records ``delta_vs_phase11`` = this row's metric minus the matching Phase 11
  row (R1->S1, R2->S2, R3->S3) for ``mrr`` / ``map`` / ``ndcg`` only. A
  positive, zero, or negative delta is recorded honestly; nothing is tuned to
  force a positive number;
- because the model is an external dependency, this profile is *not* promised
  byte-identical across environments. The runner runs the whole matrix twice
  on two freshly constructed providers/rerankers/paths, compares the two
  evidence dicts, and records ``same_environment_reproducible`` = true *only*
  when the two passes are exactly equal; otherwise it reports the honest
  result and still writes evidence.

Interpretation guard (recorded verbatim in the evidence ``notes``): this is a
*plumbing and ranking-stage validation*, not evidence that CrossEncoder
reranking is universally superior. The corpus is small (6 documents) and
lexical-overlap-heavy; one case is ``missing::0``, so the observable recall
ceiling is 7/8 = 0.875. ``deterministic=false`` is recorded even when the same
machine re-runs identically.

Run from the repository root (models download on first use unless cached):

    python benchmark/run_semantic_rerank_profile.py

An optional positional output path overrides the default evidence location
(used by the integration tests so they never overwrite the committed file).
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

# Bootstrapping for the uninstalled src/ layout (mirrors the pyproject
# ``pythonpath = ["src"]`` pytest setting), so the script runs as-is from the
# repo root without ``pip install``. Stdlib only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zmrag import (
    DenseRetriever,
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
from zmrag.semantic import SemanticEmbeddingProvider
from zmrag.semantic_rerank import CrossEncoderReranker

#: Sibling Phase 10B runner: the canonical corpus ingestion + boundary checks.
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmark" / "datasets" / "retrieval_small.json"
PHASE11_PATH = ROOT / "benchmark" / "results" / "retrieval_semantic_profile.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_semantic_rerank_profile.json"

#: Stable relative identifiers used in the evidence (never an absolute path).
EXPERIMENT_ID = "retrieval_semantic_rerank_profile"

#: The locked embedding (Phase 11) and the locked reranker. Never substituted,
#: never compared against other models.
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Fixed experiment environment -- identical corpus/chunking to the frozen
#: deterministic baseline and Phase 11. Candidate pool is the full corpus depth
#: (33); the reranker narrows the pool down to top_k=8.
CHUNK_SIZE = 120
OVERLAP = 20
TOP_K = 8
CANDIDATE_K = 33
EXPECTED_DOCUMENTS = 6
EXPECTED_TOTAL_CHUNKS = 33
EXPECTED_CASES = 8

#: The locked rerank matrix -- fixed in Step 1 of Phase 12 Step 3, never tuned
#: against results. First-stage configuration is Phase 11's S1/S2/S3; only the
#: reranker stage is added. One reranker model for all three rows.
CONFIGS = (
    {"id": "R1", "retriever": "dense", "fusion": None, "reranker": "cross_encoder"},
    {"id": "R2", "retriever": "hybrid", "fusion": "weighted", "reranker": "cross_encoder"},
    {"id": "R3", "retriever": "hybrid", "fusion": "rrf", "reranker": "cross_encoder"},
)

#: Phase 11 delta baseline row per rerank row (read-only, never overwritten).
BASELINE_ROW_BY_ID = {"R1": "S1", "R2": "S2", "R3": "S3"}

#: The three Phase 8 metrics whose Phase 11 delta is recorded (no new metrics).
DELTA_KEYS = ("mrr", "map", "ndcg")

#: Honest interpretation guard, recorded verbatim in the evidence notes.
NOTES = [
    "plumbing and ranking-stage validation, not evidence that CrossEncoder "
    "reranking is universally superior",
    "corpus is small (6 documents) and lexical-overlap-heavy",
    "observable recall ceiling is 7/8 = 0.875 (one case is missing::0)",
]

#: Environment facts recorded for reproducibility. No host / user / path /
#: timestamp -- only versions that explain why a re-run may differ.
_ENV_DEPENDENCIES = ("sentence_transformers", "torch", "numpy", "transformers")


def _env_facts() -> dict:
    facts = {"python": platform.python_version()}
    for dist in _ENV_DEPENDENCIES:
        try:
            facts[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            facts[dist] = "not installed"
    return facts


class RetrievalPath:
    """One freshly built, self-contained two-stage retrieval configuration.

    Every instance reads the corpus, chunks + embeds + indexes it (with the
    real semantic provider) into brand-new stores, wires only the public
    components its ``spec`` names, and holds the shared ``CrossEncoderReranker``
    for the ranking stage. Nothing else is shared between instances.
    """

    def __init__(
        self,
        spec: dict,
        provider: SemanticEmbeddingProvider,
        reranker: CrossEncoderReranker,
    ) -> None:
        chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        self.vector_store = InMemoryVectorStore()
        self.lexical_index = InMemoryLexicalIndex()

        documents = _baseline.ingest_corpus()
        _baseline.verify_corpus(documents)
        for document in documents:
            for chunk in chunker.chunk(document):
                self.vector_store.upsert(chunk, provider.embed(chunk.text))
                self.lexical_index.upsert(chunk)
        if not (
            self.vector_store.count() == EXPECTED_TOTAL_CHUNKS
            and self.lexical_index.count() == EXPECTED_TOTAL_CHUNKS
        ):
            raise RuntimeError(
                f"indexed {self.vector_store.count()}/{self.lexical_index.count()} "
                f"chunks, expected {EXPECTED_TOTAL_CHUNKS}"
            )

        dense = DenseRetriever(provider, self.vector_store)
        lexical = LexicalRetriever(self.lexical_index)
        self._dense = dense
        self._hybrid = None
        if spec["retriever"] == "dense":
            pass
        elif spec["retriever"] == "lexical":
            raise ValueError(
                f"{spec['id']}: lexical-only is not in the locked rerank matrix"
            )
        else:  # hybrid
            fusion = (
                WeightedScoreFusion() if spec["fusion"] == "weighted" else RRFFusion()
            )
            self._hybrid = HybridRetriever(dense, lexical, fusion)
        self._reranker = reranker

    def run_case(self, query: str) -> list:
        """Two-stage results for one query: pool up to ``CANDIDATE_K``, then rerank.

        Stage 1 recalls a candidate pool of up to ``CANDIDATE_K=33`` (dense
        directly, hybrid via ``top_k=candidate_k=33`` so fusion is *not* allowed
        to truncate to 8). Stage 2 re-ranks that pool with the CrossEncoder and
        returns at most ``top_k=8``. The pool is never truncated to 8 before the
        reranker, so the 33 -> 8 two-stage shape is real.
        """
        if self._hybrid is not None:
            pool = self._hybrid.retrieve(
                query, top_k=CANDIDATE_K, candidate_k=CANDIDATE_K
            )
        else:
            pool = self._dense.retrieve(query, top_k=CANDIDATE_K)
        return self._reranker.rerank(query, pool, top_k=TOP_K)


def _evaluate(retrieval: RetrievalPath, case_set) -> object:
    """Run every case through ``retrieval`` and delegate to the real evaluator."""
    predictions = []
    for case in case_set:
        predictions.append(
            EvaluationPrediction(query=case.query, results=retrieval.run_case(case.query))
        )
    harness = EvaluationHarness(RetrievalEvaluator())
    return harness.run(list(case_set.cases), predictions)


def _delta(result, baseline_row: dict) -> dict:
    """Honest per-metric delta vs the Phase 11 baseline row.

    ``rerank_metric - phase11_metric`` for mrr / map / ndcg only. A negative or
    zero delta is recorded exactly as measured -- no tuning to force an
    improvement.
    """
    metrics = result.to_dict()
    return {
        f"{key}_delta": metrics[key] - baseline_row[key]
        for key in DELTA_KEYS
    }


def build_row(spec: dict, result, baseline_row: dict) -> dict:
    """Assemble one evidence row: config + real metrics + Phase 11 delta."""
    row = {
        "id": spec["id"],
        "retriever": spec["retriever"],
        "fusion": spec["fusion"],
        "reranker": spec["reranker"],
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
    }
    row.update(result.to_dict())
    row["deterministic"] = False
    row["delta_vs_phase11"] = _delta(result, baseline_row)
    return row


def _run_profile(baseline_by_id: dict) -> dict:
    """One full two-stage pass: fresh provider + reranker + dataset, all 3 rows."""
    provider = SemanticEmbeddingProvider(model_name=EMBEDDING_MODEL_ID)
    reranker = CrossEncoderReranker(model_name=RERANKER_MODEL_ID)
    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(
            f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}"
        )
    rows = []
    for spec in CONFIGS:
        baseline_id = BASELINE_ROW_BY_ID[spec["id"]]
        retrieval = RetrievalPath(spec, provider, reranker)
        rows.append(
            build_row(spec, _evaluate(retrieval, case_set), baseline_by_id[baseline_id])
        )

    first = rows[0]
    return {
        "profile": "semantic_rerank",
        "experiment": EXPERIMENT_ID,
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "embedding_dimension": provider.dimension,
        "reranker_model_id": RERANKER_MODEL_ID,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "total_cases": first["total_cases"],
        "evaluated_cases": first["evaluated_cases"],
        "deterministic": False,
        "notes": NOTES,
        "environment": _env_facts(),
        "configs": rows,
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the two-stage matrix twice, compare passes, write honest evidence."""
    output = Path(out_path)

    # Phase 11 evidence is read-only; only its metrics are read, never written.
    phase11 = json.loads(PHASE11_PATH.read_text(encoding="utf-8"))
    baseline_by_id = {row["id"]: row for row in phase11["configs"]}

    evidence1 = _run_profile(baseline_by_id)
    evidence2 = _run_profile(baseline_by_id)
    reproducible = evidence1 == evidence2
    evidence = dict(evidence1)
    evidence["same_environment_reproducible"] = reproducible
    if not reproducible:
        # Honest report: semantic output is pinned to an environment, not
        # promised byte-identical across processes. Do not paper over drift.
        print(
            "WARNING: two same-environment semantic rerank passes differed; "
            "evidence written reflects the first pass only.",
            file=sys.stderr,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVIDENCE_PATH
    evidence = main(target)
    print(
        f"Evidence written to: {target} "
        f"[same_environment_reproducible="
        f"{evidence['same_environment_reproducible']}]"
    )
    print(json.dumps(evidence, indent=2))
