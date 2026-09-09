"""Semantic retrieval profile: real sentence-transformers embeddings, E2E.

Phase 11 Step 3 proves that a *real* semantic embedding (not the stdlib
``DeterministicEmbedder`` stand-in) flows through the existing retrieval
stack end to end and produces a usable dense profile. It is deliberately a
separate, non-deterministic evidence channel:

- corpus / dataset / chunking / evaluation are the *same canonical ones* used
  by the frozen Phase 10B/10C/10D runs (``run_retrieval_small_baseline``
  ingestion + ``benchmark/datasets/retrieval_small.json``);
- the embedder is ``SemanticEmbeddingProvider`` over
  ``sentence-transformers/all-MiniLM-L6-v2`` (real weights, downloaded on
  first use). No monkeypatch, no fake backend, no ``DeterministicEmbedder``
  anywhere in this pipeline;
- retrieval components are the existing public ones, unchanged and unpatched:
  ``DenseRetriever`` / ``LexicalRetriever`` / ``HybridRetriever`` /
  ``WeightedScoreFusion`` / ``RRFFusion`` / ``InMemoryVectorStore`` /
  ``InMemoryLexicalIndex``;
- only retrieval *configuration* varies, in a minimal 3-cell matrix locked in
  Step 1 of Phase 11 Step 3 and never tuned against results::

      S1 Dense-only          S2 Hybrid + Weighted          S3 Hybrid + RRF

  unified candidate_k=33, top_k=8, reranker=None (no reranker, no new fusion,
  no new metric -- the six Phase 8 metrics come verbatim from
  ``EvaluationResult.to_dict()``);
- metrics are computed by the real evaluator (``EvaluationHarness`` ->
  ``RetrievalEvaluator`` -> ``EvaluationResult``); nothing is hand-computed;
- because the model is an external dependency, this profile is *not* promised
  byte-identical across environments. The runner runs the whole profile twice
  on two freshly constructed providers/paths, compares the two evidence
  dicts, and records ``same_environment_reproducible`` = true *only* when the
  two passes are exactly equal; otherwise it reports the honest result and
  still writes evidence. It never asserts the frozen deterministic files and
  never writes to them.

Interpretation guard: this 6-document / 8-case corpus is *not* a semantic-
superiority benchmark. One case is ``missing::0`` (recall ceiling 7/8) and
the other queries are lexical-overlap-heavy, so the results answer "does a
real semantic embedding reach and drive the retrieval stack?" -- never
"Dense > Lexical". The evidence records ``deterministic=false`` even though
the same machine re-runs identically.

Run from the repository root (the model is downloaded on first run unless
already cached):

    python benchmark/run_semantic_profile.py

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

#: Sibling Phase 10B runner: the canonical corpus ingestion + boundary checks.
import run_retrieval_small_baseline as _baseline

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "benchmark" / "datasets" / "retrieval_small.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_semantic_profile.json"

#: Stable relative identifier used in the evidence (never an absolute path).
EXPERIMENT_ID = "retrieval_semantic_profile"

#: The one external model this profile is pinned to. Never silently substituted.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = MODEL_NAME

#: Fixed experiment environment -- identical corpus/chunking to the frozen
#: deterministic baseline, but top_k narrowed to 8 (candidate_k stays 33).
CHUNK_SIZE = 120
OVERLAP = 20
TOP_K = 8
CANDIDATE_K = 33
EXPECTED_DOCUMENTS = 6
EXPECTED_TOTAL_CHUNKS = 33
EXPECTED_CASES = 8

#: The minimal semantic matrix -- locked in Step 1, only configuration varies.
CONFIGS = (
    {"id": "S1", "retriever": "dense", "fusion": None, "reranker": None},
    {"id": "S2", "retriever": "hybrid", "fusion": "weighted", "reranker": None},
    {"id": "S3", "retriever": "hybrid", "fusion": "rrf", "reranker": None},
)

#: Environment facts recorded for reproducibility. No host / user / path /
#: timestamp -- only versions that explain why a re-run may differ.
_ENV_DEPENDENCIES = ("sentence_transformers", "torch", "numpy")


def _env_facts() -> dict:
    facts = {"python": platform.python_version()}
    for dist in _ENV_DEPENDENCIES:
        try:
            facts[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            facts[dist] = "not installed"
    return facts


class RetrievalPath:
    """One freshly built, self-contained semantic retrieval configuration.

    Every instance reads the corpus, chunks + embeds + indexes it (with the
    real semantic provider) into brand-new stores, and wires only the public
    components its ``spec`` names. Nothing is shared between instances.
    """

    def __init__(self, spec: dict, provider: SemanticEmbeddingProvider) -> None:
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
        self._hybrid = None
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

        Single-source retrievers are asked for ``top_k`` directly; hybrid
        retrieval widens to ``candidate_k`` then fuses down to ``top_k``.
        No reranker exists in this profile.
        """
        if self._hybrid is not None:
            return self._hybrid.retrieve(query, top_k=TOP_K, candidate_k=CANDIDATE_K)
        return self._retrieve(query, top_k=TOP_K)


def _evaluate(retrieval: RetrievalPath, case_set) -> object:
    """Run every case through ``retrieval`` and delegate to the real evaluator."""
    predictions = []
    for case in case_set:
        predictions.append(
            EvaluationPrediction(query=case.query, results=retrieval.run_case(case.query))
        )
    harness = EvaluationHarness(RetrievalEvaluator())
    return harness.run(list(case_set.cases), predictions)


def build_row(spec: dict, result) -> dict:
    """Assemble one evidence row: fixed config description + real metrics."""
    row = {
        "id": spec["id"],
        "retriever": spec["retriever"],
        "fusion": spec["fusion"],
        "reranker": spec["reranker"],
    }
    row.update(result.to_dict())
    row["deterministic"] = False
    return row


def _run_profile() -> dict:
    """One full semantic pass: fresh provider + fresh dataset, all 3 configs."""
    provider = SemanticEmbeddingProvider(model_name=MODEL_NAME)
    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(
            f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}"
        )
    rows = []
    for spec in CONFIGS:
        retrieval = RetrievalPath(spec, provider)
        rows.append(build_row(spec, _evaluate(retrieval, case_set)))

    first = rows[0]
    return {
        "profile": "semantic",
        "experiment": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "embedding_dimension": provider.dimension,
        "corpus_documents": EXPECTED_DOCUMENTS,
        "corpus_chunks": EXPECTED_TOTAL_CHUNKS,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "total_cases": first["total_cases"],
        "evaluated_cases": first["evaluated_cases"],
        "deterministic": False,
        "environment": _env_facts(),
        "configs": rows,
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Run the semantic matrix twice, compare passes, write honest evidence."""
    output = Path(out_path)

    evidence1 = _run_profile()
    evidence2 = _run_profile()
    reproducible = evidence1 == evidence2
    evidence = dict(evidence1)
    evidence["same_environment_reproducible"] = reproducible
    if not reproducible:
        # Honest report: semantic output is pinned to an environment, not
        # promised byte-identical across processes. Do not paper over drift.
        print(
            "WARNING: two same-environment semantic passes differed; "
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
