"""Reproducible retrieval baseline: real E2E run -> machine-readable evidence.

Phase 10B turns the Phase 10A local retrieval benchmark into an experiment
baseline that can be re-run, re-verified, and re-produced at any time. This
script is the *canonical execution path*:

- it rebuilds the whole chain from scratch on every run (no caching, no
  global state, no reading of a previous result);
- it ingests the canonical 6-document corpus through the real public API
  (PlainTextFileIngestor -> Document -> FixedSizeChunker ->
  DeterministicEmbedder -> InMemoryVectorStore + InMemoryLexicalIndex ->
  DenseRetriever / LexicalRetriever / WeightedScoreFusion / HybridRetriever);
- it verifies the corpus boundary (6 documents, 33 chunks,
  ``doc_<letter>::<integer>`` ids, no absolute paths) and the dataset boundary
  (8 cases) *before* any evidence is produced -- on mismatch it stops;
- it runs the real benchmark (DeterministicRAGPipeline ->
  RetrievalEvaluationRunner -> EvaluationHarness -> RetrievalEvaluator ->
  EvaluationResult) twice on one instance and once on a fully fresh instance,
  and refuses to write evidence unless same-instance and cross-instance runs
  are identical;
- it writes only the fixed, machine-readable evidence schema below; every
  metric comes verbatim from ``EvaluationResult.to_dict()`` -- nothing is
  hand-computed, and no timestamp / hostname / username / path / environment
  information leaks into the artifact.

Canonical experiment configuration is fixed (never derived at runtime):
``FixedSizeChunker(chunk_size=120, overlap=20)`` over ``benchmark/corpus``
(33 chunks), ``WeightedScoreFusion()``, ``top_k=33`` / ``candidate_k=33``,
``benchmark/datasets/retrieval_small.json`` (8 cases).

Run from the repository root:

    python benchmark/run_retrieval_small_baseline.py

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
    DeterministicBenchmarkRunner,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    EvaluationHarness,
    FixedSizeChunker,
    HybridRetriever,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    PlainTextFileIngestor,
    RetrievalEvaluationRunner,
    RetrievalEvaluator,
    WeightedScoreFusion,
)

#: Repo root, resolved from this file so the script runs from any cwd.
ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "benchmark" / "corpus"
DATASET_PATH = ROOT / "benchmark" / "datasets" / "retrieval_small.json"
DEFAULT_EVIDENCE_PATH = ROOT / "benchmark" / "results" / "retrieval_small_baseline.json"

#: Stable relative identifier used in the evidence (never an absolute path).
DATASET_ID = "benchmark/datasets/retrieval_small.json"

#: Canonical corpus: the six hand-written synthetic documents, in order.
CORPUS_FILES = (
    "doc_a.txt",
    "doc_b.txt",
    "doc_c.txt",
    "doc_d.txt",
    "doc_e.txt",
    "doc_f.txt",
)

#: Canonical experiment configuration (fixed, never derived at runtime).
CHUNK_SIZE = 120
OVERLAP = 20
EXPECTED_DOCUMENTS = 6
EXPECTED_CHUNKS_BY_DOC = {"doc_a": 6, "doc_b": 5, "doc_c": 6, "doc_d": 6, "doc_e": 5, "doc_f": 5}
EXPECTED_TOTAL_CHUNKS = 33
EXPECTED_CASES = 8
TOP_K = 33
CANDIDATE_K = 33


def ingest_corpus() -> list:
    """Read the six canonical TXT files through PlainTextFileIngestor.

    ``document_id`` is the file stem (``doc_a``, ...) -- explicit, stable,
    and never an absolute path. No bytes are read and no Document/Chunk is
    hand-built here; the public ingestion API is used unchanged.
    """
    ingestor = PlainTextFileIngestor()
    documents = []
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        documents.append(ingestor.ingest(path, document_id=path.stem))
    return documents


def verify_corpus(documents: list) -> None:
    """Stop-and-report guard: the real corpus must match the canonical config.

    Raises RuntimeError (never silently adjusts retrieval / chunking) when the
    actual ingested corpus is not 6 documents / 33 chunks with canonical
    per-document counts and clean ``doc_<letter>::<integer>`` chunk ids.
    """
    if len(documents) != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"corpus has {len(documents)} documents, expected {EXPECTED_DOCUMENTS}"
        )
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    per_doc = {}
    for document in documents:
        chunks = chunker.chunk(document)
        per_doc[document.id] = len(chunks)
        # ids must be doc_<letter>::<integer> in sequential order, with no path.
        expected_ids = [f"{document.id}::{i}" for i in range(len(chunks))]
        actual_ids = [chunk.id for chunk in chunks]
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"chunk ids for {document.id!r} are not canonical: {actual_ids!r}"
            )
        for cid in actual_ids:
            if "/" in cid or "\\" in cid:
                raise RuntimeError(f"chunk id contains a path separator: {cid!r}")
    if per_doc != EXPECTED_CHUNKS_BY_DOC:
        raise RuntimeError(
            f"per-document chunk counts {per_doc} != canonical {EXPECTED_CHUNKS_BY_DOC}"
        )
    total = sum(per_doc.values())
    if total != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(
            f"corpus has {total} chunks, expected {EXPECTED_TOTAL_CHUNKS}"
        )


def build_chain(documents: list) -> DeterministicBenchmarkRunner:
    """Build one fresh deterministic pipeline + evaluation chain, then ingest.

    Everything is reconstructed here on every call -- no singleton, no cache,
    no shared store. The wiring is exactly the canonical configuration.
    """
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    embedder = DeterministicEmbedder()
    vector_store = InMemoryVectorStore()
    lexical_index = InMemoryLexicalIndex()
    hybrid = HybridRetriever(
        DenseRetriever(embedder, vector_store),
        LexicalRetriever(lexical_index),
        WeightedScoreFusion(),
    )
    pipeline = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    for document in documents:
        pipeline.ingest(document)
    harness = EvaluationHarness(RetrievalEvaluator())
    evaluation_runner = RetrievalEvaluationRunner(pipeline=pipeline, harness=harness)
    return DeterministicBenchmarkRunner(evaluation_runner=evaluation_runner)


def _assert_equal_results(label: str, a, b) -> None:
    """Stop-and-report guard: two runs must be identical in every view."""
    checks = {
        "result equality": a == b,
        "to_dict equality": a.to_dict() == b.to_dict(),
        "summary equality": a.summary() == b.summary(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"{label}: determinism failed on {', '.join(failed)}")


def build_evidence(result, documents: int, chunks: int) -> dict:
    """Assemble the fixed evidence schema from a real EvaluationResult.

    Every metric is copied verbatim from ``result.to_dict()`` -- nothing is
    recomputed or hand-written. The dict literal fixes the key order.
    """
    metrics = result.to_dict()
    return {
        "dataset": DATASET_ID,
        "corpus_documents": documents,
        "corpus_chunks": chunks,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "total_cases": metrics["total_cases"],
        "evaluated_cases": metrics["evaluated_cases"],
        "hit_rate": metrics["hit_rate"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "mrr": metrics["mrr"],
        "map": metrics["map"],
        "ndcg": metrics["ndcg"],
        "deterministic": True,
    }


def main(out_path: str | Path = DEFAULT_EVIDENCE_PATH) -> dict:
    """Re-run the canonical baseline and write the evidence JSON; returns it."""
    output = Path(out_path)

    documents = ingest_corpus()
    verify_corpus(documents)

    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    if len(case_set) != EXPECTED_CASES:
        raise RuntimeError(
            f"dataset has {len(case_set)} cases, expected {EXPECTED_CASES}"
        )

    # First real benchmark (same instance).
    benchmark = build_chain(documents)
    result1 = benchmark.run(case_set, top_k=TOP_K, candidate_k=CANDIDATE_K)
    # Second real benchmark on the *same* instance -> same-instance determinism.
    result2 = benchmark.run(case_set, top_k=TOP_K, candidate_k=CANDIDATE_K)
    _assert_equal_results("same-instance", result1, result2)

    # Cross-instance determinism: an entirely fresh pipeline + fresh load.
    fresh_documents = ingest_corpus()
    verify_corpus(fresh_documents)
    benchmark2 = build_chain(fresh_documents)
    case_set2 = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    result3 = benchmark2.run(case_set2, top_k=TOP_K, candidate_k=CANDIDATE_K)
    _assert_equal_results("cross-instance", result1, result3)

    # Only now is deterministic=True evidence produced.
    evidence = build_evidence(result1, len(documents), EXPECTED_TOTAL_CHUNKS)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVIDENCE_PATH
    evidence = main(target)
    print(f"Evidence written to: {target}")
    print(json.dumps(evidence, indent=2))
