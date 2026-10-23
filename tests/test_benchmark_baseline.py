"""Phase 10B reproducible-baseline tests.

Phase 10A already pinned the corpus / dataset / E2E chain. This module pins
the *baseline evidence* contracts added by Phase 10B on top of that same real
chain, without re-implementing any metric and without touching production code:

- the canonical corpus boundary that the evidence asserts (6 documents, 33
  chunks, per-document counts, clean ``doc_<letter>::<integer>`` ids);
- the canonical dataset boundary (8 cases, order preserved);
- real indexing: the full TXT -> FileIngestor -> Document -> Pipeline.ingest
  -> chunker/embedder/vector+lexical stores chain actually holds 33 chunks;
- a real benchmark run's metrics come straight from
  ``EvaluationResult.to_dict()`` and equal the committed evidence --
  nothing in this file recomputes a metric;
- same-instance and cross-instance runs are identical in every view
  (result / to_dict / summary), the condition under which evidence is written;
- the evidence JSON has a fixed key set, fixed key order, only int/float
  metrics, ``deterministic: true``, and no timestamp / path / host / user /
  environment leakage;
- the canonical runner script itself (re-run as a subprocess to a fresh output
  path, so the committed evidence is never overwritten) reproduces the
  committed evidence exactly.

The committed evidence file is an *output* this module verifies; it is never
asserted against itself -- every comparison re-runs the real chain.
"""

import json
import subprocess
import sys
from pathlib import Path

from zmwiki import (
    DenseRetriever,
    DeterministicBenchmarkRunner,
    DeterministicContextBuilder,
    DeterministicEmbedder,
    DeterministicRAGPipeline,
    DeterministicReranker,
    EvaluationCaseSet,
    EvaluationHarness,
    EvaluationResult,
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

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmark"
CORPUS_DIR = BENCH_DIR / "corpus"
DATASET_PATH = BENCH_DIR / "datasets" / "retrieval_small.json"
EVIDENCE_PATH = BENCH_DIR / "results" / "retrieval_small_baseline.json"
RUNNER_SCRIPT = BENCH_DIR / "run_retrieval_small_baseline.py"

#: Canonical configuration -- fixed, shared with the evidence and the runner.
CHUNK_SIZE = 120
OVERLAP = 20
TOP_K = 33
CANDIDATE_K = 33
DOC_NAMES = [
    "doc_a.txt",
    "doc_b.txt",
    "doc_c.txt",
    "doc_d.txt",
    "doc_e.txt",
    "doc_f.txt",
]
EXPECTED_CHUNKS_BY_DOC = {
    "doc_a": 6,
    "doc_b": 5,
    "doc_c": 6,
    "doc_d": 6,
    "doc_e": 5,
    "doc_f": 5,
}
TOTAL_CHUNKS = 33
DATASET_ID = "benchmark/datasets/retrieval_small.json"

#: The dataset's 8 queries, in their exact file order.
EXPECTED_QUERIES = [
    "database B-tree index query",
    "cache eviction memory",
    "producer consumer message queue",
    "embedding vector similarity retrieval",
    "BM25 lexical vector ranking fusion",
    "document chunk context",
    "nearest neighbor search representation",
    "quantum submarine agriculture",
]

#: The metric + count fields the evidence shares with EvaluationResult.to_dict.
RESULT_FIELDS = [
    "total_cases",
    "evaluated_cases",
    "hit_rate",
    "recall",
    "precision",
    "mrr",
    "map",
    "ndcg",
]

#: The evidence's fixed top-level keys, in their fixed order.
EVIDENCE_KEYS = [
    "dataset",
    "corpus_documents",
    "corpus_chunks",
    "top_k",
    "candidate_k",
] + RESULT_FIELDS + ["deterministic"]

DOC_IDS = tuple(name[:-4] for name in DOC_NAMES)  # doc_a .. doc_f


def _load_documents():
    """Read the six canonical corpus files through the real file ingestor."""
    ingestor = PlainTextFileIngestor()
    return [
        ingestor.ingest(CORPUS_DIR / name, document_id=Path(name).stem)
        for name in DOC_NAMES
    ]


def _chunks_by_doc(documents):
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    return {doc.id: chunker.chunk(doc) for doc in documents}


def _build_pipeline(documents):
    """A fresh canonical deterministic pipeline, then ingest ``documents``."""
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    embedder = DeterministicEmbedder()
    vstore, lindex = InMemoryVectorStore(), InMemoryLexicalIndex()
    hybrid = HybridRetriever(
        DenseRetriever(embedder, vstore),
        LexicalRetriever(lindex),
        WeightedScoreFusion(),
    )
    pipeline = DeterministicRAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vstore,
        lexical_index=lindex,
        retriever=hybrid,
        reranker=DeterministicReranker(),
        context_builder=DeterministicContextBuilder(),
    )
    for document in documents:
        pipeline.ingest(document)
    return pipeline, vstore, lindex


def _make_benchmark(pipeline):
    harness = EvaluationHarness(RetrievalEvaluator())
    runner = RetrievalEvaluationRunner(pipeline=pipeline, harness=harness)
    return DeterministicBenchmarkRunner(evaluation_runner=runner)


def _fresh_setup():
    """One entirely fresh, independent wiring: pipeline + stores + benchmark."""
    documents = _load_documents()
    pipeline, vstore, lindex = _build_pipeline(documents)
    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    return _make_benchmark(pipeline), case_set, pipeline, vstore, lindex


def _read_evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


# --- A/B. Corpus boundary asserted by the evidence ----------------------------


def test_corpus_ingests_into_the_six_canonical_documents():
    documents = _load_documents()
    assert [d.id for d in documents] == list(DOC_IDS)
    assert all(d.content for d in documents)


def test_corpus_chunks_to_33_with_canonical_per_document_counts():
    per_doc = {
        doc_id: len(chunks) for doc_id, chunks in _chunks_by_doc(_load_documents()).items()
    }
    assert per_doc == EXPECTED_CHUNKS_BY_DOC
    assert sum(per_doc.values()) == TOTAL_CHUNKS


def test_chunk_ids_are_doc_letter_integer_with_no_path():
    documents = _load_documents()
    for doc in documents:
        for chunk in _chunks_by_doc(documents)[doc.id]:
            prefix, sep, rest = chunk.id.partition("::")
            assert sep == "::", chunk.id
            assert prefix == doc.id, chunk.id  # document id, never a path
            assert rest.isdigit(), chunk.id
            assert "/" not in chunk.id and "\\" not in chunk.id, chunk.id
            assert chunk.document_id == doc.id


# --- C. Dataset boundary ------------------------------------------------------


def test_dataset_loads_8_cases_in_file_order():
    case_set = JsonEvaluationDatasetLoader().load(DATASET_PATH)
    assert isinstance(case_set, EvaluationCaseSet)
    assert len(case_set) == 8
    assert [case.query for case in case_set] == EXPECTED_QUERIES


# --- D. Real indexing of the whole corpus ------------------------------------


def test_real_pipeline_indexes_33_chunks_and_retrieves_them():
    _, _, pipeline, vstore, lindex = _fresh_setup()
    assert vstore.count() == TOTAL_CHUNKS == lindex.count()
    # The dense + lexical path returns real, retrievable chunk objects.
    results = pipeline.retrieve("B-tree index", top_k=TOP_K, candidate_k=CANDIDATE_K)
    assert results
    for result in results:
        assert result.chunk.text  # a real stored chunk, not a fabricated one
        assert result.chunk.document_id in DOC_IDS


# --- E/G. Real benchmark determinism (same + cross instance) ------------------


def test_real_benchmark_same_and_cross_instance_are_identical():
    benchmark, case_set, _, _, _ = _fresh_setup()

    first = benchmark.run(case_set, top_k=TOP_K, candidate_k=CANDIDATE_K)
    assert isinstance(first, EvaluationResult)
    assert (first.total_cases, first.evaluated_cases) == (8, 8)

    # Same instance, second run.
    second = benchmark.run(case_set, top_k=TOP_K, candidate_k=CANDIDATE_K)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()

    # Cross instance: a fully independent pipeline + stores + benchmark.
    other_benchmark, other_case_set, _, _, _ = _fresh_setup()
    third = other_benchmark.run(other_case_set, top_k=TOP_K, candidate_k=CANDIDATE_K)
    assert first == third
    assert first.to_dict() == third.to_dict()
    assert first.summary() == third.summary()


# --- F/H. Evidence metrics come from a fresh real run (never recomputed) ------

_TOP_LEVEL_CONSTANTS = {"corpus_documents": 6, "corpus_chunks": 33, "top_k": 33, "candidate_k": 33}


def test_evidence_metrics_equal_a_fresh_real_run_verbatim():
    benchmark, case_set, _, _, _ = _fresh_setup()
    fresh = benchmark.run(case_set, top_k=TOP_K, candidate_k=CANDIDATE_K)
    metrics = fresh.to_dict()
    evidence = _read_evidence()
    # Every metric/count in the evidence equals the fresh run's to_dict() -- the
    # test computes nothing; it reads the evaluator's own output both times.
    for field in RESULT_FIELDS:
        assert evidence[field] == metrics[field], field
    assert evidence["total_cases"] == evidence["evaluated_cases"] == 8


# --- I. Evidence schema: fixed, typed, leak-free ------------------------------


def test_evidence_schema_is_fixed_typed_and_leak_free():
    evidence = _read_evidence()
    assert isinstance(evidence, dict)
    assert list(evidence) == EVIDENCE_KEYS  # fixed key set AND fixed order
    assert evidence["deterministic"] is True
    assert evidence["dataset"] == DATASET_ID  # stable relative id, not a path
    assert {k: evidence[k] for k in _TOP_LEVEL_CONSTANTS} == _TOP_LEVEL_CONSTANTS
    for field in ("total_cases", "evaluated_cases", "corpus_documents", "corpus_chunks", "top_k", "candidate_k"):
        assert isinstance(evidence[field], int) and not isinstance(evidence[field], bool)
    for field in ("hit_rate", "recall", "precision", "mrr", "map", "ndcg"):
        assert isinstance(evidence[field], (int, float)) and not isinstance(evidence[field], bool)
        assert 0.0 <= evidence[field] <= 1.0
    # No timestamp / path / host / user / environment leakage in the artifact.
    raw = EVIDENCE_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL / remote reference


# --- J. The canonical runner reproduces the committed evidence ---------------


def test_runner_script_reproduces_evidence_to_a_fresh_output(tmp_path):
    out = tmp_path / "rerun.json"
    proc = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    rerun = json.loads(out.read_text(encoding="utf-8"))
    assert rerun == _read_evidence()  # full dict equality, key order included
