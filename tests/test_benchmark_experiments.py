"""Phase 10C controlled retrieval experiments tests.

Phase 10B already pinned the canonical corpus / dataset / frozen baseline. This
module pins what Phase 10C adds on top of that same real chain, without
re-implementing any metric and without touching production code:

- the committed experiment evidence is a valid JSON dict with a fixed top-level
  schema and exactly the locked E1..E6 matrix, in order;
- each configuration description matches the matrix locked in Step 1
  (retriever / fusion / reranker labels, top_k = candidate_k = 33);
- corpus (6 docs / 33 chunks) and dataset (8 cases) invariants hold;
- every metric in the evidence comes from a *fresh real re-run* of the named
  configuration through the runner's own public-component path and equals the
  committed row -- nothing is fake, nothing is recomputed by hand;
- E5 (the baseline configuration) recomputes the frozen Phase 10B baseline
  metrics exactly, and the baseline file is never overwritten (no E0 row);
- same-instance and cross-instance determinism hold on real components;
- single-source configurations (E1 / E2) are raw DenseRetriever /
  LexicalRetriever output -- no pipeline and no hidden reranker;
- the evidence schema is fixed, typed, leak-free (no timestamp / host / user /
  path / environment), and the runner source imports no pipeline and no
  third-party / random / time machinery;
- the canonical experiment runner (re-run as a subprocess to a fresh output
  path, so the committed evidence is never overwritten) reproduces the
  committed evidence exactly.

The committed evidence file is an *output* this module verifies; it is never
asserted against itself -- every comparison re-runs the real chain.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make benchmark/ (and its sibling baseline runner) importable under pytest.
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_retrieval_experiments as experiments  # noqa: E402

from zmwiki import (  # noqa: E402
    DenseRetriever,
    DeterministicEmbedder,
    EvaluationCaseSet,
    EvaluationResult,
    InMemoryVectorStore,
    JsonEvaluationDatasetLoader,
    LexicalRetriever,
    PlainTextFileIngestor,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = BENCH_DIR / "corpus"
DATASET_PATH = BENCH_DIR / "datasets" / "retrieval_small.json"
EVIDENCE_PATH = BENCH_DIR / "results" / "retrieval_small_experiments.json"
BASELINE_PATH = BENCH_DIR / "results" / "retrieval_small_baseline.json"
RUNNER_SCRIPT = BENCH_DIR / "run_retrieval_experiments.py"
DOC_NAMES = [
    "doc_a.txt",
    "doc_b.txt",
    "doc_c.txt",
    "doc_d.txt",
    "doc_e.txt",
    "doc_f.txt",
]

#: The metric/count keys EvaluationResult.to_dict emits (and both evidence share).
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

#: Evidence top-level keys, in their fixed order.
TOP_KEYS = [
    "experiment",
    "corpus_documents",
    "corpus_chunks",
    "top_k",
    "candidate_k",
    "total_cases",
    "evaluated_cases",
    "configs",
]

#: Each configuration row's fixed keys, in their fixed order.
ROW_KEYS = [
    "id",
    "retriever",
    "fusion",
    "reranker",
    "hit_rate",
    "recall",
    "precision",
    "mrr",
    "map",
    "ndcg",
    "total_cases",
    "evaluated_cases",
    "deterministic",
]

#: The matrix locked in Phase 10C Step 1.
MATRIX = {
    "E1": ("dense", None, None),
    "E2": ("lexical", None, None),
    "E3": ("hybrid", "weighted", None),
    "E4": ("hybrid", "rrf", None),
    "E5": ("hybrid", "weighted", "deterministic"),
    "E6": ("hybrid", "rrf", "deterministic"),
}


def _load_documents():
    """Read the six canonical corpus files through the real file ingestor."""
    ingestor = PlainTextFileIngestor()
    return [
        ingestor.ingest(CORPUS_DIR / name, document_id=Path(name).stem)
        for name in DOC_NAMES
    ]


def _load_case_set():
    return JsonEvaluationDatasetLoader().load(DATASET_PATH)


def _read_evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _read_baseline():
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _row_for(evidence, config_id):
    return next(row for row in evidence["configs"] if row["id"] == config_id)


# --- A. Evidence file: valid JSON, fixed schema, locked matrix -----------------


def test_experiments_evidence_is_valid_json_with_fixed_top_keys():
    evidence = _read_evidence()
    assert isinstance(evidence, dict)
    assert list(evidence) == TOP_KEYS  # fixed key set AND fixed order
    assert evidence["experiment"] == experiments.EXPERIMENT_ID
    assert evidence["top_k"] == evidence["candidate_k"] == 33


def test_matrix_has_exactly_e1_to_e6_in_order():
    evidence = _read_evidence()
    configs = evidence["configs"]
    assert isinstance(configs, list)
    assert [row["id"] for row in configs] == ["E1", "E2", "E3", "E4", "E5", "E6"]
    assert all(list(row) == ROW_KEYS for row in configs)  # fixed row key order


def test_config_descriptions_match_the_locked_matrix():
    evidence = _read_evidence()
    for row in evidence["configs"]:
        expected = MATRIX[row["id"]]
        assert (row["retriever"], row["fusion"], row["reranker"]) == expected, row["id"]


def test_fixed_environment_and_counts_in_evidence():
    evidence = _read_evidence()
    assert evidence["corpus_documents"] == 6
    assert evidence["corpus_chunks"] == 33
    assert evidence["total_cases"] == evidence["evaluated_cases"] == 8
    for row in evidence["configs"]:
        assert row["total_cases"] == row["evaluated_cases"] == 8


def test_corpus_and_dataset_invariants_hold():
    # Real corpus ingests to the canonical six documents.
    documents = _load_documents()
    assert len(documents) == 6
    chunker = experiments.FixedSizeChunker(
        chunk_size=experiments.CHUNK_SIZE, overlap=experiments.OVERLAP
    )
    total = sum(len(chunker.chunk(doc)) for doc in documents)
    assert total == 33
    # Real dataset loads to the canonical eight cases in file order.
    case_set = _load_case_set()
    assert isinstance(case_set, EvaluationCaseSet)
    assert len(case_set) == 8
    assert len(_read_evidence()["configs"]) == 6


# --- B. Metrics: typed, in range, deterministic flag ---------------------------


def test_all_metrics_are_numeric_in_range_and_counts_int():
    for row in _read_evidence()["configs"]:
        for field in ("total_cases", "evaluated_cases"):
            assert isinstance(row[field], int) and not isinstance(row[field], bool)
        for field in ("hit_rate", "recall", "precision", "mrr", "map", "ndcg"):
            assert isinstance(row[field], (int, float)) and not isinstance(row[field], bool)
            assert 0.0 <= row[field] <= 1.0
        assert row["deterministic"] is True


# --- C. Real per-configuration re-runs equal the committed evidence ------------


@pytest.mark.parametrize("config_id", ["E1", "E2", "E3", "E4", "E5", "E6"])
def test_config_reruns_real_components_and_matches_evidence(config_id):
    """A fresh build of the named config through real components == committed row.

    This is the "no fakes" guard: each configuration is rebuilt from the real
    corpus + public components and evaluated through the real harness/evaluator,
    and its metrics must equal the committed evidence verbatim.
    """
    spec = next(c for c in experiments.CONFIGS if c["id"] == config_id)
    case_set = _load_case_set()
    retrieval = experiments.RetrievalPath(spec)
    result = experiments._evaluate(retrieval, case_set)
    assert isinstance(result, EvaluationResult)
    row = _row_for(_read_evidence(), config_id)
    assert {field: result.to_dict()[field] for field in RESULT_FIELDS} == {
        field: row[field] for field in RESULT_FIELDS
    }, config_id


# --- D. E5 recompute guard vs the frozen Phase 10B baseline --------------------


def test_e5_recomputes_frozen_baseline_metrics_exactly():
    evidence = _read_evidence()
    baseline = _read_baseline()
    e5 = _row_for(evidence, "E5")
    for field in RESULT_FIELDS:
        assert e5[field] == baseline[field], field
    assert e5["deterministic"] is True
    # The E5 configuration description is the baseline configuration.
    assert (e5["retriever"], e5["fusion"], e5["reranker"]) == (
        "hybrid",
        "weighted",
        "deterministic",
    )


def test_no_e0_row_and_baseline_file_untouched():
    evidence = _read_evidence()
    baseline = _read_baseline()
    # The experiments evidence never fabricates an E0 / baseline duplicate row.
    assert [row["id"] for row in evidence["configs"]] == ["E1", "E2", "E3", "E4", "E5", "E6"]
    # The frozen baseline file is byte-preserved: it still carries its committed
    # metrics, still flagged deterministic, and its metric keys match E5.
    assert baseline["deterministic"] is True
    e5 = _row_for(evidence, "E5")
    assert all(baseline[field] == e5[field] for field in RESULT_FIELDS)


# --- E. Real determinism: same instance and cross instance ---------------------


def test_same_instance_determinism_on_real_components():
    spec = next(c for c in experiments.CONFIGS if c["id"] == "E3")
    case_set = _load_case_set()
    retrieval = experiments.RetrievalPath(spec)  # one fresh build
    first = experiments._evaluate(retrieval, case_set)
    second = experiments._evaluate(retrieval, case_set)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()


def test_cross_instance_determinism_on_real_components():
    spec = next(c for c in experiments.CONFIGS if c["id"] == "E4")
    first = experiments._evaluate(experiments.RetrievalPath(spec), _load_case_set())
    third = experiments._evaluate(experiments.RetrievalPath(spec), _load_case_set())
    assert first.to_dict() == third.to_dict()
    assert first.summary() == third.summary()


# --- F. Single-source configs are raw retriever output (no pipeline/reranker) --


def test_e1_is_raw_dense_retriever_output():
    spec = next(c for c in experiments.CONFIGS if c["id"] == "E1")
    retrieval = experiments.RetrievalPath(spec)
    case = _load_case_set().cases[0]
    got = retrieval.run_case(case.query)
    # An independent raw DenseRetriever over the *same* freshly built store must
    # return exactly the same ordered results -- proving E1 skips the pipeline
    # and any reranker.
    embedder = DeterministicEmbedder()  # same default dim 8 -> same query vector
    raw = DenseRetriever(embedder, retrieval.vector_store)
    expected = raw.retrieve(case.query, top_k=experiments.TOP_K)
    assert [(r.chunk.id, r.score) for r in got] == [
        (r.chunk.id, r.score) for r in expected
    ]


def test_e2_is_raw_lexical_retriever_output():
    spec = next(c for c in experiments.CONFIGS if c["id"] == "E2")
    retrieval = experiments.RetrievalPath(spec)
    case = _load_case_set().cases[0]
    got = retrieval.run_case(case.query)
    raw = LexicalRetriever(retrieval.lexical_index)
    expected = raw.retrieve(case.query, top_k=experiments.TOP_K)
    assert [(r.chunk.id, r.score) for r in got] == [
        (r.chunk.id, r.score) for r in expected
    ]
    # Lexical-only may legitimately return fewer than the corpus size (BM25 only
    # surfaces chunks sharing a query term) -- real results are never padded.
    assert len(got) <= experiments.TOP_K


# --- G. Reproducible from the runner script ------------------------------------


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


# --- H. No environment leakage, no pipeline / third-party in the runner --------


def test_evidence_has_no_environment_leakage():
    raw = EVIDENCE_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid", "seed"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL / remote reference
    assert "git" not in raw.lower()  # no repository metadata


def test_runner_source_has_no_pipeline_random_time_or_third_party():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    # The direct-composition path never *imports* the pipeline. (The docstring
    # may mention the class to explain why it is not used -- import lines only.)
    import_lines = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert not any(
        "RAGPipeline" in ln for ln in import_lines
    ), import_lines
    # No nondeterminism / network / heavyweight third-party imports. Scanned on
    # import lines so local paths like ``benchmark/datasets/`` are not flagged.
    banned = (
        "random", "uuid", "datetime", "time.", "hash(", "requests", "urllib",
        "socket", "http", "numpy", "scipy", "sklearn", "torch", "transformers",
        "datasets", "evaluate",
    )
    assert not any(any(token in ln for token in banned) for ln in import_lines), import_lines
    # The runner module itself exposes no pipeline attribute.
    assert not hasattr(experiments, "DeterministicRAGPipeline")


# --- I. InMemoryVectorStore import sanity (stores used are real, not fakes) ----


def test_real_store_types_are_used():
    spec = next(c for c in experiments.CONFIGS if c["id"] == "E1")
    retrieval = experiments.RetrievalPath(spec)
    assert isinstance(retrieval.vector_store, InMemoryVectorStore)
    assert isinstance(retrieval.lexical_index, experiments.InMemoryLexicalIndex)
