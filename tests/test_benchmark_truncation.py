"""Phase 10D controlled truncation experiments tests.

Phase 10C pinned the canonical corpus / dataset / frozen baseline + full-pool
experiments at top_k = candidate_k = 33 (where hit_rate / recall saturate).
Phase 10D measures the same real chain under candidate_k / top_k cutoffs. This
module pins what Phase 10D adds, without re-implementing any metric and without
touching production code:

- the committed truncation evidence is a valid JSON dict with a fixed top-level
  schema and exactly the locked 18-cell matrix, in fixed order (D1 = 2,
  D2 = 2, D3 = 7, D4 = 7);
- configuration descriptions are correct: D1 dense / D2 lexical / D3 hybrid +
  weighted / D4 hybrid + rrf, reranker always null, D1/D2 candidate_k null,
  D3/D4 candidate_k in {4, 8, 16, 33} with top_k in {4, 8} and top_k <=
  candidate_k;
- every metric comes from a *fresh real re-run* of the named cell through the
  runner's own public-component path and equals the committed row -- nothing is
  fake, nothing is recomputed by hand;
- same-instance and cross-instance determinism hold on real components;
- the E3 cross-phase guard recomputes the frozen Phase 10C E3 row (Hybrid +
  Weighted @ 33/33) exactly, and the Phase 10C evidence file is never
  overwritten;
- single-source cells are raw DenseRetriever / LexicalRetriever output -- no
  pipeline and no hidden reranker;
- the evidence schema is fixed, typed, leak-free (no timestamp / host / user /
  path / environment), and the runner source imports no pipeline, no reranker
  and no third-party / random / time machinery;
- the canonical runner (re-run as a subprocess to a fresh output path, so the
  committed evidence is never overwritten) reproduces the committed evidence
  exactly.

The committed evidence file is an *output* this module verifies; it is never
asserted against itself -- every comparison re-runs the real chain.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

# Make benchmark/ (and its sibling baseline runner) importable under pytest.
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_retrieval_truncation as truncation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = BENCH_DIR / "corpus"
DATASET_PATH = BENCH_DIR / "datasets" / "retrieval_small.json"
EVIDENCE_PATH = BENCH_DIR / "results" / "retrieval_truncation.json"
EXPERIMENTS_PATH = BENCH_DIR / "results" / "retrieval_small_experiments.json"
RUNNER_SCRIPT = BENCH_DIR / "run_retrieval_truncation.py"

#: The six metric keys each evidence row carries (counts live at the top level).
METRIC_FIELDS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")

#: The metric/count keys EvaluationResult.to_dict emits (shared by evidence).
RESULT_FIELDS = truncation.RESULT_FIELDS

#: Evidence top-level keys, in their fixed order.
TOP_KEYS = [
    "experiment",
    "reference",
    "corpus_documents",
    "corpus_chunks",
    "total_cases",
    "evaluated_cases",
    "results",
]

#: Each result row's fixed keys, in their fixed order.
ROW_KEYS = [
    "id",
    "retriever",
    "fusion",
    "reranker",
    "candidate_k",
    "top_k",
    "hit_rate",
    "recall",
    "precision",
    "mrr",
    "map",
    "ndcg",
    "deterministic",
]

#: Locked matrix structure: config label -> (retriever, fusion) -> row count.
CONFIG_COUNTS = {
    ("D1", "dense", None): 2,
    ("D2", "lexical", None): 2,
    ("D3", "hybrid", "weighted"): 7,
    ("D4", "hybrid", "rrf"): 7,
}

#: Representative cells re-run on real components (at least the locked six).
REPRESENTATIVE_CELLS = [
    "D1-t4",
    "D2-t4",
    "D3-c4-t4",
    "D3-c33-t4",
    "D4-c4-t4",
    "D4-c33-t8",
]

CANDIDATE_K_SET = {4, 8, 16, 33}
TOP_K_SET = {4, 8}


def _read_evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _read_experiments():
    return json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))


def _load_case_set():
    return truncation.JsonEvaluationDatasetLoader().load(DATASET_PATH)


def _spec_for(cell_id):
    return next(spec for spec in truncation.ROWS if spec["id"] == cell_id)


def _config_label(row):
    return row["id"].split("-")[0]


def _rerun_cell(spec):
    """A fresh build of the named cell through real components == EvaluationResult."""
    retrieval = truncation.RetrievalPath(spec["retriever"], spec["fusion"])
    return truncation._evaluate(
        retrieval, _load_case_set(), spec["top_k"], spec["candidate_k"]
    )


# --- A. Evidence file: valid JSON, fixed schema, locked matrix -----------------


def test_evidence_is_valid_json_with_fixed_top_keys():
    evidence = _read_evidence()
    assert isinstance(evidence, dict)
    assert list(evidence) == TOP_KEYS  # fixed key set AND fixed order
    assert evidence["experiment"] == truncation.EXPERIMENT_ID == "retrieval_truncation"
    assert evidence["reference"] == truncation.REFERENCE_EXPERIMENT
    assert evidence["corpus_documents"] == 6
    assert evidence["corpus_chunks"] == 33
    assert evidence["total_cases"] == evidence["evaluated_cases"] == 8


def test_matrix_has_18_rows_in_fixed_order():
    evidence = _read_evidence()
    results = evidence["results"]
    assert isinstance(results, list)
    assert len(results) == 18
    assert [row["id"] for row in results] == [spec["id"] for spec in truncation.ROWS]
    assert all(list(row) == ROW_KEYS for row in results)  # fixed row key order


def test_config_counts_match_the_locked_matrix():
    evidence = _read_evidence()
    rows = evidence["results"]
    # Config identity from structure (retriever / fusion), not from the id.
    counts = Counter((_config_label(r), r["retriever"], r["fusion"]) for r in rows)
    assert counts == Counter(CONFIG_COUNTS)
    # The id prefix must also agree with the structural config label.
    expected_label = {
        ("dense", None): "D1",
        ("lexical", None): "D2",
        ("hybrid", "weighted"): "D3",
        ("hybrid", "rrf"): "D4",
    }
    for row in rows:
        assert _config_label(row) == expected_label[(row["retriever"], row["fusion"])]


def test_config_descriptions_are_correct():
    for row in _read_evidence()["results"]:
        assert row["reranker"] is None  # Phase 10D never reranks.
        if _config_label(row) in ("D1", "D2"):
            assert row["fusion"] is None
            assert row["candidate_k"] is None  # single-source has no candidate_k
            assert row["top_k"] in TOP_K_SET
        else:  # D3 / D4 hybrid
            assert row["fusion"] in ("weighted", "rrf")
            assert row["candidate_k"] in CANDIDATE_K_SET
            assert row["top_k"] in TOP_K_SET
            assert row["candidate_k"] >= row["top_k"]


def test_no_full_pool_33_33_rows_are_regenerated():
    # The full-pool control (33, 33) is Phase 10C's E1/E2/E3/E4 -- referenced,
    # not duplicated as Phase 10D rows.
    rows = _read_evidence()["results"]
    assert all(not (row["candidate_k"] == 33 and row["top_k"] == 33) for row in rows)


def test_corpus_and_dataset_invariants_hold():
    # Real corpus ingests to the canonical six documents / 33 chunks.
    documents = truncation._baseline.ingest_corpus()
    truncation._baseline.verify_corpus(documents)
    # Real dataset loads to the canonical eight cases.
    assert len(_load_case_set()) == truncation.EXPECTED_CASES == 8


# --- B. Metrics: typed, in range, deterministic flag ---------------------------


def test_all_metrics_are_numeric_in_range_and_deterministic():
    for row in _read_evidence()["results"]:
        for field in METRIC_FIELDS:
            assert isinstance(row[field], (int, float)) and not isinstance(
                row[field], bool
            )
            assert 0.0 <= row[field] <= 1.0
        assert row["deterministic"] is True


# --- C. Real per-cell re-runs equal the committed evidence --------------------


@pytest.mark.parametrize("cell_id", REPRESENTATIVE_CELLS)
def test_cell_reruns_real_components_and_matches_evidence(cell_id):
    """A fresh build of the named cell through real components == committed row."""
    spec = _spec_for(cell_id)
    result = _rerun_cell(spec)
    row = next(r for r in _read_evidence()["results"] if r["id"] == cell_id)
    got = result.to_dict()
    assert {f: got[f] for f in METRIC_FIELDS} == {f: row[f] for f in METRIC_FIELDS}
    assert got["total_cases"] == got["evaluated_cases"] == 8


# --- D. Real determinism: same instance and cross instance --------------------


def test_same_instance_determinism_on_real_components():
    spec = _spec_for("D3-c8-t4")
    retrieval = truncation.RetrievalPath(spec["retriever"], spec["fusion"])  # one build
    first = truncation._evaluate(
        retrieval, _load_case_set(), spec["top_k"], spec["candidate_k"]
    )
    second = truncation._evaluate(
        retrieval, _load_case_set(), spec["top_k"], spec["candidate_k"]
    )
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.summary() == second.summary()


def test_cross_instance_determinism_on_real_components():
    spec = _spec_for("D4-c16-t8")
    first = truncation._evaluate(
        truncation.RetrievalPath(spec["retriever"], spec["fusion"]),
        _load_case_set(),
        spec["top_k"],
        spec["candidate_k"],
    )
    # Fully fresh build: fresh corpus ingestion, stores, retrievers, fusion,
    # dataset loader -- everything is reconstructed inside RetrievalPath.
    third = truncation._evaluate(
        truncation.RetrievalPath(spec["retriever"], spec["fusion"]),
        _load_case_set(),
        spec["top_k"],
        spec["candidate_k"],
    )
    assert first.to_dict() == third.to_dict()
    assert first.summary() == third.summary()


# --- E. E3 cross-phase guard vs the frozen Phase 10C evidence -----------------


def test_e3_cross_phase_recompute_matches_frozen_phase10c_row():
    frozen = _read_experiments()
    e3 = next(row for row in frozen["configs"] if row["id"] == "E3")
    assert (e3["retriever"], e3["fusion"], e3["reranker"]) == (
        "hybrid",
        "weighted",
        None,
    )
    # Recomputed through the Phase 10D runner's own direct-composition path.
    result = truncation._evaluate(
        truncation.RetrievalPath("hybrid", "weighted"),
        _load_case_set(),
        candidate_k=33,
        top_k=33,
    )
    got = result.to_dict()
    for field in RESULT_FIELDS:
        assert got[field] == e3[field], field


def test_e3_guard_function_passes_on_recomputed_result():
    result = truncation._evaluate(
        truncation.RetrievalPath("hybrid", "weighted"),
        _load_case_set(),
        candidate_k=33,
        top_k=33,
    )
    truncation._check_e3_against_experiments(result)  # raises RuntimeError on mismatch


# --- F. Single-source cells are raw retriever output (no pipeline/reranker) ----


def test_d1_is_raw_dense_retriever_output():
    spec = _spec_for("D1-t4")
    retrieval = truncation.RetrievalPath(spec["retriever"], spec["fusion"])
    case = _load_case_set().cases[0]
    got = retrieval.retrieve(case.query, top_k=spec["top_k"], candidate_k=None)
    # An independent raw DenseRetriever over the *same* freshly built store must
    # return exactly the same ordered results -- proving D1 skips the pipeline
    # and any reranker.
    raw = retrieval._dense
    expected = raw.retrieve(case.query, top_k=spec["top_k"])
    assert [(r.chunk.id, r.score) for r in got] == [
        (r.chunk.id, r.score) for r in expected
    ]


def test_d2_is_raw_lexical_retriever_output():
    spec = _spec_for("D2-t4")
    retrieval = truncation.RetrievalPath(spec["retriever"], spec["fusion"])
    case = _load_case_set().cases[0]
    got = retrieval.retrieve(case.query, top_k=spec["top_k"], candidate_k=None)
    raw = retrieval._lexical
    expected = raw.retrieve(case.query, top_k=spec["top_k"])
    assert [(r.chunk.id, r.score) for r in got] == [
        (r.chunk.id, r.score) for r in expected
    ]
    # Lexical-only may return fewer than top_k when few chunks match -- never
    # padded to fill the requested depth.
    assert len(got) <= spec["top_k"]


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


# --- H. No environment leakage, no pipeline/reranker/third-party imports --------


def test_evidence_has_no_environment_leakage():
    raw = EVIDENCE_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid", "seed"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL / remote reference
    assert "git" not in raw.lower()  # no repository metadata


def test_runner_source_has_no_pipeline_reranker_random_time_or_third_party():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    # Import lines only: prose may explain why the pipeline is not used, and
    # local path strings (e.g. ``.../retrieval_small_experiments.json``) must
    # not be misflagged.
    import_lines = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert not any(
        ("RAGPipeline" in ln or "Reranker" in ln) for ln in import_lines
    ), import_lines
    banned = (
        "random", "uuid", "datetime", "time.", "hash(", "requests", "urllib",
        "httpx", "socket", "subprocess", "numpy", "scipy", "sklearn", "torch",
        "transformers", "datasets", "evaluate",
    )
    assert not any(any(token in ln for token in banned) for ln in import_lines), import_lines
    # The runner module itself exposes no pipeline and no reranker attribute.
    assert not hasattr(truncation, "DeterministicRAGPipeline")
    assert not hasattr(truncation, "DeterministicReranker")


# --- I. Real store types are used (stores are real, not fakes) -----------------


def test_real_store_types_are_used():
    retrieval = truncation.RetrievalPath("dense", None)
    assert isinstance(retrieval.vector_store, truncation.InMemoryVectorStore)
    assert isinstance(retrieval.lexical_index, truncation.InMemoryLexicalIndex)
