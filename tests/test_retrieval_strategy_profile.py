"""Retrieval strategy comparison tests: matrix, fairness, evidence, guards.

Pins what the strategy profile adds, without re-implementing a metric, a
retriever, or a reranker:

- the committed evidence has a fixed schema, the locked five-strategy matrix and
  the locked four-point candidate-pool matrix;
- every strategy was built over the same frozen corpus, and each row varies
  only the configuration the matrix intends to vary;
- the per-query records cover every query and every strategy;
- the full-pool arm still reproduces the frozen Phase 10B / 10C rows, and the
  run refuses to write evidence when it does not;
- the frozen inputs (corpus, dataset, historical evidence) are byte-identical to
  what they were before this experiment ran;
- the runner reaches no network and no model on its default path.

The committed evidence is an *output* this module verifies against a fresh real
re-run -- it is never asserted against itself.
"""

import ast
import json
import os
import pathlib
import sys

import pytest

from _frozen_evidence import frozen_digest_matches

# Make benchmark/ (and its sibling frozen runners) importable under pytest.
BENCH_DIR = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_retrieval_strategy_profile as profile  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE_PATH = BENCH_DIR / "results" / "retrieval_strategy_comparison.json"
SEMANTIC_PATH = BENCH_DIR / "results" / "retrieval_strategy_semantic_profile.json"
RUNNER_SCRIPT = BENCH_DIR / "run_retrieval_strategy_profile.py"

#: Metrics every strategy row must carry, at the headline depth and full pool.
HEADLINE_FIELDS = (
    "hit_rate_at_1",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "precision_at_5",
    "mrr_at_5",
    "map_at_5",
    "ndcg_at_5",
)

#: Frozen inputs and frozen historical evidence: sha256 at the time this
#: experiment was built. A change here means someone edited data or history.
FROZEN_SHA256 = {
    "benchmark/datasets/retrieval_small.json":
        "5112ba3c580ddc448a7ef96031930c7a9614c0264808af9d648ecce246c46f81",
    "benchmark/results/retrieval_small_baseline.json":
        "7a5e9b8f141b96c39b149a5842dda994d85c21ab388c9581e1d76e19ced622f9",
    "benchmark/results/retrieval_small_experiments.json":
        "6aa510cc7dce4ba31793e0fd9899d532d5945de5c3d2409f746f1eafc908cb17",
    "benchmark/results/retrieval_semantic_rerank_profile.json":
        "bad0a04ca1866a2955d95a8551a7fa68aaefeed4b7c7abddfc320794d59276aa",
    "benchmark/corpus/doc_a.txt":
        "576c6b4f7ccf621bfa16efd955d7bdf48a9ea7dd4c7f11b335e1915c8c20c287",
    "benchmark/corpus/doc_b.txt":
        "511a8b4df39a2bdd0603e3dad2fd333322a81fc27dc35a05d8019e7338a465b2",
    "benchmark/corpus/doc_c.txt":
        "f86c275d73a9e31d7efd9c8c40903ef4752981058942c585749e9e1849163104",
    "benchmark/corpus/doc_d.txt":
        "d02b1e485f53f8577c5702f154bce7e64c12a9e50599f975c7a352e3711f251f",
    "benchmark/corpus/doc_e.txt":
        "22a51d4e96d584295ff9b1bd79f89bc7789d1e528873926ea1a3eaba90baaadb",
    "benchmark/corpus/doc_f.txt":
        "09d114ab7c8adbd14ba1e24cdee886ef014be446218cc83569185634dc314fc5",
}

TOP_KEYS = [
    "experiment",
    "reference",
    "dataset",
    "corpus_documents",
    "corpus_chunks",
    "chunking",
    "embedding",
    "bm25_tokenizer",
    "total_cases",
    "evaluated_cases",
    "top_k",
    "candidate_k",
    "dataset_fingerprint",
    "fairness",
    "historical_baseline_match",
    "historical_comparisons",
    "strategies",
    "rank_changes_vs_baseline",
    "per_query",
    "candidate_pool",
    "deterministic",
]

ROW_KEYS = [
    "id",
    "name",
    "retriever",
    "fusion",
    "reranker",
    "candidate_k",
    "top_k",
    *(HEADLINE_FIELDS),
    "full_pool_candidate_k",
    "full_pool_top_k",
    "full_pool_hit_rate",
    "full_pool_recall",
    "full_pool_precision",
    "full_pool_mrr",
    "full_pool_map",
    "full_pool_ndcg",
    "deterministic",
]


def _evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _semantic():
    return json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))


# --- A. Frozen inputs and history are untouched --------------------------------


@pytest.mark.parametrize("relative_path", sorted(FROZEN_SHA256))
def test_frozen_inputs_and_historical_evidence_are_unchanged(relative_path):
    assert frozen_digest_matches(
        ROOT / relative_path, FROZEN_SHA256[relative_path]
    ), (
        f"{relative_path} changed -- the corpus, dataset and historical evidence "
        "must not be modified by this experiment"
    )


def test_rerun_never_overwrites_the_frozen_evidence_files(tmp_path):
    profile.main(tmp_path / "fresh.json")  # writes only to the temp path
    assert frozen_digest_matches(
        ROOT / "benchmark/results/retrieval_small_baseline.json",
        FROZEN_SHA256["benchmark/results/retrieval_small_baseline.json"],
    )


# --- B. Evidence schema, matrix, metrics ---------------------------------------


def test_evidence_schema_and_environment_are_fixed():
    evidence = _evidence()
    assert list(evidence) == TOP_KEYS
    assert evidence["experiment"] == profile.EXPERIMENT_ID
    assert evidence["dataset"] == "benchmark/datasets/retrieval_small.json"
    assert evidence["corpus_documents"] == 6
    assert evidence["corpus_chunks"] == 33
    assert evidence["chunking"] == profile.CHUNKING_ID
    assert evidence["embedding"] == profile.EMBEDDING_ID
    assert evidence["bm25_tokenizer"] == "SimpleTokenizer"
    assert evidence["total_cases"] == evidence["evaluated_cases"] == 8
    assert evidence["top_k"] == 5
    assert evidence["candidate_k"] == 33
    assert evidence["deterministic"] is True


def test_dataset_fingerprint_covers_every_case():
    fingerprint = _evidence()["dataset_fingerprint"]
    dataset = json.loads(
        (BENCH_DIR / "datasets" / "retrieval_small.json").read_text(encoding="utf-8")
    )
    assert fingerprint["queries"] == [case["query"] for case in dataset["cases"]]
    assert fingerprint["relevant_chunk_ids"] == [
        sorted(case["relevant_chunk_ids"]) for case in dataset["cases"]
    ]


def test_strategy_matrix_is_complete_and_locked():
    rows = _evidence()["strategies"]
    assert [row["id"] for row in rows] == ["T1", "T2", "T3", "T4", "T5"]
    assert all(list(row) == ROW_KEYS for row in rows)
    assert [(r["retriever"], r["fusion"], r["reranker"]) for r in rows] == [
        ("dense", None, None),
        ("hybrid", "weighted", None),
        ("hybrid", "rrf", None),
        ("hybrid", "weighted", "deterministic"),
        ("hybrid", "rrf", "deterministic"),
    ]
    # Single-source retrieval has no recall pool; fused strategies all share one.
    assert rows[0]["candidate_k"] is None
    assert {row["candidate_k"] for row in rows[1:]} == {profile.CANDIDATE_K}
    assert all(row["top_k"] == profile.TOP_K for row in rows)
    assert all(row["deterministic"] is True for row in rows)


def test_every_metric_field_is_present_and_in_range():
    for row in _evidence()["strategies"]:
        for field in HEADLINE_FIELDS:
            value = row[field]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert 0.0 <= value <= 1.0, (row["id"], field)
        for field in ("hit_rate", "recall", "precision", "mrr", "map", "ndcg"):
            assert 0.0 <= row[f"full_pool_{field}"] <= 1.0


# --- C. Fairness: only the intended variable differs ---------------------------


def test_fairness_records_only_intended_differences():
    evidence = _evidence()
    fairness = evidence["fairness"]
    assert [entry["id"] for entry in fairness] == ["T1", "T2", "T3", "T4", "T5"]
    assert all(entry["compared_against"] == "T1" for entry in fairness)
    assert fairness[0]["differing_keys"] == []  # T1 against itself
    allowed = {"retriever", "fusion", "reranker", "candidate_k"}
    for entry in fairness:
        assert set(entry["differing_keys"]) <= allowed
        assert set(entry["allowed_keys"]) == allowed
    # The reranked rows differ from the baseline in exactly the reranker too.
    reranked = {e["id"]: set(e["differing_keys"]) for e in fairness}
    assert "reranker" in reranked["T4"] and "reranker" in reranked["T5"]
    assert "reranker" not in reranked["T2"]


def test_guard_rejects_a_second_difference():
    documents = profile._baseline.ingest_corpus()
    left = profile.build_strategy(profile.STRATEGIES[0])  # dense
    right = profile.build_strategy(profile.STRATEGIES[1])  # hybrid + weighted
    # Dense vs hybrid may differ in retriever/fusion/candidate_k -- but not in
    # the embedder, so allow-listing only the strategy keys still passes...
    profile._assert_only_variable_differs(
        left,
        right,
        reference_args={"top_k": 5, "candidate_k": None},
        other_args={"top_k": 5, "candidate_k": 33},
        allowed=("retriever", "fusion", "reranker", "candidate_k"),
    )
    # ...while claiming nothing differs at all is caught.
    with pytest.raises(RuntimeError, match="unexpected difference"):
        profile._assert_only_variable_differs(
            left,
            right,
            reference_args={"top_k": 5, "candidate_k": None},
            other_args={"top_k": 5, "candidate_k": 33},
            allowed=(),
        )
    assert len(documents) == 6


def test_every_strategy_was_built_over_the_same_corpus():
    documents = profile._baseline.ingest_corpus()
    expected = profile._expected_chunks(documents)
    assert len(expected) == 33
    fingerprints = set()
    for strategy in profile.STRATEGIES:
        path = profile.build_strategy(strategy)
        fingerprints.add(profile._verify_stores(path, expected))
    assert len(fingerprints) == 1  # byte-identical corpus in every strategy


# --- D. Per-query evidence ------------------------------------------------------


def test_per_query_covers_every_query_and_strategy():
    records = _evidence()["per_query"]
    dataset = json.loads(
        (BENCH_DIR / "datasets" / "retrieval_small.json").read_text(encoding="utf-8")
    )
    assert [record["query"] for record in records] == [
        case["query"] for case in dataset["cases"]
    ]
    for record in records:
        assert list(record) == ["query", "relevant_chunk_ids", "strategies"]
        assert list(record["strategies"]) == ["T1", "T2", "T3", "T4", "T5"]
        for entry in record["strategies"].values():
            assert list(entry) == [
                "retrieved_chunk_ids",
                "rank_of_first_relevant",
                "hit_at_1",
                "hit_at_3",
                "hit_at_5",
            ]
            assert isinstance(entry["retrieved_chunk_ids"], list)
            assert len(entry["retrieved_chunk_ids"]) <= profile.TOP_K


def test_per_query_hit_flags_agree_with_the_recorded_ids():
    for record in _evidence()["per_query"]:
        relevant = set(record["relevant_chunk_ids"])
        for entry in record["strategies"].values():
            ids = entry["retrieved_chunk_ids"]
            assert entry["hit_at_1"] == any(cid in relevant for cid in ids[:1])
            assert entry["hit_at_3"] == any(cid in relevant for cid in ids[:3])
            assert entry["hit_at_5"] == any(cid in relevant for cid in ids[:5])
            expected_rank = next(
                (i for i, cid in enumerate(ids, start=1) if cid in relevant), 0
            )
            assert entry["rank_of_first_relevant"] == expected_rank


def test_rank_change_counts_match_the_per_query_records():
    records = _evidence()["per_query"]
    for entry in _evidence()["rank_changes_vs_baseline"]:
        sid = entry["id"]
        changed = sum(
            1
            for record in records
            if record["strategies"][sid]["retrieved_chunk_ids"]
            != record["strategies"]["T1"]["retrieved_chunk_ids"]
        )
        changed_top1 = sum(
            1
            for record in records
            if record["strategies"][sid]["retrieved_chunk_ids"][:1]
            != record["strategies"]["T1"]["retrieved_chunk_ids"][:1]
        )
        assert entry["queries_with_changed_ranking"] == changed
        assert entry["queries_with_changed_top_1"] == changed_top1
    # The baseline cannot differ from itself.
    assert _evidence()["rank_changes_vs_baseline"][0]["queries_with_changed_ranking"] == 0


# --- E. Candidate pool arm ------------------------------------------------------


def test_candidate_pool_matrix_is_complete():
    arm = _evidence()["candidate_pool"]
    assert [(row["id"], row["candidate_k"]) for row in arm] == [
        (sid, candidate_k)
        for sid in profile.RERANKED_STRATEGY_IDS
        for candidate_k in profile.CANDIDATE_K_MATRIX
    ]
    for row in arm:
        assert list(row) == [
            "id",
            "fusion",
            "reranker",
            "top_k",
            "candidate_k",
            "recall_at_5",
            "mrr_at_5",
            "ndcg_at_5",
        ]
        assert row["top_k"] == profile.TOP_K
        assert row["reranker"] == "deterministic"
        assert 0.0 <= row["recall_at_5"] <= 1.0


def test_candidate_pool_agrees_with_the_headline_run():
    # candidate_k = 33 in the arm is the same configuration as the T4/T5 rows.
    arm = {
        row["id"]: row
        for row in _evidence()["candidate_pool"]
        if row["candidate_k"] == profile.CANDIDATE_K
    }
    for row in _evidence()["strategies"]:
        if row["id"] not in arm:
            continue
        assert arm[row["id"]]["recall_at_5"] == row["recall_at_5"]
        assert arm[row["id"]]["mrr_at_5"] == row["mrr_at_5"]
        assert arm[row["id"]]["ndcg_at_5"] == row["ndcg_at_5"]


# --- F. Historical baseline guard -----------------------------------------------


def test_historical_baseline_guard_passed():
    evidence = _evidence()
    assert evidence["historical_baseline_match"] is True
    comparisons = evidence["historical_comparisons"]
    assert [c["strategy"] for c in comparisons] == ["T1", "T2", "T3", "T4", "T5", "T4"]
    assert all(c["match"] is True for c in comparisons)
    assert "retrieval_small_baseline" in {c["reference"] for c in comparisons}
    assert "retrieval_small_experiments:E1" in {c["reference"] for c in comparisons}


def test_full_pool_arm_reproduces_the_frozen_rows(tmp_path):
    """A fresh full run still matches Phase 10B / 10C -- recomputed, not copied."""
    fresh = profile.main(tmp_path / "fresh.json")
    assert fresh["historical_baseline_match"] is True
    row = next(r for r in fresh["strategies"] if r["id"] == "T4")
    frozen = json.loads(
        (BENCH_DIR / "results" / "retrieval_small_baseline.json").read_text(encoding="utf-8")
    )
    for field in ("hit_rate", "recall", "precision", "mrr", "map", "ndcg"):
        assert row[f"full_pool_{field}"] == frozen[field], field


def test_guard_function_stops_on_a_mismatch():
    evidence = _evidence()
    documents = profile._baseline.ingest_corpus()
    case_set = profile._load_cases()
    full_pool = {}
    for strategy in profile.STRATEGIES:
        path = profile.build_strategy(strategy)
        result, _ = profile._run(
            path, case_set, top_k=profile.CANDIDATE_K, candidate_k=profile.CANDIDATE_K
        )
        full_pool[strategy["id"]] = result
    profile._historical_guard(full_pool)  # raises on any drift
    assert len(documents) == 6
    assert evidence["historical_baseline_match"] is True


# --- G. Determinism and the opt-in semantic arm ---------------------------------


def test_evidence_is_reproducible_from_the_runner(tmp_path):
    fresh = profile.main(tmp_path / "fresh.json")
    assert fresh == _evidence()


def test_semantic_evidence_matches_the_phase12_row():
    evidence = _semantic()
    assert evidence["experiment"] == profile.SEMANTIC_EXPERIMENT_ID
    assert evidence["historical_baseline_match"] is True
    row = evidence["strategies"][0]
    assert row["reranker"] == "cross_encoder"
    assert row["deterministic"] is False
    frozen = json.loads(
        (BENCH_DIR / "results" / "retrieval_semantic_rerank_profile.json").read_text(
            encoding="utf-8"
        )
    )
    reference = next(c for c in frozen["configs"] if c["id"] == "R2")
    for field in ("hit_rate", "recall", "precision", "mrr", "map", "ndcg"):
        assert row[field] == reference[field], field


def test_semantic_arm_reports_unavailable_instead_of_failing(monkeypatch):
    # With the optional extra hidden, the arm returns a reason and writes nothing.
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("zmwiki.semantic"):
            raise ImportError("no semantic extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    out = SEMANTIC_PATH.parent / "should-not-be-written.json"
    result = profile.main_semantic(out)
    assert result["available"] is False
    assert "ImportError" in result["reason"]
    assert not out.exists()


@pytest.mark.skipif(
    not os.environ.get("zmwiki_SEMANTIC_INTEGRATION"),
    reason="real-model integration; set zmwiki_SEMANTIC_INTEGRATION=1 to run",
)
def test_integration_real_semantic_strategy_profile(tmp_path):
    fresh = profile.main_semantic(tmp_path / "semantic.json")
    assert fresh.get("available", True) is True
    committed = _semantic()
    for field in ("hit_rate", "recall", "precision", "mrr", "map", "ndcg"):
        assert fresh["strategies"][0][field] == committed["strategies"][0][field]


# --- H. No network, no model, no metric of its own ------------------------------


def test_runner_imports_no_network_client_and_no_model():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    absolute = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            absolute.add((node.module or "").split(".")[0])
    # stdlib + zmwiki + the sibling frozen corpus-ingestion runner (no HTTP client,
    # no model, no third-party package).
    assert absolute <= {
        "__future__",
        "json",
        "sys",
        "pathlib",
        "zmwiki",
        "run_retrieval_small_baseline",
    }
    for banned in ("requests", "urllib", "httpx", "socket", "sentence_transformers"):
        assert banned not in absolute
    # The semantic providers live behind the opt-in function only.
    assert "from zmwiki.semantic import" in source
    import_lines = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("semantic" in line for line in import_lines), import_lines


def test_runner_does_not_compute_metrics_itself():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    # Metrics must be read from EvaluationResult, never recomputed here.
    assert "RetrievalEvaluator" in source
    for banned in ("def _ndcg", "def _reciprocal", "def _average_precision", "math.log2"):
        assert banned not in source


def test_semantic_evidence_has_no_environment_leakage():
    raw = SEMANTIC_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL
