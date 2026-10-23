"""Baseline vs DeterministicQueryRewriter benchmark tests.

Pins what the query-rewriter experiment adds, without re-implementing any
metric and without touching production code:

- the committed evidence has a fixed schema and exactly the locked matrix
  (2 deterministic scenarios x 2 variants x 4 depths = 16 rows);
- both variants carry byte-equal metrics at every depth -- the experiment's
  whole claim;
- the per-query records show an identical retrieval query and an identical
  33-deep ranking for every case;
- the mechanism probe reports what it measured: the rewriter fires on every
  de-normalized probe, changes no pipeline and no lexical ranking, and does
  move the dense ordering of the hash-based stand-in embedder;
- the runner reproduces the committed evidence from a fresh output path;
- the opt-in semantic evidence is pinned too, and its real re-run is skipped by
  default (``zmwiki_SEMANTIC_INTEGRATION=1`` to enable), matching Phase 11/12.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Make benchmark/ (and its sibling baseline runner) importable under pytest.
BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH_DIR))

import run_query_rewriter_profile as qr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = BENCH_DIR / "results" / "retrieval_query_rewriter_profile.json"
SEMANTIC_PATH = BENCH_DIR / "results" / "retrieval_query_rewriter_semantic_profile.json"
RUNNER_SCRIPT = BENCH_DIR / "run_query_rewriter_profile.py"

METRIC_FIELDS = ("hit_rate", "recall", "precision", "mrr", "map", "ndcg")

TOP_KEYS = [
    "experiment",
    "reference",
    "dataset",
    "rewriter",
    "only_variable",
    "corpus_documents",
    "corpus_chunks",
    "candidate_k",
    "total_cases",
    "evaluated_cases",
    "queries_changed_by_rewriter",
    "variants_identical",
    "deterministic",
    "results",
    "per_query",
    "mechanism_probe",
]

ROW_KEYS = [
    "id",
    "variant",
    "retriever",
    "fusion",
    "reranker",
    "candidate_k",
    "top_k",
    *(METRIC_FIELDS),
    "deterministic",
]

PER_QUERY_KEYS = [
    "query",
    "retrieval_query",
    "rewrite_identical",
    "relevant_chunk_ids",
    "baseline_top_5",
    "query_rewritten_top_5",
    "ranking_identical",
    "rank_of_first_relevant",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_k_changed",
    "config",
]


def _evidence():
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _semantic():
    return json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))


# --- A. Evidence: fixed schema, locked matrix --------------------------------


def test_evidence_schema_is_fixed():
    evidence = _evidence()
    assert list(evidence) == TOP_KEYS
    assert evidence["experiment"] == qr.EXPERIMENT_ID
    assert evidence["rewriter"] == "DeterministicQueryRewriter"
    assert evidence["only_variable"] == "query_rewriter"
    assert evidence["dataset"] == "benchmark/datasets/retrieval_small.json"
    assert evidence["corpus_documents"] == 6
    assert evidence["corpus_chunks"] == 33
    assert evidence["candidate_k"] == 33
    assert evidence["total_cases"] == evidence["evaluated_cases"] == 8
    assert evidence["deterministic"] is True


def test_matrix_is_the_locked_two_scenarios_by_two_variants_by_four_depths():
    rows = _evidence()["results"]
    assert len(rows) == 16
    assert all(list(row) == ROW_KEYS for row in rows)
    assert [row["id"] for row in rows] == [
        f"{config_id}-{variant}-t{top_k}"
        for config_id in ("Q1", "Q2")
        for variant in (qr.BASELINE_VARIANT, qr.REWRITTEN_VARIANT)
        for top_k in qr.DEPTHS
    ]
    assert {row["fusion"] for row in rows} == {"weighted", "rrf"}
    assert all(row["reranker"] == "deterministic" for row in rows)
    assert all(row["candidate_k"] == 33 for row in rows)


def test_metrics_are_numeric_in_range():
    for row in _evidence()["results"]:
        for field in METRIC_FIELDS:
            value = row[field]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert 0.0 <= value <= 1.0
        assert row["deterministic"] is True


# --- B. The claim: the rewriter changes nothing -------------------------------


def test_variants_carry_identical_metrics_at_every_depth():
    rows = _evidence()["results"]
    paired = {}
    for row in rows:
        key = (row["fusion"], row["top_k"])
        paired.setdefault(key, {})[row["variant"]] = {
            f: row[f] for f in METRIC_FIELDS
        }
    assert len(paired) == 8  # 2 scenarios x 4 depths
    for key, variants in paired.items():
        assert variants[qr.BASELINE_VARIANT] == variants[qr.REWRITTEN_VARIANT], key


def test_evidence_records_no_change_honestly():
    evidence = _evidence()
    assert evidence["queries_changed_by_rewriter"] == 0
    assert evidence["variants_identical"] is True
    # The flag is a measurement, not a constant: the runner sets it False the
    # moment two variants disagree.
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    assert "variants_identical = False" in source


def test_q1_baseline_reproduces_the_frozen_phase10b_row():
    frozen = json.loads(
        (BENCH_DIR / "results" / "retrieval_small_baseline.json").read_text(encoding="utf-8")
    )
    row = next(r for r in _evidence()["results"] if r["id"] == "Q1-baseline-t33")
    for field in METRIC_FIELDS:
        assert row[field] == frozen[field], field


# --- C. Per-query records -----------------------------------------------------


def test_per_query_records_are_complete_and_unchanged():
    records = _evidence()["per_query"]
    assert len(records) == 16  # 2 scenarios x 8 cases
    assert all(list(record) == PER_QUERY_KEYS for record in records)
    for record in records:
        assert record["rewrite_identical"] is True
        assert record["retrieval_query"] == record["query"]
        assert record["ranking_identical"] is True
        assert record["hit_at_k_changed"] is False
        assert record["baseline_top_5"] == record["query_rewritten_top_5"]


def test_per_query_covers_every_dataset_query_once_per_scenario():
    dataset = json.loads(
        (BENCH_DIR / "datasets" / "retrieval_small.json").read_text(encoding="utf-8")
    )
    queries = [case["query"] for case in dataset["cases"]]
    records = _evidence()["per_query"]
    for config_id in ("Q1", "Q2"):
        recorded = [r["query"] for r in records if r["config"] == config_id]
        assert recorded == queries  # dataset order, nothing added or dropped


def test_per_query_hit_flags_agree_with_the_recorded_rank():
    for record in _evidence()["per_query"]:
        relevant = set(record["relevant_chunk_ids"])
        expected = {
            k: any(cid in relevant for cid in record["baseline_top_5"][:k])
            for k in (1, 3, 5)
        }
        assert (record["hit_at_1"], record["hit_at_3"], record["hit_at_5"]) == (
            expected[1],
            expected[3],
            expected[5],
        )


# --- D. Mechanism probe: what happens when the rewriter fires -----------------


def test_mechanism_probe_reports_its_measurements():
    probe = _evidence()["mechanism_probe"]
    assert probe["probes"] == 8
    assert probe["probes_where_rewrite_changed_the_string"] == 8  # it does fire
    assert probe["probes_where_pipeline_ranking_changed"] == 0
    assert probe["probes_where_lexical_ranking_changed"] == 0
    # DeterministicEmbedder hashes raw text, so the dense ordering does move;
    # recorded as measured rather than smoothed over.
    assert probe["probes_where_dense_ranking_changed"] == 8


def test_mechanism_probe_queries_are_derived_diagnostics_not_dataset_edits():
    dataset = json.loads(
        (BENCH_DIR / "datasets" / "retrieval_small.json").read_text(encoding="utf-8")
    )
    originals = [case["query"] for case in dataset["cases"]]
    probe = _evidence()["mechanism_probe"]
    assert [r["query"] for r in probe["queries"]] == originals
    for record in probe["queries"]:
        # Each probe is the original with whitespace perturbed -- nothing else.
        assert " ".join(record["denormalized_query"].split()) == record["query"]
        assert record["retrieval_query"] == record["query"]
        assert record["rewrite_changed_the_string"] is True


# --- E. Cross-phase guard and faithfulness to the frozen evidence -------------


def test_q1_baseline_passes_the_phase10b_cross_phase_guard():
    # Recompute through the runner's own path and run its guard function.
    documents = qr._baseline.ingest_corpus()
    qr._baseline.verify_corpus(documents)
    _, chain = qr.build_pipeline(documents, fusion="weighted", rewriter=None)
    result = chain.run(qr._load_cases(), top_k=33, candidate_k=33)
    qr._check_q1_against_phase10b(result)  # raises RuntimeError on mismatch


def test_runner_builds_both_variants_from_one_function_with_one_variable():
    documents = qr._baseline.ingest_corpus()
    baseline, _ = qr.build_pipeline(documents, fusion="weighted", rewriter=None)
    rewritten, _ = qr.build_pipeline(
        documents, fusion="weighted", rewriter=qr.DeterministicQueryRewriter()
    )
    qr._assert_only_variable_differs(baseline, rewritten)  # raises on any drift
    assert baseline.query_rewriter is None
    assert isinstance(rewritten.query_rewriter, qr.DeterministicQueryRewriter)


def test_only_variable_guard_rejects_a_second_difference():
    documents = qr._baseline.ingest_corpus()
    baseline, _ = qr.build_pipeline(documents, fusion="weighted", rewriter=None)
    rewritten, _ = qr.build_pipeline(
        documents, fusion="weighted", rewriter=qr.DeterministicQueryRewriter()
    )
    # Swapped roles: the "baseline" side must not carry a rewriter.
    with pytest.raises(RuntimeError, match="baseline pipeline must have"):
        qr._assert_only_variable_differs(rewritten, rewritten)
    # A second difference is caught, not averaged away.
    rewritten.reranker = None
    with pytest.raises(RuntimeError, match="collaborator 'reranker' differs"):
        qr._assert_only_variable_differs(baseline, rewritten)


# --- F. Runner hygiene --------------------------------------------------------


def test_runner_loads_no_model_outside_the_opt_in_path():
    source = RUNNER_SCRIPT.read_text(encoding="utf-8")
    import_lines = [
        ln for ln in source.splitlines() if ln.startswith(("import ", "from "))
    ]
    # The semantic providers exist, but only inside the opt-in function.
    assert "from zmwiki.semantic import" in source
    assert "from zmwiki.semantic_rerank import" in source
    assert not any("semantic" in ln for ln in import_lines), import_lines
    banned = ("requests", "urllib", "httpx", "socket", "random", "uuid", "datetime")
    assert not any(any(token in ln for token in banned) for ln in import_lines), import_lines


def test_evidence_has_no_environment_leakage():
    raw = EVIDENCE_PATH.read_text(encoding="utf-8")
    for leaked in ("timestamp", "hostname", "username", "datetime", "uuid", "seed"):
        assert leaked not in raw.lower()
    assert "\\" not in raw  # no Windows absolute path
    assert "://" not in raw  # no URL / remote reference


# --- G. Reproducible from the runner script -----------------------------------


def test_runner_reproduces_committed_evidence_to_a_fresh_output(tmp_path):
    out = tmp_path / "rerun.json"
    proc = subprocess.run(
        [sys.executable, str(RUNNER_SCRIPT), str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == _evidence()


def test_main_reproduces_committed_evidence_in_process(tmp_path):
    fresh = qr.main(tmp_path / "in_process.json")
    assert fresh == _evidence()


# --- H. Opt-in semantic evidence ----------------------------------------------


def test_semantic_evidence_schema_and_identical_variants():
    evidence = _semantic()
    assert evidence["experiment"] == qr.SEMANTIC_EXPERIMENT_ID
    assert evidence["deterministic"] is False
    assert evidence["queries_changed_by_rewriter"] == 0
    assert evidence["variants_identical"] is True
    assert [row["variant"] for row in evidence["results"]] == [
        qr.BASELINE_VARIANT,
        qr.REWRITTEN_VARIANT,
    ]
    assert all(row["reranker"] == "cross_encoder" for row in evidence["results"])
    assert all(row["top_k"] == 8 for row in evidence["results"])
    for field in METRIC_FIELDS:
        assert evidence["results"][0][field] == evidence["results"][1][field], field


def test_semantic_baseline_reproduces_the_frozen_phase12_r2_row():
    frozen = json.loads(
        (BENCH_DIR / "results" / "retrieval_semantic_rerank_profile.json").read_text(
            encoding="utf-8"
        )
    )
    r2 = next(c for c in frozen["configs"] if c["id"] == "R2")
    baseline = _semantic()["results"][0]
    for field in METRIC_FIELDS:
        assert baseline[field] == r2[field], field


@pytest.mark.skipif(
    not os.environ.get("zmwiki_SEMANTIC_INTEGRATION"),
    reason="real-model integration; set zmwiki_SEMANTIC_INTEGRATION=1 to run",
)
def test_integration_real_semantic_query_rewriter_profile(tmp_path):
    """Re-run the real CrossEncoder scenario and compare it to the committed one."""
    fresh = qr.main_semantic(tmp_path / "semantic_rerun.json")
    committed = _semantic()
    assert fresh["queries_changed_by_rewriter"] == committed["queries_changed_by_rewriter"]
    assert fresh["variants_identical"] is True
    for field in METRIC_FIELDS:
        assert fresh["results"][0][field] == committed["results"][0][field], field
        assert fresh["results"][1][field] == committed["results"][1][field], field
